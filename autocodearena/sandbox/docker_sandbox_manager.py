"""
Docker-based sandbox manager for secure code execution.
This replaces the Firejail implementation with a cleaner Docker-based approach.
"""

import os
import sys
import time
import signal
import subprocess
import threading
import tempfile
import shutil
import socket
import json
import uuid
import logging
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import docker
from docker.errors import DockerException, ContainerError, ImageNotFound
import threading
import time
import signal

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Silence noisy urllib3 connection pool warnings
try:
    import logging as _logging
    _logging.getLogger("urllib3.connectionpool").setLevel(_logging.ERROR)
except Exception:
    pass

# Global port manager for thread-safe port allocation
_port_manager_lock = threading.Lock()
_allocated_ports = set()

# Global sandbox tracking for proper cleanup
_sandbox_registry_lock = threading.Lock()
_active_sandboxes: Dict[str, 'DockerSandbox'] = {}

# Docker client (initialized lazily)
_docker_client = None

# Limit concurrent container creations to avoid overwhelming Docker daemon
_creation_semaphore = threading.BoundedSemaphore(int(os.environ.get("SANDBOX_MAX_CONCURRENT_CREATES", "2")))

# Watchdog config (seconds)
_SANDBOX_TTL_SECONDS = int(os.environ.get("SANDBOX_TTL_SECONDS", "300"))  # max lifetime
_SANDBOX_IDLE_SECONDS = int(os.environ.get("SANDBOX_IDLE_SECONDS", "180"))  # max idle
_WATCHDOG_INTERVAL = int(os.environ.get("SANDBOX_WATCH_INTERVAL", "30"))
_watchdog_started = False

def reset_docker_client():
    """Reset the global Docker client (used after timeouts)."""
    global _docker_client
    try:
        if _docker_client is not None:
            try:
                # Close underlying pool if possible
                _docker_client.api.close()
            except Exception:
                pass
    finally:
        _docker_client = None

def get_docker_client():
    """Get or create Docker client with timeouts."""
    global _docker_client
    if _docker_client is None:
        try:
            _docker_client = docker.from_env(
                timeout=120,  # Increase timeout for slow daemon
                max_pool_size=32  # Allow more concurrent HTTP connections
            )
            # Test connection with timeout
            _docker_client.ping()
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Docker: {e}")
    return _docker_client

def find_available_port(start_port: int = 8000, max_attempts: int = 1000) -> int:
    """Find an available port starting from start_port with thread-safe allocation."""
    with _port_manager_lock:
        for port in range(start_port, start_port + max_attempts):
            if port not in _allocated_ports:
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.bind(('localhost', port))
                        _allocated_ports.add(port)
                        return port
                except OSError:
                    continue
        
        # Try dynamic allocation from user port range
        import random
        for _ in range(1000):
            port = random.randint(1024, 65535)
            if port not in _allocated_ports:
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.bind(('localhost', port))
                        _allocated_ports.add(port)
                        return port
                except OSError:
                    continue
        
        raise RuntimeError(f"Could not find available port")

def find_random_available_port() -> int:
    """Find a random available port in the user port range."""
    return find_available_port(1024, 1000)

def release_port(port: int):
    """Release a port back to the available pool."""
    with _port_manager_lock:
        _allocated_ports.discard(port)

def register_sandbox(sandbox: 'DockerSandbox'):
    """Register a sandbox in the global registry for tracking."""
    with _sandbox_registry_lock:
        _active_sandboxes[sandbox.sandbox_id] = sandbox

def unregister_sandbox(sandbox_id: str):
    """Unregister a sandbox from the global registry."""
    with _sandbox_registry_lock:
        _active_sandboxes.pop(sandbox_id, None)

@dataclass
class DockerImageConfig:
    """Configuration for Docker images used in sandboxes."""
    name: str
    dockerfile_content: str
    base_image: str = "ubuntu:22.04"

# Predefined Docker images for different languages
DOCKER_IMAGES = {
    "python": DockerImageConfig(
        name="bigcode_python",
        base_image="mcr.microsoft.com/playwright:v1.50.0-noble",
        dockerfile_content="""
FROM mcr.microsoft.com/playwright:v1.50.0-noble

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \\
    python3-pip \\
    python3-venv \\
    python3-dev \\
    build-essential \\
    curl \\
    git \\
    wget \\
    ca-certificates \\
    && rm -rf /var/lib/apt/lists/*

# Install pip first, then uv, create venv, and install packages with uv
RUN apt-get update && \\
    apt-get install -y python3-pip net-tools procps curl && \\
    python3 -m pip install --break-system-packages uv && \\
    uv venv /opt/venv && \\
    uv pip install --python /opt/venv/bin/python \\
    playwright==1.50.0 \\
    streamlit \\
    gradio \\
    fastapi \\
    uvicorn \\
    pandas \\
    numpy \\
    matplotlib \\
    seaborn \\
    plotly \\
    pillow \\
    requests \\
    aiohttp \\
    scikit-learn \\
    jupyter && \\
    rm -rf /var/lib/apt/lists/*

# Add venv to PATH so packages are available globally
ENV PATH="/opt/venv/bin:$PATH"

# Set environment variables to use pre-installed browsers
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

# Create python symlink for compatibility
RUN ln -s /usr/bin/python3 /usr/bin/python

# Create sandbox user with proper permissions
RUN useradd -m -s /bin/bash sandbox && \\
    chown -R sandbox:sandbox /home/sandbox

# Set working directory
WORKDIR /sandbox

# Switch to sandbox user
USER sandbox

# Add paths for installed packages
ENV PATH="/usr/local/bin:/home/sandbox/.local/bin:$PATH"
ENV PYTHONPATH="/usr/local/lib/python3.12/dist-packages:$PYTHONPATH"

# Default command
CMD ["/bin/bash"]
"""
    ),
    
    "node": DockerImageConfig(
        name="bigcode_node",
        base_image="mcr.microsoft.com/playwright:v1.50.0-noble",
        dockerfile_content="""
FROM mcr.microsoft.com/playwright:v1.50.0-noble

# Install Node.js 22
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \\
    apt-get install -y nodejs

# Install Python pip and uv, create venv, and install playwright with uv
RUN apt-get update && \\
    apt-get install -y python3-pip net-tools procps curl && \\
    python3 -m pip install --break-system-packages uv && \\
    uv venv /opt/venv && \\
    uv pip install --python /opt/venv/bin/python playwright==1.50.0 && \\
    rm -rf /var/lib/apt/lists/*

# Add venv to PATH so packages are available globally
ENV PATH="/opt/venv/bin:$PATH"

# Install common Node.js packages globally
RUN npm install -g \\
    create-react-app \\
    @vue/cli \\
    express \\
    typescript \\
    vite

# Set environment variables to use pre-installed browsers
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

# Create python symlink for compatibility
RUN ln -s /usr/bin/python3 /usr/bin/python

# Create sandbox user with proper permissions
RUN useradd -m -s /bin/bash sandbox && \\
    chown -R sandbox:sandbox /home/sandbox

# Set working directory
WORKDIR /sandbox

# Switch to sandbox user
USER sandbox

# Add paths for installed packages
ENV PATH="/usr/local/bin:/home/sandbox/.local/bin:$PATH"

# Default command
CMD ["/bin/bash"]
"""
    ),
    
    "multi": DockerImageConfig(
        name="bigcode_multi",
        base_image="mcr.microsoft.com/playwright:v1.50.0-noble",
        dockerfile_content="""
FROM mcr.microsoft.com/playwright:v1.50.0-noble

# Install system dependencies and programming languages
RUN apt-get update && apt-get install -y \\
    python3-pip \\
    python3-venv \\
    python3-dev \\
    build-essential \\
    curl \\
    git \\
    openjdk-11-jdk \\
    rustc \\
    golang-go \\
    gcc \\
    g++ \\
    wget \\
    ca-certificates \\
    ffmpeg \\
    && rm -rf /var/lib/apt/lists/*

# Install Node.js 22
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \\
    apt-get install -y nodejs

# Install pip first, then uv, create venv, and install Python packages with uv
RUN apt-get update && \\
    apt-get install -y python3-pip net-tools procps curl && \\
    python3 -m pip install --break-system-packages uv && \\
    uv venv /opt/venv && \\
    uv pip install --python /opt/venv/bin/python \\
    playwright==1.50.0 \\
    streamlit \\
    gradio \\
    fastapi \\
    uvicorn \\
    pandas \\
    numpy \\
    matplotlib \\
    pillow \\
    requests \\
    flask && \\
    rm -rf /var/lib/apt/lists/*

# Add venv to PATH so packages are available globally
ENV PATH="/opt/venv/bin:$PATH"

# Pre-install common Node.js packages globally
RUN npm install -g \\
    create-react-app \\
    @vue/cli \\
    express \\
    typescript \\
    vite

# Set environment variables to use pre-installed browsers
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

# Create python symlink for compatibility
RUN ln -s /usr/bin/python3 /usr/bin/python

# Create sandbox user with proper permissions
RUN useradd -m -s /bin/bash sandbox && \\
    chown -R sandbox:sandbox /home/sandbox

# Set working directory
WORKDIR /sandbox

# Switch to sandbox user
USER sandbox

# Add paths for installed packages (both global and user)
ENV PATH="/usr/local/bin:/home/sandbox/.local/bin:$PATH"
ENV PYTHONPATH="/usr/local/lib/python3.12/dist-packages:$PYTHONPATH"

# Default command
CMD ["/bin/bash"]
"""
    )
}

class DockerSandboxCommands:
    """Command execution interface for Docker sandbox."""
    
    def __init__(self, sandbox: 'DockerSandbox'):
        self.sandbox = sandbox
    
    def run(self, cmd: str, cwd: Optional[str] = None, 
            on_stdout=None, on_stderr=None, timeout: int = 120):
        """Run a command in the Docker container."""
        try:
            # Ensure container is healthy before running commands
            try:
                self.sandbox.ensure_running()
            except Exception:
                # One more silent attempt
                self.sandbox.ensure_running()
            self.sandbox.mark_used()
            if cwd is None:
                cwd = "/sandbox"
            
            # Execute command in the container
            exec_result = self.sandbox.container.exec_run(
                f"bash -c 'cd {cwd} && {cmd}'",
                stdout=True,
                stderr=True,
                stream=True,
                user="sandbox"
            )
            
            stdout_lines = []
            stderr_lines = []
            
            # Stream output
            for line in exec_result.output:
                line_str = line.decode('utf-8', errors='ignore').rstrip()
                if line_str:
                    # Docker doesn't separate stdout/stderr in exec_run stream
                    # For simplicity, we'll treat all output as stdout
                    stdout_lines.append(line_str)
                    if on_stdout:
                        on_stdout(line_str)
            
            # Get exit code
            exit_code = exec_result.exit_code
            
            return CommandResult(exit_code, '\n'.join(stdout_lines), '\n'.join(stderr_lines))
            
        except Exception as e:
            raise RuntimeError(f"Failed to run command: {e}")

class DockerSandboxFiles:
    """File operations interface for Docker sandbox."""
    
    def __init__(self, sandbox: 'DockerSandbox'):
        self.sandbox = sandbox
    
    def make_dir(self, path: str):
        """Create a directory in the sandbox."""
        # Create directory on host side with proper permissions
        full_path = os.path.join(self.sandbox.working_dir, path)
        os.makedirs(full_path, exist_ok=True)
        
        # Fix permissions for the sandbox user
        try:
            os.chown(full_path, 1000, 1000)
            os.chmod(full_path, 0o755)
        except (OSError, PermissionError):
            # If we can't change ownership, at least make it accessible
            os.chmod(full_path, 0o777)
    
    def write(self, path: str, content: str):
        """Write content to a file in the sandbox."""
        # Use the mounted volume to write files
        full_path = os.path.join(self.sandbox.working_dir, path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Fix permissions - make writable by everyone (safe for temp dir)
        try:
            os.chmod(full_path, 0o666)
        except (OSError, PermissionError):
            pass

class CommandResult:
    """Result of a command execution."""
    
    def __init__(self, exit_code: int, stdout: str, stderr: str):
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr

class DockerSandbox:
    """A Docker-based sandbox for secure code execution."""
    
    def __init__(self, sandbox_id: str, image_type: str = "multi"):
        self.sandbox_id = sandbox_id
        self.image_type = image_type
        self.working_dir = tempfile.mkdtemp(prefix=f"docker_sandbox_{sandbox_id}_")
        self.container = None
        self.allocated_port: Optional[int] = None
        self.commands = DockerSandboxCommands(self)
        self.files = DockerSandboxFiles(self)
        self.created_at = time.time()
        self.last_used_at = self.created_at
        
        # Fix working directory permissions - make it writable by everyone (safe for temp dir)
        try:
            os.chmod(self.working_dir, 0o777)
        except (OSError, PermissionError):
            pass
        # Ensure Docker image exists
        self._ensure_image()
        # Create and start container
        self._create_container()
        # Register this sandbox for tracking
        register_sandbox(self)
        _maybe_start_watchdog()
    
    def _ensure_image(self):
        """Ensure the required Docker image exists."""
        client = get_docker_client()
        image_config = DOCKER_IMAGES[self.image_type]
        
        try:
            # Check if image exists - no building, just verification
            client.images.get(image_config.name)  # Check if image exists
        except Exception as e:
            error_msg = f"Docker image {image_config.name} not found. Please run 'python build_docker_images.py' first to build all required images."
            print(f"❌ {error_msg}")
            raise RuntimeError(error_msg)
    
    def _create_container(self):
        """Create and start the Docker container."""
        client = get_docker_client()
        image_config = DOCKER_IMAGES[self.image_type]
        
        try:
            # Gate container creations to avoid daemon overload
            _creation_semaphore.acquire()
            # Container configuration with reduced resource limits for parallel processing
            container_config = {
                'image': image_config.name,
                'detach': True,
                'auto_remove': True,  # Automatically remove when stopped
                'volumes': {
                    self.working_dir: {
                        'bind': '/sandbox',
                        'mode': 'rw'
                    }
                },
                'working_dir': '/sandbox',
                'user': 'sandbox',  # Back to original sandbox user
                'network_mode': 'bridge',  # Allow network access but isolated
                'mem_limit': '512m',  # Reduced from 1g to 512m for parallel processing
                'cpu_quota': 50000,  # Reduced from 100000 to 50000 (50% of one core)
                'cpu_period': 100000,
                'security_opt': ['no-new-privileges:true'],  # Security constraint
                'cap_drop': ['ALL'],  # Drop all capabilities
                'cap_add': ['CHOWN', 'SETUID', 'SETGID'],  # Add only necessary capabilities
                'read_only': False,  # Allow writes to mounted volume
                'tmpfs': {'/tmp': 'size=50M,exec'},  # Reduced from 100M to 50M
                'command': 'sleep infinity',  # Keep container running
                'publish_all_ports': True  # Expose all ports to the host environment
            }
            
            # Create container with timeout protection
            import threading
            import time
            
            container_created = [False]
            container_exception = [None]
            
            def create_container():
                try:
                    # Create container first, then start it
                    self.container = client.containers.create(**container_config)
                    self.container.start()
                    
                    # Verify container is running
                    self.container.reload()
                    if self.container.status != 'running':
                        raise RuntimeError(f"Container failed to start. Status: {self.container.status}")
                    
                    container_created[0] = True
                except Exception as e:
                    container_exception[0] = e
            
            # Run container creation in a thread with timeout
            thread = threading.Thread(target=create_container)
            thread.daemon = True
            thread.start()
            
            # Wait for completion or timeout
            thread.join(timeout=60)  # 60 second timeout for container creation
            
            if thread.is_alive():
                # Timeout occurred - reset client and raise
                try:
                    reset_docker_client()
                except Exception:
                    pass
                raise RuntimeError("Container creation timed out after 60 seconds")
            
            if container_exception[0]:
                raise container_exception[0]
            
            if not container_created[0]:
                raise RuntimeError("Container creation failed")
            
        except Exception as e:
            raise RuntimeError(f"Failed to create Docker container: {e}")
        finally:
            try:
                _creation_semaphore.release()
            except Exception:
                pass
    
    def ensure_running(self, max_attempts: int = 3):
        """Ensure the sandbox container is running; retry creating if needed without noisy logs."""
        import time as _time
        attempts = 0
        while attempts < max_attempts:
            try:
                if self.container is not None:
                    try:
                        self.container.reload()
                        if self.container.status == 'running':
                            return
                    except Exception:
                        pass
                # (Re)create container if not running
                self._create_container()
                return
            except Exception:
                backoff = min(2 ** attempts, 10)
                _time.sleep(backoff)
                attempts += 1
        # Final attempt; let exception bubble if it fails
        self._create_container()
    
    def get_host(self) -> str:
        """Get the host identifier for this sandbox."""
        return "localhost"
    
    def set_allocated_port(self, port: int):
        """Set the allocated port for this sandbox."""
        self.allocated_port = port
        self.last_used_at = time.time()
    
    def kill(self):
        """Stop and clean up the Docker container."""
        try:
            # Stop and remove container
            if self.container:
                # First fix permissions while container is still running
                try:
                    self.container.exec_run(f"chown -R 1000:1000 /sandbox", user="root")
                    self.container.exec_run(f"chmod -R 755 /sandbox", user="root")
                except Exception:
                    pass
                
                # Stop the container
                try:
                    self.container.stop(timeout=5)
                except Exception as e:
                    # Force kill if stop fails
                    try:
                        self.container.kill()
                    except Exception:
                        pass
                
                # Explicitly remove the container (auto_remove doesn't always work)
                try:
                    self.container.remove(force=True)
                except Exception:
                    pass
                    
                self.container = None
            
            # Release allocated port
            if self.allocated_port is not None:
                release_port(self.allocated_port)
                self.allocated_port = None
            
            # Clean up working directory
            if os.path.exists(self.working_dir):
                try:
                    # Try to fix permissions on host side - be more aggressive
                    try:
                        import subprocess
                        # First try to change ownership of directory itself
                        subprocess.run(['sudo', 'chown', '-R', f'{os.getuid()}:{os.getgid()}', self.working_dir], 
                                     check=False, capture_output=True, timeout=10)
                        # Then fix permissions
                        subprocess.run(['chmod', '-R', '755', self.working_dir], 
                                     check=False, capture_output=True, timeout=10)
                    except:
                        pass
                    
                    shutil.rmtree(self.working_dir)
                except Exception as e:
                    # Try force removal with sudo
                    try:
                        import subprocess
                        subprocess.run(['sudo', 'rm', '-rf', self.working_dir], 
                                     check=False, capture_output=True, timeout=10)
                    except:
                        pass
            
            # Unregister from global registry
            unregister_sandbox(self.sandbox_id)
            
        except Exception as e:
            pass

    def mark_used(self):
        self.last_used_at = time.time()

def check_docker_health():
    """Check if Docker daemon is healthy and responsive."""
    try:
        client = get_docker_client()
        # Quick health check
        client.ping()
        
        # Check if Docker daemon is responsive
        containers = client.containers.list(limit=1)
        return True
    except Exception:
        return False

def wait_for_docker_health(max_wait: int = 30):
    """Wait for Docker daemon to become healthy."""
    import time
    
    start_time = time.time()
    while time.time() - start_time < max_wait:
        if check_docker_health():
            return True
        time.sleep(2)
    return False

def create_sandbox_with_retry(image_type: str = "multi", max_retries: int = 3) -> DockerSandbox:
    """Create a new Docker sandbox with exponential backoff retry."""
    import time
    
    for attempt in range(max_retries):
        try:
            # Wait for Docker to be healthy before creating container
            if not wait_for_docker_health():
                # Silence warnings per user request; still try to proceed
                pass
            
            return DockerSandbox(str(uuid.uuid4()), image_type)
        except Exception as e:
            if "timeout" in str(e).lower() or "read timed out" in str(e).lower():
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # 1, 2, 4 seconds
                    # Don't print noisy logs; backoff silently
                    time.sleep(wait_time)
                    continue
            raise e
    
    raise RuntimeError("Failed to create sandbox after all retries")

def create_sandbox(image_type: str = "multi") -> DockerSandbox:
    """Create a new Docker sandbox with retry logic."""
    return create_sandbox_with_retry(image_type)

def reuse_or_create_sandbox(sandbox_id: Optional[str] = None, image_type: str = "multi") -> DockerSandbox:
    """Create a new Docker sandbox (reuse not implemented for simplicity)."""
    # Clean up excess sandboxes to prevent resource exhaustion
    # cleanup_excess_sandboxes()
    
    # Always create fresh containers for security and simplicity
    return create_sandbox(image_type)

def cleanup_excess_sandboxes(max_sandboxes: int = 5):
    """Clean up excess sandboxes to prevent resource exhaustion in parallel."""
    with _sandbox_registry_lock:
        active_sandboxes = list(_active_sandboxes.values())
        
        if len(active_sandboxes) <= max_sandboxes:
            return
        
        excess_count = len(active_sandboxes) - max_sandboxes
        oldest_sandboxes = active_sandboxes[:excess_count]
        
        if not oldest_sandboxes:
            return
        
        # Use parallel cleanup for excess sandboxes
        import concurrent.futures
        import threading
        
        def kill_excess_sandbox(sandbox):
            """Kill a single excess sandbox with error handling"""
            try:
                sandbox.kill()
                return True
            except Exception as e:
                return False
        
        # Use reasonable number of workers for excess cleanup
        max_workers = min(len(oldest_sandboxes), 8, threading.active_count() + 2)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all kill tasks
            future_to_sandbox = {executor.submit(kill_excess_sandbox, sandbox): sandbox 
                               for sandbox in oldest_sandboxes}
            
            # Wait for all to complete
            killed_count = 0
            for future in concurrent.futures.as_completed(future_to_sandbox):
                try:
                    if future.result():
                        killed_count += 1
                except Exception:
                    pass
            
            logger.info(f"Cleaned up {killed_count}/{excess_count} excess sandboxes in parallel")

def run_command_in_sandbox(
    sandbox: DockerSandbox,
    command: str,
    working_directory: Optional[str] = None,
    timeout: int = 120,
    print_output: bool = False,
) -> Tuple[bool, List[str], List[str]]:
    """Run a command in the Docker sandbox."""
    is_run_success = False
    stdouts: List[str] = []
    stderrs: List[str] = []
    
    def on_stdout(message: str):
        stdouts.append(message)
            
    def on_stderr(message: str):
        stderrs.append(message)
    
    try:
        command_result = sandbox.commands.run(
            cmd=command,
            cwd=working_directory,
            on_stdout=on_stdout,
            on_stderr=on_stderr,
            timeout=timeout
        )
        if command_result and command_result.exit_code == 0:
            is_run_success = True
    except Exception as e:
        stderrs.append(str(e))
        is_run_success = False
    
    return is_run_success, stdouts, stderrs

def run_background_command_with_timeout(
    sandbox: DockerSandbox,
    command: str,
    timeout: int = 30,
    custom_env: dict = None
) -> str:
    
    try:
        # Ensure container is healthy before running background command
        try:
            sandbox.ensure_running()
        except Exception:
            # Second silent attempt
            sandbox.ensure_running()
        sandbox.mark_used()
        # Set environment variables
        env_vars = ""
        if custom_env:
            env_vars = " ".join([f"{k}={v}" for k, v in custom_env.items()])
            command = f"env {env_vars} {command}"
        
        # Run command in background
        exec_result = sandbox.container.exec_run(
            f"bash -c 'cd /sandbox && nohup {command} > /tmp/bg_stdout.log 2> /tmp/bg_stderr.log & echo $! > /tmp/bg_pid.txt'",
            detach=True,
            user="sandbox"
        )
        
        # Wait for the timeout period to check for startup errors
        # Use the full timeout instead of just 5 seconds
        time.sleep(min(timeout, 10))  # Increased from 5 to 10 seconds
        
        # Check for stderr output continuously during the timeout
        stderr_output = ""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                # Check if process is still running
                pid_result = sandbox.container.exec_run(
                    "cat /tmp/bg_pid.txt 2>/dev/null || echo ''",
                    user="sandbox"
                )
                pid = pid_result.output.decode('utf-8', errors='ignore').strip()
                
                if pid and pid.isdigit():
                    # Check if process is still running
                    ps_result = sandbox.container.exec_run(
                        f"ps -p {pid} >/dev/null 2>&1; echo $?",
                        user="sandbox"
                    )
                    if ps_result.output.decode('utf-8', errors='ignore').strip() != "0":
                        # Process died, read stderr immediately
                        break
                
                # Read current stderr content
                stderr_result = sandbox.container.exec_run(
                    "cat /tmp/bg_stderr.log",
                    user="sandbox"
                )
                current_stderr = stderr_result.output.decode('utf-8', errors='ignore')
                if current_stderr:
                    stderr_output = current_stderr
                
                # Also check stdout for Gradio errors
                stdout_result = sandbox.container.exec_run(
                    "cat /tmp/bg_stdout.log",
                    user="sandbox"
                )
                current_stdout = stdout_result.output.decode('utf-8', errors='ignore')
                if "error" in current_stdout.lower() or "exception" in current_stdout.lower():
                    stderr_output += f"\nSTDOUT: {current_stdout}"
                
                time.sleep(1)  # Check every second
                
            except Exception:
                break
        
        return stderr_output
            
    except Exception as e:
        return f"Error running background command: {str(e)}"

def install_python_dependencies(sandbox: DockerSandbox, dependencies: List[str]) -> List[str]:
    """Install Python dependencies in the Docker container using uv."""
    if not dependencies:
        return []
    
    install_errors = []
    
    for dep in dependencies:
        if not dep.strip():
            continue
            
        try:
            # Try uv pip first as sandbox user
            result = sandbox.container.exec_run(
                f"uv pip install --python /opt/venv/bin/python {dep}",
                user="sandbox"
            )
            
            # If that fails with permission error, try as root with uv
            if result.exit_code != 0:
                error_msg = result.output.decode('utf-8', errors='ignore')
                if "Permission denied" in error_msg or "permission denied" in error_msg.lower():
                    # Try installing as root using uv with the virtual environment
                    result = sandbox.container.exec_run(
                        f"uv pip install --python /opt/venv/bin/python {dep}",
                        user="root"
                    )
                    
                    if result.exit_code != 0:
                        error_msg = result.output.decode('utf-8', errors='ignore')
                        install_errors.append(f"Failed to install {dep}: {error_msg}")
                else:
                    install_errors.append(f"Failed to install {dep}: {error_msg}")
                
        except Exception as e:
            install_errors.append(f"Error installing {dep}: {e}")
    
    return install_errors

def install_npm_dependencies(sandbox: DockerSandbox, dependencies: List[str]) -> List[str]:
    """Install npm dependencies in the Docker container."""
    if not dependencies:
        return []
    
    install_errors = []
    
    # Filter out empty dependencies
    valid_dependencies = [dep.strip() for dep in dependencies if dep.strip()]
    
    if not valid_dependencies:
        return install_errors
    
    try:
        # Install all dependencies at once with better dependency management
        # --save-exact: Locks exact versions to prevent future conflicts
        # --legacy-peer-deps: Better peer dependency resolution
        # --force: Ensures installation even with conflicts
        deps_string = " ".join(valid_dependencies)
        result = sandbox.container.exec_run(
            f"npm install {deps_string} --save-exact --force --no-audit --no-fund",
            user="sandbox",
            workdir="/sandbox"
        )
        
        if result.exit_code != 0:
            error_msg = result.output.decode('utf-8', errors='ignore')
            install_errors.append(f"Failed to install dependencies: {error_msg}")
            
            # Try fallback installation with more permissive flags
            fallback_result = sandbox.container.exec_run(
                f"npm install {deps_string} --save --force --no-audit --no-fund",
                user="sandbox",
                workdir="/sandbox"
            )
            
            if fallback_result.exit_code != 0:
                fallback_error = fallback_result.output.decode('utf-8', errors='ignore')
                install_errors.append(f"Fallback installation also failed: {fallback_error}")
            else:
                print(f"Fallback installation succeeded for {len(valid_dependencies)} npm packages")

            
    except Exception as e:
        print(f"Error during npm install: {e}")
        install_errors.append(f"Error during npm install: {e}")
    
    return install_errors

def get_sandbox_app_url(sandbox: DockerSandbox, port: int) -> str:
    """Get the URL for a sandbox application."""
    # check the port is available in the container
    try:
        # Use curl to check if the port is responding
        result = sandbox.container.exec_run(
            f"curl -s --connect-timeout 1 --max-time 2 http://localhost:{port}",
            user="sandbox"
        )
        
        # If curl succeeds (exit code 0), the port is running
        if result.exit_code == 0:
            return f"http://localhost:{port}"
        else:
            return ""
            
    except Exception:
        # If there's any error checking the port, assume it's not running
        return ""

def take_internal_screenshot(sandbox: DockerSandbox, port: int, screenshot_filename: str, wait_time: int = 5) -> str:
    """
    Take a screenshot from inside the Docker container using Playwright.
    
    Args:
        sandbox: The Docker sandbox instance
        port: The port where the web app is running inside the container
        screenshot_filename: Name of the screenshot file to save
        wait_time: Time to wait for the page to load
    
    Returns:
        str: Path to the saved screenshot on the host system
    """
    import uuid
    # Generate unique script name to avoid conflicts
    script_id = str(uuid.uuid4())[:8]
    script_name = f"screenshot_{script_id}.py"
    
    # Create Python script to run Playwright inside container
    screenshot_script = f'''
import asyncio
import sys
import os
import time
import stat
import shutil

async def take_screenshot():
    try:
        # Use playwright with the pre-installed browsers from the official image
        try:
            from playwright.async_api import async_playwright
            
            async with async_playwright() as p:
                browser = await p.chromium.launch(
                    headless=True,
                    args=[
                        '--no-sandbox',
                        '--disable-dev-shm-usage',
                        '--disable-gpu',
                        '--disable-web-security',
                        '--disable-features=VizDisplayCompositor',
                        '--disable-background-timer-throttling',
                        '--disable-backgrounding-occluded-windows',
                        '--disable-renderer-backgrounding'
                    ]
                )
                
                page = await browser.new_page(viewport={{'width': 1024, 'height': 1024}})
                
                # Navigate to the local app
                url = "http://localhost:{port}"
                
                try:
                    await page.goto(url, wait_until='domcontentloaded', timeout=15000)
                    
                    # Wait for content to load
                    await page.wait_for_timeout({wait_time * 1000})
                    
                    # Save screenshot directly to /tmp (always writable)
                    temp_path = "/tmp/{screenshot_filename}"
                    await page.screenshot(path=temp_path, full_page=True)
                    
                except Exception as e:
                    # Try to save anyway with a simple page
                    try:
                        await page.goto("data:text/html,<h1>Screenshot Test</h1>")
                        await page.screenshot(path="/sandbox/{screenshot_filename}")
                    except:
                        pass
                    sys.exit(1)
                finally:
                    await browser.close()
                    
        except ImportError:
            # Fallback to simple browser command - try different browser names
            import subprocess
            
            browser_names = ['chromium', 'chromium-browser', 'google-chrome']
            for browser in browser_names:
                try:
                    result = subprocess.run([
                        browser, '--headless', '--disable-gpu', '--no-sandbox',
                        '--virtual-time-budget=5000', '--run-all-compositor-stages-before-draw',
                        f'--screenshot=/sandbox/{screenshot_filename}',
                        f'http://localhost:{port}'
                    ], capture_output=True, text=True, timeout=30)
                    
                    if result.returncode == 0:
                        break
                except Exception as e:
                    continue
            else:
                sys.exit(1)
                
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(take_screenshot())
'''
    
    # Write script directly inside container to avoid permission issues
    # Use base64 encoding to safely write the script inside the container
    import base64
    encoded_script = base64.b64encode(screenshot_script.encode('utf-8')).decode('ascii')
    
    # Write script to /tmp inside container (always writable)
    tmp_script_path = f"/tmp/{script_name}"
    write_command = f"""python3 -c "
import base64
content = base64.b64decode('{encoded_script}').decode('utf-8')
with open('{tmp_script_path}', 'w') as f:
    f.write(content)
" """
    
    result = sandbox.container.exec_run(
        write_command,
        user="root"
    )

    if result.exit_code != 0:
        return None

    # Execute screenshot script inside container
    try:
        result = sandbox.container.exec_run(
            f"python3 {tmp_script_path}",
            user="root",  # Use root to avoid permission issues
            environment=[
                'DISPLAY=:99',  # Virtual display for headless mode
                'PLAYWRIGHT_BROWSERS_PATH=/ms-playwright'  # Use pre-installed browsers
            ]
        )
        # Check if screenshot was created
        host_screenshot_path = os.path.join(sandbox.working_dir, screenshot_filename)
        if os.path.exists(host_screenshot_path):
            return host_screenshot_path
        else:
            # Try to copy from container's /tmp to volume mount
            try:
                # First check if file exists in /tmp
                check_result = sandbox.container.exec_run(f"ls -la /tmp/{screenshot_filename}", user="root")
                
                # Change ownership of /sandbox to root temporarily
                chown_result = sandbox.container.exec_run(f"chown root:root /sandbox", user="root")
                
                # Copy file
                copy_result = sandbox.container.exec_run(f"cp /tmp/{screenshot_filename} /sandbox/", user="root")
                
                if copy_result.exit_code == 0:
                    # Change ownership back and make readable
                    sandbox.container.exec_run(f"chmod 644 /sandbox/{screenshot_filename}", user="root")
                    sandbox.container.exec_run(f"chown 1000:1000 /sandbox/{screenshot_filename}", user="root")
                    
                    # Wait a moment for file system to sync
                    import time
                    time.sleep(0.1)
                    
                    if os.path.exists(host_screenshot_path):
                        # Ensure host can read the file
                        try:
                            os.chmod(host_screenshot_path, 0o644)
                        except (OSError, PermissionError):
                            pass
                        return host_screenshot_path
            except Exception as e:
                print(f"Error copying screenshot: {e}")
                pass
            
            return None
            
    except Exception as e:
        return None
    finally:
        # Clean up script file
        try:
            sandbox.container.exec_run(f"rm -f /sandbox/{script_name}", user="sandbox")
        except:
            pass

def get_sandbox_stats() -> dict:
    """Get statistics about active sandboxes."""
    with _sandbox_registry_lock:
        active_count = len(_active_sandboxes)
        
        return {
            'active_sandboxes': active_count,
            'allocated_ports': len(_allocated_ports)
        }

def _maybe_start_watchdog():
    global _watchdog_started
    if _watchdog_started:
        return
    _watchdog_started = True

    def _watchdog_loop():
        while True:
            try:
                now = time.time()
                with _sandbox_registry_lock:
                    sandboxes = list(_active_sandboxes.values())
                for sb in sandboxes:
                    try:
                        # Kill if exceeded TTL
                        if _SANDBOX_TTL_SECONDS > 0 and (now - sb.created_at) > _SANDBOX_TTL_SECONDS:
                            sb.kill()
                            continue
                        # Kill if idle too long
                        if _SANDBOX_IDLE_SECONDS > 0 and (now - sb.last_used_at) > _SANDBOX_IDLE_SECONDS:
                            sb.kill()
                            continue
                        # Kill if container not running
                        try:
                            if sb.container is None:
                                continue
                            sb.container.reload()
                            if sb.container.status != 'running':
                                sb.kill()
                        except Exception:
                            try:
                                sb.kill()
                            except Exception:
                                pass
                    except Exception:
                        pass
            except Exception:
                pass
            time.sleep(max(5, _WATCHDOG_INTERVAL))

    t = threading.Thread(target=_watchdog_loop, daemon=True)
    t.start()

def kill_all_sandboxes():
    """Kill all active sandboxes in parallel for faster cleanup."""
    with _sandbox_registry_lock:
        sandboxes_to_kill = list(_active_sandboxes.values())
    
    if not sandboxes_to_kill:
        return
    
    # Use ThreadPoolExecutor for parallel cleanup
    import concurrent.futures
    import threading
    
    def kill_single_sandbox(sandbox):
        """Kill a single sandbox with error handling"""
        try:
            sandbox.kill()
            return True
        except Exception as e:
            return False
    
    # Use reasonable number of workers (don't overwhelm the system)
    max_workers = min(len(sandboxes_to_kill), 16, threading.active_count() + 4)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all kill tasks
        future_to_sandbox = {executor.submit(kill_single_sandbox, sandbox): sandbox 
                           for sandbox in sandboxes_to_kill}
        
        # Wait for all to complete
        killed_count = 0
        for future in concurrent.futures.as_completed(future_to_sandbox):
            try:
                if future.result():
                    killed_count += 1
            except Exception:
                pass
        
        logger.info(f"Killed {killed_count}/{len(sandboxes_to_kill)} sandboxes in parallel")

def cleanup_docker_resources():
    """Clean up Docker resources (containers, images, etc.) in parallel for faster cleanup."""
    try:
        client = get_docker_client()
        
        # Remove stopped, created, and running containers in parallel
        # Handle each status separately since Docker API filter doesn't work well with multiple statuses
        sandbox_containers = []
        
        # Get exited containers
        exited_containers = client.containers.list(all=True, filters={'status': 'exited'})
        for container in exited_containers:
            if any(image_name in str(container.image.tags) for image_name in [img.name for img in DOCKER_IMAGES.values()]):
                sandbox_containers.append(container)
        
        # Get created containers
        created_containers = client.containers.list(all=True, filters={'status': 'created'})
        for container in created_containers:
            if any(image_name in str(container.image.tags) for image_name in [img.name for img in DOCKER_IMAGES.values()]):
                sandbox_containers.append(container)
        
        # Get running containers (these need to be stopped first)
        running_containers = client.containers.list(filters={'status': 'running'})
        for container in running_containers:
            if any(image_name in str(container.image.tags) for image_name in [img.name for img in DOCKER_IMAGES.values()]):
                sandbox_containers.append(container)
        
        if sandbox_containers:
            import concurrent.futures
            import threading
            
            def remove_single_container(container):
                """Remove a single container with error handling"""
                try:
                    # Stop running containers first, then remove
                    if container.status == 'running':
                        container.stop(timeout=5)
                    container.remove()
                    return True
                except Exception as e:
                    logger.warning(f"Failed to remove container {container.name}: {e}")
                    return False
            
            # Use reasonable number of workers
            max_workers = min(len(sandbox_containers), 8, threading.active_count() + 2)
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all removal tasks
                future_to_container = {executor.submit(remove_single_container, container): container 
                                     for container in sandbox_containers}
                
                # Wait for all to complete
                removed_count = 0
                for future in concurrent.futures.as_completed(future_to_container):
                    try:
                        if future.result():
                            removed_count += 1
                    except Exception:
                        pass
                
                logger.info(f"Removed {removed_count}/{len(sandbox_containers)} containers (stopped/created/running) in parallel")
        
        # Clean up volumes and networks (these are fast operations, no need for parallelization)
        try:
            client.volumes.prune()
        except Exception as e:
            print(f"Failed to prune volumes: {e}")
        
    except Exception as e:
        logger.warning(f"Error during Docker resource cleanup: {e}")

def cleanup_containers_by_pattern(pattern: str, max_workers: int = 8):
    """Clean up Docker containers matching a pattern in parallel."""
    try:
        client = get_docker_client()
        
        # Find containers matching the pattern
        containers = client.containers.list(all=True)
        matching_containers = []
        
        for container in containers:
            if pattern in container.name or pattern in str(container.image.tags):
                matching_containers.append(container)
        
        if not matching_containers:
            logger.info(f"No containers found matching pattern: {pattern}")
            return 0
        
        # Use parallel cleanup
        import concurrent.futures
        import threading
        
        def remove_container(container):
            """Remove a single container with error handling"""
            try:
                if container.status == 'running':
                    container.stop(timeout=5)
                container.remove(force=True)
                return True
            except Exception as e:
                return False
        
        # Use reasonable number of workers
        actual_workers = min(len(matching_containers), max_workers, threading.active_count() + 2)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=actual_workers) as executor:
            # Submit all removal tasks
            future_to_container = {executor.submit(remove_container, container): container 
                                 for container in matching_containers}
            
            # Wait for all to complete
            removed_count = 0
            for future in concurrent.futures.as_completed(future_to_container):
                try:
                    if future.result():
                        removed_count += 1
                except Exception:
                    pass
            
            logger.info(f"Cleaned up {removed_count}/{len(matching_containers)} containers matching '{pattern}' in parallel")
            return removed_count
            
    except Exception as e:
        logger.error(f"Error during pattern-based container cleanup: {e}")
        return 0

# Cleanup handler for graceful shutdown
def signal_handler(signum, frame):
    """Handle shutdown signals."""
    kill_all_sandboxes()
    cleanup_docker_resources()
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
