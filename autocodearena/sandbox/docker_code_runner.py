"""
Docker-based code runner for secure code execution.
This replaces the local_code_runner.py Firejail implementation with Docker containers.
"""

from typing import Any, Generator, Literal, TypeAlias, TypedDict, Set, Tuple, List, Optional
import uuid
import base64
import os
import subprocess
import time
import threading
import tempfile
import shutil
import html
import requests
import json
import glob
import mimetypes
from pathlib import Path

from sandbox.docker_sandbox_manager import (
    DockerSandbox, create_sandbox, reuse_or_create_sandbox,
    run_command_in_sandbox, run_background_command_with_timeout,
    install_python_dependencies, install_npm_dependencies,
    get_sandbox_app_url, find_random_available_port, take_internal_screenshot,
    logger
)
from sandbox.code_analyzer import (
    SandboxEnvironment, extract_code_from_markdown, 
    extract_installation_commands, extract_java_class_name, 
    extract_js_imports, extract_python_imports, 
    replace_placeholder_urls, validate_dependencies
)
from sandbox.constants import PYTHON_EXECUTABLE, PYTHON_HTTP_SERVER_MODULE

# Constants

def exec_run_with_timeout(container, command, timeout_seconds, **kwargs):
    """
    Execute a command in a Docker container with timeout handling.
    
    Args:
        container: Docker container object
        command: Command to execute
        timeout_seconds: Timeout in seconds
        **kwargs: Additional arguments for exec_run
        
    Returns:
        exec_run result or None if timeout
    """
    import threading
    import time
    
    result = [None]
    exception = [None]
    
    def execute():
        try:
            result[0] = container.exec_run(command, **kwargs)
        except Exception as e:
            exception[0] = e
    
    # Start execution in a thread
    thread = threading.Thread(target=execute)
    thread.daemon = True
    thread.start()
    
    # Wait for completion or timeout
    thread.join(timeout_seconds)
    
    if thread.is_alive():
        # Timeout occurred - the thread will continue running but we return None
        # The container will be cleaned up when the sandbox is killed
        # This is safer than stopping/starting the container
        return None
    
    if exception[0]:
        raise exception[0]
    
    return result[0]

def detect_visual_outputs_in_directory(sandbox_dir: str, initial_files: set) -> list:
    """
    Detect visual outputs generated in the sandbox directory.
    
    Args:
        sandbox_dir: Path to the sandbox directory
        initial_files: Set of files that existed before code execution
        
    Returns:
        List of visual output dictionaries with 'type' and 'data' keys
    """
    visual_outputs = []
    
    # Get all files in the directory after execution
    current_files = set()
    for root, dirs, files in os.walk(sandbox_dir):
        for file in files:
            current_files.add(os.path.join(root, file))
    
    # Find newly created files
    new_files = current_files - initial_files
    
    # Simple content-based deduplication across files detected in this run
    seen_digests = set()
    for file_path in new_files:
        try:
            # Determine file type
            mime_type, _ = mimetypes.guess_type(file_path)
            file_ext = Path(file_path).suffix.lower()

            # Compute binary digest for deduplication
            import hashlib
            with open(file_path, 'rb') as fbin:
                raw_bytes = fbin.read()
            digest = hashlib.sha256(raw_bytes).hexdigest()
            if digest in seen_digests:
                continue
            seen_digests.add(digest)

            # Read file content appropriately and append
            if file_ext in ['.png', '.jpg', '.jpeg', '.gif']:
                encoded_data = base64.b64encode(raw_bytes).decode()
                if file_ext == '.png':
                    visual_outputs.append({'type': 'png', 'data': encoded_data})
                elif file_ext in ['.jpg', '.jpeg']:
                    visual_outputs.append({'type': 'jpeg', 'data': encoded_data})
                elif file_ext == '.gif':
                    visual_outputs.append({'type': 'gif', 'data': encoded_data})
            elif file_ext == '.svg':
                try:
                    svg_content = raw_bytes.decode('utf-8', errors='ignore')
                except Exception:
                    svg_content = ''
                visual_outputs.append({'type': 'svg', 'data': svg_content})
            elif file_ext in ['.html', '.htm']:
                html_content = raw_bytes.decode('utf-8', errors='ignore')
                visual_outputs.append({'type': 'html', 'data': html_content})
            elif file_ext == '.js':
                js_content = raw_bytes.decode('utf-8', errors='ignore')
                visual_outputs.append({'type': 'javascript', 'data': js_content})
            elif file_ext == '.json':
                json_content = raw_bytes.decode('utf-8', errors='ignore')
                visual_outputs.append({'type': 'json', 'data': json_content})
            elif file_ext in ['.md', '.markdown']:
                md_content = raw_bytes.decode('utf-8', errors='ignore')
                visual_outputs.append({'type': 'markdown', 'data': md_content})
            elif file_ext in ['.tex', '.latex']:
                latex_content = raw_bytes.decode('utf-8', errors='ignore')
                visual_outputs.append({'type': 'latex', 'data': latex_content})
                    
        except Exception as e:
            # Skip files that can't be read
            continue
    
    return visual_outputs

def detect_visual_outputs_in_stdout(stdout: str) -> list:
    """
    Detect visual outputs embedded in stdout.
    
    Args:
        stdout: The stdout string from code execution
        
    Returns:
        List of visual output dictionaries
    """
    visual_outputs = []
    
    # Look for base64 encoded images in stdout
    import re
    
    # Pattern for base64 PNG data
    png_pattern = r'data:image/png;base64,([A-Za-z0-9+/=]+)'
    png_matches = re.findall(png_pattern, stdout)
    for match in png_matches:
        visual_outputs.append({
            'type': 'png',
            'data': match
        })
    
    # Pattern for base64 JPEG data
    jpeg_pattern = r'data:image/jpeg;base64,([A-Za-z0-9+/=]+)'
    jpeg_matches = re.findall(jpeg_pattern, stdout)
    for match in jpeg_matches:
        visual_outputs.append({
            'type': 'jpeg',
            'data': match
        })
    
    # Pattern for SVG content
    svg_pattern = r'<svg[^>]*>.*?</svg>'
    svg_matches = re.findall(svg_pattern, stdout, re.DOTALL | re.IGNORECASE)
    for match in svg_matches:
        visual_outputs.append({
            'type': 'svg',
            'data': match
        })
    
    # Pattern for HTML content
    html_pattern = r'<!DOCTYPE html>.*?</html>|<html[^>]*>.*?</html>'
    html_matches = re.findall(html_pattern, stdout, re.DOTALL | re.IGNORECASE)
    for match in html_matches:
        visual_outputs.append({
            'type': 'html',
            'data': match
        })
    
    return visual_outputs

def mermaid_to_html(mermaid_code: str, theme: str = 'default') -> str:
    """
    Convert Mermaid diagram code to a minimal HTML document.

    Args:
        mermaid_code: The Mermaid diagram syntax
        theme: Theme name ('default', 'dark', 'forest', 'neutral', etc.)

    Returns:
        str: Complete HTML document with embedded Mermaid diagram
    """
    html_template = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <script type="module">
    import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
    </script>

    <script>
        mermaid.initialize({{
            startOnLoad: true,
            theme: '{theme}',
            securityLevel: 'loose',
            fontFamily: 'Arial, sans-serif'
        }});
    </script>
</head>
<body>
    <div class="mermaid">
{mermaid_code}
    </div>
</body>
</html>'''
    return html_template

def javascript_to_html(javascript_code: str) -> str:
    """Convert JavaScript code to a minimal HTML document that executes the code."""
    html_template = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>JavaScript Code Execution</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        #output {{
            background-color: white;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 15px;
            margin-top: 20px;
            white-space: pre-wrap;
            font-family: monospace;
        }}
        .error {{
            color: red;
        }}
        .log {{
            color: black;
        }}
    </style>
</head>
<body>
    <h1>JavaScript Code Execution</h1>
    <div id="output"></div>
    
    <script>
        // Override console methods to capture output
        const outputDiv = document.getElementById('output');
        const originalConsole = {{}};
        
        ['log', 'error', 'warn', 'info'].forEach(function(method) {{
            originalConsole[method] = console[method];
            console[method] = function(...args) {{
                const message = args.map(function(arg) {{ 
                    return typeof arg === 'object' ? JSON.stringify(arg, null, 2) : String(arg);
                }}).join(' ');
                
                const span = document.createElement('span');
                span.className = method === 'error' ? 'error' : 'log';
                span.textContent = '[' + method.toUpperCase() + '] ' + message + '\\n';
                outputDiv.appendChild(span);
                
                // Also call original console method
                originalConsole[method].apply(console, args);
            }};
        }});
        
        // Capture uncaught errors
        window.addEventListener('error', function(e) {{
            const span = document.createElement('span');
            span.className = 'error';
            span.textContent = '[ERROR] ' + e.message + ' at line ' + e.lineno + '\\n';
            outputDiv.appendChild(span);
        }});
        
        try {{
            // Execute the user's JavaScript code
{javascript_code}
        }} catch (error) {{
            console.error('Execution error:', error.message);
        }}
    </script>
</body>
</html>'''
    return html_template

def render_result_local(visual_output: dict) -> str:
    """Render a visual output similar to the remote code runner's render_result function."""
    output_type = visual_output['type']
    data = visual_output['data']
    
    if output_type == 'png':
        return f"![png image](data:image/png;base64,{data})"
    elif output_type == 'jpeg':
        return f"![jpeg image](data:image/jpeg;base64,{data})"
    elif output_type == 'gif':
        return f"![gif image](data:image/gif;base64,{data})"
    elif output_type == 'svg':
        if isinstance(data, str):
            svg_data = data
        else:
            svg_data = data.decode() if hasattr(data, 'decode') else str(data)
        svg_base64 = base64.b64encode(svg_data.encode()).decode()
        return f"![svg image](data:image/svg+xml;base64,{svg_base64})"
    elif output_type == 'html':
        return data
    elif output_type == 'markdown':
        return f"```markdown\n{data}\n```"
    elif output_type == 'latex':
        return f"```latex\n{data}\n```"
    elif output_type == 'json':
        return f"```json\n{data}\n```"
    elif output_type == 'javascript':
        return data  # Return raw JavaScript
    else:
        return str(data)

def ensure_vue_sfc_structure(code: str) -> str:
    """Ensure Vue code has proper Single File Component (SFC) structure."""
    code = code.strip()
    
    import re
    
    # Check for template tags (case insensitive, with potential attributes)
    has_template = bool(re.search(r'<template[^>]*>', code, re.IGNORECASE))
    # Check for script tags (case insensitive, with potential attributes like setup, lang="ts")
    has_script = bool(re.search(r'<script[^>]*>', code, re.IGNORECASE))
    
    # If it already has both template and script tags, return as is
    if has_template and has_script:
        return code
    
    # If it has neither template nor script, check if it's pure HTML/template content
    if not has_template and not has_script:
        # Check if it looks like Vue template syntax (has Vue directives or mustache syntax)
        vue_template_patterns = [
            r'v-[a-zA-Z-]+',  # Vue directives like v-if, v-for, v-model
            r'@[a-zA-Z-]+',   # Event handlers like @click
            r':[a-zA-Z-]+',   # Prop bindings like :class, :style
            r'\{\{.*?\}\}',   # Mustache interpolation
            r'<[a-zA-Z-]+[^>]*\s+v-',  # Elements with Vue directives
        ]
        
        is_vue_template = any(re.search(pattern, code) for pattern in vue_template_patterns)
        
        if is_vue_template or '<div' in code or '<template' in code.lower():
            # Wrap as template with basic script
            return f"""<template>
{code}
</template>

<script>
export default {{
  name: "App"
}};
</script>"""
        else:
            # Might be JavaScript/TypeScript code, wrap with minimal template
            return f"""<template>
  <div class="min-h-screen bg-gray-100 flex justify-center items-center p-4">
    <div class="bg-white rounded-lg shadow-lg p-6 w-full max-w-md">
      <h1 class="text-2xl font-bold mb-4 text-center">Vue Component</h1>
      <p class="text-center">Component loaded successfully</p>
    </div>
  </div>
</template>

<script>
{code}
</script>"""
    
    # If it has template but no script, add a minimal script
    if has_template and not has_script:
        return f"""{code}

<script>
export default {{
  name: "App"
}};
</script>"""
    
    # If it has script but no template, add a minimal template
    if not has_template and has_script:
        return f"""<template>
  <div class="min-h-screen bg-gray-100 flex justify-center items-center p-4">
    <div class="bg-white rounded-lg shadow-lg p-6 w-full max-w-md">
      <h1 class="text-2xl font-bold mb-4 text-center">Vue Component</h1>
      <p class="text-center">Component loaded successfully</p>
    </div>
  </div>
</template>

{code}"""
    
    return code

def check_url_responds(url: str, timeout: int = 5) -> bool:
    """Check if a URL responds successfully within the timeout period."""
    try:
        response = requests.get(url, timeout=timeout)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

class CodeRunResult(TypedDict):
    """The result of running the code in the sandbox."""
    sandbox_id: str
    """The sandbox id to run the code."""
    sandbox_url: str
    """The sandbox url to access the rendered results."""
    is_run_success: bool
    """Whether the code run is successful."""
    stderr: str
    """The stderr output from the sandbox."""

def run_code_interpreter(code: str, code_language: str | None, code_dependencies: tuple[list[str], list[str]], timeout_seconds: int = 120) -> tuple[str, str, list, str]:
    """Executes the provided code within a Docker sandbox environment."""
    # Determine the appropriate Docker image based on language
    if code_language == 'python':
        image_type = 'python'
    elif code_language == 'javascript':
        image_type = 'node'
    else:
        image_type = 'multi'
    
    sandbox = create_sandbox(image_type)
    
    try:
        # Capture initial files in the sandbox directory before execution
        initial_files = set()
        for root, dirs, files in os.walk(sandbox.working_dir):
            for file in files:
                initial_files.add(os.path.join(root, file))
        
        # Install dependencies
        python_dependencies, npm_dependencies = code_dependencies
        
        stderrs = []
        if python_dependencies:
            pip_install_errs = install_python_dependencies(sandbox, python_dependencies)
            # stderrs.extend(pip_install_errs)
        if npm_dependencies:
            npm_install_errs = install_npm_dependencies(sandbox, npm_dependencies)
            # stderrs.extend(npm_install_errs)
        
        # Create a code file
        if code_language == 'python':
            file_ext = '.py'
            code_file = f'code{file_ext}'
            cmd = f"python3 {code_file}"
        elif code_language == 'javascript':
            file_ext = '.js'
            code_file = f'code{file_ext}'
            cmd = f"node {code_file}"
        else:
            file_ext = '.txt'
            code_file = f'code{file_ext}'
            cmd = f"echo '{code}'"
        
        # Write code to file with instrumentation to auto-save visual outputs
        if code_language == 'python':
            instrumentation_preamble = """
import uuid, atexit
_saved_fig_ids = set()
_plotly_saved_ids = set()
try:
    # Matplotlib: force non-interactive backend and auto-save on show()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _plt_original_show = getattr(plt, 'show', None)
    def _plt_auto_save_show(*args, **kwargs):
        try:
            figs = [plt.figure(num) for num in plt.get_fignums()]
            for i, fig in enumerate(figs):
                fid = id(fig)
                if fid in _saved_fig_ids:
                    continue
                base = f"mpl_{uuid.uuid4().hex[:8]}"
                name = f"{base}.png" if i == 0 else f"{base}_{i}.png"
                try:
                    fig.savefig(name, bbox_inches='tight')
                    _saved_fig_ids.add(fid)
                    # Close to avoid re-saving at atexit and free memory
                    try:
                        plt.close(fig)
                    except Exception:
                        pass
                except Exception:
                    pass
        except Exception:
            pass
        # Do not block; skip calling the original interactive show
    if _plt_original_show is not None:
        plt.show = _plt_auto_save_show

    # Also save any remaining open figures at process exit
    def _save_remaining_figs():
        try:
            figs = [plt.figure(num) for num in plt.get_fignums()]
            for i, fig in enumerate(figs):
                fid = id(fig)
                if fid in _saved_fig_ids:
                    continue
                base = f"mpl_{uuid.uuid4().hex[:8]}_final"
                name = f"{base}.png" if i == 0 else f"{base}_{i}.png"
                try:
                    fig.savefig(name, bbox_inches='tight')
                    _saved_fig_ids.add(fid)
                    try:
                        plt.close(fig)
                    except Exception:
                        pass
                except Exception:
                    pass
        except Exception:
            pass
    atexit.register(_save_remaining_figs)
except Exception:
    pass

try:
    # Plotly: capture pio.show as HTML, optionally PNG if kaleido installed
    import plotly.io as _pio
    _pio_original_show = getattr(_pio, 'show', None)
    def _pio_auto_show(fig, *args, **kwargs):
        try:
            fid = id(fig)
        except Exception:
            fid = None
        if fid is not None and fid in _plotly_saved_ids:
            return
        base = f"plotly_{uuid.uuid4().hex[:8]}"
        html_name = f"{base}.html"
        try:
            _pio.write_html(fig, file=html_name, auto_open=False, include_plotlyjs='cdn')
        except Exception:
            pass
        try:
            png_name = f"{base}.png"
            _pio.write_image(fig, png_name, format='png', scale=2)
        except Exception:
            pass
        if fid is not None:
            _plotly_saved_ids.add(fid)
    if _pio_original_show is not None:
        _pio.show = _pio_auto_show
except Exception:
    pass
"""
            augmented_code = instrumentation_preamble + "\n" + code
        else:
            augmented_code = code
        sandbox.files.write(code_file, augmented_code)
        
        # Add the code file to initial files so we don't treat it as a visual output
        initial_files.add(os.path.join(sandbox.working_dir, code_file))
        
        # Run the code with custom timeout handling
        try:
            if code_language in ['python', 'javascript']:
                result = exec_run_with_timeout(
                    sandbox.container,
                    f"bash -c 'cd /sandbox && {cmd}'",
                    timeout_seconds,
                    user="sandbox"
                )
                
                if result is None:
                    # Timeout occurred
                    stdout = ""
                    stderr = "Execution timed out"
                else:
                    stdout = result.output.decode('utf-8', errors='ignore')
                    stderr = ""
                    
                    if result.exit_code != 0:
                        stderr = stdout
                        stdout = ""
            else:
                stdout = ""
                stderr = "Unsupported language"
                
        except Exception as e:
            stdout = ""
            stderr = str(e)
        
        # Detect visual outputs using both directory scanning and stdout parsing
        visual_outputs = []
        
        # 1. Check for new files created in the sandbox directory
        file_outputs = detect_visual_outputs_in_directory(sandbox.working_dir, initial_files)
        visual_outputs.extend(file_outputs)
        
        # 2. Check stdout for embedded visual content
        stdout_outputs = detect_visual_outputs_in_stdout(stdout)
        visual_outputs.extend(stdout_outputs)
        
        # Return sandbox_id for proper cleanup management
        return stdout, stderr, visual_outputs, sandbox.sandbox_id
        
    except Exception as e:
        # Kill sandbox on exception
        sandbox.kill()
        raise e

def run_html_sandbox(code: str, code_dependencies: tuple[list[str], list[str]], existing_sandbox_id: str | None = None, timeout_seconds: int = 120) -> tuple[str, str, str]:
    """Run HTML code in a Docker sandbox."""
    sandbox = reuse_or_create_sandbox(image_type="multi")
    
    try:
        stderrs = []
        
        # Create the HTML file
        sandbox.files.write("index.html", code)
        
        # Install dependencies if specified
        python_dependencies, npm_dependencies = code_dependencies
        
        if python_dependencies:
            pip_install_errs = install_python_dependencies(sandbox, python_dependencies)
            # stderrs.extend(pip_install_errs)
        if npm_dependencies:
            npm_install_errs = install_npm_dependencies(sandbox, npm_dependencies)
            # stderrs.extend(npm_install_errs)
        
        # Find an available port and start HTTP server
        port = find_random_available_port()
        sandbox.set_allocated_port(port)
        
        # Start HTTP server in the background
        stderr = run_background_command_with_timeout(
            sandbox,
            f"python3 -m http.server {port}",
            timeout=30
        )
        stderrs.append(stderr)
        
        url = get_sandbox_app_url(sandbox, port)
        filtered_stderrs = [s for s in stderrs if s is not None]
        return (url, sandbox.sandbox_id, '\n'.join(filtered_stderrs))
        
    except Exception as e:
        sandbox.kill()
        raise e

def run_react_sandbox(code: str, code_dependencies: tuple[list[str], list[str]], existing_sandbox_id: str | None = None, timeout_seconds: int = 120) -> tuple[str, str, str]:
    """Run React code in a Docker sandbox using Vite build system."""
    sandbox = reuse_or_create_sandbox(image_type="node")
    
    try:
        stderrs = []
        
        # Find an available port
        port = find_random_available_port()
        sandbox.set_allocated_port(port)
        
        # Copy React template to sandbox
        template_source = os.path.join(os.path.dirname(__file__), "..", "e2b_sandbox_template", "react_app")
        
        if os.path.exists(template_source):
            # Copy template files
            for item in os.listdir(template_source):
                if item in ['node_modules', 'package-lock.json']:
                    continue
                    
                source_path = os.path.join(template_source, item)
                
                if os.path.isdir(source_path):
                    # Copy directory recursively
                    dest_path = os.path.join(sandbox.working_dir, item)
                    if os.path.exists(dest_path):
                        shutil.rmtree(dest_path)
                    shutil.copytree(source_path, dest_path)
                else:
                    # Copy file
                    with open(source_path, 'r') as f:
                        content = f.read()
                    sandbox.files.write(item, content)
        
        # Replace App.tsx with user's code
        sandbox.files.write("src/App.tsx", code.strip())
        
        # Update vite.config.ts
        vite_config_content = '''import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  base: '/',
  plugins: [react()],
  build: {
    rollupOptions: {
      external: [],
      output: {
        globals: {}
      }
    }
  },
  optimizeDeps: {
    include: ['react-is']
  }
})
'''
        sandbox.files.write("vite.config.ts", vite_config_content)
        
        # Fix permissions for npm install and Vite build process
        try:
            # Ensure sandbox user owns the directory and all files
            sandbox.container.exec_run("chown -R sandbox:sandbox /sandbox", user="root")
            # Set proper permissions for npm and Vite temp files
            sandbox.container.exec_run("chmod -R 755 /sandbox", user="root")
        except Exception as e:
            stderrs.append(f"Permission fix error: {e}")
        
        # Install dependencies
        try:
            result = sandbox.container.exec_run(
                "npm install --include=dev --legacy-peer-deps --prefer-offline --no-audit --no-fund",
                user="sandbox",
                workdir="/sandbox"
            )
            
            if result.exit_code != 0:
                error_msg = result.output.decode('utf-8', errors='ignore')
                stderrs.append(f"npm install failed: {error_msg}")
        except Exception as e:
            stderrs.append(f"Error installing dependencies: {e}")
        
        # Install additional dependencies
        _, npm_dependencies = code_dependencies
        if npm_dependencies:
            npm_install_errs = install_npm_dependencies(sandbox, npm_dependencies)
            stderrs.extend(npm_install_errs)
        
        # Build the project
        try:
            result = sandbox.container.exec_run(
                "npm run build",
                user="sandbox",
                workdir="/sandbox"
            )
            
            if result.exit_code != 0:
                error_msg = result.output.decode('utf-8', errors='ignore')
                stderrs.append(f"Build failed: {error_msg}")
                raise Exception(f"Build failed: {error_msg}")
                
        except Exception as e:
            stderrs.append(f"Build error: {e}")
            raise e
        
        # Serve the built application
        stderr = run_background_command_with_timeout(
            sandbox,
            f"python3 -m http.server {port} --directory dist",
            timeout=30
        )
        stderrs.append(stderr)
        
        # Wait for server to start
        time.sleep(5)
        
        url = f"http://localhost:{port}"
        return url, sandbox.sandbox_id, "\n".join(stderrs) if stderrs else ""
        
    except Exception as e:
        if sandbox:
            try:
                sandbox.kill()
            except:
                pass
        raise e

def run_vue_sandbox(code: str, code_dependencies: tuple[list[str], list[str]], existing_sandbox_id: str | None = None, timeout_seconds: int = 120) -> tuple[str, str, str]:
    """Run Vue code in a Docker sandbox using Vite build system."""
    sandbox = reuse_or_create_sandbox(image_type="node")
    
    try:
        stderrs = []
        
        # Find an available port
        port = find_random_available_port()
        sandbox.set_allocated_port(port)
        
        # Copy Vue template to sandbox
        template_source = os.path.join(os.path.dirname(__file__), "..", "e2b_sandbox_template", "vue_app")
        
        if os.path.exists(template_source):
            # Copy template files
            for item in os.listdir(template_source):
                if item in ['node_modules', 'package-lock.json']:
                    continue
                    
                source_path = os.path.join(template_source, item)
                
                if os.path.isdir(source_path):
                    # Copy directory recursively
                    dest_path = os.path.join(sandbox.working_dir, item)
                    if os.path.exists(dest_path):
                        shutil.rmtree(dest_path)
                    shutil.copytree(source_path, dest_path)
                else:
                    # Copy file
                    with open(source_path, 'r') as f:
                        content = f.read()
                    sandbox.files.write(item, content)
        
        # Ensure proper Vue SFC structure and replace App.vue
        code = ensure_vue_sfc_structure(code.strip())
        sandbox.files.write("src/App.vue", code)
        
        # Fix permissions for npm install and Vite build process
        try:
            # Ensure sandbox user owns the directory and all files
            sandbox.container.exec_run("chown -R sandbox:sandbox /sandbox", user="root")
            # Set proper permissions for npm and Vite temp files
            sandbox.container.exec_run("chmod -R 755 /sandbox", user="root")
        except Exception as e:
            stderrs.append(f"Permission fix error: {e}")
        
        # Install additional dependencies
        _, npm_dependencies = code_dependencies
        if npm_dependencies:
            npm_install_errs = install_npm_dependencies(sandbox, npm_dependencies)
        
        # Install dependencies
        try:
            result = sandbox.container.exec_run(
                "npm install --include=dev --legacy-peer-deps --no-audit --no-fund",
                user="sandbox",
                workdir="/sandbox"
            )
            
            if result.exit_code != 0:
                error_msg = result.output.decode('utf-8', errors='ignore')
                stderrs.append(f"npm install failed: {error_msg}")
        except Exception as e:
            stderrs.append(f"Error installing dependencies: {e}")

        # Build the project
        try:
            result = sandbox.container.exec_run(
                "npm run build",
                user="sandbox",
                workdir="/sandbox"
            )
            
            if result.exit_code != 0:
                error_msg = result.output.decode('utf-8', errors='ignore')
                stderrs.append(f"Build failed: {error_msg}")
                raise Exception(f"Build failed: {error_msg}")
                
        except Exception as e:
            stderrs.append(f"Build error: {e}")
            raise e
        
        # Serve the built application
        stderr = run_background_command_with_timeout(
            sandbox,
            f"python3 -m http.server {port} --directory dist",
            timeout=30
        )
        stderrs.append(stderr)
        
        # Wait for server to start
        time.sleep(5)
        
        url = f"http://localhost:{port}"
        return url, sandbox.sandbox_id, "\n".join(stderrs) if stderrs else ""
        
    except Exception as e:
        if sandbox:
            try:
                sandbox.kill()
            except:
                pass
        raise e

def run_gradio_sandbox(code: str, code_dependencies: tuple[list[str], list[str]], existing_sandbox_id: str | None = None, timeout_seconds: int = 120) -> tuple[str, str, str]:
    """Run Gradio code in a Docker sandbox."""

    sandbox = create_sandbox("python")
    try:
        stderrs = []
        
        # Find an available port first
        port = find_random_available_port()
        sandbox.set_allocated_port(port)

        # Create the Python file with proper server configuration
        # Remove any hardcoded server_port and server_name from demo.launch()
        import re
        
        # Remove server_port and server_name arguments from demo.launch() calls
        code = re.sub(r'\.launch\([^)]*server_port\s*=\s*[^,)]+[,\s]*', '.launch(', code)
        code = re.sub(r'\.launch\([^)]*server_name\s*=\s*[^,)]+[,\s]*', '.launch(', code)
        
        # Ensure demo.launch() uses the correct server configuration
        if 'demo.launch(' in code:
            # Replace demo.launch() with proper configuration
            code = re.sub(
                r'demo\.launch\([^)]*\)',
                f'demo.launch(server_name="0.0.0.0", server_port={port}, share=False)',
                code
            )
        elif '.launch(' in code:
            # Handle other patterns like app.launch(), interface.launch(), etc.
            code = re.sub(
                r'(\w+)\.launch\([^)]*\)',
                rf'\1.launch(server_name="0.0.0.0", server_port={port}, share=False)',
                code
            )
        else:
            # If no launch() call found, add one at the end
            code += f'\n\n# Auto-added launch configuration\nif __name__ == "__main__":\n    demo.launch(server_name="0.0.0.0", server_port={port}, share=False)'

        sandbox.files.write("app.py", code)
        
        # Install dependencies
        python_dependencies, npm_dependencies = code_dependencies
        
        # Add gradio if not present
        all_python_deps = list(python_dependencies) if python_dependencies else []
        if "gradio" not in str(python_dependencies).lower():
            all_python_deps.append("gradio")
        
        if all_python_deps:
            pip_install_errs = install_python_dependencies(sandbox, all_python_deps)
            # stderrs.extend(pip_install_errs)
        
        # Start Gradio app
        stderr = run_background_command_with_timeout(
            sandbox,
            "python3 app.py",
            timeout=15
        )
        stderrs.append(stderr)
        url = get_sandbox_app_url(sandbox, port)

        return (url, sandbox.sandbox_id, '\n'.join(stderrs))
        
    except Exception as e:
        sandbox.kill()
        raise e

def run_streamlit_sandbox(code: str, code_dependencies: tuple[list[str], list[str]], existing_sandbox_id: str | None = None, timeout_seconds: int = 120) -> tuple[str, str, str]:
    """Run Streamlit code in a Docker sandbox."""
    sandbox = reuse_or_create_sandbox(image_type="python")
    
    try:
        stderrs = []
        
        # Create the Python file
        sandbox.files.write("app.py", code)
        
        # Install dependencies
        python_dependencies, npm_dependencies = code_dependencies
        
        # Add streamlit if not present
        all_python_deps = list(python_dependencies) if python_dependencies else []
        if "streamlit" not in str(python_dependencies).lower():
            all_python_deps.append("streamlit")
        
        if all_python_deps:
            pip_install_errs = install_python_dependencies(sandbox, all_python_deps)
            # stderrs.extend(pip_install_errs)
        
        # Find an available port
        port = find_random_available_port()
        sandbox.set_allocated_port(port)
        
        # Start Streamlit app
        stderr = run_background_command_with_timeout(
            sandbox,
            f"streamlit run app.py --server.port {port} --server.headless true --server.runOnSave false",
            timeout=30
        )
        stderrs.append(stderr)
        
        url = get_sandbox_app_url(sandbox, port)
        filtered_stderrs = [s for s in stderrs if s is not None]
        return (url, sandbox.sandbox_id, '\n'.join(filtered_stderrs))
        
    except Exception as e:
        sandbox.kill()
        raise e

def run_pygame_sandbox(code: str, code_dependencies: tuple[list[str], list[str]], existing_sandbox_id: str | None = None, timeout_seconds: int = 120) -> tuple[str, str, str]:
    """Run Pygame code in a Docker sandbox."""
    sandbox = create_sandbox("python")
    
    try:
        stderrs = []
        
        # Find an available port
        port = find_random_available_port()
        sandbox.set_allocated_port(port)
        
        # Create the pygame_app directory in /sandbox (volume mount)
        try:
            # Create directory structure on host (which maps to /sandbox in container)
            pygame_dir = os.path.join(sandbox.working_dir, "pygame_app")
            os.makedirs(pygame_dir, exist_ok=True)
            
            # Write the Python file to the host directory
            with open(os.path.join(pygame_dir, "main.py"), "w") as f:
                f.write(code)
                
        except Exception as e:
            stderrs.append(f"Error creating pygame app structure: {e}")
        
        # Install dependencies
        python_dependencies, npm_dependencies = code_dependencies
        
        # Add pygame dependencies
        all_python_deps = list(python_dependencies) if python_dependencies else []
        all_python_deps.extend(["pygbag", "pygame-ce"])
        
        if all_python_deps:
            pip_install_errs = install_python_dependencies(sandbox, all_python_deps)
            # stderrs.extend(pip_install_errs)
        
        # Fix permissions for the pygame_app directory
        try:
            sandbox.container.exec_run("chown -R sandbox:sandbox /sandbox/pygame_app", user="root")
            sandbox.container.exec_run("chmod -R 755 /sandbox/pygame_app", user="root")
        except Exception as e:
            stderrs.append(f"Permission fix error: {e}")
        
        # Build and serve the Pygame app from /sandbox
        # Use a more direct approach to run pygbag in background
        try:
            # Run pygbag in background with proper logging
            result = sandbox.container.exec_run(
                f"bash -c 'cd /sandbox && nohup python3 -m pygbag --port {port} pygame_app > /tmp/pygbag_out.log 2> /tmp/pygbag_err.log &'",
                user="sandbox",
                detach=True
            )
            
            # Wait longer for pygbag to build and start serving
            time.sleep(20)
                            
        except Exception as e:
            stderrs.append(f"Error starting pygbag: {e}")
        
        url = get_sandbox_app_url(sandbox, port)
        return (url, sandbox.sandbox_id, "\n".join(stderrs) if stderrs else "")
        
    except Exception as e:
        sandbox.kill()
        raise e

# Additional language support functions (simplified for Docker)
def take_screenshot_for_web_app(sandbox: DockerSandbox, port: int, screenshot_filename: str = "screenshot.png") -> str:
    """
    Take a screenshot of a web application running in the Docker sandbox.
    
    Args:
        sandbox: The Docker sandbox instance
        port: Port where the web app is running inside the container
        screenshot_filename: Name for the screenshot file
        
    Returns:
        str: Path to the screenshot file on the host system (via volume mount)
    """
    return take_internal_screenshot(sandbox, port, screenshot_filename, wait_time=5)

def take_screenshot_for_pygame(sandbox: DockerSandbox, port: int, screenshot_filename: str = "pygame_screenshot.png") -> str:
    """
    Take a screenshot of a PyGame application running in the Docker sandbox.
    Includes PyGame-specific actions like clicking the canvas to give it focus.
    
    Args:
        sandbox: The Docker sandbox instance
        port: Port where the PyGame web app is running inside the container
        screenshot_filename: Name for the screenshot file
        
    Returns:
        str: Path to the screenshot file on the host system (via volume mount)
    """
    import uuid
    
    # Generate unique script name to avoid conflicts
    script_id = str(uuid.uuid4())[:8]
    script_name = f"pygame_screenshot_{script_id}.py"
    
    # Create Python script to run Playwright inside container with PyGame-specific actions
    pygame_screenshot_script = f'''
import asyncio
import sys
import os
import time

async def take_pygame_screenshot():
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
            
            # Navigate to the PyGame app
            url = "http://localhost:{port}"
            
            try:
                await page.goto(url, wait_until='domcontentloaded', timeout=15000)
                
                # Wait for initial load
                await page.wait_for_timeout(3000)
                
                # PyGame-specific actions: Click on canvas elements to give them focus
                canvas_elements = await page.query_selector_all('canvas')
                if canvas_elements:
                    for i, canvas in enumerate(canvas_elements):
                        try:
                            await canvas.click()
                            await page.wait_for_timeout(500)  # Brief pause between clicks
                        except Exception as e:
                            pass
                else:
                    await page.click('body')
                
                # Additional PyGame interactions
                # Try to trigger any mouse events that might activate the game
                await page.mouse.move(512, 384)  # Move to center
                await page.mouse.click(512, 384)  # Click center
                await page.wait_for_timeout(1000)
                
                # Try some keyboard events that might be useful for games
                await page.keyboard.press('Space')  # Common game interaction
                await page.wait_for_timeout(500)
                
                # Move mouse around a bit to trigger any hover effects
                await page.mouse.move(300, 200)
                await page.wait_for_timeout(500)
                await page.mouse.move(700, 500)
                await page.wait_for_timeout(500)
                
                # Final wait for any animations or state changes
                await page.wait_for_timeout(2000)
                
                # Take the screenshot
                temp_path = "/tmp/{screenshot_filename}"
                await page.screenshot(path=temp_path, full_page=True)
                
            except Exception as e:
                sys.exit(1)
            finally:
                await browser.close()
                
    except ImportError:
        sys.exit(1)
    except Exception as e:
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(take_pygame_screenshot())
'''
    
    # Write the script to the container
    try:
        import base64
        encoded_script = base64.b64encode(pygame_screenshot_script.encode('utf-8')).decode('ascii')
        
        # Write script to container
        result = sandbox.container.exec_run(
            f"python3 -c \"import base64; open('/tmp/{script_name}', 'w').write(base64.b64decode('{encoded_script}').decode('utf-8'))\"",
            user="root"
        )
        
        if result.exit_code != 0:
            return None
        
        # Make script executable and run it
        sandbox.container.exec_run(f"chmod +x /tmp/{script_name}", user="root")
        
        # Execute the screenshot script
        screenshot_result = sandbox.container.exec_run(
            f"python3 /tmp/{script_name}",
            user="root"
        )
        
        # Use the same approach as take_internal_screenshot for file handling
        temp_screenshot_path = f"/tmp/{screenshot_filename}"
        
        # Check if file exists in container
        check_result = sandbox.container.exec_run(f"ls -la {temp_screenshot_path}", user="root")
        
        if check_result.exit_code == 0:
            # Use the same file handling logic as take_internal_screenshot
            host_screenshot_path = os.path.join(sandbox.working_dir, screenshot_filename)
            
            if os.path.exists(host_screenshot_path):
                return host_screenshot_path
            else:
                # Try to copy from container's /tmp to volume mount
                try:
                    # First check if file exists in /tmp
                    check_result = sandbox.container.exec_run(f"ls -la {temp_screenshot_path}", user="root")
                    
                    # Change ownership of /sandbox to root temporarily
                    chown_result = sandbox.container.exec_run(f"chown root:root /sandbox", user="root")
                    
                    # Copy file
                    copy_result = sandbox.container.exec_run(f"cp {temp_screenshot_path} /sandbox/", user="root")
                    
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
                    pass
                
                return None
        else:
            return None
        
    except Exception as e:
        return None

def run_c_code(code: str, existing_sandbox_id: str | None = None, timeout_seconds: int = 120) -> tuple[str, str, list, str]:
    """Execute C code in a Docker sandbox."""
    sandbox = create_sandbox("multi")
    
    try:
        # Write code to file in /tmp using Python inside container
        import base64
        encoded_code = base64.b64encode(code.encode('utf-8')).decode('ascii')
        result = sandbox.container.exec_run(
            f"python3 -c \"import base64; open('/tmp/main.c', 'w').write(base64.b64decode('{encoded_code}').decode('utf-8'))\"",
            user="sandbox"
        )
        
        # Compile
        compile_result = sandbox.container.exec_run(
            "gcc main.c -o program",
            user="sandbox",
            workdir="/tmp"
        )
        
        if compile_result.exit_code != 0:
            return "", f"Compilation failed: {compile_result.output.decode()}", [], sandbox.sandbox_id
        
        # Make executable
        chmod_result = sandbox.container.exec_run(
            "chmod +x program",
            user="sandbox",
            workdir="/tmp"
        )
        
        # Run with proper stdout/stderr separation and timeout
        run_result = exec_run_with_timeout(
            sandbox.container,
            "./program",
            timeout_seconds,
            user="sandbox",
            workdir="/tmp",
            demux=True
        )
        
        if run_result is None:
            return "", "Execution timed out", [], sandbox.sandbox_id
        
        stdout, stderr = run_result.output
        stdout = stdout.decode('utf-8', errors='ignore') if stdout else ""
        stderr = stderr.decode('utf-8', errors='ignore') if stderr else ""
        
        return stdout, stderr, [], sandbox.sandbox_id
        
    except Exception as e:
        # Kill sandbox on exception
        try:
            sandbox.kill()
        except:
            pass
        raise e

def run_cpp_code(code: str, existing_sandbox_id: str | None = None, timeout_seconds: int = 120) -> tuple[str, str, list, str]:
    """Execute C++ code in a Docker sandbox."""
    sandbox = create_sandbox("multi")
    
    try:
        # Write code to file in /tmp using Python inside container
        import base64
        encoded_code = base64.b64encode(code.encode('utf-8')).decode('ascii')
        result = sandbox.container.exec_run(
            f"python3 -c \"import base64; open('/tmp/main.cpp', 'w').write(base64.b64decode('{encoded_code}').decode('utf-8'))\"",
            user="sandbox"
        )
        
        # Compile
        compile_result = sandbox.container.exec_run(
            "g++ main.cpp -o program",
            user="sandbox",
            workdir="/tmp"
        )
        
        if compile_result.exit_code != 0:
            return "", f"Compilation failed: {compile_result.output.decode()}", [], sandbox.sandbox_id
        
        # Make executable
        chmod_result = sandbox.container.exec_run(
            "chmod +x program",
            user="sandbox",
            workdir="/tmp"
        )
        # Run with proper stdout/stderr separation and timeout
        run_result = exec_run_with_timeout(
            sandbox.container,
            "./program",
            timeout_seconds,
            user="sandbox",
            workdir="/tmp",
            demux=True
        )
        
        if run_result is None:
            return "", "Execution timed out", [], sandbox.sandbox_id
        
        stdout, stderr = run_result.output
        stdout = stdout.decode('utf-8', errors='ignore') if stdout else ""
        stderr = stderr.decode('utf-8', errors='ignore') if stderr else ""
        
        return stdout, stderr, [], sandbox.sandbox_id
        
    except Exception as e:
        # Kill sandbox on exception
        try:
            sandbox.kill()
        except:
            pass
        raise e

def run_java_code(code: str, existing_sandbox_id: str | None = None, timeout_seconds: int = 120) -> tuple[str, str, list, str]:
    """Execute Java code in a Docker sandbox."""
    sandbox = create_sandbox("multi")
    
    try:
        class_name = extract_java_class_name(code)
        
        # Write code to file in /tmp using Python inside container
        import base64
        encoded_code = base64.b64encode(code.encode('utf-8')).decode('ascii')
        result = sandbox.container.exec_run(
            f"python3 -c \"import base64; open('/tmp/{class_name}.java', 'w').write(base64.b64decode('{encoded_code}').decode('utf-8'))\"",
            user="sandbox"
        )
        
        # Compile
        compile_result = sandbox.container.exec_run(
            f"javac {class_name}.java",
            user="sandbox",
            workdir="/tmp"
        )
        
        if compile_result.exit_code != 0:
            return "", f"Compilation failed: {compile_result.output.decode()}", [], sandbox.sandbox_id
        
        # Run with proper stdout/stderr separation and timeout
        run_result = exec_run_with_timeout(
            sandbox.container,
            f"java {class_name}",
            timeout_seconds,
            user="sandbox",
            workdir="/tmp",
            demux=True
        )
        
        if run_result is None:
            return "", "Execution timed out", [], sandbox.sandbox_id
        
        stdout, stderr = run_result.output
        stdout = stdout.decode('utf-8', errors='ignore') if stdout else ""
        stderr = stderr.decode('utf-8', errors='ignore') if stderr else ""
        
        return stdout, stderr, [], sandbox.sandbox_id
        
    except Exception as e:
        # Kill sandbox on exception
        try:
            sandbox.kill()
        except:
            pass
        raise e

def run_golang_code(code: str, existing_sandbox_id: str | None = None, timeout_seconds: int = 120) -> tuple[str, str, list, str]:
    """Execute Go code in a Docker sandbox."""
    sandbox = create_sandbox("multi")
    
    try:
        # Write code to file in /tmp using Python inside container
        import base64
        encoded_code = base64.b64encode(code.encode('utf-8')).decode('ascii')
        result = sandbox.container.exec_run(
            f"python3 -c \"import base64; open('/tmp/main.go', 'w').write(base64.b64decode('{encoded_code}').decode('utf-8'))\"",
            user="sandbox"
        )
        
        # Run (go run compiles and runs in one step) with proper stdout/stderr separation and timeout
        run_result = exec_run_with_timeout(
            sandbox.container,
            "go run main.go",
            timeout_seconds,
            user="sandbox",
            workdir="/tmp",
            demux=True
        )
        
        if run_result is None:
            return "", "Execution timed out", [], sandbox.sandbox_id
        
        stdout, stderr = run_result.output
        stdout = stdout.decode('utf-8', errors='ignore') if stdout else ""
        stderr = stderr.decode('utf-8', errors='ignore') if stderr else ""
        
        return stdout, stderr, [], sandbox.sandbox_id
        
    except Exception as e:
        # Kill sandbox on exception
        try:
            sandbox.kill()
        except:
            pass
        raise e

def run_rust_code(code: str, existing_sandbox_id: str | None = None, timeout_seconds: int = 120) -> tuple[str, str, list, str]:
    """Execute Rust code in a Docker sandbox."""
    sandbox = create_sandbox("multi")
    
    try:
        # Write code to file in /tmp using Python inside container
        import base64
        encoded_code = base64.b64encode(code.encode('utf-8')).decode('ascii')
        result = sandbox.container.exec_run(
            f"python3 -c \"import base64; open('/tmp/main.rs', 'w').write(base64.b64decode('{encoded_code}').decode('utf-8'))\"",
            user="sandbox"
        )
        
        # Compile
        compile_result = sandbox.container.exec_run(
            "rustc main.rs -o program",
            user="sandbox",
            workdir="/tmp"
        )
        
        if compile_result.exit_code != 0:
            return "", f"Compilation failed: {compile_result.output.decode()}", [], sandbox.sandbox_id
        
        # Make executable
        chmod_result = sandbox.container.exec_run(
            "chmod +x program",
            user="sandbox",
            workdir="/tmp"
        )
        
        # Run with proper stdout/stderr separation and timeout
        run_result = exec_run_with_timeout(
            sandbox.container,
            "./program",
            timeout_seconds,
            user="sandbox",
            workdir="/tmp",
            demux=True
        )
        
        if run_result is None:
            return "", "Execution timed out", [], sandbox.sandbox_id
        
        stdout, stderr = run_result.output
        stdout = stdout.decode('utf-8', errors='ignore') if stdout else ""
        stderr = stderr.decode('utf-8', errors='ignore') if stderr else ""
        
        return stdout, stderr, [], sandbox.sandbox_id
        
    except Exception as e:
        # Kill sandbox on exception
        try:
            sandbox.kill()
        except:
            pass
        raise e
