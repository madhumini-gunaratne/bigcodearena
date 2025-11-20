#!/usr/bin/env python3
"""
Lightweight solution executor for running code solutions in different environments.
Wraps existing sandbox infrastructure from gen_execution.py for unified execution.
"""

import os
import json
import time
import threading
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys

# Add parent directory to path for sandbox imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from sandbox.docker_code_runner import (
    run_html_sandbox, run_react_sandbox, run_pygame_sandbox,
    run_streamlit_sandbox, run_vue_sandbox, run_gradio_sandbox,
    run_code_interpreter, run_c_code, run_cpp_code, run_java_code,
    run_golang_code, run_rust_code, mermaid_to_html, javascript_to_html
)
from sandbox.docker_sandbox_manager import kill_all_sandboxes, cleanup_docker_resources


@dataclass
class ExecutionResult:
    """Result from executing a solution"""
    uid: str
    environment: str
    success: bool
    stdout: str
    stderr: str
    screenshot_path: Optional[str] = None
    visual_outputs: List[str] = None
    execution_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'uid': self.uid,
            'environment': self.environment,
            'success': self.success,
            'stdout': self.stdout,
            'stderr': self.stderr,
            'screenshot_path': self.screenshot_path,
            'visual_outputs': self.visual_outputs or [],
            'execution_time': self.execution_time
        }


class SolutionExecutor:
    """Execute solutions in different environments"""

    def __init__(self, timeout_seconds: int = 120, output_dir: str = None):
        """
        Initialize the executor.

        Args:
            timeout_seconds: Maximum execution time per solution
            output_dir: Directory to save outputs (screenshots, visuals)
        """
        self.timeout_seconds = timeout_seconds
        self.output_dir = output_dir or "execution_outputs"
        self.screenshots_dir = os.path.join(self.output_dir, "screenshots")
        self.visuals_dir = os.path.join(self.output_dir, "visual_outputs")

        # Create output directories
        os.makedirs(self.screenshots_dir, exist_ok=True)
        os.makedirs(self.visuals_dir, exist_ok=True)

        # Sandbox routing
        self.sandbox_envs = {
            "HTML": self._run_html,
            "React": self._run_react,
            "Vue": self._run_vue,
            "PyGame": self._run_pygame,
            "Streamlit": self._run_streamlit,
            "Gradio": self._run_gradio,
            "Mermaid": self._run_mermaid,
            "Javascript Runner": self._run_javascript,
        }

        self.runner_envs = {
            "Python Runner": self._run_python,
            "C Runner": self._run_c,
            "C++ Runner": self._run_cpp,
            "Java Runner": self._run_java,
            "Golang Runner": self._run_golang,
            "Rust Runner": self._run_rust,
        }

    def _run_html(self, code: str, dependencies: Tuple) -> Tuple[str, str, str, str]:
        """Run HTML code - returns (stdout, stderr, url, sandbox_id)"""
        try:
            result = run_html_sandbox(code, dependencies, None, self.timeout_seconds)
            if result and len(result) >= 3:
                url, sandbox_id, stderr = result
                return "", stderr, url, sandbox_id
            return "", "HTML sandbox failed", "", None
        except Exception as e:
            return "", str(e), "", None

    def _run_react(self, code: str, dependencies: Tuple) -> Tuple[str, str, str, str]:
        """Run React code"""
        try:
            result = run_react_sandbox(code, dependencies, None, self.timeout_seconds)
            if result and len(result) >= 3:
                url, sandbox_id, stderr = result
                return "", stderr, url, sandbox_id
            return "", "React sandbox failed", "", None
        except Exception as e:
            return "", str(e), "", None

    def _run_vue(self, code: str, dependencies: Tuple) -> Tuple[str, str, str, str]:
        """Run Vue code"""
        try:
            result = run_vue_sandbox(code, dependencies, None, self.timeout_seconds)
            if result and len(result) >= 3:
                url, sandbox_id, stderr = result
                return "", stderr, url, sandbox_id
            return "", "Vue sandbox failed", "", None
        except Exception as e:
            return "", str(e), "", None

    def _run_pygame(self, code: str, dependencies: Tuple) -> Tuple[str, str, str, str]:
        """Run PyGame code"""
        try:
            result = run_pygame_sandbox(code, dependencies, None, self.timeout_seconds)
            if result and len(result) >= 3:
                url, sandbox_id, stderr = result
                return "", stderr, url, sandbox_id
            return "", "PyGame sandbox failed", "", None
        except Exception as e:
            return "", str(e), "", None

    def _run_streamlit(self, code: str, dependencies: Tuple) -> Tuple[str, str, str, str]:
        """Run Streamlit code"""
        try:
            result = run_streamlit_sandbox(code, dependencies, None, self.timeout_seconds)
            if result and len(result) >= 3:
                url, sandbox_id, stderr = result
                return "", stderr, url, sandbox_id
            return "", "Streamlit sandbox failed", "", None
        except Exception as e:
            return "", str(e), "", None

    def _run_gradio(self, code: str, dependencies: Tuple) -> Tuple[str, str, str, str]:
        """Run Gradio code"""
        try:
            result = run_gradio_sandbox(code, dependencies, None, self.timeout_seconds)
            if result and len(result) >= 3:
                url, sandbox_id, stderr = result
                return "", stderr, url, sandbox_id
            return "", "Gradio sandbox failed", "", None
        except Exception as e:
            return "", str(e), "", None

    def _run_mermaid(self, code: str, dependencies: Tuple) -> Tuple[str, str, str, str]:
        """Run Mermaid diagram"""
        try:
            html_code = mermaid_to_html(code)
            result = run_html_sandbox(html_code, dependencies, None, self.timeout_seconds)
            if result and len(result) >= 3:
                url, sandbox_id, stderr = result
                return "", stderr, url, sandbox_id
            return "", "Mermaid sandbox failed", "", None
        except Exception as e:
            return "", str(e), "", None

    def _run_javascript(self, code: str, dependencies: Tuple) -> Tuple[str, str, str, str]:
        """Run JavaScript code"""
        try:
            html_code = javascript_to_html(code)
            result = run_html_sandbox(html_code, dependencies, None, self.timeout_seconds)
            if result and len(result) >= 3:
                url, sandbox_id, stderr = result
                return "", stderr, url, sandbox_id
            return "", "JavaScript sandbox failed", "", None
        except Exception as e:
            return "", str(e), "", None

    def _run_python(self, code: str, dependencies: Tuple) -> Tuple[str, str, List, str]:
        """Run Python code - returns (stdout, stderr, visual_outputs, sandbox_id)"""
        try:
            result = run_code_interpreter(code, 'python', dependencies, self.timeout_seconds)
            if result and len(result) >= 4:
                stdout, stderr, visual_outputs, sandbox_id = result
                return stdout, stderr, visual_outputs or [], sandbox_id
            return "", "Python execution failed", [], None
        except Exception as e:
            return "", str(e), [], None

    def _run_c(self, code: str, dependencies: Tuple) -> Tuple[str, str, List, str]:
        """Run C code"""
        try:
            result = run_c_code(code, None, self.timeout_seconds)
            if result and len(result) >= 4:
                stdout, stderr, visual_outputs, sandbox_id = result
                return stdout, stderr, visual_outputs or [], sandbox_id
            return "", "C execution failed", [], None
        except Exception as e:
            return "", str(e), [], None

    def _run_cpp(self, code: str, dependencies: Tuple) -> Tuple[str, str, List, str]:
        """Run C++ code"""
        try:
            result = run_cpp_code(code, None, self.timeout_seconds)
            if result and len(result) >= 4:
                stdout, stderr, visual_outputs, sandbox_id = result
                return stdout, stderr, visual_outputs or [], sandbox_id
            return "", "C++ execution failed", [], None
        except Exception as e:
            return "", str(e), [], None

    def _run_java(self, code: str, dependencies: Tuple) -> Tuple[str, str, List, str]:
        """Run Java code"""
        try:
            result = run_java_code(code, None, self.timeout_seconds)
            if result and len(result) >= 4:
                stdout, stderr, visual_outputs, sandbox_id = result
                return stdout, stderr, visual_outputs or [], sandbox_id
            return "", "Java execution failed", [], None
        except Exception as e:
            return "", str(e), [], None

    def _run_golang(self, code: str, dependencies: Tuple) -> Tuple[str, str, List, str]:
        """Run Go code"""
        try:
            result = run_golang_code(code, None, self.timeout_seconds)
            if result and len(result) >= 4:
                stdout, stderr, visual_outputs, sandbox_id = result
                return stdout, stderr, visual_outputs or [], sandbox_id
            return "", "Go execution failed", [], None
        except Exception as e:
            return "", str(e), [], None

    def _run_rust(self, code: str, dependencies: Tuple) -> Tuple[str, str, List, str]:
        """Run Rust code"""
        try:
            result = run_rust_code(code, None, self.timeout_seconds)
            if result and len(result) >= 4:
                stdout, stderr, visual_outputs, sandbox_id = result
                return stdout, stderr, visual_outputs or [], sandbox_id
            return "", "Rust execution failed", [], None
        except Exception as e:
            return "", str(e), [], None

    def execute(
        self,
        uid: str,
        code: str,
        environment: str,
        dependencies: Tuple = None
    ) -> ExecutionResult:
        """
        Execute a solution in the specified environment.

        Args:
            uid: Unique identifier for the solution
            code: Solution code to execute
            environment: Target environment (HTML, React, Python Runner, etc.)
            dependencies: Optional (pip_packages, system_packages) tuple

        Returns:
            ExecutionResult with execution details and outputs
        """
        if not code or not code.strip():
            return ExecutionResult(
                uid=uid,
                environment=environment,
                success=False,
                stdout="",
                stderr="No code to execute",
                execution_time=0
            )

        if dependencies is None:
            dependencies = ([], [])

        start_time = time.time()
        stdout, stderr, url, visual_outputs, sandbox_id = "", "", "", [], None
        screenshot_path = None

        try:
            # Route to appropriate environment
            if environment in self.sandbox_envs:
                # Web-based environments
                stdout, stderr, url, sandbox_id = self.sandbox_envs[environment](
                    code, dependencies
                )
                
                # Wait for app to load and take screenshot
                if url and sandbox_id:
                    time.sleep(2)  # Give app time to load
                    screenshot_path = self._take_screenshot(
                        sandbox_id, url, uid, environment
                    )

            elif environment in self.runner_envs:
                # Code runner environments
                stdout, stderr, visual_outputs, sandbox_id = self.runner_envs[
                    environment
                ](code, dependencies)
                
                # Save visual outputs
                if visual_outputs:
                    visual_paths = self._save_visuals(uid, visual_outputs)
                    visual_outputs = visual_paths

            else:
                stderr = f"Unsupported environment: {environment}"

            execution_time = time.time() - start_time
            success = not stderr or "error" not in stderr.lower()

            return ExecutionResult(
                uid=uid,
                environment=environment,
                success=success,
                stdout=stdout,
                stderr=stderr,
                screenshot_path=screenshot_path,
                visual_outputs=visual_outputs,
                execution_time=execution_time
            )

        except Exception as e:
            return ExecutionResult(
                uid=uid,
                environment=environment,
                success=False,
                stdout="",
                stderr=f"Execution error: {str(e)}",
                execution_time=time.time() - start_time
            )

        finally:
            # Cleanup sandbox if needed
            if sandbox_id:
                try:
                    from sandbox.docker_sandbox_manager import _active_sandboxes
                    if sandbox_id in _active_sandboxes:
                        sandbox = _active_sandboxes[sandbox_id]
                        sandbox.kill()
                except:
                    pass

    def _take_screenshot(
        self,
        sandbox_id: str,
        url: str,
        uid: str,
        environment: str
    ) -> Optional[str]:
        """Take screenshot of running web app"""
        try:
            from sandbox.docker_sandbox_manager import _active_sandboxes
            import urllib.parse
            from sandbox.docker_sandbox_manager import (
                take_screenshot_for_web_app,
                take_screenshot_for_pygame
            )

            if sandbox_id not in _active_sandboxes:
                return None

            parsed_url = urllib.parse.urlparse(url)
            port = parsed_url.port

            if not port:
                return None

            screenshot_filename = f"screenshot_{uid}.png"
            screenshot_path = os.path.join(self.screenshots_dir, screenshot_filename)

            # Use appropriate screenshot function
            if environment.lower() == 'pygame':
                result = take_screenshot_for_pygame(
                    _active_sandboxes[sandbox_id], port, screenshot_filename
                )
            else:
                result = take_screenshot_for_web_app(
                    _active_sandboxes[sandbox_id], port, screenshot_filename
                )

            if result and os.path.exists(result):
                if result != screenshot_path:
                    import shutil
                    shutil.copy2(result, screenshot_path)
                    try:
                        os.remove(result)
                    except:
                        pass
                return screenshot_path

            return None

        except Exception as e:
            return None

    def _save_visuals(self, uid: str, visual_outputs: List) -> List[str]:
        """Save visual outputs and return paths"""
        saved_paths = []
        seen_hashes = set()

        for i, output in enumerate(visual_outputs):
            try:
                if isinstance(output, dict):
                    output_type = output.get('type', 'png')
                    output_data = output.get('data')
                elif isinstance(output, bytes):
                    output_type = 'png'
                    output_data = output
                else:
                    continue

                if not output_data:
                    continue

                # Compute hash for deduplication
                import hashlib
                if isinstance(output_data, bytes):
                    data_hash = hashlib.sha256(output_data).hexdigest()
                else:
                    data_hash = hashlib.sha256(str(output_data).encode()).hexdigest()

                if data_hash in seen_hashes:
                    continue

                seen_hashes.add(data_hash)

                filename = f"visual_{uid}_{i}.{output_type}"
                filepath = os.path.join(self.visuals_dir, filename)

                if output_type in ['png', 'jpeg']:
                    import base64
                    if isinstance(output_data, str):
                        try:
                            bytes_data = base64.b64decode(output_data)
                        except:
                            bytes_data = output_data.encode()
                    else:
                        bytes_data = output_data

                    with open(filepath, 'wb') as f:
                        f.write(bytes_data)
                else:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(str(output_data))

                saved_paths.append(filepath)

            except Exception:
                continue

        return saved_paths

    def execute_batch(
        self,
        solutions: List[Dict[str, Any]],
        max_workers: int = 5,
        save_results: bool = True,
        results_file: str = None
    ) -> List[ExecutionResult]:
        """
        Execute multiple solutions in parallel.

        Args:
            solutions: List of solution dicts with keys: uid, code, environment, dependencies
            max_workers: Number of parallel workers
            save_results: Whether to save results to JSON file
            results_file: Path to save results (default: execution_outputs/results.jsonl)

        Returns:
            List of ExecutionResult objects
        """
        results = []
        results_file = results_file or os.path.join(self.output_dir, "results.jsonl")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self.execute,
                    sol.get('uid', ''),
                    sol.get('code', ''),
                    sol.get('environment', 'Python Runner'),
                    sol.get('dependencies', ([], []))
                ): sol
                for sol in solutions
            }

            from tqdm import tqdm
            with tqdm(total=len(solutions), desc="Executing solutions") as pbar:
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)

                        if save_results:
                            with open(results_file, 'a') as f:
                                f.write(json.dumps(result.to_dict(), ensure_ascii=False) + '\n')

                    except Exception as e:
                        pass
                    finally:
                        pbar.update(1)

        return results

    def cleanup(self):
        """Clean up all sandboxes and resources"""
        kill_all_sandboxes()
        cleanup_docker_resources()


def main():
    """Example usage"""
    import argparse

    parser = argparse.ArgumentParser(description="Execute solutions in different environments")
    parser.add_argument("--generation_file", default="../data/autocodearena_local/model_answer/generated_questions/generation.jsonl",
                        help="Path to generation.jsonl with environments")
    parser.add_argument("--output_dir", default="execution_outputs")
    parser.add_argument("--max_workers", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=120)

    args = parser.parse_args()

    # Load from generation.jsonl (has correct environments)
    solutions = []
    with open(args.generation_file) as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                solutions.append({
                    'uid': record.get('uid', ''),
                    'code': record.get('code_to_execute', ''),
                    'environment': record.get('environment', 'Python Runner'),
                    'dependencies': tuple(record.get('code_dependencies', []))
                })

    print(f"📁 Loaded {len(solutions)} solutions from {args.generation_file}")
    print(f"📊 Environment distribution:")
    env_counts = {}
    for sol in solutions:
        env = sol['environment']
        env_counts[env] = env_counts.get(env, 0) + 1
    for env, count in sorted(env_counts.items()):
        print(f"   {env}: {count}")

    # Execute
    executor = SolutionExecutor(
        timeout_seconds=args.timeout,
        output_dir=args.output_dir
    )

    try:
        results = executor.execute_batch(
            solutions,
            max_workers=args.max_workers,
            save_results=True
        )

        # Print summary
        successful = sum(1 for r in results if r.success)
        print(f"\n✅ Results: {successful}/{len(results)} successful")
        print(f"📁 Outputs saved to {args.output_dir}")

    finally:
        executor.cleanup()


if __name__ == "__main__":
    main()
