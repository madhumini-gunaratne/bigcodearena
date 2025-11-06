#!/usr/bin/env python3
"""
Simple code execution system - clean version of execute_code.py
Focuses on core functionality without the complexity.
"""

import os
import json
import time
import threading
import concurrent.futures
from typing import Dict, Any, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Import the core sandbox functionality
from sandbox.docker_code_runner import (
    run_html_sandbox, run_react_sandbox, run_pygame_sandbox, 
    run_streamlit_sandbox, run_vue_sandbox, run_gradio_sandbox,
    run_code_interpreter, run_c_code, run_cpp_code, run_java_code,
    run_golang_code, run_rust_code, mermaid_to_html, javascript_to_html,
    take_screenshot_for_web_app, take_screenshot_for_pygame
)
from sandbox.docker_sandbox_manager import cleanup_docker_resources, kill_all_sandboxes, DOCKER_IMAGES, get_docker_client
from sandbox.code_analyzer import extract_code_from_markdown

# Constants
GENERATION_FILE = "generation.jsonl"
RESULTS_FILE = "execution_results.jsonl"

def check_docker_images():
    """Check if all required Docker images exist"""
    print("🔍 Checking Docker images...")
    
    try:
        client = get_docker_client()
        missing_images = []
        
        for image_type, image_config in DOCKER_IMAGES.items():
            try:
                client.images.get(image_config.name)
                print(f"   ✅ {image_type}: {image_config.name}")
            except Exception:
                print(f"   ❌ {image_type}: {image_config.name} - MISSING")
                missing_images.append(image_config.name)
        
        if missing_images:
            print(f"\n❌ Missing {len(missing_images)} Docker images:")
            for img in missing_images:
                print(f"   - {img}")
            print("\n💡 Please run 'python build_docker_images.py' first to build all required images.")
            return False
        
        print("✅ All required Docker images are available!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to check Docker images: {e}")
        return False

def cleanup_existing_results(model_dir: str):
    """Clean up existing results file and screenshots directory if overwrite is enabled"""
    results_path = os.path.join(model_dir, RESULTS_FILE)
    screenshots_dir = os.path.join(model_dir, 'screenshots')
    visual_outputs_dir = os.path.join(model_dir, 'visual_outputs')
    
    if os.path.exists(results_path):
        os.remove(results_path)
        print(f"🗑️  Removed existing results file: {RESULTS_FILE}")
    
    if os.path.exists(screenshots_dir):
        import shutil
        try:
            shutil.rmtree(screenshots_dir)
            print(f"🗑️  Removed existing screenshots directory")
        except OSError as e:
            print(f"⚠️  Could not remove screenshots directory: {e}")
    
    if os.path.exists(visual_outputs_dir):
        import shutil
        try:
            shutil.rmtree(visual_outputs_dir)
            print(f"🗑️  Removed existing visual outputs directory")
        except OSError as e:
            print(f"⚠️  Could not remove visual outputs directory: {e}")

def reclassify_all_environments(model_dir: str):
    """Reclassify environment for all generation records by re-extracting code and determining environment using extract_code_from_markdown"""
    generation_file = os.path.join(model_dir, GENERATION_FILE)
    
    if not os.path.exists(generation_file):
        print(f"No generation file found: {generation_file}")
        return []
    
    reclassified_records = []
    updated_records = []
    
    print(f"🔄 Reclassifying environments for all samples using extract_code_from_markdown...")
    
    # Read all generation records
    with open(generation_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                original_env = record.get('environment', 'Unknown')
                
                # Extract the assistant's message content
                assistant_content = None
                if "messages" in record and len(record["messages"]) > 0:
                    # Find the assistant's response
                    for msg in reversed(record["messages"]):
                        if msg["role"] == "assistant":
                            content = msg.get("content", "")
                            # Handle both string content and dict content with "answer" key
                            if isinstance(content, dict) and "answer" in content:
                                assistant_content = content["answer"]
                            elif isinstance(content, str):
                                assistant_content = content
                            break
                
                if assistant_content:
                    # Use extract_code_from_markdown to reclassify the environment
                    extraction_result = extract_code_from_markdown(assistant_content)
                    if extraction_result:
                        code, code_language, code_dependencies, new_env = extraction_result
                        
                        # Update the record with new classification
                        record["code_to_execute"] = code
                        record["code_dependencies"] = code_dependencies
                        record["language"] = code_language
                        record["environment"] = str(new_env) if new_env else original_env
                        
                        if str(new_env) != original_env and new_env is not None:
                            reclassified_records.append(record)
                            print(f"  🔄 {record.get('uid', 'unknown')}: {original_env} → {new_env}")
                        
                        updated_records.append(record)
                    else:
                        print(f"No code found in {record.get('uid', 'unknown')}, keeping original environment: {original_env}")
                        updated_records.append(record)
                else:
                    updated_records.append(record)
    
    # Write back the updated records to generation file
    if updated_records:
        with open(generation_file, 'w', encoding='utf-8') as f:
            for record in updated_records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"✅ Reclassified {len(reclassified_records)} samples")
    return reclassified_records

class SimpleExecutor:
    def __init__(self, max_workers: int = 10, timeout_seconds: int = 120):
        self.max_workers = max_workers
        self.timeout_seconds = timeout_seconds
        self.processed = 0
        self.failed = 0
        self.active_sandboxes = {}  # sandbox_id -> thread_id
        self.lock = threading.Lock()
    
    def register_sandbox(self, sandbox_id: str, thread_id: int):
        """Register a sandbox for cleanup tracking"""
        if sandbox_id:
            with self.lock:
                self.active_sandboxes[sandbox_id] = thread_id
    
    def unregister_sandbox(self, sandbox_id: str):
        """Unregister a sandbox"""
        if sandbox_id:
            with self.lock:
                if sandbox_id in self.active_sandboxes:
                    del self.active_sandboxes[sandbox_id]
    
    def cleanup_sandbox(self, sandbox_id: str):
        """Clean up a specific sandbox"""
        if not sandbox_id:
            return
        
        try:
            from sandbox.docker_sandbox_manager import _active_sandboxes
            if sandbox_id in _active_sandboxes:
                sandbox = _active_sandboxes[sandbox_id]
                sandbox.kill()
            self.unregister_sandbox(sandbox_id)
        except Exception as e:
            pass  # Silent cleanup

def load_generations(model_dir: str, environment_filter: str = None, reclassify: bool = False) -> List[Dict[str, Any]]:
    """Load generation data from JSONL file"""
    generation_file = os.path.join(model_dir, GENERATION_FILE)
    results_file = os.path.join(model_dir, RESULTS_FILE)
    
    if not os.path.exists(generation_file):
        print(f"No generation file found: {generation_file}")
        return []
    
    # If reclassify is enabled, reclassify all environments first
    if reclassify:
        reclassified_records = reclassify_all_environments(model_dir)
        # Clear the results file so all records will be reprocessed with new environments
        if os.path.exists(results_file):
            os.remove(results_file)
            print(f"🗑️  Removed existing results file for reprocessing with reclassified environments")
    
    # Load all generations (after reclassification if enabled)
    generations = []
    with open(generation_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                generations.append(json.loads(line))
    
    # Find already processed UIDs
    processed_uids = set()
    if os.path.exists(results_file):
        with open(results_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    processed_uids.add(record.get('uid'))
    
    # Filter unprocessed generations
    unprocessed = [g for g in generations if g.get('uid') not in processed_uids]
    
    # Apply environment filtering if specified
    if environment_filter:
        original_count = len(unprocessed)
        unprocessed = [g for g in unprocessed if g.get('environment', '') == environment_filter]
        filtered_count = len(unprocessed)
        print(f"🔍 Environment filter '{environment_filter}' applied")
        percentage = (filtered_count/original_count*100) if original_count > 0 else 0
        print(f"📊 Filtered from {original_count} to {filtered_count} samples ({percentage:.1f}%)")
        
        if filtered_count == 0:
            print(f"⚠️  No samples found for environment '{environment_filter}' in {os.path.basename(model_dir)}")
            return []
    
    print(f"Found {len(unprocessed)} unprocessed out of {len(generations)} total")
    return unprocessed

def run_code_in_environment(code: str, environment: str, dependencies: tuple, thread_id: int, timeout_seconds: int = 120) -> Tuple[str, str, str, List, str]:
    """
    Run code in the specified environment
    Returns: (stdout, stderr, url, visual_outputs, sandbox_id)
    """
    
    # Sandbox environments (return url, sandbox_id, stderr)
    sandbox_envs = {
        "HTML": lambda: run_html_sandbox(code, dependencies, None, timeout_seconds),
        "React": lambda: run_react_sandbox(code, dependencies, None, timeout_seconds),
        "PyGame": lambda: run_pygame_sandbox(code, dependencies, None, timeout_seconds),
        "Vue": lambda: run_vue_sandbox(code, dependencies, None, timeout_seconds),
        "Gradio": lambda: run_gradio_sandbox(code, dependencies, None, timeout_seconds),
        "Streamlit": lambda: run_streamlit_sandbox(code, dependencies, None, timeout_seconds),
        "Mermaid": lambda: run_html_sandbox(mermaid_to_html(code), dependencies, None, timeout_seconds),
        "Javascript Runner": lambda: run_html_sandbox(javascript_to_html(code), dependencies, None, timeout_seconds),
    }
    
    # Runner environments (return stdout, stderr, visual_outputs, sandbox_id)
    runner_envs = {
        "Python Runner": lambda: run_code_interpreter(code, 'python', dependencies, timeout_seconds),
        "C Runner": lambda: run_c_code(code, None, timeout_seconds),
        "C++ Runner": lambda: run_cpp_code(code, None, timeout_seconds),
        "Java Runner": lambda: run_java_code(code, None, timeout_seconds),
        "Golang Runner": lambda: run_golang_code(code, None, timeout_seconds),
        "Rust Runner": lambda: run_rust_code(code, None, timeout_seconds),
    }
    
    try:
        if environment in sandbox_envs:
            result = sandbox_envs[environment]()
            if result and len(result) >= 3:
                url, sandbox_id, stderr = result
                return "", stderr, url, [], sandbox_id
            else:
                return "", "Sandbox creation failed", "", [], None
                
        elif environment in runner_envs:
            result = runner_envs[environment]()
            if result and len(result) >= 4:
                stdout, stderr, visual_outputs, sandbox_id = result
                return stdout, stderr, "", visual_outputs or [], sandbox_id
            elif result and len(result) >= 2:
                stdout, stderr = result[:2]
                return stdout, stderr, "", [], None
            else:
                return "", "Runner execution failed", "", [], None
        else:
            return "", f"Unsupported environment: {environment}", "", [], None
            
    except Exception as e:
        return "", f"Execution error: {str(e)}", "", [], None

def take_screenshot(sandbox_id: str, url: str, screenshot_path: str, environment: str) -> bool:
    """Take a screenshot of the running application"""
    if not url or not sandbox_id:
        return False
    
    try:
        from sandbox.docker_sandbox_manager import _active_sandboxes
        import urllib.parse
        
        if sandbox_id not in _active_sandboxes:
            return False
        
        sandbox = _active_sandboxes[sandbox_id]
        parsed_url = urllib.parse.urlparse(url)
        port = parsed_url.port
        
        if not port:
            return False
        
        # Use appropriate screenshot function
        screenshot_filename = os.path.basename(screenshot_path)
        
        if environment.lower() == 'pygame':
            result = take_screenshot_for_pygame(sandbox, port, screenshot_filename)
        else:
            result = take_screenshot_for_web_app(sandbox, port, screenshot_filename)
        
        if result and os.path.exists(result):
            # Move screenshot to target location
            os.makedirs(os.path.dirname(screenshot_path), exist_ok=True)
            if result != screenshot_path:
                import shutil
                shutil.copy2(result, screenshot_path)
                try:
                    os.remove(result)
                except:
                    pass
            return True
        return False
        
    except Exception as e:
        return False

def save_visual_outputs(visual_outputs: List, uid: str, model_dir: str) -> List[str]:
    """Save visual outputs and return relative paths"""
    if not visual_outputs:
        return []
    
    visual_dir = os.path.join(model_dir, 'visual_outputs')
    os.makedirs(visual_dir, exist_ok=True)
    
    saved_paths = []
    # Content/perceptual de-duplication within a single result
    seen_signatures = set()
    
    def _image_signature(bytes_data: bytes) -> str:
        """Compute a perceptual-like signature for image bytes; fall back to sha256 if PIL unavailable."""
        try:
            from PIL import Image
            import io
            import numpy as np
            with Image.open(io.BytesIO(bytes_data)) as img:
                img = img.convert('L').resize((32, 32))
                arr = np.array(img, dtype='float32')
                avg = arr.mean()
                bits = (arr > avg).astype('uint8')
                # Pack bits into bytes for compact signature
                packed = np.packbits(bits.flatten())
                return 'ahash:' + packed.tobytes().hex()
        except Exception:
            import hashlib
            return 'sha256:' + hashlib.sha256(bytes_data).hexdigest()
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
            
            filename = f"visual_{uid}_{i}.{output_type}"
            filepath = os.path.join(visual_dir, filename)
            
            if output_type in ['png', 'jpeg']:
                # Compute signature to avoid duplicates (perceptual if possible)
                import base64
                if isinstance(output_data, str):
                    try:
                        bytes_data = base64.b64decode(output_data)
                    except Exception:
                        bytes_data = output_data.encode()
                else:
                    bytes_data = output_data
                signature = _image_signature(bytes_data)
                if signature in seen_signatures:
                    continue
                seen_signatures.add(signature)
                with open(filepath, 'wb') as f:
                    f.write(bytes_data)
            else:
                # Text-like outputs: dedupe by content hash
                import hashlib
                text_data = str(output_data)
                digest = hashlib.sha256(text_data.encode('utf-8', errors='ignore')).hexdigest()
                if digest in seen_signatures:
                    continue
                seen_signatures.add(digest)
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(text_data)
            
            relative_path = os.path.relpath(filepath, model_dir)
            saved_paths.append(relative_path)
            
        except Exception as e:
            continue
    
    return saved_paths

def process_single_generation(generation: Dict[str, Any], model_dir: str, executor: SimpleExecutor) -> Tuple[bool, Dict[str, Any]]:
    """Process a single generation record"""
    uid = generation.get('uid', '')
    environment = generation.get('environment', '')
    code = generation.get('code_to_execute', '')
    dependencies = generation.get('code_dependencies', ([], []))
    thread_id = threading.get_ident()
    
    # Validate required fields
    if not uid or not environment:
        return False, {
            **generation,
            'stdout': '',
            'stderr': 'Missing required fields',
            'screenshot_path': None,
            'visual_outputs': []
        }
    
    if not code:
        return True, {
            **generation,
            'stdout': '',
            'stderr': 'No code to execute',
            'screenshot_path': None,
            'visual_outputs': []
        }
    
    # Processing sample silently
    
    sandbox_id = None
    try:
        # Run the code
        stdout, stderr, url, visual_outputs, sandbox_id = run_code_in_environment(
            code, environment, dependencies, thread_id, executor.timeout_seconds
        )
        
        # Register sandbox for cleanup
        if sandbox_id:
            executor.register_sandbox(sandbox_id, thread_id)
        
        # Prepare result
        result = {
            **generation,
            'stdout': stdout,
            'stderr': stderr,
            'screenshot_path': None,
            'visual_outputs': []
        }
        
        # Save visual outputs for runner environments
        if visual_outputs:
            visual_paths = save_visual_outputs(visual_outputs, uid, model_dir)
            result['visual_outputs'] = visual_paths
        
        # Take screenshot for web environments
        if url:
            screenshot_dir = os.path.join(model_dir, 'screenshots')
            os.makedirs(screenshot_dir, exist_ok=True)
            screenshot_path = os.path.join(screenshot_dir, f"screenshot_{uid}.png")
            
            # Wait a bit for app to load
            time.sleep(3)
            
            if take_screenshot(sandbox_id, url, screenshot_path, environment):
                relative_path = os.path.relpath(screenshot_path, model_dir)
                result['screenshot_path'] = relative_path
            else:
                result['stderr'] += "\nScreenshot failed"
        
        return True, result
        
    except Exception as e:
        return False, {
            **generation,
            'stdout': '',
            'stderr': f'Processing error: {str(e)}',
            'screenshot_path': None,
            'visual_outputs': []
        }
    
    finally:
        # Always cleanup sandbox
        if sandbox_id:
            executor.cleanup_sandbox(sandbox_id)

def save_result(result: Dict[str, Any], model_dir: str):
    """Save a single result to the results file"""
    results_file = os.path.join(model_dir, RESULTS_FILE)
    
    with open(results_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(result, ensure_ascii=False) + '\n')

def process_model(model_dir: str, max_workers: int = 10, environment_filter: str = None, timeout_seconds: int = 120, overwrite: bool = False, reclassify: bool = False):
    """Process all generations for a model"""
    print(f"\n{'='*60}")
    print(f"Processing model: {os.path.basename(model_dir)}")
    print(f"{'='*60}")
    
    # Clean up existing results if overwrite is enabled
    if overwrite:
        print(f"🗑️  Cleaning up existing results for {os.path.basename(model_dir)}...")
        cleanup_existing_results(model_dir)
    
    # Load generations to process
    generations = load_generations(model_dir, environment_filter, reclassify)
    if not generations:
        print("No generations to process")
        return
    
    # Create executor
    executor = SimpleExecutor(max_workers, timeout_seconds)
    
    # Process with thread pool
    with ThreadPoolExecutor(max_workers=max_workers) as thread_pool:
        # Submit all tasks
        future_to_generation = {
            thread_pool.submit(process_single_generation, gen, model_dir, executor): gen
            for gen in generations
        }
        
        # Process results with progress bar
        with tqdm(total=len(generations), desc="Processing") as pbar:
            for future in as_completed(future_to_generation):
                try:
                    success, result = future.result()
                    
                    # Save result
                    save_result(result, model_dir)
                    
                    if success:
                        executor.processed += 1
                    else:
                        executor.failed += 1
                    
                    pbar.update(1)
                    
                except Exception as e:
                    executor.failed += 1
                    pbar.update(1)
    
    print(f"\n✅ Completed: {executor.processed} successful, {executor.failed} failed")
    
    # Final cleanup
    remaining_sandboxes = list(executor.active_sandboxes.keys())
    for sandbox_id in remaining_sandboxes:
        executor.cleanup_sandbox(sandbox_id)

def main():
    """Main function"""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Simple code execution system")
    parser.add_argument("--data_path", default="data/autocodearena_local/model_answer", 
                       help="Path to data directory")
    parser.add_argument("--model_name", help="Specific model to process")
    parser.add_argument("--max_workers", type=int, default=10, 
                       help="Number of worker threads")
    parser.add_argument("--environment", type=str, default=None,
                       help="Process only samples with this specific environment (e.g., 'HTML', 'React', 'Python Runner', etc.). Skip all other environments.")
    parser.add_argument("--timeout", type=int, default=120,
                       help="Maximum execution time in seconds for code execution (default: 120)")
    parser.add_argument("--cleanup", action="store_true",
                       help="Clean up Docker resources before starting")
    parser.add_argument("--skip_image_check", action="store_true",
                       help="Skip Docker image availability check")
    parser.add_argument("--overwrite", action="store_true",
                       help="Overwrite existing execution results and remove existing screenshots folder")
    parser.add_argument("--reclassify", action="store_true",
                       help="Reclassify the environment for all samples by re-extracting code and determining environment from the model's answer. This will update the environment field in generation records and reprocess based on the updated environment.")
    
    args = parser.parse_args()
    
    if args.cleanup:
        kill_all_sandboxes()
        cleanup_docker_resources()
    
    # Check Docker images unless skipped
    if not args.skip_image_check:
        if not check_docker_images():
            print("\n❌ Cannot proceed without required Docker images.")
            print("   Run 'python build_docker_images.py' to build all images, or use --skip_image_check to bypass.")
            sys.exit(1)
    
    try:
        if args.model_name:
            # Process specific model
            model_dir = os.path.join(args.data_path, args.model_name)
            if os.path.exists(model_dir):
                process_model(model_dir, args.max_workers, args.environment, args.timeout, args.overwrite, args.reclassify)
            else:
                print(f"Model directory not found: {model_dir}")
        else:
            # Process all models
            if not os.path.exists(args.data_path):
                print(f"Data path not found: {args.data_path}")
                return
            
            for model_name in os.listdir(args.data_path):
                model_dir = os.path.join(args.data_path, model_name)
                if os.path.isdir(model_dir):
                    process_model(model_dir, args.max_workers, args.environment, args.timeout, args.overwrite, args.reclassify)
    
    except KeyboardInterrupt:
        pass
    
    finally:
        kill_all_sandboxes()
        cleanup_docker_resources()

if __name__ == "__main__":
    main()
