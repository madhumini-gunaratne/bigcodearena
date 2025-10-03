#!/usr/bin/env python3
"""
Hugging Face Data Evaluation Script

This script:
1. Loads data directly from Hugging Face dataset
2. Evaluates data using the refactored CodeArenaJudger
3. Supports multithreaded processing and checkpoint recovery
4. Outputs results to results/{model_name}/{with_execution|without_execution}/ directory
"""

import json
import os
from pathlib import Path
import random
from typing import Dict, List, Tuple, Optional, Set
from tqdm import tqdm
import concurrent.futures
import threading
import time
from queue import Queue
import hashlib
from datetime import datetime
import base64
import re
import yaml
from datasets import load_dataset

from code_arena_judger import CodeArenaJudger


class ThreadSafeCounter:
    """Thread-safe counter for concurrent operations."""
    
    def __init__(self):
        self._value = 0
        self._lock = threading.Lock()
    
    def increment(self) -> int:
        """Increment counter and return new value."""
        with self._lock:
            self._value += 1
            return self._value
    
    def get_value(self) -> int:
        """Get current counter value."""
        with self._lock:
            return self._value


class ProgressTracker:
    """Progress tracker with checkpoint recovery support."""
    
    def __init__(self, progress_file: str = "results/hf_evaluation_progress.json"):
        self.progress_file = Path(progress_file)
        self.lock = threading.Lock()
        self.completed_records: Set[str] = set()
        self.failed_records: Set[str] = set()
        self.total_records = 0
        self.total_votes_processed = 0
        self.start_time = None
        self.last_save_time = 0
        self.save_interval = 10  # Save progress every 10 seconds
        
        # Load existing progress
        self.load_progress()
    
    def generate_record_id(self, record_data: Dict) -> str:
        """Generate unique and consistent ID for a record."""
        try:
            # Use chat_session_id as primary ID
            chat_session_id = record_data.get('chat_session_id', '')
            if chat_session_id:
                return chat_session_id
            
            # Fallback: generate ID from other fields
            instruction = record_data.get('instruction', '')
            model_a = record_data.get('model_A', '')
            model_b = record_data.get('model_B', '')
            
            content = f"{instruction}_{model_a}_{model_b}"
            return hashlib.md5(content.encode('utf-8')).hexdigest()[:16]
            
        except Exception as e:
            print(f"Error generating record ID: {e}")
            # Last resort: use timestamp-based ID
            return hashlib.md5(str(time.time()).encode('utf-8')).hexdigest()[:16]
    
    def load_progress(self):
        """Load progress from file."""
        if not self.progress_file.exists():
            print(f"No existing progress file found at {self.progress_file}")
            return
        
        try:
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.completed_records = set(data.get('completed_records', []))
            self.failed_records = set(data.get('failed_records', []))
            self.total_records = data.get('total_records', 0)
            self.total_votes_processed = data.get('total_votes_processed', 0)
            
            print(f"Loaded progress: {len(self.completed_records)} completed, {len(self.failed_records)} failed records")
            print(f"Total votes processed so far: {self.total_votes_processed}")
            
        except Exception as e:
            print(f"Error loading progress file: {e}")
            print("Starting fresh...")
    
    def save_progress(self, force: bool = False):
        """Save progress to file (thread-safe with rate limiting)."""
        current_time = time.time()
        
        # Rate limiting: save only at intervals unless forced
        if not force and (current_time - self.last_save_time) < self.save_interval:
            return
        
        with self.lock:
            try:
                progress_data = {
                    'completed_records': list(self.completed_records),
                    'failed_records': list(self.failed_records),
                    'total_records': self.total_records,
                    'total_votes_processed': self.total_votes_processed,
                    'last_updated': datetime.now().isoformat(),
                    'start_time': self.start_time.isoformat() if self.start_time else None,
                    'elapsed_time': (current_time - self.start_time.timestamp()) if self.start_time else 0
                }
                
                # Atomic write: write to temp file first, then rename
                temp_file = self.progress_file.with_suffix('.tmp')
                # Ensure directory exists
                temp_file.parent.mkdir(parents=True, exist_ok=True)
                
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(progress_data, f, indent=2, ensure_ascii=False)
                
                temp_file.replace(self.progress_file)
                self.last_save_time = current_time
                
            except Exception as e:
                print(f"Error saving progress: {e}")
    
    def mark_completed(self, record_id: str, votes_processed: int = 1):
        """Mark record as completed."""
        with self.lock:
            self.completed_records.add(record_id)
            self.failed_records.discard(record_id)  # Remove from failed list
            self.total_votes_processed += votes_processed
        
        self.save_progress()
    
    def mark_failed(self, record_id: str):
        """Mark record as failed."""
        with self.lock:
            self.failed_records.add(record_id)
            self.completed_records.discard(record_id)  # Remove from completed list
        
        self.save_progress()
    
    def is_completed(self, record_id: str) -> bool:
        """Check if record is completed."""
        with self.lock:
            return record_id in self.completed_records
    
    def should_skip_record(self, record_id: str, retry_failed: bool = False) -> bool:
        """Check if record should be skipped."""
        with self.lock:
            # Skip if already completed
            if record_id in self.completed_records:
                return True
            
            # Skip if failed and not retrying failed records
            if record_id in self.failed_records and not retry_failed:
                return True
            
            return False
    
    def get_stats(self) -> Dict:
        """Get statistics."""
        with self.lock:
            completed = len(self.completed_records)
            failed = len(self.failed_records)
            remaining = self.total_records - completed - failed
            
            return {
                'total_records': self.total_records,
                'completed': completed,
                'failed': failed,
                'remaining': remaining,
                'total_votes_processed': self.total_votes_processed,
                'completion_rate': completed / self.total_records if self.total_records > 0 else 0
            }
    
    def set_total_records(self, total: int):
        """Set total number of records."""
        with self.lock:
            self.total_records = total
            if not self.start_time:
                self.start_time = datetime.now()


class HFDataJudgerProcessor:
    """Process data downloaded from Hugging Face and apply model-as-a-judge evaluation"""
    
    def __init__(self, judger, judge_model_name: str, max_workers: int = 16, 
                 progress_file: str = "results/hf_evaluation_progress.json", 
                 max_records: int = None, retry_failed: bool = False, include_output: bool = True):
        self.judger = judger
        self.judge_model_name = judge_model_name
        self.include_output = include_output
        self.max_workers = max_workers
        self.processed_counter = ThreadSafeCounter()
        self.total_counter = ThreadSafeCounter()
        self.lock = threading.Lock()
        self.progress_tracker = ProgressTracker(progress_file)
        self.max_records = max_records
        self.retry_failed = retry_failed
        
        # Incremental save related
        self.incremental_save_interval = 150
        self.processed_count_since_save = 0
        self.output_file_path = None
        self.failed_file_path = None
        self.incremental_save_lock = threading.Lock()
    
    def extract_hf_data_fields(self, record: Dict) -> Tuple[str, str, str, str, str, str, str, str, str]:
        """
        Extract required fields from HF data record
        Args:
            record: HF data record
        Returns:
            Tuple: (instruction, model_a_name, model_b_name, model_a_code, model_b_code, 
                   model_a_output, model_b_output, model_a_screenshot, model_b_screenshot)
        """
        # Basic fields
        instruction = record.get('instruction', '')
        model_a_name = record.get('model_A', 'unknown')
        model_b_name = record.get('model_B', 'unknown')
        
        # Extract code and other information from states
        states = record.get('states', {})
        model_a_state = states.get('model_A', {})
        model_b_state = states.get('model_B', {})
        
        # Extract code (from the last Assistant message in messages)
        model_a_code = self._extract_code_from_messages(model_a_state.get('messages', []))
        model_b_code = self._extract_code_from_messages(model_b_state.get('messages', []))
        
        # Extract output and screenshot (from sandbox_logs_by_round)
        model_a_output, model_a_screenshot = self._extract_execution_info(model_a_state.get('sandbox_logs_by_round', {}))
        model_b_output, model_b_screenshot = self._extract_execution_info(model_b_state.get('sandbox_logs_by_round', {}))
        
        return (instruction, model_a_name, model_b_name, model_a_code, model_b_code,
                model_a_output, model_b_output, model_a_screenshot, model_b_screenshot)
    
    def _extract_code_from_messages(self, messages: List) -> str:
        """
        Extract code from message list (find code blocks in the last Assistant message).
        
        Args:
            messages: List of messages
            
        Returns:
            str: Extracted code, empty string if not found
        """
        return messages[-1][-1]
    
    def _extract_execution_info(self, sandbox_logs: Dict) -> Tuple[str, str]:
        """
        Extract execution output and screenshot from sandbox logs.
        
        Args:
            sandbox_logs: Sandbox logs dictionary
            
        Returns:
            Tuple[str, str]: (execution_output, screenshot_base64)
        """
        output = ''
        screenshot = ''
        
        if not sandbox_logs:
            return output, screenshot
        
        # Find the latest execution result
        for round_key in sorted(sandbox_logs.keys(), reverse=True):
            round_data = sandbox_logs.get(round_key)
            if round_data and isinstance(round_data, dict):
                sandbox_state = round_data.get('sandbox_state', {})
                if sandbox_state:
                    # Extract output
                    sandbox_output = sandbox_state.get('sandbox_output', '')
                    if sandbox_output:
                        output = sandbox_output
                    
                    # Extract screenshot
                    screenshot_base64 = sandbox_state.get('screenshot_base64', '')
                    if screenshot_base64:
                        screenshot = screenshot_base64
                    
                    # If we found valid execution results, use them
                    if output or screenshot:
                        break
        
        return output, screenshot
    
    def sanitize_record_for_json(self, record: Dict) -> Dict:
        """
        Sanitize a record for JSON serialization, converting PIL Images to base64.
        
        Args:
            record: Record to sanitize
            
        Returns:
            Dict: Sanitized record safe for JSON serialization
        """
        from PIL import Image
        import base64
        import io
        
        def sanitize_value(value):
            """Recursively sanitize a value for JSON serialization."""
            if value is None or isinstance(value, (str, int, float, bool)):
                return value
            elif isinstance(value, dict):
                return {k: sanitize_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [sanitize_value(item) for item in value]
            elif isinstance(value, Image.Image):
                # Convert PIL Image to base64
                try:
                    buffer = io.BytesIO()
                    value.save(buffer, format='PNG')
                    img_bytes = buffer.getvalue()
                    return base64.b64encode(img_bytes).decode('utf-8')
                except Exception as e:
                    return f"[PIL Image conversion error: {e}]"
            else:
                # Try to see if it's JSON serializable
                try:
                    json.dumps(value)
                    return value
                except (TypeError, ValueError):
                    return f"[Non-serializable object: {type(value).__name__}]"
        
        return sanitize_value(record)
    
    def save_processed_record_incremental(self, record_id: str, updated_record: Dict):
        """
        Incrementally save processed record to output file.
        
        Args:
            record_id: Record ID
            updated_record: Updated record data
        """
        if not self.output_file_path:
            return
        
        with self.incremental_save_lock:
            try:
                # Ensure output directory exists
                output_path = Path(self.output_file_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Sanitize record for JSON serialization
                sanitized_record = self.sanitize_record_for_json(updated_record)
                
                # Append single record
                with open(self.output_file_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(sanitized_record, separators=(',', ':')) + '\n')
                    
                    # Sync file after processing a certain number of records
                    self.processed_count_since_save += 1
                    if self.processed_count_since_save >= self.incremental_save_interval:
                        f.flush()
                        if hasattr(f, 'fileno'):
                            os.fsync(f.fileno())
                        self.processed_count_since_save = 0
                    
            except Exception as e:
                print(f"Error in incremental save: {e}")
    
    def initialize_output_file(self, output_file: str):
        """
        Initialize output file (clear or create).
        
        Args:
            output_file: Output file path
        """
        self.output_file_path = output_file
        
        # If only retrying failed records, don't clear existing file
        if self.retry_failed:
            print(f"Retry mode: Will append to existing output file: {output_file}")
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            # Ensure file exists but don't clear it
            if not output_path.exists():
                with open(output_file, 'w', encoding='utf-8') as f:
                    pass
            return
        
        # If output file exists, back it up
        output_path = Path(output_file)
        if output_path.exists():
            backup_path = output_path.with_suffix(f'.backup_{int(time.time())}.jsonl')
            print(f"Backing up existing output file to: {backup_path}")
            import shutil
            shutil.copy2(output_file, backup_path)
        
        # Clear output file (or create new file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            pass  # Create empty file
        
        print(f"Initialized output file: {output_file}")
    
    def save_failed_record_incremental(self, record_id: str, original_record: Dict, error_message: str = ""):
        """
        Incrementally save failed record to separate file (for debugging).
        
        Args:
            record_id: Record ID
            original_record: Original record data
            error_message: Error message
        """
        if not self.failed_file_path:
            return
        
        with self.incremental_save_lock:
            try:
                # Ensure output directory exists
                failed_path = Path(self.failed_file_path)
                failed_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Add failure information
                failed_record = original_record.copy()
                failed_record['processing_error'] = {
                    'error_message': error_message,
                    'timestamp': datetime.now().isoformat(),
                    'judge_model': self.judge_model_name,
                }
                
                # Sanitize record for JSON serialization
                sanitized_failed_record = self.sanitize_record_for_json(failed_record)
                
                # Append failed record
                with open(self.failed_file_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(sanitized_failed_record, separators=(',', ':')) + '\n')
                    
            except Exception as e:
                print(f"Error saving failed record: {e}")
    
    def initialize_failed_file(self, failed_file: str):
        """
        Initialize failed records file.
        
        Args:
            failed_file: Failed records file path
        """
        self.failed_file_path = failed_file
        
        # If failed file exists, back it up
        failed_path = Path(failed_file)
        if failed_path.exists():
            backup_path = failed_path.with_suffix(f'.backup_{int(time.time())}.jsonl')
            print(f"Backing up existing failed file to: {backup_path}")
            import shutil
            shutil.copy2(failed_file, backup_path)
        
        # Clear failed file (or create new file)
        failed_path.parent.mkdir(parents=True, exist_ok=True)
        with open(failed_file, 'w', encoding='utf-8') as f:
            pass  # Create empty file
        
        print(f"Initialized failed records file: {failed_file}")
    
    def process_hf_record(self, record: Dict, record_id: str) -> Tuple[Optional[Dict], bool]:
        """
        Process single HF record and generate judgment.
        
        Args:
            record: HF data record
            record_id: Unique record ID
            
        Returns:
            Tuple of (Updated record with model-as-a-judge results, success_flag)
        """
        # Choose namespace based on include_output parameter
        if self.include_output:
            feedback_key = "model_as_judge_unified"  # Include execution results, use original naming
            judger_type = "llm_judge_unified"
        else:
            feedback_key = "model_as_judge_unified_no_execution"  # Exclude execution results, use new naming
            judger_type = "llm_judge_unified_no_execution"
        
        # Check if already processed
        if 'feedback' in record and feedback_key in record['feedback']:
            existing_results = record['feedback'][feedback_key]
            if self.judge_model_name in existing_results:
                existing_judgment = existing_results[self.judge_model_name].get('judgment')
                if existing_judgment in ["vote_left", "vote_right", "vote_tie", "vote_both_bad"]:
                    print(f"[Thread-{threading.current_thread().name}] Skip already processed record: {record.get('model_A', 'unknown')} vs {record.get('model_B', 'unknown')}")
                    return record, True  # Skip already processed records
        
        try:
            # Extract HF data fields
            (instruction, model_a_name, model_b_name, model_a_code, model_b_code,
             model_a_output, model_b_output, model_a_screenshot, model_b_screenshot) = self.extract_hf_data_fields(record)
            
            if not instruction:
                print(f"[Thread-{threading.current_thread().name}] No instruction found in record")
                return record, False
            
            if not model_a_code or not model_b_code:
                print(f"[Thread-{threading.current_thread().name}] Missing code in record")
                # return record, False
            
            # Call judger (using new HF data adaptation method)
            judgment, detailed_scores, messages = self.judger.judge_hf_data(
                instruction=instruction,
                model_a_code=model_a_code,
                model_b_code=model_b_code,
                model_a_name=model_a_name,
                model_b_name=model_b_name,
                model_a_output=model_a_output if self.include_output else None,
                model_b_output=model_b_output if self.include_output else None,
                model_a_screenshot=model_a_screenshot if self.include_output else None,
                model_b_screenshot=model_b_screenshot if self.include_output else None
            )
            
            # Check if judgment was successful
            if judgment == "error":
                print(f"[Thread-{threading.current_thread().name}] Failed to get valid judgment for: {model_a_name} vs {model_b_name}")
                return record, False
            
            # Update record
            updated_record = record.copy()
            if 'feedback' not in updated_record:
                updated_record['feedback'] = {}
            
            # Use corresponding namespace
            if feedback_key not in updated_record['feedback']:
                updated_record['feedback'][feedback_key] = {}
            
            updated_record['feedback'][feedback_key][self.judge_model_name] = {
                'judgment': judgment,
                'model1_name': model_a_name,
                'model2_name': model_b_name,
                'instruction': instruction,
                'judger_type': judger_type,
                'judge_model': self.judge_model_name,
                'judge_messages': self.sanitize_messages_for_saving(messages),
                'model1_output_type': 'image' if model_a_screenshot else 'text',
                'model2_output_type': 'image' if model_b_screenshot else 'text',
                'version': '2.0',  # Add version number to distinguish new/old systems
                'timestamp': datetime.now().isoformat()  # Add timestamp
            }
            
            # Add detailed scoring data (if available)
            if detailed_scores:
                updated_record['feedback'][feedback_key][self.judge_model_name]['detailed_scores'] = detailed_scores
            
            print(f"[Thread-{threading.current_thread().name}] Successfully processed record: {model_a_name} vs {model_b_name}")
            
            return updated_record, True
            
        except Exception as e:
            print(f"[Thread-{threading.current_thread().name}] Error processing record: {e}")
            import traceback
            print(traceback.format_exc())
            
            return record, False
    
    def sanitize_messages_for_saving(self, messages: List[Dict]) -> List[Dict]:
        """
        Sanitize messages for JSON serialization, removing non-serializable objects.
        
        Args:
            messages: Original message list
            
        Returns:
            List[Dict]: Sanitized message list safe for JSON serialization
        """
        import json
        
        def is_json_serializable(obj):
            """Check if an object is JSON serializable."""
            try:
                json.dumps(obj)
                return True
            except (TypeError, ValueError):
                return False
        
        def sanitize_object(obj):
            """Recursively sanitize an object for JSON serialization."""
            if is_json_serializable(obj):
                return obj
            elif isinstance(obj, dict):
                return {k: sanitize_object(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [sanitize_object(item) for item in obj]
            elif hasattr(obj, '__class__') and 'PIL' in str(obj.__class__):
                # Handle PIL Image objects
                return f"[PIL Image object: {obj.__class__.__name__}]"
            else:
                return f"[Non-serializable object: {type(obj).__name__}]"
        
        try:
            # First try to serialize the original messages
            json.dumps(messages)
            return messages
        except (TypeError, ValueError):
            # If serialization fails, sanitize the messages
            return sanitize_object(messages)
    
    def process_record_worker(self, task_data: Tuple[Dict, str]) -> Tuple[str, bool, Dict]:
        """
        Worker function for processing a single record (for thread pool).
        
        Args:
            task_data: Tuple of (record, record_id)
            
        Returns:
            Tuple of (record_id, success_flag, updated_record)
        """
        record, record_id = task_data
        
        try:
            updated_record, success = self.process_hf_record(record, record_id)
            
            if success:
                # Incrementally save processed record
                self.save_processed_record_incremental(record_id, updated_record)
                
                # Mark as completed
                self.progress_tracker.mark_completed(record_id, 1)
            else:
                # Save failed record for debugging
                self.save_failed_record_incremental(record_id, record, "Processing failed")
                
                # Mark as failed
                self.progress_tracker.mark_failed(record_id)
            
            return record_id, success, updated_record
            
        except Exception as e:
            error_message = f"Exception during processing: {str(e)}"
            print(f"[Thread-{threading.current_thread().name}] Error processing record {record_id}: {e}")
            self.save_failed_record_incremental(record_id, record, error_message)
            self.progress_tracker.mark_failed(record_id)
            return record_id, False, record
    
    def load_and_filter_hf_records(self, dataset_name: str, split: str = "train") -> List[Tuple[Dict, str]]:
        """
        Load records directly from HuggingFace dataset and filter records that need processing.
        
        Args:
            dataset_name: HuggingFace dataset name
            split: Dataset split to load (default: "train")
            
        Returns:
            List of (record, record_id) tuples to process
        """
        print(f"Loading records from HuggingFace dataset: {dataset_name} (split: {split})")
        if self.max_records:
            print(f"Limiting to first {self.max_records} records")
        
        # Load dataset from HuggingFace
        try:
            dataset = load_dataset(dataset_name, split=split)
            print(f"Loaded {len(dataset)} records from HuggingFace")
        except Exception as e:
            print(f"Error loading dataset from HuggingFace: {e}")
            return []
        
        all_records = []
        skipped_count = 0
        
        for idx, example in enumerate(tqdm(dataset, desc="Processing records")):
            try:
                record = dict(example)
                
                record_id = self.progress_tracker.generate_record_id(record)
                
                # Use new skip logic
                if self.progress_tracker.should_skip_record(record_id, retry_failed=self.retry_failed):
                    skipped_count += 1
                    continue
                
                all_records.append((record, record_id))
                
                # If max records is set, check if limit is reached
                if self.max_records and len(all_records) >= self.max_records:
                    print(f"Reached max_records limit ({self.max_records}), stopping record collection")
                    break
                    
            except Exception as e:
                print(f"Error processing record {idx}: {e}")
                continue
        
        print(f"\nRecord loading summary:")
        print(f"Total records in dataset: {len(dataset)}")
        print(f"Already completed or permanently failed: {skipped_count}")
        print(f"Remaining to process: {len(all_records)}")
        if self.max_records:
            print(f"Limited by max_records: {self.max_records}")
        
        return all_records
    
    def process_hf_dataset(self, dataset_name: str, split: str = "train", output_file: str = None):
        """
        Process all records in HF dataset.
        
        Args:
            dataset_name: HuggingFace dataset name
            split: Dataset split to process (default: "train")
            output_file: Output file path (must be provided)
        """
        if output_file is None:
            # Auto-generate output filename according to new directory structure
            output_mode = "with_execution" if self.include_output else "without_execution"
            # New directory structure: results/{model_name}/{output_mode}/
            model_dir = f"results/{self.judge_model_name}/{output_mode}"
            output_file = f"{model_dir}/{split}-judge_{self.judge_model_name}_{output_mode}.jsonl"
        
        print(f"Processing HF dataset: {dataset_name} (split: {split})")
        print(f"Output file: {output_file}")
        print(f"Using judge model: {self.judge_model_name}")
        print(f"Using {self.max_workers} worker threads")
        print(f"Progress file: {self.progress_tracker.progress_file}")
        
        # Initialize output file for incremental saving
        self.initialize_output_file(output_file)
        
        # Initialize failed records file
        output_path = Path(output_file)
        failed_output_file = output_path.parent / f"{output_path.stem}_failed.jsonl"
        self.initialize_failed_file(str(failed_output_file))
        
        # Load records that need processing
        records_to_process = self.load_and_filter_hf_records(dataset_name, split)
        
        if not records_to_process:
            print("No records to process. All records may already be completed.")
            return
        
        # Set total record count
        stats = self.progress_tracker.get_stats()
        
        # Fix: avoid double-counting failed records in retry-failed mode
        if self.retry_failed:
            # When retrying failed records, records_to_process already contains failed records
            # So total_records = unprocessed records + completed records
            total_records = len(records_to_process) + stats['completed']
        else:
            # In normal mode, records_to_process doesn't contain failed records
            # So total_records = unprocessed records + completed records + failed records
            total_records = len(records_to_process) + stats['completed'] + stats['failed']
        
        self.progress_tracker.set_total_records(total_records)
        
        print(f"\nStarting multithreaded processing with incremental saving...")
        print(f"Records to process: {len(records_to_process)}")
        print(f"Results will be saved incrementally to: {output_file}")
        print(f"Failed records will be saved to: {failed_output_file}")
        
        # Store all updated records (kept for statistics)
        updated_records = {}
        processed_records = 0
        total_processed_votes = 0
        start_time = time.time()
        
        # Use thread pool to process records
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_record = {
                    executor.submit(self.process_record_worker, task): task 
                    for task in records_to_process
                }
                
                # Process completed tasks
                with tqdm(total=len(records_to_process), desc="Processing records") as pbar:
                    for future in concurrent.futures.as_completed(future_to_record):
                        try:
                            record_id, success, updated_record = future.result()
                            
                            # Store updated record (for statistics)
                            updated_records[record_id] = updated_record
                            
                            if success:
                                processed_records += 1
                                total_processed_votes += 1
                            
                            # Update progress bar
                            current_stats = self.progress_tracker.get_stats()
                            pbar.update(1)
                            pbar.set_postfix({
                                'Completed': f"{current_stats['completed']}/{current_stats['total_records']}",
                                'Votes': current_stats['total_votes_processed'],
                                'Failed': f"{current_stats['failed']}"
                            })
                            
                        except Exception as e:
                            print(f"Task generated an exception: {e}")
                            pbar.update(1)
                            
        except KeyboardInterrupt:
            print("\n⚠️  Processing interrupted by user")
            print("Progress has been saved. Processed results are already saved incrementally.")
            print("You can resume later by running the script again.")
        except Exception as e:
            print(f"\n❌ Error during processing: {e}")
        finally:
            self.progress_tracker.save_progress(force=True)
            # Final file sync
            if self.output_file_path:
                try:
                    with open(self.output_file_path, 'a', encoding='utf-8') as f:
                        f.flush()
                        if hasattr(f, 'fileno'):
                            os.fsync(f.fileno())
                except:
                    pass
        
        # Display statistics
        end_time = time.time()
        processing_time = end_time - start_time
        final_stats = self.progress_tracker.get_stats()
        
        print(f"\n🎉 Processing session complete!")
        print(f"📊 Session Statistics:")
        print(f"  - Records processed this session: {processed_records}")
        print(f"  - Votes processed this session: {total_processed_votes}")
        print(f"  - Session time: {processing_time:.2f} seconds")
        print(f"  - Results saved incrementally to: {output_file}")
        print(f"  - Failed records saved to: {failed_output_file}")
        
        print(f"\n📈 Overall Progress:")
        print(f"  - Total records: {final_stats['total_records']}")
        print(f"  - Completed: {final_stats['completed']} ({final_stats['completion_rate']:.1%})")
        print(f"  - Failed (total): {final_stats['failed']}")
        print(f"  - Remaining: {final_stats['remaining']}")
        print(f"  - Total votes processed: {final_stats['total_votes_processed']}")
        
        if processed_records > 0:
            print(f"\n⚡ Performance:")
            print(f"  - Average time per record: {processing_time/processed_records:.3f} seconds")


def main():
    """Main function to run the HF data model judger processor."""
    import argparse
    
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description='HF Data Model-as-a-Judge Evaluation System')
    parser.add_argument('--judge-model', type=str, default='sonnet35v2', help='Judge model to use (default: sonnet35v2)')
    parser.add_argument('--max-records', type=int, help='Maximum number of records to process')
    parser.add_argument('--output', type=str, help='Output file path (default: auto-generated based on judge model)')
    parser.add_argument('--workers', type=int, default=1, help='Number of worker threads (default: 1)')
    parser.add_argument('--retry-failed', action='store_true', help='Retry previously failed records instead of skipping them')
    parser.add_argument('--include-output', action='store_true', default=True, help='Include model outputs in evaluation (default: True)')
    parser.add_argument('--no-output', action='store_true', help='Exclude model outputs from evaluation (equivalent to --include-output=False)')
    parser.add_argument('--dataset', type=str, default='bigcode/bigcodereward', help='HuggingFace dataset name (default: bigcode/bigcodereward)')
    parser.add_argument('--split', type=str, default='train', help='Dataset split to process (default: train)')
    parser.add_argument('--config', type=str, default='config/bigcodereward.yaml', help='Path to config file')
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    else:
        print(f"Config file {args.config} not found, using defaults")
        config = {}
    
    # Handle include_output parameter logic
    if args.no_output:
        include_output = False
    else:
        include_output = args.include_output
    
    # Configure paths
    judge_prompts_dir = config.get('paths', {}).get('prompts_dir', 'prompts')
    
    # Get configuration from command line arguments
    judge_model = args.judge_model
    max_records = args.max_records
    output_file = args.output
    max_workers = args.workers
    retry_failed = args.retry_failed
    dataset_name = args.dataset
    split = args.split
    
    # Determine output mode description
    output_mode = "with_execution" if include_output else "without_execution"
    
    # Generate independent progress file for each judge model and output mode combination
    # New directory structure: results/{model_name}/{output_mode}/
    results_dir = config.get('paths', {}).get('results_dir', 'results')
    model_dir = f"{results_dir}/{judge_model}/{output_mode}"
    progress_file = f"{model_dir}/jsonl_experiment_progress_{judge_model}_{output_mode}.json"
    
    # If no output file specified, auto-generate
    if not output_file:
        output_file = f"{model_dir}/{split}-judge_{judge_model}_{output_mode}.jsonl"
        print(f"📁 Auto-generated output file: {output_file}")
    
    # Display configuration information
    if max_records:
        print(f"🔍 Processing only first {max_records} records")
        
    print(f"📁 Dataset: {dataset_name} (split: {split})")
    print(f"📁 Output file: {output_file}")
    print(f"📁 Progress file: {progress_file}")
    print(f"🤖 Judge model: {judge_model}")
    print(f"📊 Include execution: {include_output} ({output_mode})")
    print(f"⚡ Using {max_workers} worker threads")
    
    # Create LLM judger
    print(f"Initializing LLM Judge with model: {judge_model}")
    judger = CodeArenaJudger(judge_model_id=judge_model, prompts_dir=judge_prompts_dir, include_output=include_output)
    
    # Display simplified content truncation strategy
    print(f"\n📊 Simplified Content Truncation Strategy:")
    print(f"  - Model: {judge_model}")
    print(f"  - Include execution results: {include_output} ({output_mode})")
    print(f"  - Instruction truncation: max {judger.max_instruction_length:,} characters")
    print(f"  - Code truncation: max {judger.max_code_length:,} characters")
    print(f"  - Output truncation: max {judger.max_output_length:,} characters")
    print(f"  - Truncation strategy: simple character length truncation")
    
    # Initialize processor
    processor = HFDataJudgerProcessor(
        judger=judger,
        judge_model_name=judge_model,
        max_workers=max_workers,
        progress_file=progress_file,
        max_records=max_records,
        retry_failed=retry_failed,
        include_output=include_output
    )
    
    # Display current progress status
    stats = processor.progress_tracker.get_stats()
    if stats['total_records'] > 0:
        print(f"\n📋 Current Progress Status:")
        print(f"  - Completed: {stats['completed']}/{stats['total_records']} ({stats['completion_rate']:.1%})")
        print(f"  - Failed (total): {stats['failed']}")
        print(f"  - Remaining: {stats['remaining']}")
        print(f"  - Total votes processed: {stats['total_votes_processed']}")
    
    # Process HF dataset
    processor.process_hf_dataset(dataset_name, split, output_file)


if __name__ == "__main__":
    main()