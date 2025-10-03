# code_arena_judger.py
"""Code Arena Judger for comparing model outputs in coding tasks."""

import json
import os
from pathlib import Path
import re
import time
from typing import Dict, List, Tuple, Optional
import litellm
from litellm import completion
import threading
import random
import base64
from PIL import Image
import io
import openai
import yaml

# Disable debug info output
litellm.suppress_debug_info = True

class CodeArenaJudger:
    """Code LLM Arena judger for comparing code output quality between two models (thread-safe version)."""
    
    def __init__(self, judge_model_id="sonnet35v2", prompts_dir="prompts", include_output=True, 
                 simple_mode=False, config_path="config/judge_model_config.yaml"):
        """
        Initialize Code Arena Judger.
        
        Args:
            judge_model_id: Model ID for judging
            prompts_dir: Directory path for judge prompts
            include_output: Whether to include model output results in judgment 
                          (True: include execution results, False: code only)
            simple_mode: Whether to use simplified mode for speed testing 
                        (True: use fixed short text, False: use complete prompt)
            config_path: Path to the judge model configuration YAML file
        """
        self.judge_model_id = judge_model_id
        self.prompts_dir = Path(prompts_dir)
        self.include_output = include_output  # Control whether to include output
        self.simple_mode = simple_mode  # Control whether to use simplified mode
        
        # Load configuration from YAML file
        self.config = self.load_config(config_path)
        self.model_config = self.config['model_mappings'].get(judge_model_id, {})
        
        # Screenshot processing configuration from config
        image_settings = self.config.get('image_settings', {})
        self.enable_screenshot_compression = image_settings.get('enable_screenshot_compression', True)
        self.screenshot_max_size = tuple(image_settings.get('screenshot_max_size', [1024, 768]))
        self.screenshot_quality = image_settings.get('screenshot_quality', 70)
        
        # Load unified prompt template
        self.judge_prompts = self.load_judge_prompts()
        
        # Thread safety related
        self._lock = threading.Lock()
        self._last_request_time = 0
        # Set request intervals based on model config
        self._min_request_interval = self.model_config.get('min_request_interval', 1.0)
        self._request_count = 0
        
        # Context window management parameters from config
        self.model_context_limits = {
            model_name: config.get('context_limit', 128000) 
            for model_name, config in self.config['model_mappings'].items()
        }
        
        # Content limits from config
        content_limits = self.config.get('content_limits', {})
        self.max_code_length = content_limits.get('max_code_length', 20000)
        self.max_instruction_length = content_limits.get('max_instruction_length', 5000)
        self.max_output_length = content_limits.get('max_output_length', 3000)
        self.reserved_tokens = content_limits.get('reserved_tokens', 10000)
        
        # Image compression settings from config
        self.max_image_size = tuple(image_settings.get('max_image_size', [1024, 1024]))
        self.jpeg_quality = image_settings.get('jpeg_quality', 85)
    
    def load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        config_file = Path(config_path)
        
        if not config_file.exists():
            print(f"Warning: Config file {config_path} not found, using default configuration")
            return self.get_default_config()
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            print(f"Error loading config file {config_path}: {e}")
            print("Using default configuration")
            return self.get_default_config()
    
    def get_default_config(self) -> Dict:
        """Return default configuration if config file is not found."""
        return {
            'model_mappings': {},
            'content_limits': {
                'max_code_length': 20000,
                'max_instruction_length': 5000,
                'max_output_length': 3000,
                'reserved_tokens': 10000
            },
            'image_settings': {
                'enable_screenshot_compression': True,
                'screenshot_max_size': [1024, 768],
                'screenshot_quality': 70,
                'max_image_size': [1024, 1024],
                'jpeg_quality': 85
            }
        }
        
    def get_model_id(self, model: str) -> str:
        """Get identifier for different models from config."""
        model_config = self.config['model_mappings'].get(model)
        if model_config:
            return model_config.get('model_id', 'error')
        return "error"  # Return error if no mapping found
    
    def load_judge_prompts(self) -> str:
        """Load unified judge prompts (Visual mode, compatible with all situations)."""
        try:
            # Choose different prompt files based on include_output parameter
            if self.include_output:
                prompt_file = self.prompts_dir / "judge_prompt_unified.md"
                prompt_type = "with output"
            else:
                prompt_file = self.prompts_dir / "judge_prompt_unified_no_output.md"
                prompt_type = "without output"
            
            if prompt_file.exists():
                with open(prompt_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    print(f"Loaded judge prompt ({prompt_type}): {prompt_file}")
                    return content
            else:
                print(f"Warning: Prompt file {prompt_file} not found, falling back to basic prompt")
                # Use basic prompt as fallback
                basic_prompt_file = self.prompts_dir / "judge_prompt.md"
                if basic_prompt_file.exists():
                    with open(basic_prompt_file, 'r', encoding='utf-8') as f:
                        return f.read().strip()
                else:
                    print(f"Warning: No prompt files found in {self.prompts_dir}")
                    return "Default unified prompt placeholder"
                    
        except Exception as e:
            print(f"Error loading judge prompt from {self.prompts_dir}: {e}")
            return "Default unified prompt placeholder"
    
    def _rate_limit_request(self):
        """Simple request rate limiting (thread-safe)."""
        with self._lock:
            current_time = time.time()
            time_since_last = current_time - self._last_request_time
            
            if time_since_last < self._min_request_interval:
                sleep_time = self._min_request_interval - time_since_last
                time.sleep(sleep_time)
            
            self._last_request_time = time.time()
            self._request_count += 1
    
    def encode_image_to_base64(self, image_path: str) -> Optional[str]:
        """Encode image to base64 string for API usage."""
        try:
            with open(image_path, 'rb') as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            return None
    
    def get_image_mime_type(self, image_path: str) -> str:
        """Get MIME type based on file extension."""
        extension = Path(image_path).suffix.lower()
        mime_types = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.bmp': 'image/bmp',
            '.webp': 'image/webp'
        }
        return mime_types.get(extension, 'image/png')
    
    def build_messages_from_hf_data(self, instruction: str, model_a_code: str, model_b_code: str, 
                                   model_a_output: str = None, model_b_output: str = None,
                                   model_a_screenshot: str = None, model_b_screenshot: str = None) -> List[Dict]:
        """
        Build messages from Hugging Face data, adapted for new data format.
        
        Args:
            instruction: User instruction
            model_a_code: Model A's code
            model_b_code: Model B's code
            model_a_output: Model A's output (optional)
            model_b_output: Model B's output (optional)
            model_a_screenshot: Model A's screenshot base64 (optional)
            model_b_screenshot: Model B's screenshot base64 (optional)
            
        Returns:
            List of messages for API call
        """
        # If in simple mode, use fixed short text for speed testing
        if self.simple_mode:
            return self.build_simple_messages_from_hf_data(model_a_screenshot, model_b_screenshot)
        
        # Build complete answers for models A and B
        answer_a = self.build_complete_answer_from_hf_data(model_a_code, model_a_output)
        answer_b = self.build_complete_answer_from_hf_data(model_b_code, model_b_output)
        
        # Build conditional screenshot sections
        screenshot_a_section = self.build_screenshot_section_from_hf_data(model_a_screenshot, "A")
        screenshot_b_section = self.build_screenshot_section_from_hf_data(model_b_screenshot, "B")
        visual_a_section = ""  # HF data doesn't need additional visual sections for now
        visual_b_section = ""
        
        # Replace placeholders in template
        text_filled = self.judge_prompts.replace("{INSTRUCTION}", instruction)
        text_filled = text_filled.replace("{ANSWER_A}", answer_a)
        text_filled = text_filled.replace("{ANSWER_B}", answer_b)
        text_filled = text_filled.replace("{SCREENSHOT_A_SECTION}", screenshot_a_section)
        text_filled = text_filled.replace("{SCREENSHOT_B_SECTION}", screenshot_b_section)
        text_filled = text_filled.replace("{VISUAL_A_SECTION}", visual_a_section)
        text_filled = text_filled.replace("{VISUAL_B_SECTION}", visual_b_section)
        
        # Build message content, handle screenshot placeholders
        content = []
        
        # Split and handle screenshot placeholders
        if "{SCREENSHOT_A}" in text_filled:
            parts = text_filled.split("{SCREENSHOT_A}")
            content.append({"type": "text", "text": parts[0]})
            content.append(self.get_image_content_from_hf_data(model_a_screenshot))
            remaining = parts[1]
            
            if "{SCREENSHOT_B}" in remaining:
                parts = remaining.split("{SCREENSHOT_B}")
                content.append({"type": "text", "text": parts[0]})
                content.append(self.get_image_content_from_hf_data(model_b_screenshot))
                content.append({"type": "text", "text": parts[1]})
            else:
                content.append({"type": "text", "text": remaining})
        elif "{SCREENSHOT_B}" in text_filled:
            parts = text_filled.split("{SCREENSHOT_B}")
            content.append({"type": "text", "text": parts[0]})
            content.append(self.get_image_content_from_hf_data(model_b_screenshot))
            content.append({"type": "text", "text": parts[1]})
        else:
            # No screenshots, return text directly
            content.append({"type": "text", "text": text_filled})
        
        return [{"role": "user", "content": content}]
    
    def build_complete_answer_from_hf_data(self, code: str, output: str = None) -> str:
        """
        Build model's complete answer from HF data.
        
        Args:
            code: Model code
            output: Model output (optional)
            
        Returns:
            str: Formatted complete answer
        """
        # Build answer
        answer_parts = []
        
        # Add code (simple truncation)
        if code:
            if len(code) > self.max_code_length:
                code = code[:self.max_code_length] + "\n...[Code truncated]"
            answer_parts.append(f"<|The Start of Code|>\n{code}\n<|The End of Code|>")
        
        # Decide whether to include execution results based on include_output parameter
        if self.include_output and output:
            # Add execution results (simple truncation)
            if len(output) > self.max_output_length:
                output = output[:self.max_output_length] + "\n...[Truncated]"
            
            answer_parts.append(f"<|The Start of Execution Results|>\n{output}\n<|The End of Execution Results|>")
        
        return "\n\n".join(answer_parts) if answer_parts else "[No response found]"
    
    def build_screenshot_section_from_hf_data(self, screenshot_base64: str, assistant_label: str) -> str:
        """
        Build screenshot section from HF data.
        
        Args:
            screenshot_base64: Base64 encoded screenshot
            assistant_label: Assistant label (A or B)
            
        Returns:
            str: Screenshot section text
        """
        # If not including output, return empty string directly
        if not self.include_output:
            return ""
        
        # Check if screenshot exists
        if screenshot_base64:
            return f"\n<|The Start of Assistant {assistant_label}'s Artifact Screenshot|>\n{{SCREENSHOT_{assistant_label}}}\n<|The End of Assistant {assistant_label}'s Artifact Screenshot|>"
        return ""
    
    def get_image_content_from_hf_data(self, screenshot_base64: str) -> Dict:
        """
        Get image content message format from HF data.
        
        Args:
            screenshot_base64: Base64 encoded screenshot
            
        Returns:
            Dict: Image message content
        """
        if not screenshot_base64:
            return {"type": "text", "text": "[No image output]"}
        
        # Compress image to reduce data size
        compressed_b64 = self.compress_image_base64(screenshot_base64)
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{compressed_b64}"
            }
        }
    
    def compress_image_base64(self, image_base64: str, max_size: tuple = None, quality: int = None) -> str:
        """
        Compress base64 encoded image to reduce data size.
        
        Args:
            image_base64: Original base64 encoded image
            max_size: Maximum size (width, height), defaults to instance configuration
            quality: JPEG quality (1-100), defaults to instance configuration
            
        Returns:
            str: Compressed base64 encoding
        """
        if not self.enable_screenshot_compression:
            return image_base64
        
        max_size = max_size or self.screenshot_max_size
        quality = quality or self.screenshot_quality
        
        try:
            # Decode base64
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB mode (if needed)
            if image.mode in ('RGBA', 'LA', 'P'):
                image = image.convert('RGB')
            
            # Resize
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Compress and re-encode
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=quality, optimize=True)
            compressed_data = buffer.getvalue()
            
            # Re-encode to base64
            compressed_base64 = base64.b64encode(compressed_data).decode('utf-8')
            
            # Calculate compression ratio
            original_size = len(image_base64)
            compressed_size = len(compressed_base64)
            compression_ratio = compressed_size / original_size
            
            print(f"Image compressed: {original_size:,} -> {compressed_size:,} bytes ({compression_ratio:.2%})")
            
            return compressed_base64
            
        except Exception as e:
            print(f"Error compressing image: {e}")
            return image_base64  # Return original data as fallback
    
    def call_judge_model(self, messages: List[Dict], max_retries=100, **kwargs) -> Optional[str]:
        """
        Call judge model (thread-safe version).
        
        Args:
            messages: Message list
            max_retries: Maximum retry count
            
        Returns:
            str: Model reply content, None if failed, "CONTEXT_WINDOW_EXCEEDED" if context window error
        """
        thread_name = threading.current_thread().name
        
        # Check if it's SGLang model, if so set local API base
        model_id = self.get_model_id(self.judge_model_id)
        is_sglang_model = self.model_config.get('api_type') == 'sglang'
        
        # Save original settings
        original_api_base = getattr(litellm, 'api_base', None)
        original_api_key = getattr(litellm, 'api_key', None)
        
        try:
            # 如果是SGLang模型，设置本地API配置
            if is_sglang_model:
                api_base = self.model_config.get('api_base', 'http://localhost:30000/v1')
                litellm.api_base = api_base
                litellm.api_key = "dummy"
                print(f"[Thread-{thread_name}] Using SGLang API for model: {self.judge_model_id} at {api_base}")
        
            for attempt in range(max_retries):
                try:
                    # 应用频率限制
                    self._rate_limit_request()
                    
                    # 添加随机抖动以减少并发冲突
                    if attempt > 0:
                        jitter = random.uniform(0.1, 0.5) * attempt
                        time.sleep(jitter)
                    
                    print(f"[Thread-{thread_name}] Making API call, attempt {attempt + 1}/{max_retries}")
                    
                    # 对于本地模型，需要使用实际的模型路径
                    actual_model = model_id
                    # Check if custom model path is specified in config
                    if is_sglang_model and 'custom_model_path' in self.model_config:
                        actual_model = self.model_config['custom_model_path']
                    
                    # 为 OpenAI 兼容 API 设置额外参数
                    completion_kwargs = {
                        'model': actual_model,
                        'messages': messages,
                        'temperature': 0,
                        'max_tokens': 4000,  # 增加token数量以支持JSON输出
                        'timeout': 300 if is_sglang_model else 60,  # SGLang模型使用更长超时
                        **kwargs
                    }
                    
                    # 对于本地 OpenAI 兼容 API，需要设置 custom_llm_provider
                    if is_sglang_model and 'custom_model_path' in self.model_config:
                        completion_kwargs['custom_llm_provider'] = 'openai'
                        completion_kwargs['api_base'] = self.model_config.get('api_base', 'http://localhost:30000/v1')
                        completion_kwargs['api_key'] = "dummy"
                    
                    response = completion(**completion_kwargs)
                    
                    if response['choices'][0]['message']['content']:
                        print(f"[Thread-{thread_name}] API call successful")
                        return response['choices'][0]['message']['content']
                    else:
                        raise Exception("No response from judge model")
                        
                except Exception as e:
                    error_str = str(e).lower()
                    print(f"[Thread-{thread_name}] Judge model error on attempt {attempt + 1}: {e}")
                    
                    # 检查是否是context window exceeded错误
                    if ("context window" in error_str or 
                        "input is too long" in error_str or 
                        "context_length_exceeded" in error_str or
                        "maximum context" in error_str or
                        "string too long" in error_str or  # OpenAI字符串长度限制
                        "maximum length" in error_str):  # OpenAI最大长度限制
                        print(f"[Thread-{thread_name}] Context window/length exceeded error detected!")
                        # 对于context window/length错误，返回特殊状态，让上层进行更激进的优化
                        return "CONTEXT_WINDOW_EXCEEDED"
                    
                    if attempt == max_retries - 1:
                        print(f"[Thread-{thread_name}] Max retries reached for judge model")
                        return None
                    
                    # 根据错误类型决定退避时间
                    if "rate_limit" in error_str or "429" in error_str:
                        backoff_time = min(60, 5 * (attempt + 1))  # Rate limit: 递增退避，最多60秒
                    elif "500" in error_str or "502" in error_str or "503" in error_str:
                        backoff_time = min(30, 3 * (attempt + 1))  # Server error: 较短退避
                    else:
                        backoff_time = 10  # 其他错误：固定退避
                    
                    print(f"[Thread-{thread_name}] Waiting {backoff_time} seconds before retry...")
                    time.sleep(backoff_time)
        
        finally:
            # 恢复原始API设置
            if is_sglang_model:
                if original_api_base is not None:
                    litellm.api_base = original_api_base
                else:
                    if hasattr(litellm, 'api_base'):
                        delattr(litellm, 'api_base')
                
                if original_api_key is not None:
                    litellm.api_key = original_api_key
                else:
                    if hasattr(litellm, 'api_key'):
                        delattr(litellm, 'api_key')
        
        return None

    def call_conversion_model(self, messages: List[Dict], max_retries=3, **kwargs):
        """
        专门调用gpt-4o-mini进行JSON格式转换（简化版本）
        Args:
            messages: 消息列表
            max_retries: 最大重试次数
        Returns:
            str: 模型回复内容，如果失败返回None，如果context window错误返回"CONTEXT_WINDOW_EXCEEDED"
        """
        thread_name = threading.current_thread().name
        conversion_model = "gpt-4.1-mini-2025-04-14" 
        
        # 从环境变量获取API密钥
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print(f"[Thread-{thread_name}] Error: OPENAI_API_KEY not found in environment variables")
            return None
        
        client = openai.OpenAI(api_key=api_key)
        
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    time.sleep(0.5)  # 转换任务间隔较短
                
                print(f"[Thread-{thread_name}] Making conversion API call to {conversion_model}, attempt {attempt + 1}/{max_retries}")
                
                response = client.chat.completions.create(
                    model=conversion_model,
                    messages=messages,
                    temperature=0,
                    max_tokens=2000,
                    timeout=30
                )
                
                content = response.choices[0].message.content
                if content:
                    print(f"[Thread-{thread_name}] Conversion API call successful")
                    return content
                else:
                    raise Exception("No response from conversion model")
                    
            except Exception as e:
                error_str = str(e).lower()
                print(f"[Thread-{thread_name}] Conversion model error on attempt {attempt + 1}: {e}")
                
                # 检查是否是context window exceeded错误
                if ("context window" in error_str or 
                    "input is too long" in error_str or 
                    "context_length_exceeded" in error_str or
                    "maximum context" in error_str or
                    "string too long" in error_str or
                    "maximum length" in error_str):
                    print(f"[Thread-{thread_name}] Context window/length exceeded error in conversion!")
                    return "CONTEXT_WINDOW_EXCEEDED"
                
                if attempt == max_retries - 1:
                    print(f"[Thread-{thread_name}] Max retries reached for conversion model")
                    return None
                
                # 简单的退避策略
                backoff_time = 2 * (attempt + 1)
                print(f"[Thread-{thread_name}] Waiting {backoff_time} seconds before retry...")
                time.sleep(backoff_time)
        
        return None

    def convert_to_json_format(self, original_response: str) -> str:
        """
        调用gpt-4.1-mini将原始响应转换为JSON格式
        Args:
            original_response: 原始的模型响应
        Returns:
            str: 转换后的JSON格式响应，如果失败返回None
        """
        thread_name = threading.current_thread().name
        
        conversion_prompt = f"""The following is a response from a judge model that should be in JSON format, but it's not properly formatted. 
        Please convert it to the required JSON format. "reasoning" is a single paragraph explanation without line breaks. Any quotation marks within the text should be properly escaped for a valid JSON format.

The expected JSON format should be:
{{
    "Overall": {{
        "winner": "A" | "B" | "TIE",
        "reasoning": "explanation for the overall judgment"
    }},
}}

Original response:
{original_response}

Please output ONLY the JSON format, no additional text or explanation."""

        messages = [{"role": "user", "content": conversion_prompt}]
        
        try:
            print(f"[Thread-{thread_name}] Attempting to convert response to JSON format using gpt-4.1-mini...")
            converted_response = self.call_conversion_model(messages, max_retries=3)
            
            if converted_response and converted_response != "CONTEXT_WINDOW_EXCEEDED":
                print(f"[Thread-{thread_name}] JSON conversion successful")
                return converted_response
            else:
                print(f"[Thread-{thread_name}] JSON conversion failed - no response from model")
                return None
                
        except Exception as e:
            print(f"[Thread-{thread_name}] JSON conversion failed with error: {e}")
            return None

    def parse_json_judgment(self, judge_response: str) -> Tuple[Dict, str]:
        """
        从判断回复中解析JSON格式的判断结果（线程安全）
        Args:
            judge_response: 判断模型的回复
        Returns:
            Tuple of (parsed_json_dict, overall_judgment)
        """
        thread_name = threading.current_thread().name
        
        if not judge_response:
            return {}, "error"
        
        try:
            # 尝试直接解析JSON
            json_data = json.loads(judge_response)
            print(f"[Thread-{thread_name}] Successfully parsed JSON directly")
        except json.JSONDecodeError as e:
            print(f"[Thread-{thread_name}] Direct JSON parsing failed: {e}")
            print(f"[Thread-{thread_name}] Response starts with: {judge_response}")
            # 如果直接解析失败，尝试提取JSON部分
            try:
                # 先尝试移除markdown代码块标记
                response = judge_response.strip()
                
                # 更好的markdown代码块处理
                if response.startswith('```json'):
                    response = response[7:]  # 移除 '```json'
                elif response.startswith('```'):
                    response = response[3:]  # 移除 '```'
                
                if response.endswith('```'):
                    response = response[:-3]  # 移除结尾的 '```'
                
                # 清理换行和空格
                response = response.strip()
                
                # 修复常见的JSON格式问题
                # 1. 处理未转义的单引号问题
                response = response.replace("\\'", "'")  # 将 \' 替换为 '
                
                # 2. 处理LaTeX数学表达式中的未转义反斜杠问题
                # 这是导致 "Invalid \escape" 错误的主要原因
                # 需要将单个反斜杠转义为双反斜杠，但要小心不要破坏已经正确转义的内容
                # 匹配未转义的反斜杠（不是 \\ 的单个 \）
                # 使用负向后查找确保不匹配已经转义的反斜杠
                response = re.sub(r'(?<!\\)\\(?!\\|["\'/bfnrt])', r'\\\\', response)
                
                # 3. 处理可能的多余空格和换行（保持JSON结构）
                # 先尝试直接解析清理后的响应
                try:
                    json_data = json.loads(response)
                    print(f"[Thread-{thread_name}] Successfully parsed JSON after markdown cleanup")
                except json.JSONDecodeError:
                    # 如果还是不能直接解析，则查找JSON块
                    if not response.startswith('{'):
                        start_idx = response.find('{')
                        if start_idx == -1:
                            raise ValueError("No JSON block found")
                        
                        # 找到匹配的结束括号
                        bracket_count = 0
                        end_idx = start_idx
                        for i, char in enumerate(response[start_idx:], start_idx):
                            if char == '{':
                                bracket_count += 1
                            elif char == '}':
                                bracket_count -= 1
                                if bracket_count == 0:
                                    end_idx = i
                                    break
                        
                        response = response[start_idx:end_idx+1]
                    
                    json_data = json.loads(response)
            except (json.JSONDecodeError, ValueError) as e:
                print(f"[Thread-{thread_name}] Failed to parse JSON from response: {e}")
                print(f"[Thread-{thread_name}] Attempting to convert response to JSON format...")
                
                # 尝试通过模型转换格式
                print(f"[Thread-{thread_name}] Attempting JSON conversion via model...")
                converted_response = self.convert_to_json_format(judge_response)
                
                if converted_response and converted_response != "CONTEXT_WINDOW_EXCEEDED":
                    try:
                        # 尝试解析转换后的响应，需要先清理格式
                        cleaned_response = converted_response.strip()
                        if cleaned_response.startswith('```json'):
                            cleaned_response = cleaned_response[7:]  # 移除 '```json'
                        if cleaned_response.startswith('```'):
                            cleaned_response = cleaned_response[3:]  # 移除 '```'
                        if cleaned_response.endswith('```'):
                            cleaned_response = cleaned_response[:-3]  # 移除结尾的 '```'
                        cleaned_response = cleaned_response.strip()
                        
                        json_data = json.loads(cleaned_response)
                        print(f"[Thread-{thread_name}] Successfully parsed JSON after conversion")
                    except json.JSONDecodeError as conversion_error:
                        print(f"[Thread-{thread_name}] Failed to parse converted JSON: {conversion_error}")
                        print(f"[Thread-{thread_name}] Original response: {judge_response}")
                        print(f"[Thread-{thread_name}] Converted response: {converted_response}")
                        return {}, "error"
                else:
                    print(f"[Thread-{thread_name}] JSON conversion failed or unavailable")
                    print(f"[Thread-{thread_name}] Skipping conversion and returning error")
                    print(f"[Thread-{thread_name}] Original response: {judge_response}")
                    return {}, "error"
        
        # 计算总体判断结果
        overall_judgment = self.calculate_overall_judgment(json_data)
        
        return json_data, overall_judgment
    
    def calculate_overall_judgment(self, json_data: Dict) -> str:
        """
        根据JSON数据计算总体判断结果（线程安全）
        Args:
            json_data: 解析后的JSON数据
        Returns:
            str: "vote_left", "vote_right", "vote_tie", "vote_both_bad", 或 "error"
        """
        if not json_data:
            return "error"
        
        # 获取Overall维度的判断结果
        overall_dimension = json_data.get("Overall", {})
        
        # 处理两种可能的格式
        if isinstance(overall_dimension, dict):
            # 期望的格式: {"winner": "A", "reasoning": "..."}
            winner = overall_dimension.get("winner", "").upper()
        elif isinstance(overall_dimension, str):
            # 实际返回的格式: "A"
            winner = overall_dimension.upper()
        else:
            print(f"[Thread-{threading.current_thread().name}] Unexpected Overall format: {type(overall_dimension)}")
            return "error"
        
        # 根据winner映射到最终结果
        if winner == "A":
            return "vote_left"  # Model A (left) wins
        elif winner == "B":
            return "vote_right"  # Model B (right) wins
        elif winner == "TIE":
            return "vote_tie"  # Tie
        elif winner == "BOTH_BAD":
            return "vote_both_bad"  # Both solutions are bad
        else:
            print(f"[Thread-{threading.current_thread().name}] Unexpected winner value: {winner}")
            return "error"
    
    def extract_judgment(self, judge_response: str, prompt) -> str:
        """
        从判断回复中提取最终判断结果（保持向后兼容，线程安全）
        Args:
            judge_response: 判断模型的回复
        Returns:
            str: "vote_left", "vote_right", "vote_tie", "vote_both_bad", 或 "error"
        """
        thread_name = threading.current_thread().name
        
        # 如果是简化模式，优先处理简单格式的回复
        if self.simple_mode:
            # 在简化模式下，模型可能直接返回 A、B、TIE、BOTH_BAD
            response_upper = judge_response.upper().strip()
            if response_upper == "A" or "SOLUTION A" in response_upper:
                return "vote_left"
            elif response_upper == "B" or "SOLUTION B" in response_upper:
                return "vote_right"
            elif response_upper == "TIE":
                return "vote_tie"
            elif response_upper == "BOTH_BAD":
                return "vote_both_bad"
            # 如果简单格式失败，继续尝试完整的解析
        
        # 首先尝试解析JSON格式
        print(f'judge_response: {judge_response}')
        json_data, overall_judgment = self.parse_json_judgment(judge_response)
        
        if overall_judgment != "error":
            return overall_judgment
        
        # 如果JSON解析失败，回退到原来的文本解析方法
        if not judge_response:
            return "error"
        
        response_lower = judge_response.lower()
        
        # 查找最终判断
        if "vote_left" in response_lower:
            return "vote_left"
        elif "vote_right" in response_lower:
            return "vote_right"
        elif "vote_tie" in response_lower:
            return "vote_tie"
        elif "vote_both_bad" in response_lower:
            return "vote_both_bad"
        else:
            # If no clear judgment, try other keywords
            if "model a" in response_lower and "better" in response_lower:
                return "vote_left"
            elif "model b" in response_lower and "better" in response_lower:
                return "vote_right"
            elif "equal" in response_lower or "equivalent" in response_lower:
                return "vote_tie"
            elif "both_bad" in response_lower or "both bad" in response_lower or "neither" in response_lower:
                return "vote_both_bad"
            else:
                print(f"❗️[Thread-{thread_name}] Could not extract clear judgment from: {judge_response} | response_lower: {response_lower}")
                print(f'\n\n')
                print(f'='*10)
                return "error"
    
    def judge_hf_data(self, instruction: str, model_a_code: str, model_b_code: str,
                      model_a_name: str, model_b_name: str,
                      model_a_output: str = None, model_b_output: str = None,
                      model_a_screenshot: str = None, model_b_screenshot: str = None) -> Tuple[str, Dict, List[Dict]]:
        """
        Judge quality of two model outputs (adapted for HF data format, thread-safe).
        
        Args:
            instruction: User instruction
            model_a_code: Model A's code
            model_b_code: Model B's code
            model_a_name: Model A name
            model_b_name: Model B name
            model_a_output: Model A's output (optional)
            model_b_output: Model B's output (optional)
            model_a_screenshot: Model A's screenshot base64 (optional)
            model_b_screenshot: Model B's screenshot base64 (optional)
            
        Returns:
            Tuple[str, Dict, List[Dict]]: (judgment result, detailed scores, messages passed to model)
        """
        thread_name = threading.current_thread().name
        
        print(f"[Thread-{thread_name}] Judging: {model_a_name} vs {model_b_name}")
        print(f"[Thread-{thread_name}] Instruction: {instruction}")
        
        # Simple truncation for overly long instructions
        if len(instruction) > self.max_instruction_length:
            instruction = instruction[:self.max_instruction_length] + "...[Instruction truncated]"
        
        # Build messages
        messages = self.build_messages_from_hf_data(
            instruction=instruction,
            model_a_code=model_a_code,
            model_b_code=model_b_code,
            model_a_output=model_a_output,
            model_b_output=model_b_output,
            model_a_screenshot=model_a_screenshot,
            model_b_screenshot=model_b_screenshot
        )
        
        # Call judge model
        print(f"[Thread-{thread_name}] Calling judge model...")
        judge_response = self.call_judge_model(messages)
        
        if judge_response is None:
            print(f"[Thread-{thread_name}] Judge model failed")
            return "error", {}, messages
        
        print(f"[Thread-{thread_name}] Judge response received, extracting judgment...")
        
        # Parse judgment result
        judgment = self.extract_judgment(judge_response, str(messages))
        
        # Try to parse detailed JSON scores
        json_data, _ = self.parse_json_judgment(judge_response)
        
        print(f"[Thread-{thread_name}] Final judgment: {judgment}")
        
        return judgment, json_data, messages
    
    def build_simple_messages_from_hf_data(self, model_a_screenshot: str, model_b_screenshot: str) -> List[Dict]:
        """
        Build simplified messages from HF data for speed testing - using fixed short text + images.
        
        Args:
            model_a_screenshot: Model A's screenshot base64
            model_b_screenshot: Model B's screenshot base64
            
        Returns:
            List of messages for API call
        """
        # Use fixed short text (about 100 words)
        simple_text = """Please compare these two code solutions. 
        
Solution A: Basic implementation approach.
Solution B: Alternative implementation method.

Please analyze which solution is better and respond with A, B, TIE, or BOTH_BAD.

Image A:
"""
        
        # Build message content
        content = []
        
        # 添加开始文字
        content.append({"type": "text", "text": simple_text})
        
        # 添加第一个图片（如果存在）
        image_a_content = self.get_image_content_from_hf_data(model_a_screenshot)
        if image_a_content.get("type") == "image_url":
            content.append(image_a_content)
        else:
            content.append({"type": "text", "text": "[No image A]"})
        
        # Add middle text
        content.append({"type": "text", "text": "\n\nImage B:\n"})
        
        # Add second image (if exists)
        image_b_content = self.get_image_content_from_hf_data(model_b_screenshot)
        if image_b_content.get("type") == "image_url":
            content.append(image_b_content)
        else:
            content.append({"type": "text", "text": "[No image B]"})
        
        # Add ending text
        content.append({"type": "text", "text": "\n\nYour judgment:"})
        
        return [{"role": "user", "content": content}]
    
    def get_statistics(self) -> Dict:
        """
        Get statistics (thread-safe).
        
        Returns:
            Dict: Statistics dictionary
        """
        with self._lock:
            return {
                "total_requests": self._request_count,
                "judge_model_id": self.judge_model_id,
            }
