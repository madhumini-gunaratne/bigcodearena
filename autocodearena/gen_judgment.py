import json
import yaml
import argparse
import os
import base64
import concurrent.futures
from PIL import Image
import io


from tqdm import tqdm

from utils.completion import (
    load_questions_from_hf,
    registered_api_completion,
    load_model_answers_and_execution_results,
    load_model_judgements,
    get_endpoint,
    make_config,
)

from utils.judge_utils import JUDGE_SETTINGS


BENCH_CATEGORY = "swe-arena"


def image_path_to_base64_png(image_path):
    try:
        with open(image_path, "rb") as img_file:
            file_data = img_file.read()
            if not file_data:
                # print(f"Warning: Image file {image_path} is empty")
                return None
            
            # Check minimum file size (valid image files should be at least a few dozen bytes)
            if len(file_data) < 20:
                return None
        
        # Open image with PIL to check dimensions and rescale if needed
        image = Image.open(io.BytesIO(file_data))
        
        # Check if rescaling is needed (cap at 8000 pixels for either dimension)
        max_dimension = 8000
        width, height = image.size
        
        if width > max_dimension or height > max_dimension:
            # Calculate new dimensions while maintaining aspect ratio
            if width > height:
                new_width = max_dimension
                new_height = int((height * max_dimension) / width)
            else:
                new_height = max_dimension
                new_width = int((width * max_dimension) / height)
            
            # Resize the image
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Ensure the final image size is under 5MB (5,242,880 bytes)
        # We need to account for base64 encoding overhead (~33% increase)
        max_file_size = 5242880  # 5MB in bytes
        max_raw_size = int(max_file_size * 0.75)  # Account for base64 overhead
        
        # Try different compression strategies to get under the size limit
        output_buffer = io.BytesIO()
        image_format = 'PNG'
        
        # First try PNG with optimization
        image.save(output_buffer, format='PNG', optimize=True)
        file_data = output_buffer.getvalue()
        
        # If still too large, try JPEG with progressively lower quality
        if len(file_data) > max_raw_size:
            image_format = 'JPEG'
            for quality in [95, 85, 75, 65, 55, 45, 35, 25]:
                output_buffer = io.BytesIO()
                # Convert to RGB if necessary for JPEG
                if image.mode in ('RGBA', 'LA', 'P'):
                    rgb_image = Image.new('RGB', image.size, (255, 255, 255))
                    if image.mode == 'P':
                        image = image.convert('RGBA')
                    rgb_image.paste(image, mask=image.split()[-1] if image.mode in ('RGBA', 'LA') else None)
                    image = rgb_image
                
                image.save(output_buffer, format='JPEG', quality=quality, optimize=True)
                file_data = output_buffer.getvalue()
                
                if len(file_data) <= max_raw_size:
                    break
            
            # If still too large, resize further
            if len(file_data) > max_raw_size:
                scale_factor = 0.8
                while len(file_data) > max_raw_size and scale_factor > 0.1:
                    new_width = int(image.size[0] * scale_factor)
                    new_height = int(image.size[1] * scale_factor)
                    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    
                    output_buffer = io.BytesIO()
                    resized_image.save(output_buffer, format='JPEG', quality=75, optimize=True)
                    file_data = output_buffer.getvalue()
                    
                    scale_factor -= 0.1
                    image = resized_image
                
        b64_data = base64.b64encode(file_data).decode("utf-8")
        if not b64_data:
            # print(f"Warning: Failed to encode image file {image_path} to base64")
            return None
            
        # Ensure we have reasonable amount of base64 data
        if len(b64_data) < 50:
            return None
        
        # Final size check
        final_size = len(b64_data)
        if final_size > max_file_size:
            print(f"Warning: Image {image_path} still exceeds 5MB after compression ({final_size} bytes)")
            return None
                
        data_url = f"data:image/{image_format.lower()};base64,{b64_data}"
        return data_url
    except FileNotFoundError:
        # print(f"Warning: Image file not found: {image_path}")
        return None
    except Exception as e:
        # print(f"Error processing image file {image_path}: {e}")
        return None


def build_complete_answer(answer_record, model_dir, include_execution_results=True):
    """
    Build complete answer including both model response and execution results.
    
    Args:
        answer_record: The answer record containing messages, execution results, etc.
        model_dir: The model directory path
        include_execution_results: Whether to include stdout, stderr, and visual outputs
    
    Returns:
        str: Formatted complete answer with model response and execution results
    """
    result_parts = []
    
    # Extract model response from messages
    model_response = "[No response found]"
    if answer_record.get("messages") and len(answer_record["messages"]) > 0:
        last_message = answer_record["messages"][-1]
        if isinstance(last_message.get("content"), dict):
            model_response = last_message["content"].get("answer", "[No response found]")
        elif isinstance(last_message.get("content"), str):
            model_response = last_message["content"]
    
    # Add model response first
    if model_response and model_response != "[No response found]":
        result_parts.append(f"<|The Start of Model Response|>\n{model_response}\n<|The End of Model Response|>")
    
    # Only include execution results if requested
    if include_execution_results:
        # Build execution results
        execution_parts = []
        
        # Add stdout if it exists
        if answer_record.get("stdout"):
            stdout = answer_record['stdout']
            if len(stdout) > 1000:
                stdout = stdout[:1000] + "\n...[Truncated]"
            execution_parts.append(f"<|The Start of Execution Results|>\n{stdout}\n<|The End of Execution Results|>")
        
        # If no stdout/stderr, indicate execution failure
        if not answer_record.get("stdout", "") and not answer_record.get("visual_outputs", []) and not answer_record.get("screenshot_path", None):
            stderr = answer_record.get("stderr", "")
            if stderr:
                if len(stderr) > 1000:
                    stderr = stderr[:1000] + "\n...[Truncated]"
                execution_parts.append(f"<|The Start of Execution Errors and Warnings|>\n{stderr}\n<|The End of Execution Errors and Warnings|>")

        # Combine execution parts
        if execution_parts:
            execution_result = "\n\n".join(execution_parts)
            result_parts.append(execution_result)
    
    result = "\n\n".join(result_parts)
    
    return result


def build_execution_result(execution_record, model_dir, include_execution_results=True):
    """
    Legacy function for backward compatibility.
    Now calls build_complete_answer for full context.
    """
    return build_complete_answer(execution_record, model_dir, include_execution_results)


def get_score(judgment, patterns):
    import re
    for pattern in patterns:
        pattern = re.compile(pattern)
        
        matches = pattern.findall(judgment.upper())
        matches = [m for m in matches if m != ""]
        
        if len(set(matches)) > 0:
            return matches[-1].strip("\n")
    return None


def get_judge_message(prompt_template, question, answer_a, screenshot_a, answer_b, screenshot_b, visual_outputs_a=None, visual_outputs_b=None, model_dir_a=None, model_dir_b=None):
    """
    Build judge message with proper image handling for both screenshots and visual outputs.
    
    Args:
        prompt_template: Template string with placeholders
        question: The question being judged
        answer_a: Answer from model A
        screenshot_a: Screenshot path from model A
        answer_b: Answer from model B  
        screenshot_b: Screenshot path from model B
        visual_outputs_a: List of visual output paths from model A
        visual_outputs_b: List of visual output paths from model B
        model_dir_a: Directory containing model A's files
        model_dir_b: Directory containing model B's files
    
    Returns:
        List of content items for the judge message
    """
    
    def process_image(image_path, visual_outputs=None, model_dir=None):
        """Helper function to process images and convert to base64."""
        if image_path:
            full_path = os.path.join(model_dir, image_path) if model_dir else image_path
            base64_data = image_path_to_base64_png(full_path)
            if base64_data:
                return {"type": "image_url", "image_url": {"url": base64_data}}
        
        # Fallback to visual outputs if no screenshot
        if visual_outputs and len(visual_outputs) > 0:
            visual_path = visual_outputs[0]
            full_path = os.path.join(model_dir, visual_path) if model_dir else visual_path
            base64_data = image_path_to_base64_png(full_path)
            if base64_data:
                return {"type": "image_url", "image_url": {"url": base64_data}}
        
        return None

    
    # Check if this is the new adaptive template format
    is_adaptive_template = "{SCREENSHOT_A_SECTION}" in prompt_template and "{SCREENSHOT_B_SECTION}" in prompt_template
    
    if not is_adaptive_template:
        # Fallback to old template format for backwards compatibility
        assert all(ph in prompt_template for ph in ("{ANSWER_A}", "{ANSWER_B}", "{SCREENSHOT_A}", "{SCREENSHOT_B}")), \
            "Prompt template must contain `{ANSWER_A}`, `{ANSWER_B}`, `{SCREENSHOT_A}`, and `{SCREENSHOT_B}` placeholders."

        # Replace text placeholders
        answer_a = answer_a if answer_a else "[No model response found]"
        answer_b = answer_b if answer_b else "[No model response found]"

        text_filled = prompt_template.replace("{QUESTION}", question).replace("{ANSWER_A}", answer_a).replace("{ANSWER_B}", answer_b)

        # Split by image placeholders
        parts = text_filled.split("{SCREENSHOT_A}")
        before_screenshot_a, rest = parts[0], parts[1]
        parts = rest.split("{SCREENSHOT_B}")
        between_screenshots, after_screenshot_b = parts[0], parts[1]

        content = [
            {"type": "text", "text": before_screenshot_a},
        ]
        
        # Only add images if they exist
        image_a = process_image(screenshot_a, visual_outputs_a, model_dir_a)
        if image_a:
            content.append(image_a)
        
        content.append({"type": "text", "text": between_screenshots})
        
        image_b = process_image(screenshot_b, visual_outputs_b, model_dir_b)
        if image_b:
            content.append(image_b)
        
        content.append({"type": "text", "text": after_screenshot_b})
        
        # Add visual outputs as additional images if they exist
        if visual_outputs_a and len(visual_outputs_a) > 1:
            for i, visual_path in enumerate(visual_outputs_a[1:], 1):  # Skip first one as it's used as main image
                full_path = os.path.join(model_dir_a, visual_path) if model_dir_a else visual_path
                base64_data = image_path_to_base64_png(full_path)
                if base64_data:
                    content.append({"type": "text", "text": f"\n<|The Start of Additional Visual Output {i} for Assistant A|>"})
                    content.append({"type": "image_url", "image_url": {"url": base64_data}})
                    content.append({"type": "text", "text": f"<|The End of Additional Visual Output {i} for Assistant A|>"})
        
        if visual_outputs_b and len(visual_outputs_b) > 1:
            for i, visual_path in enumerate(visual_outputs_b[1:], 1):  # Skip first one as it's used as main image
                full_path = os.path.join(model_dir_b, visual_path) if model_dir_b else visual_path
                base64_data = image_path_to_base64_png(full_path)
                if base64_data:
                    content.append({"type": "text", "text": f"\n<|The Start of Additional Visual Output {i} for Assistant B|>"})
                    content.append({"type": "image_url", "image_url": {"url": base64_data}})
                    content.append({"type": "text", "text": f"<|The End of Additional Visual Output {i} for Assistant B|>"})
        
        content = [item for item in content if item is not None]
        return content
    
    # New adaptive template handling
    assert all(ph in prompt_template for ph in ("{ANSWER_A}", "{ANSWER_B}", "{SCREENSHOT_A_SECTION}", "{SCREENSHOT_B_SECTION}", "{VISUAL_A_SECTION}", "{VISUAL_B_SECTION}")), \
        "Adaptive prompt template must contain `{ANSWER_A}`, `{ANSWER_B}`, `{SCREENSHOT_A_SECTION}`, `{SCREENSHOT_B_SECTION}`, `{VISUAL_A_SECTION}`, and `{VISUAL_B_SECTION}` placeholders."

    # Replace text placeholders
    answer_a = answer_a if answer_a else "[No code output]"
    answer_b = answer_b if answer_b else "[No code output]"
    
    # Build conditional sections
    def build_screenshot_section(screenshot_path, visual_outputs=None, assistant_label="A"):
        if screenshot_path or (visual_outputs and len(visual_outputs) > 0):
            return f"\n<|The Start of Assistant {assistant_label}'s UI Screenshot|>\n{{SCREENSHOT_{assistant_label}}}\n<|The End of Assistant {assistant_label}'s UI Screenshot|>"
        return ""
    
    # For adaptive templates, we need to handle visual outputs differently
    # Instead of building text sections, we'll build the content directly
    screenshot_a_section = build_screenshot_section(screenshot_a, visual_outputs_a, "A")
    screenshot_b_section = build_screenshot_section(screenshot_b, visual_outputs_b, "B")
    
    # Replace all placeholders
    text_filled = prompt_template.replace("{QUESTION}", question).replace("{ANSWER_A}", answer_a).replace("{ANSWER_B}", answer_b)
    text_filled = text_filled.replace("{SCREENSHOT_A_SECTION}", screenshot_a_section)
    text_filled = text_filled.replace("{SCREENSHOT_B_SECTION}", screenshot_b_section)
    
    # Remove visual section placeholders since we'll handle them directly
    text_filled = text_filled.replace("{VISUAL_A_SECTION}", "")
    text_filled = text_filled.replace("{VISUAL_B_SECTION}", "")
    
    # Build content, handling conditional screenshots and visual outputs
    content = []
    
    # Check if we actually have screenshots to process
    has_screenshot_a = screenshot_a or (visual_outputs_a and len(visual_outputs_a) > 0)
    has_screenshot_b = screenshot_b or (visual_outputs_b and len(visual_outputs_b) > 0)
    
    # Split by screenshot placeholders and build content only if screenshots exist
    if "{SCREENSHOT_A}" in text_filled and has_screenshot_a:
        parts = text_filled.split("{SCREENSHOT_A}")
        content.append({"type": "text", "text": parts[0]})
        
        # Only add image if process_image returns a valid result
        image_a = process_image(screenshot_a, visual_outputs_a, model_dir_a)
        if image_a:
            content.append(image_a)
        
        remaining = parts[1]
        
        if "{SCREENSHOT_B}" in remaining and has_screenshot_b:
            parts = remaining.split("{SCREENSHOT_B}")
            content.append({"type": "text", "text": parts[0]})
            
            # Only add image if process_image returns a valid result
            image_b = process_image(screenshot_b, visual_outputs_b, model_dir_b)
            if image_b:
                content.append(image_b)
            
            content.append({"type": "text", "text": parts[1]})
        else:
            content.append({"type": "text", "text": remaining})
    elif "{SCREENSHOT_B}" in text_filled and has_screenshot_b:
        parts = text_filled.split("{SCREENSHOT_B}")
        content.append({"type": "text", "text": parts[0]})
        
        # Only add image if process_image returns a valid result
        image_b = process_image(screenshot_b, visual_outputs_b, model_dir_b)
        if image_b:
            content.append(image_b)
        
        content.append({"type": "text", "text": parts[1]})
    else:
        # No screenshots to replace, just return as text
        content.append({"type": "text", "text": text_filled})

    # Add visual outputs as additional images if they exist
    if visual_outputs_a and len(visual_outputs_a) > 1:
        for i, visual_path in enumerate(visual_outputs_a[1:], 1):  # Skip first one as it's used as main image
            full_path = os.path.join(model_dir_a, visual_path) if model_dir_a else visual_path
            base64_data = image_path_to_base64_png(full_path)
            if base64_data:
                content.append({"type": "text", "text": f"\n<|The Start of Visual Output {i} for Assistant A|>"})
                content.append({"type": "image_url", "image_url": {"url": base64_data}})
                content.append({"type": "text", "text": f"<|The End of Visual Output {i} for Assistant A|>"})
    
    if visual_outputs_b and len(visual_outputs_b) > 1:
        for i, visual_path in enumerate(visual_outputs_b[1:], 1):  # Skip first one as it's used as main image
            full_path = os.path.join(model_dir_b, visual_path) if model_dir_b else visual_path
            base64_data = image_path_to_base64_png(full_path)
            if base64_data:
                content.append({"type": "text", "text": f"\n<|The Start of Visual Output {i} for Assistant B|>"})
                content.append({"type": "image_url", "image_url": {"url": base64_data}})
                content.append({"type": "text", "text": f"<|The End of Visual Output {i} for Assistant B|>"})

    return content


def pairwise_judgment(question, baseline, answer, configs, settings, model_dir_a, model_dir_b, screenshot_a, screenshot_b, visual_outputs_a, visual_outputs_b, include_execution_results=True):
    # Build complete answers (includes model response + execution results)
    baseline_complete_answer = build_complete_answer(baseline, model_dir_a, include_execution_results)
    answer_complete_answer = build_complete_answer(answer, model_dir_b, include_execution_results)
    # print(model_dir_a, model_dir_b)
    # exit(0)
    
    # Disable screenshots and visual outputs if execution results are disabled
    if not include_execution_results:
        screenshot_a = None
        screenshot_b = None
        visual_outputs_a = None
        visual_outputs_b = None
    
    # Build initial messages
    prompt_args = {
        "QUESTION": question['instruction'],
        "ANSWER_A": baseline_complete_answer,
        "SCREENSHOT_A": screenshot_a,
        "ANSWER_B": answer_complete_answer,
        "SCREENSHOT_B": screenshot_b,
    }

    user_prompt = get_judge_message(
        configs["prompt_template"],
        question=prompt_args["QUESTION"],
        answer_a=prompt_args["ANSWER_A"],
        screenshot_a=prompt_args["SCREENSHOT_A"],
        answer_b=prompt_args["ANSWER_B"],
        screenshot_b=prompt_args["SCREENSHOT_B"],
        visual_outputs_a=visual_outputs_a,
        visual_outputs_b=visual_outputs_b,
        model_dir_a=model_dir_a,
        model_dir_b=model_dir_b,
    )
    messages = [
        {
            "role": "system", 
            "content": JUDGE_SETTINGS[BENCH_CATEGORY]["system_prompt"],
        },
        {
            "role": "user", 
            "content": user_prompt,
        }
    ]

    # build arguments for api completions
    kwargs = settings | {
        "api_dict": get_endpoint(settings["endpoints"]),
        "messages": messages,
    }
    kwargs['temperature'] = configs['temperature']
    kwargs['max_tokens'] = configs['max_tokens']
    
    api_completion_func = registered_api_completion[settings["api_type"]]
    output = api_completion_func(**kwargs)
    
    if output is None:
        return None

    score = get_score(output['answer'], configs["regex_patterns"])

    result = {
        "score": score,
        "judgment": output,
        "prompt": messages,
    }
    return result


def judgment(args):
    answer = args['answer']
    baseline = args['baseline']
    include_execution_results = args.get('include_execution_results', True)
    
    output = {
        "uid": args['question']["uid"],
        "environment": answer.get("environment", "Unknown"),
        "judge": args['configs']['judge_model'],
        "model": answer["model"],
        "baseline": baseline["model"],
        "games": []
    }

    # round 1
    result = pairwise_judgment(
        question=args['question'],
        baseline=baseline,
        answer=answer,
        configs=args['configs'],
        settings=args['settings'],
        model_dir_a=os.path.join(os.path.dirname(args["model_dir"]), args['configs']["baseline_model"]),
        model_dir_b=args["model_dir"],
        screenshot_a=baseline.get("screenshot_path"),
        screenshot_b=answer.get("screenshot_path"),
        visual_outputs_a=baseline.get("visual_outputs"),
        visual_outputs_b=answer.get("visual_outputs"),
        include_execution_results=include_execution_results
    )
    output["games"].append(result)

    # round 2
    result = pairwise_judgment(
        question=args['question'],
        baseline=answer,
        answer=baseline,
        configs=args['configs'],
        settings=args['settings'],
        model_dir_a=args["model_dir"],
        model_dir_b=os.path.join(os.path.dirname(args["model_dir"]), args['configs']["baseline_model"]),
        screenshot_a=answer.get("screenshot_path"),
        screenshot_b=baseline.get("screenshot_path"),
        visual_outputs_a=answer.get("visual_outputs"),
        visual_outputs_b=baseline.get("visual_outputs"),
        include_execution_results=include_execution_results
    )
    output["games"].append(result)

    with open(args['output_file'], "a", encoding="utf-8") as f:
        f.write(json.dumps(output, ensure_ascii=False) + "\n")


def show_demo_input(configs, questions, model_answers, answer_dir, include_execution_results=True):
    """Show one example input for demonstration purposes"""
    print("\n" + "="*80)
    print("DEMO: Example Input to Judge")
    print("="*80)
    
    # Find first available question with both baseline and target model answers
    baseline_model = configs["baseline_model"]
    models = [model for model in configs["model_list"]]
    
    demo_found = False
    for question in questions:
        uid = question["uid"]
        for model in models:
            if (model in model_answers and uid in model_answers[model] and
                baseline_model in model_answers and uid in model_answers[baseline_model]):
                
                print(f"\nQuestion UID: {uid}")
                print(f"Baseline Model: {baseline_model}")
                print(f"Target Model: {model}")
                print(f"\nQuestion: {question['instruction']}")
                
                baseline = model_answers[baseline_model][uid]
                answer = model_answers[model][uid]
                baseline_model_dir = os.path.join(answer_dir, baseline_model)
                target_model_dir = os.path.join(answer_dir, model)
                
                # Build complete answers (model response + execution results)
                baseline_complete_answer = build_complete_answer(baseline, baseline_model_dir, include_execution_results)
                answer_complete_answer = build_complete_answer(answer, target_model_dir, include_execution_results)
                
                print(f"\n--- BASELINE ANSWER ({baseline_model}) ---")
                print(baseline_complete_answer)
                if include_execution_results:
                    if baseline.get("screenshot_path"):
                        print(f"\nBaseline Screenshot: {baseline['screenshot_path']}")
                    if baseline.get("visual_outputs"):
                        print(f"Baseline Visual Outputs: {baseline['visual_outputs']}")
                
                print(f"\n--- TARGET ANSWER ({model}) ---") 
                print(answer_complete_answer)
                if include_execution_results:
                    if answer.get("screenshot_path"):
                        print(f"\nTarget Screenshot: {answer['screenshot_path']}")
                    if answer.get("visual_outputs"):
                        print(f"Target Visual Outputs: {answer['visual_outputs']}")
                
                print(f"\n--- JUDGE PROMPT TEMPLATE ---")
                print(configs["prompt_template"])
                
                # Demo complete
                print(f"\n--- COMPLETE ---")
                print("✅ Demo completed successfully")
                
                demo_found = True
                break
        if demo_found:
            break
    
    if not demo_found:
        print("No suitable example found for demo.")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting-file", type=str, default="config/autocodearena.yaml")
    parser.add_argument("--endpoint-file", type=str, default="config/api_config.yaml")
    parser.add_argument("--demo", action="store_true", help="Show one example input and exit")
    parser.add_argument("--no-execution-results", action="store_true", help="Disable execution results (stdout, stderr, visuals) in model judgment")
    parser.add_argument(
        "--dataset", type=str, default="bigcode/autocodearena-v0",
        help="HuggingFace dataset repository ID"
    )
    args = parser.parse_args()
    print(args)

    configs = make_config(args.setting_file)
    endpoint_list = make_config(args.endpoint_file)

    print(f'judge model: {configs["judge_model"]}, temperature: {configs["temperature"]}, max tokens: {configs["max_tokens"]}')

    answer_dir = os.path.join("data", configs["bench_name"], "model_answer")

    # Load questions from HuggingFace dataset
    print(f"Loading questions from HuggingFace dataset: {args.dataset}")
    questions = load_questions_from_hf(repo_id=args.dataset)
    
    model_answers = load_model_answers_and_execution_results(answer_dir)
    
    # Handle demo mode
    if args.demo:
        show_demo_input(configs, questions, model_answers, answer_dir, not args.no_execution_results)
        exit(0)
    
    # if user choose a set of models, only judge those models
    models = [model for model in configs["model_list"]]
    
    output_files = {}
    if args.no_execution_results:
        output_dir = f"data/{configs['bench_name']}/model_judgment_no_results/{configs['judge_model']}"
    else:
        output_dir = f"data/{configs['bench_name']}/model_judgment/{configs['judge_model']}"
    
    for model in models:
        output_files[model] = os.path.join(
            output_dir,
            f"{model}.jsonl",
        )

    for output_file in output_files.values():
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    existing_judgments = load_model_judgements(output_dir)

    endpoint_settings = endpoint_list[configs["judge_model"]]
    
    # Use parallel setting from main config if available, otherwise fall back to endpoint setting
    parallel_workers = configs.get("parallel", endpoint_settings.get("parallel", 32))

    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_workers) as executor:
        futures = []
        for model in models:
            count = 0
            for question in questions:
                uid = question["uid"]

                kwargs = {}
                kwargs["question"] = question
                if model in model_answers and not uid in model_answers[model]:
                    continue

                if model in existing_judgments and uid in existing_judgments[model]:
                    count += 1
                    continue

                kwargs["answer"] = model_answers[model][uid]
                
                # Check if baseline model has successful execution for this UID
                baseline_model = configs["baseline_model"]
                if baseline_model not in model_answers or uid not in model_answers[baseline_model]:
                    continue
                    
                kwargs["baseline"] = model_answers[baseline_model][uid]
                    
                kwargs["configs"] = configs
                kwargs["settings"] = endpoint_settings
                kwargs["output_file"] = output_files[model]
                kwargs["include_execution_results"] = not args.no_execution_results

                kwargs["model_dir"] = os.path.join(answer_dir, model)
                                
                future = executor.submit(judgment, kwargs)
                futures.append(future)

            if count > 0:
                print(f"{count} number of existing judgments")

        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            future.result()
