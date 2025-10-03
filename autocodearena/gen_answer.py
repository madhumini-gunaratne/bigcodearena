import argparse
import json
import os
import re
import time
import concurrent.futures


import shortuuid
import tqdm

from sandbox.constants import SandboxEnvironment, DEFAULT_SANDBOX_INSTRUCTIONS
from sandbox.code_analyzer import extract_code_from_markdown

from utils.completion import (
    load_questions_from_hf,
    load_model_answers,
    make_config,
    get_endpoint,
    registered_api_completion,
    registered_engine_completion,
    API_ERROR_OUTPUT,
)


def parse_and_reorg_answer_file(answer_file):
    """Extract code, sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            record = json.loads(l)
            qid = record["uid"]
            answers[qid] = record

    # Extract code from markdown
    for qid, ans in answers.items():
        content = ans["messages"][-1]["content"]["answer"]  # Get the last message which is the assistant's response
        extraction_result = extract_code_from_markdown(content)
        if not extraction_result:
            ans["code_to_execute"] = None
            ans["language"] = None
            ans["environment"] = None
            ans["code_dependencies"] = None

            continue
        code, code_language, code_dependencies, env_selection = extraction_result
        ans["code_to_execute"] = code
        ans["code_dependencies"] = code_dependencies
        ans["language"] = code_language
        ans["environment"] = env_selection
        
        answers[qid] = ans

    qids = sorted(list(answers.keys()))

    # Write as jsonl
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(json.dumps(answers[qid], ensure_ascii=False) + "\n")


def is_answer_empty(answer_content):
    """Check if the answer content is empty or just whitespace"""
    if isinstance(answer_content, dict) and "answer" in answer_content:
        return not answer_content["answer"].strip()
    elif isinstance(answer_content, str):
        return not answer_content.strip()
    return True


def get_answer(
    question: dict, answer_file: str, settings: dict
):
    code_env = "" #question["environment"]

    # build messages
    messages = []
    if "sys_prompt" in settings:  # if sys_prompt is provided, use it
        messages.append({"role": "system", "content": settings["sys_prompt"]})
    else:  # otherwise, use the default sandbox instruction
        # assert code_env in DEFAULT_SANDBOX_INSTRUCTIONS.keys(), f"Selected environment {question['environment']} not found in default sandbox instructions. Please check the benchmark question file."
        # messages.append({"role": "system", "content": DEFAULT_SANDBOX_INSTRUCTIONS[question["environment"]]})
        
        # Auto env selection
        messages.append({"role": "system", "content": DEFAULT_SANDBOX_INSTRUCTIONS[SandboxEnvironment.AUTO]})
    messages.append({"role": "user", "content": question["instruction"]})

    # retrieve the api completion function from register
    api_completion_func = registered_api_completion[settings["api_type"]]
    
    # build arguments for api completions
    kwargs = settings | {
        "api_dict": get_endpoint(settings["endpoints"]),
        "messages": messages,
    }

    output = ""
    for i in range(3):
        try:
            output = api_completion_func(**kwargs)
            if extract_code_from_markdown(output.get("answer", "")):
                break
        except Exception as e:
            print(e)
            continue
   
    if output is None or not extract_code_from_markdown(output.get("answer", "")):
        return
    
    messages.append({"role": "assistant", "content": output})

    # Dump answers
    metadata = {}

    ans = {
        "uid": question["uid"],
        "environment": code_env,
        "ans_id": shortuuid.uuid(),
        "model": settings["model_display_name"],
        "messages": messages,
        "tstamp": time.time(),
        "metadata": metadata,
    }

    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    with open(answer_file, "a", encoding="utf-8") as fout:
        fout.write(json.dumps(ans, ensure_ascii=False) + "\n")


def has_no_code(record):
    """Check if the record has no code extracted by attempting to extract from the last answer content"""
    
    # If no code was extracted, try to extract from the last answer content
    if len(record.get("messages", [])) > 0:
        last_message = record["messages"][-1]
        if last_message.get("role") == "assistant":
            content = last_message.get("content", "")
            if isinstance(content, dict) and "answer" in content:
                content = content["answer"]
            elif isinstance(content, str):
                pass  # content is already a string
            else:
                content = ""
            
            # Try to extract code from the content
            if extract_code_from_markdown(content):
                return False  # No longer has no code
    
    # If we get here, no code could be extracted
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file", type=str, default="config/gen_answer_config.yaml"
    )
    parser.add_argument(
        "--endpoint-file", type=str, default="config/api_config.yaml"
    )
    parser.add_argument(
        "--regenerate-empty", action="store_true", 
        help="Regenerate answers where the content is empty or just whitespace"
    )
    parser.add_argument(
        "--regenerate-no-code", action="store_true", 
        help="Regenerate answers where no code was extracted"
    )
    parser.add_argument(
        "--dataset", type=str, default="bigcode/autocodearena-v0",
        help="HuggingFace dataset repository ID"
    )
    args = parser.parse_args()

    config = make_config(args.config_file)
    endpoints = make_config(args.endpoint_file)

    existing_answer = load_model_answers(os.path.join("data", config["bench_name"], "model_answer"))
    
    for model in config["model_list"]:
        assert model in endpoints
        endpoint_settings = endpoints[model]
        endpoint_settings["model_display_name"] = model

        # Load questions from HuggingFace dataset
        print(f"Loading questions from HuggingFace dataset: {args.dataset}")
        questions = load_questions_from_hf(repo_id=args.dataset)

        answer_file = os.path.join("data", config["bench_name"], "model_answer", model, f"generation.jsonl")
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        print(f"Output to {answer_file}")

        # Load existing answers and identify what needs to be processed
        existing_answers = {}
        if os.path.exists(answer_file):
            with open(answer_file, "r", encoding="utf-8") as fin:
                for line in fin:
                    if line.strip():
                        record = json.loads(line)
                        uid = record["uid"]
                        existing_answers[uid] = record

        # If regenerating empty answers, first remove them from the generation file
        samples_to_regenerate = set()
        if args.regenerate_empty:
            print("🔄 Removing empty answers from generation file...")
            temp_answers = {}
            for uid, record in existing_answers.items():
                if len(record["messages"]) > 0:
                    last_message = record["messages"][-1]
                    if last_message["role"] == "assistant" and is_answer_empty(last_message["content"]):
                        samples_to_regenerate.add(uid)
                        continue  # Skip this empty answer
                temp_answers[uid] = record
            
            # Update existing_answers to exclude empty ones
            existing_answers = temp_answers
            
            # Rewrite the generation file without empty answers
            with open(answer_file, "w", encoding="utf-8") as fout:
                for uid in sorted(existing_answers.keys()):
                    fout.write(json.dumps(existing_answers[uid], ensure_ascii=False) + "\n")
            
            print(f"  Removed {len(samples_to_regenerate)} empty answers from generation file")
            print(f"  Kept {len(existing_answers)} non-empty answers")

        # If regenerating no-code answers, first remove them from the generation file
        samples_to_regenerate_no_code = set()
        if args.regenerate_no_code:
            print("🔄 Processing no-code answers from generation file...")
            temp_answers = {}
            for uid, record in existing_answers.items():
                if has_no_code(record):
                    # has_no_code already attempted code extraction, if it still returns True, we need to regenerate
                    samples_to_regenerate_no_code.add(uid)
                    continue  # Skip this no-code answer
                else:
                    # Code was successfully extracted, keep the record
                    temp_answers[uid] = record
            
            # Update existing_answers to exclude no-code ones that couldn't be fixed
            existing_answers = temp_answers
            
            # Rewrite the generation file without the unfixable no-code answers
            with open(answer_file, "w", encoding="utf-8") as fout:
                for uid in sorted(existing_answers.keys()):
                    fout.write(json.dumps(existing_answers[uid], ensure_ascii=False) + "\n")
            
            print(f"  Removed {len(samples_to_regenerate_no_code)} unfixable no-code answers from generation file")
            print(f"  Kept {len(existing_answers)} answers with code (including re-extracted ones)")

        # Identify which questions need new answers
        questions_to_process = []
        for question in questions:
            uid = question["uid"]
            if uid not in existing_answers:
                questions_to_process.append(question)
            elif args.regenerate_empty and uid in samples_to_regenerate:
                questions_to_process.append(question)
            elif args.regenerate_no_code and uid in samples_to_regenerate_no_code:
                questions_to_process.append(question)

        # Combine all samples that need regeneration
        all_samples_to_regenerate = samples_to_regenerate.union(samples_to_regenerate_no_code)
        new_samples = len(questions_to_process) - len(all_samples_to_regenerate)
        
        print(f"📊 Processing {len(questions_to_process)} questions ({new_samples} new + {len(samples_to_regenerate)} regenerating empty + {len(samples_to_regenerate_no_code)} regenerating no-code)")

        if "parallel" in endpoint_settings:
            parallel = endpoint_settings["parallel"]
        else:
            parallel = 1
            
        if 'local_engine' in endpoint_settings and endpoint_settings['local_engine']:
            local_completion_func = registered_engine_completion[endpoint_settings['api_type']]
            
            kwargs = endpoint_settings | {
                "answer_file": answer_file,
                "batch_context": questions_to_process
            }
            local_completion_func(**kwargs)
            
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
                futures = []
                for question in questions_to_process:
                    future = executor.submit(
                        get_answer,
                        question,
                        answer_file,
                        endpoint_settings,
                    )
                    futures.append(future)
                
                for future in tqdm.tqdm(
                    concurrent.futures.as_completed(futures), total=len(futures)
                ):
                    future.result()

        # After processing, rewrite the file with all answers (existing + new)
        print("🔄 Consolidating all answers...")
        all_answers = existing_answers.copy()
        
        # Read newly generated answers
        if os.path.exists(answer_file):
            with open(answer_file, "r", encoding="utf-8") as fin:
                for line in fin:
                    if line.strip():
                        record = json.loads(line)
                        uid = record["uid"]
                        all_answers[uid] = record
        
        # Write consolidated file
        with open(answer_file, "w", encoding="utf-8") as fout:
            for uid in sorted(all_answers.keys()):
                fout.write(json.dumps(all_answers[uid], ensure_ascii=False) + "\n")
        
        print(f"  Final file contains {len(all_answers)} answers")

        # Extract code and re-organize the answer file
        parse_and_reorg_answer_file(answer_file)
