#!/usr/bin/env python3
"""
Generate solutions and test cases for coding questions using LLM.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List
import yaml
from vllm import LLM, SamplingParams
from tqdm import tqdm


def load_config():
    """Load configuration from generation_config.yaml"""
    config_path = Path(__file__).parent / "config" / "generation_config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_questions_from_file(file_path: Path) -> List[Dict]:
    """Load questions from JSONL file"""
    questions = []
    with open(file_path) as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    return questions


def prepare_solution_prompt(question: Dict) -> str:
    """Prepare prompt to generate solution"""
    prompt = f"""You are an expert programmer. Generate a complete, working solution for this coding question.

QUESTION:
{question.get("instruction", "")}

REQUIREMENTS:
- Write clean, well-commented production-ready code
- Handle all edge cases
- Use efficient algorithms
- Include proper error handling

INSTRUCTIONS:
You MUST respond with ONLY the code. Do not include any explanations, notes, or preamble. Start writing the code immediately."""
    
    return prompt


def prepare_test_cases_prompt(question: Dict) -> str:
    """Prepare prompt to generate test cases"""
    prompt = f"""You are an expert QA engineer. Generate comprehensive test cases for this coding question.

QUESTION:
{question.get("instruction", "")}

REQUIREMENTS:
- Basic test cases (normal inputs)
- Edge cases (boundary conditions, empty inputs, etc.)
- Error cases (invalid inputs)
- Create 5-8 test cases total

RESPONSE FORMAT:
Respond with ONLY a valid JSON array. Each element must have "input" and "expected_output" keys. No explanations, no markdown, no extra text.

Example:
[
  {{"input": {{"n": 5}}, "expected_output": 120}},
  {{"input": {{"n": 0}}, "expected_output": 1}}
]"""
    
    return prompt


def clean_solution_text(text: str) -> str:
    """Remove instruction artifacts from generated solution"""
    # Common meta-instruction phrases that should not appear in solutions
    meta_instructions = [
        r"you must not include",
        r"wrap the entire",
        r"do not include",
        r"the code",
        r"should be",
        r"self-contained",
        r"executable",
        r"ensure the code",
        r"provide only",
        r"output only",
        r"note:",
        r"requirements:",
        r"instructions:",
        r"format as",
        r"only the code",
        r"start with",
        r"make sure",
        r"include only",
        r"this solution",
        r"the solution",
        r"valid json",
        r"json array",
        r"no explanations",
        r"no text",
        r"no markdown",
        r"no extra",
        r"no other",
    ]
    
    lines = text.split('\n')
    cleaned_lines = []
    skip_preamble = True
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        lower_stripped = stripped.lower()
        
        # Skip initial instruction lines at the very beginning
        if skip_preamble and stripped:
            # Check if this line looks like an instruction
            if any(keyword in lower_stripped for keyword in meta_instructions):
                continue
            # Once we hit actual code, stop skipping
            if any(keyword in lower_stripped for keyword in ['import ', 'def ', 'class ', '<', 'function ', 'const ', 'let ', 'var ', '{']):
                skip_preamble = False
        
        if skip_preamble and not stripped:
            # Skip empty lines at the start
            continue
        
        if not skip_preamble:
            cleaned_lines.append(line)
    
    result = '\n'.join(cleaned_lines).strip()
    
    # Remove leading/trailing code fences and markers
    while result.startswith('```'):
        first_newline = result.find('\n')
        if first_newline != -1:
            result = result[first_newline+1:].strip()
        else:
            break
    
    while result.endswith('```'):
        result = result[:-3].strip()
    
    return result


def generate_solution_and_tests(question: Dict, llm) -> Dict:
    """Generate both solution and test cases for a question"""
    
    # Generate solution
    solution_prompt = prepare_solution_prompt(question)
    test_prompt = prepare_test_cases_prompt(question)
    
    sampling_params = SamplingParams(
        temperature=0.5,
        top_p=0.95,
        max_tokens=3000
    )
    
    outputs = llm.generate([solution_prompt, test_prompt], sampling_params)
    
    solution_text = clean_solution_text(outputs[0].outputs[0].text.strip())
    test_text = outputs[1].outputs[0].text.strip()
    
    # Try to parse test cases as JSON
    test_cases = None
    try:
        # Extract JSON array from response
        start_idx = test_text.find('[')
        end_idx = test_text.rfind(']') + 1
        if start_idx != -1 and end_idx > start_idx:
            json_str = test_text[start_idx:end_idx]
            test_cases = json.loads(json_str)
    except:
        test_cases = None
    
    return {
        "uid": question.get("uid"),
        "category": question.get("category"),
        "instruction": question.get("instruction", "")[:100] + "...",
        "solution": solution_text,
        "test_cases": test_cases,
        "test_cases_raw": test_text[:500] if test_text else None
    }


def load_all_questions(results_dir: Path) -> List[Dict]:
    """Load all deduplicated questions"""
    all_questions = []
    for jsonl_file in sorted(results_dir.glob("*_generated_deduped.jsonl")):
        questions = load_questions_from_file(jsonl_file)
        all_questions.extend(questions)
    return all_questions


def main():
    parser = argparse.ArgumentParser(description="Generate solutions and test cases for questions")
    parser.add_argument("--results_dir", type=Path, default=Path(__file__).parent / "results",
                        help="Directory containing deduplicated JSONL files")
    parser.add_argument("--output_dir", type=Path, default=Path(__file__).parent / "solutions",
                        help="Output directory for solutions and test cases")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size for LLM inference")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of questions (for testing)")
    
    args = parser.parse_args()
    args.output_dir.mkdir(exist_ok=True)
    
    config = load_config()
    
    # Load questions
    print(f"Loading deduplicated questions from {args.results_dir}...")
    all_questions = load_all_questions(args.results_dir)
    print(f"Loaded {len(all_questions)} questions")
    
    if args.limit:
        all_questions = all_questions[:args.limit]
        print(f"Limited to {len(all_questions)} questions for testing")
    
    # Initialize LLM
    model_name = config["llm"]["model_name"]
    max_model_len = config["llm"].get("max_model_len", 4096)
    print(f"\nLoading LLM: {model_name}...")
    llm = LLM(
        model=model_name,
        max_model_len=max_model_len,
        tensor_parallel_size=1
    )
    
    # Generate solutions and test cases
    print(f"\nGenerating solutions and test cases for {len(all_questions)} questions...")
    results = []
    
    for i in tqdm(range(0, len(all_questions), args.batch_size)):
        batch = all_questions[i:i+args.batch_size]
        for question in batch:
            result = generate_solution_and_tests(question, llm)
            results.append(result)
    
    # Save results
    solutions_file = args.output_dir / "solutions_and_tests.jsonl"
    with open(solutions_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    
    print(f"\nSaved {len(results)} solutions and test cases to {solutions_file.name}")
    
    # Print summary
    print(f"\n{'='*80}")
    print("GENERATION SUMMARY")
    print(f"{'='*80}")
    
    has_solution = sum(1 for r in results if r.get("solution"))
    has_tests = sum(1 for r in results if r.get("test_cases"))
    
    print(f"Total questions: {len(results)}")
    print(f"Solutions generated: {has_solution}")
    print(f"Test cases generated: {has_tests}")
    print(f"Output file: {solutions_file}")


if __name__ == "__main__":
    main()
