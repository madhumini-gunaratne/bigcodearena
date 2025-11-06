#!/usr/bin/env python3
"""
Case study: Compare one passed and one failed answer
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, Tuple

def load_answers(answer_file):
    """Load generated answers from JSONL file"""
    answers = []
    with open(answer_file, 'r') as f:
        for line in f:
            if line.strip():
                answers.append(json.loads(line))
    return answers

def evaluate_answer(answer_record: Dict[str, Any]) -> Tuple[bool, str, float]:
    """Comprehensive evaluation of an answer"""
    try:
        answer_text = answer_record['messages'][-1]['content']['answer']
    except (KeyError, IndexError, TypeError):
        return False, "Invalid format", 0.0
    
    if not answer_text or answer_text.startswith("ERROR"):
        return False, "Execution error", 0.0
    
    score = 0.0
    issues = []
    
    # Check 1: Has content
    if not answer_text or answer_text.strip() == "":
        return False, "Empty answer", 0.0
    
    score += 0.2
    
    # Check 2: Contains code indicators
    has_def = 'def ' in answer_text
    has_class = 'class ' in answer_text
    has_import = 'import ' in answer_text
    has_code_block = '```' in answer_text or ('    ' in answer_text)
    
    code_quality = 0
    if has_def or has_class:
        code_quality += 0.3
    if has_import:
        code_quality += 0.2
    if has_code_block:
        code_quality += 0.2
    
    if code_quality == 0:
        issues.append("No function/class definition")
        return False, "No valid code structure", 0.1
    
    score += min(code_quality, 0.3)
    
    # Check 3: Syntax balance
    parentheses_match = answer_text.count('(') == answer_text.count(')')
    brackets_match = answer_text.count('[') == answer_text.count(']')
    braces_match = answer_text.count('{') == answer_text.count('}')
    
    if parentheses_match and brackets_match and braces_match:
        score += 0.2
    else:
        issues.append("Unbalanced brackets/parentheses/braces")
        score += 0.05
    
    # Check 4: Documentation
    has_docstring = '"""' in answer_text or "'''" in answer_text
    has_comments = '#' in answer_text
    
    if has_docstring or has_comments:
        score += 0.15
    else:
        issues.append("No documentation/comments")
        score += 0.05
    
    # Check 5: Execution success
    score += 0.1
    
    passed = score >= 0.6
    reason = "; ".join(issues) if issues else "All checks passed"
    
    return passed, reason, min(score, 1.0)

def find_examples(answers):
    """Find 2 passed, 2 failed, and 1 error answer"""
    passed_examples = []
    failed_examples = []
    error_example = None
    
    for ans in answers:
        answer_text = ans['messages'][-1]['content']['answer']
        
        if answer_text.startswith("ERROR"):
            if error_example is None:
                # Capture the actual error message
                error_reason = answer_text.strip() if len(answer_text) > 0 else "Unknown execution error"
                error_example = (ans, error_reason, 0.0)
        else:
            passed, reason, score = evaluate_answer(ans)
            
            if passed and len(passed_examples) < 2:
                passed_examples.append((ans, reason, score))
            elif not passed and len(failed_examples) < 2:
                failed_examples.append((ans, reason, score))
        
        if len(passed_examples) >= 2 and len(failed_examples) >= 2 and error_example:
            break
    
    return passed_examples, failed_examples, error_example

def save_case_study(passed_data, failed_data, error_data, output_file):
    """Save case study to a text file"""
    
    with open(output_file, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("CASE STUDY: PASSED vs FAILED vs ERROR ANSWERS\n")
        f.write("=" * 100 + "\n\n")
        
        # PASSED EXAMPLES
        f.write("PASSED ANSWERS (2 examples)\n")
        f.write("=" * 100 + "\n\n")
        for i, (ans, reason, score) in enumerate(passed_data, 1):
            question = ans.get('instruction', 'N/A')
            answer_text = ans['messages'][-1]['content']['answer']
            category = ans.get('category', 'unknown')
            
            f.write(f"PASSED EXAMPLE {i}\n")
            f.write("-" * 100 + "\n")
            f.write(f"Category: {category}\n")
            f.write(f"Score: {score:.2f}/1.00\n\n")
            
            f.write("QUESTION:\n")
            f.write(f"{question}\n\n")
            
            f.write("ANSWER:\n")
            f.write(f"{answer_text}\n\n")
            
            f.write("WHY PASSED:\n")
            f.write(f"{reason}\n")
            f.write("\n\n")
        
        # FAILED EXAMPLES
        f.write("FAILED ANSWERS (2 examples)\n")
        f.write("=" * 100 + "\n\n")
        for i, (ans, reason, score) in enumerate(failed_data, 1):
            question = ans.get('instruction', 'N/A')
            answer_text = ans['messages'][-1]['content']['answer']
            category = ans.get('category', 'unknown')
            
            f.write(f"FAILED EXAMPLE {i}\n")
            f.write("-" * 100 + "\n")
            f.write(f"Category: {category}\n")
            f.write(f"Score: {score:.2f}/1.00\n\n")
            
            f.write("QUESTION:\n")
            f.write(f"{question}\n\n")
            
            f.write("ANSWER:\n")
            f.write(f"{answer_text}\n\n")
            
            f.write("WHY FAILED:\n")
            f.write(f"{reason}\n")
            f.write("\n\n")
        
        # ERROR EXAMPLE
        if error_data:
            f.write("ERROR ANSWER (1 example)\n")
            f.write("=" * 100 + "\n\n")
            ans, reason, score = error_data
            question = ans.get('instruction', 'N/A')
            answer_text = ans['messages'][-1]['content']['answer']
            category = ans.get('category', 'unknown')
            
            f.write("ERROR EXAMPLE\n")
            f.write("-" * 100 + "\n")
            f.write(f"Category: {category}\n")
            f.write(f"Score: {score:.2f}/1.00\n\n")
            
            f.write("QUESTION:\n")
            f.write(f"{question}\n\n")
            
            f.write("ANSWER:\n")
            f.write(f"{answer_text}\n\n")
            
            f.write("WHY ERROR:\n")
            f.write(f"{reason}\n")
    
    return output_file

def main():
    if len(sys.argv) < 2:
        print("""Usage: python3 case_study.py <model_name_or_path> [--output FILE]""")
        sys.exit(1)
    
    input_arg = sys.argv[1]
    
    # Check if input is a model name or file path
    if input_arg.endswith('.jsonl') or input_arg.endswith('.json'):
        # Direct file path
        answer_file = input_arg
        model_name = Path(input_arg).parent.name
    else:
        # Model name - construct path
        model_name = input_arg
        answer_file = Path("autocodearena/data/autocodearena_local/model_answer") / input_arg / "generation.jsonl"
    
    # Generate model-specific output filename by default
    output_file = f"case_study_{model_name}.txt"
    
    # Check for custom output file
    if '--output' in sys.argv:
        output_file = sys.argv[sys.argv.index('--output') + 1]
    
    answer_file = Path(answer_file)
    
    if not answer_file.exists():
        print(f"❌ File not found: {answer_file}")
        print(f"   Check available models with: ls autocodearena/data/autocodearena_local/model_answer/")
        sys.exit(1)
    
    print(f"📥 Loading answers from: {answer_file}")
    answers = load_answers(answer_file)
    print(f"✓ Loaded {len(answers)} answers")
    print()
    
    passed_data, failed_data, error_data = find_examples(answers)
    
    if not passed_data:
        print("⚠️  No passed answers found")
    else:
        print(f"✓ Found {len(passed_data)} passed example(s)")
    
    if not failed_data:
        print("⚠️  No failed answers found")
    else:
        print(f"✓ Found {len(failed_data)} failed example(s)")
    
    if not error_data:
        print("⚠️  No error answers found")
    else:
        print(f"✓ Found 1 error example")
    
    if passed_data or failed_data or error_data:
        save_case_study(passed_data, failed_data, error_data, output_file)
        print(f"\n✅ Case study saved to: {output_file}")
    else:
        print("\n⚠️  No examples to save")

if __name__ == '__main__':
    main()
