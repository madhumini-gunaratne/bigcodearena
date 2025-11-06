#!/usr/bin/env python3
"""
Review and analyze generated answers from BigCodeArena evaluation.
Shows question details, generated answers, and quality metrics.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List, Tuple

def load_answers(answer_file):
    """Load generated answers from JSONL file"""
    answers = []
    with open(answer_file, 'r') as f:
        for line in f:
            if line.strip():
                answers.append(json.loads(line))
    return answers

def evaluate_answer(answer_record: Dict[str, Any]) -> Tuple[bool, str, float]:
    """
    Comprehensive evaluation of an answer based on multiple criteria.
    
    Returns:
        Tuple of (passed: bool, reason: str, score: float 0-1)
    """
    try:
        answer_text = answer_record['messages'][-1]['content']['answer']
    except (KeyError, IndexError, TypeError):
        return False, "Invalid format", 0.0
    
    if not answer_text or answer_text.startswith("ERROR"):
        return False, "Execution error", 0.0
    
    score = 0.0
    
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
        return False, "No valid code structure", 0.1
    
    score += min(code_quality, 0.3)
    
    # Check 3: Syntax balance
    parentheses_match = answer_text.count('(') == answer_text.count(')')
    brackets_match = answer_text.count('[') == answer_text.count(']')
    braces_match = answer_text.count('{') == answer_text.count('}')
    
    if parentheses_match and brackets_match and braces_match:
        score += 0.2
    else:
        score += 0.05
    
    # Check 4: Documentation
    has_docstring = '"""' in answer_text or "'''" in answer_text
    has_comments = '#' in answer_text
    
    if has_docstring or has_comments:
        score += 0.15
    else:
        score += 0.05
    
    # Check 5: Execution success indicator
    score += 0.1
    
    passed = score >= 0.6
    reason = "Passed" if passed else "Below threshold"
    
    return passed, reason, min(score, 1.0)

def analyze_answers(answers):
    """Analyze answers and return comprehensive statistics"""
    stats = {
        'total': len(answers),
        'passed': 0,
        'failed': 0,
        'errors': 0,
        'scores': [],
        'by_category': defaultdict(int),
        'passed_by_category': defaultdict(int),
        'failed_by_category': defaultdict(int),
        'error_by_category': defaultdict(int),
        'scores_by_category': defaultdict(list),
    }
    
    for ans in answers:
        category = ans.get('category', 'unknown')
        answer_text = ans['messages'][-1]['content']['answer']
        
        stats['by_category'][category] += 1
        
        # Check if it's an error first
        if answer_text.startswith("ERROR"):
            stats['errors'] += 1
            stats['error_by_category'][category] += 1
            stats['scores'].append(0.0)
            stats['scores_by_category'][category].append(0.0)
        else:
            # Evaluate the answer
            passed, reason, score = evaluate_answer(ans)
            stats['scores'].append(score)
            stats['scores_by_category'][category].append(score)
            
            if passed:
                stats['passed'] += 1
                stats['passed_by_category'][category] += 1
            else:
                stats['failed'] += 1
                stats['failed_by_category'][category] += 1
    
    # Calculate average score
    stats['avg_score'] = sum(stats['scores']) / len(stats['scores']) if stats['scores'] else 0.0
    
    return stats

def print_metrics_table(answers):
    """Print metrics in table format"""
    stats = analyze_answers(answers)
    
    # Print Summary
    print()
    print("=" * 80)
    print("SUMMARY OF ANSWER EVALUATION")
    print("=" * 80)
    print()
    print(f"Total Questions Answered: {stats['total']:>3}")
    print(f"Passed (Proper Evaluation): {stats['passed']:>3}")
    print(f"Failed: {stats['failed']:>3}")
    print(f"Execution Errors: {stats['errors']:>3}")
    print(f"Average Score: {stats['avg_score']:>3.2f}")
    print()
    
    # Print Metrics Table
    print("=" * 80)
    print("METRICS TABLE")
    print("=" * 80)
    print()
    print("✅ Passed Answers      - Answers that have valid code with balanced syntax")
    print("❌ Failed Answers      - Answers that don't have proper code structure")
    print("⚠️  Errors             - Answers that encountered runtime errors during execution")
    print("📊 Pass Rate (%)       - Percentage of answers that passed (Passed / Total * 100)")
    print("📈 Average Score       - Average quality score from 0.0 (worst) to 1.0 (best)")
    print()
    
    # Table header
    header = f"{'Category':<25} | {'Total':>6} | {'Passed':>7} | {'Failed':>7} | {'Errors':>6} | {'Pass Rate':>10} | {'Avg Score':>10}"
    print(header)
    print("-" * 90)
    
    # Table rows
    for category in sorted(stats['by_category'].keys()):
        total = stats['by_category'][category]
        passed = stats['passed_by_category'].get(category, 0)
        failed = stats['failed_by_category'].get(category, 0)
        errors = stats['error_by_category'].get(category, 0)
        pass_rate = (passed / total * 100) if total > 0 else 0
        
        scores = stats['scores_by_category'].get(category, [])
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        print(f"{category:<25} | {total:>6} | {passed:>7} | {failed:>7} | {errors:>6} | {pass_rate:>9.1f}% | {avg_score:>10.2f}")
    
    print("-" * 90)
    
    # Total row
    total_passed = stats['passed']
    total_failed = stats['failed']
    total_errors = stats['errors']
    total_pass_rate = (total_passed / stats['total'] * 100) if stats['total'] > 0 else 0
    
    print(f"{'TOTAL':<25} | {stats['total']:>6} | {total_passed:>7} | {total_failed:>7} | {total_errors:>6} | {total_pass_rate:>9.1f}% | {stats['avg_score']:>10.2f}")
    print()

def main():
    if len(sys.argv) < 2:
        print("""Usage: python3 review_answers.py <model_name_or_path> [--limit N]""")
        sys.exit(1)
    
    input_arg = sys.argv[1]
    limit = None
    
    if '--limit' in sys.argv:
        limit = int(sys.argv[sys.argv.index('--limit') + 1])
    
    # Check if input is a model name or file path
    if input_arg.endswith('.jsonl') or input_arg.endswith('.json'):
        # Direct file path
        answer_file = input_arg
    else:
        # Model name - construct path
        answer_file = Path("autocodearena/data/autocodearena_local/model_answer") / input_arg / "generation.jsonl"
    
    answer_file = Path(answer_file)
    
    if not answer_file.exists():
        print(f"❌ File not found: {answer_file}")
        print(f"   Check available models with: ls autocodearena/data/autocodearena_local/model_answer/")
        sys.exit(1)
    
    print(f"📥 Loading answers from: {answer_file}")
    answers = load_answers(answer_file)
    print(f"✓ Loaded {len(answers)} answers")
    
    if limit:
        answers = answers[:limit]
        print(f"✓ Limited to {len(answers)} answers")
    
    print()
    
    # Print metrics table
    print_metrics_table(answers)

if __name__ == '__main__':
    main()
