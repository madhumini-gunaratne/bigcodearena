#!/usr/bin/env python3
"""
Review and analyze generated answers from BigCodeArena evaluation.
Shows question details, generated answers, and quality metrics.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

def load_answers(answer_file):
    """Load generated answers from JSONL file"""
    answers = []
    with open(answer_file, 'r') as f:
        for line in f:
            if line.strip():
                answers.append(json.loads(line))
    return answers

def extract_code(answer_text):
    """Extract Python code from answer text"""
    if not answer_text or answer_text.startswith("ERROR"):
        return None
    
    # Try to find code blocks or Python-like content
    if 'def ' in answer_text or 'import ' in answer_text or 'class ' in answer_text:
        return answer_text[:200] + "..." if len(answer_text) > 200 else answer_text
    return None

def analyze_answers(answers):
    """Analyze the quality of generated answers"""
    stats = {
        'total': len(answers),
        'has_code': 0,
        'has_error': 0,
        'by_category': defaultdict(int),
        'code_by_category': defaultdict(int),
    }
    
    for ans in answers:
        answer_text = ans['messages'][-1]['content']['answer']
        category = ans.get('category', 'unknown')
        
        stats['by_category'][category] += 1
        
        if answer_text.startswith("ERROR"):
            stats['has_error'] += 1
        elif extract_code(answer_text):
            stats['has_code'] += 1
            stats['code_by_category'][category] += 1
    
    return stats

def print_review(answers, limit=None):
    """Print detailed review of answers"""
    print("=" * 100)
    print("BIGCODEARENA EVALUATION - DETAILED REVIEW")
    print("=" * 100)
    print()
    
    # Analyze
    stats = analyze_answers(answers)
    
    print("📊 SUMMARY STATISTICS")
    print("-" * 100)
    print(f"Total Questions Answered: {stats['total']}")
    print(f"Answers with Code: {stats['has_code']} ({100*stats['has_code']/stats['total']:.1f}%)")
    print(f"Answers with Errors: {stats['has_error']} ({100*stats['has_error']/stats['total']:.1f}%)")
    print()
    

def print_metrics_table(answers):
    """Print metrics in table format"""
    print()
    print("=" * 100)
    print("📈 METRICS TABLE")
    print("=" * 100)
    print()
    
    stats = analyze_answers(answers)
    
    # Table header
    print(f"{'Category':<20} | {'Total':>6} | {'With Code':>10} | {'Pass Rate':>10} | {'Errors':>6}")
    print("-" * 100)
    
    # Table rows
    for category, count in sorted(stats['by_category'].items()):
        code_count = stats['code_by_category'].get(category, 0)
        pass_rate = (code_count / count * 100) if count > 0 else 0
        # Error count for this category
        error_count = 0
        for ans in answers:
            if ans.get('category') == category:
                answer_text = ans['messages'][-1]['content']['answer']
                if answer_text.startswith("ERROR"):
                    error_count += 1
        
        print(f"{category:<20} | {count:>6} | {code_count:>10} | {pass_rate:>9.1f}% | {error_count:>6}")
    
    print("-" * 100)
    print(f"{'TOTAL':<20} | {stats['total']:>6} | {stats['has_code']:>10} | {(stats['has_code']/stats['total']*100):>9.1f}% | {stats['has_error']:>6}")
    print()

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 review_answers.py <path_to_generation.jsonl> [--limit N]")
        print("Example: python3 review_answers.py autocodearena/data/autocodearena_local/model_answer/phi-2-vllm/generation.jsonl --limit 10")
        sys.exit(1)
    
    answer_file = sys.argv[1]
    limit = None
    
    if '--limit' in sys.argv:
        limit = int(sys.argv[sys.argv.index('--limit') + 1])
    
    if not Path(answer_file).exists():
        print(f"❌ File not found: {answer_file}")
        sys.exit(1)
    
    print(f"📥 Loading answers from: {answer_file}")
    answers = load_answers(answer_file)
    print(f"✅ Loaded {len(answers)} answers")
    print()
    
    # Print review
    print_review(answers, limit=limit if limit else 5)
    
    # Print metrics table
    print_metrics_table(answers)

if __name__ == '__main__':
    main()
