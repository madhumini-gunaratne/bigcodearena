#!/usr/bin/env python3
"""
Compare execution results between multiple models
"""

import json
import os
from pathlib import Path
import argparse

def load_summary(model_dir):
    """Load execution summary for a model"""
    summary_file = os.path.join(model_dir, "execution_summary.json")
    
    if not os.path.exists(summary_file):
        print(f"❌ Summary file not found: {summary_file}")
        return None
    
    with open(summary_file, 'r') as f:
        return json.load(f)

def compare_models(*model_names):
    """Compare multiple models"""
    
    if len(model_names) < 2:
        print("❌ Please provide at least 2 models to compare")
        return
    
    # Load summaries for all models
    summaries = {}
    for model_name in model_names:
        model_dir = f"data/autocodearena_local/model_answer/{model_name}"
        summary = load_summary(model_dir)
        if summary:
            summaries[model_name] = summary
    
    if not summaries:
        print("❌ Could not load any model summaries")
        return
    
    # If only 2 models, use original 2-model comparison layout
    if len(model_names) == 2:
        compare_two_models(model_names[0], summaries[model_names[0]], 
                          model_names[1], summaries[model_names[1]])
    else:
        compare_multiple_models(model_names, summaries)

def compare_two_models(model1_name, summary1, model2_name, summary2):
    """Compare exactly two models with detailed layout"""
    
    # Overall comparison
    print("\n")
    print("📊 OVERALL PERFORMANCE:")
    print("-" * 90)
    print(f"{'Metric':<40} | {model1_name:<30} | {model2_name:<30}")
    print("-" * 90)
    
    success_rate1 = summary1.get('success_rate', 0)
    success_rate2 = summary2.get('success_rate', 0)
    successful1 = summary1.get('successful', 0)
    successful2 = summary2.get('successful', 0)
    total1 = summary1.get('total_executions', 0)
    total2 = summary2.get('total_executions', 0)
    
    winner1_mark = "🥇 WINNER" if success_rate1 > success_rate2 else ""
    winner2_mark = "🥇 WINNER" if success_rate2 > success_rate1 else ""
    
    print(f"{'Success Rate (%):':<40} | {success_rate1:>6.1f}% {winner1_mark:<20} | {success_rate2:>6.1f}% {winner2_mark:<20}")
    print(f"{'Successful / Total:':<40} | {successful1:>3}/{total1:<3} {' '*24} | {successful2:>3}/{total2:<3}")
    print(f"{'Failed:':<40} | {summary1.get('failed', 0):>3} {' '*24} | {summary2.get('failed', 0):>3}")
    print()
    
    # By Environment
    print("=" * 90)
    print("📌 PERFORMANCE BY ENVIRONMENT:")
    print("-" * 90)
    print(f"{'Environment':<30} | {model1_name:<25} | {model2_name:<25}")
    print(f"{'':30} | {'Success Rate':<25} | {'Success Rate':<25}")
    print("-" * 90)
    
    env1 = summary1.get('by_environment', {})
    env2 = summary2.get('by_environment', {})
    
    # Get all environments
    all_envs = set(env1.keys()) | set(env2.keys())
    
    for env in sorted(all_envs):
        e1 = env1.get(env, {})
        e2 = env2.get(env, {})
        
        rate1 = e1.get('success_rate', 0)
        rate2 = e2.get('success_rate', 0)
        
        # Determine environment winner
        env_winner = ""
        if rate1 > rate2:
            env_winner = "▶ "
        elif rate2 > rate1:
            env_winner = " ◀"
        
        total1_env = e1.get('total', 0)
        total2_env = e2.get('total', 0)
        
        print(f"{env:<30} | {rate1:>6.1f}% ({e1.get('success', 0)}/{total1_env}) {env_winner:<15} | {rate2:>6.1f}% ({e2.get('success', 0)}/{total2_env})")
    
    print()
    
    # By Category
    print("=" * 90)
    print("📂 PERFORMANCE BY CATEGORY:")
    print("-" * 90)
    print(f"{'Category':<30} | {model1_name:<25} | {model2_name:<25}")
    print(f"{'':30} | {'Success Rate':<25} | {'Success Rate':<25}")
    print("-" * 90)
    
    cat1 = summary1.get('by_category', {})
    cat2 = summary2.get('by_category', {})
    
    # Get all categories
    all_cats = set(cat1.keys()) | set(cat2.keys())
    
    for cat in sorted(all_cats):
        c1 = cat1.get(cat, {})
        c2 = cat2.get(cat, {})
        
        rate1 = c1.get('success_rate', 0)
        rate2 = c2.get('success_rate', 0)
        
        # Determine category winner
        cat_winner = ""
        if rate1 > rate2:
            cat_winner = "▶ "
        elif rate2 > rate1:
            cat_winner = " ◀"
        
        total1_cat = c1.get('total', 0)
        total2_cat = c2.get('total', 0)
        
        print(f"{cat:<30} | {rate1:>6.1f}% ({c1.get('success', 0)}/{total1_cat}) {cat_winner:<15} | {rate2:>6.1f}% ({c2.get('success', 0)}/{total2_cat})")
    
    print()
    
    # Summary insights
    print("=" * 90)
    print("🎯 KEY INSIGHTS:")
    print("=" * 90)
    
    # Calculate differences
    overall_diff = success_rate1 - success_rate2
    
    if overall_diff > 0:
        print(f"✅ {model1_name} is better overall by {abs(overall_diff):.1f} percentage points")
    elif overall_diff < 0:
        print(f"✅ {model2_name} is better overall by {abs(overall_diff):.1f} percentage points")
    else:
        print(f"✅ Both models have equal performance")
    
    print()
    
    # Best environments
    print(f"🏆 {model1_name} strongest environments:")
    env1_sorted = sorted(env1.items(), key=lambda x: x[1].get('success_rate', 0), reverse=True)[:3]
    for env, stats in env1_sorted:
        print(f"   - {env}: {stats.get('success_rate', 0):.1f}%")
    
    print()
    print(f"🏆 {model2_name} strongest environments:")
    env2_sorted = sorted(env2.items(), key=lambda x: x[1].get('success_rate', 0), reverse=True)[:3]
    for env, stats in env2_sorted:
        print(f"   - {env}: {stats.get('success_rate', 0):.1f}%")
    
    print()
    print("=" * 90)

def compare_multiple_models(model_names, summaries):
    """Compare 3+ models in a more compact layout"""
    
    # Calculate column width based on model names
    col_width = max(20, max(len(name) for name in model_names))
    metric_col_width = 25
    
    # Overall comparison - build header dynamically
    print("\n📊 OVERALL PERFORMANCE:")
    
    # Build separator line
    sep_line = "-" * (metric_col_width + 3)
    for _ in model_names:
        sep_line += "-" * (col_width + 3)
    print(sep_line)
    
    # Create header with all model names
    header = f"{'Metric':<{metric_col_width}}"
    for model_name in model_names:
        header += f" | {model_name:<{col_width}}"
    print(header)
    print(sep_line)
    
    # Success rates
    success_row = f"{'Success Rate (%):':<{metric_col_width}}"
    for model_name in model_names:
        summary = summaries[model_name]
        rate = summary.get('success_rate', 0)
        success_row += f" | {rate:>6.1f}%{' ' * (col_width - 8)}"
    print(success_row)
    
    # Successful/Total
    total_row = f"{'Successful / Total:':<{metric_col_width}}"
    for model_name in model_names:
        summary = summaries[model_name]
        successful = summary.get('successful', 0)
        total = summary.get('total_executions', 0)
        total_row += f" | {successful:>3}/{total:<3}{' ' * (col_width - 8)}"
    print(total_row)
    
    # Failed
    failed_row = f"{'Failed:':<{metric_col_width}}"
    for model_name in model_names:
        summary = summaries[model_name]
        failed = summary.get('failed', 0)
        failed_row += f" | {failed:>3}{' ' * (col_width - 4)}"
    print(failed_row)
    print(sep_line)
    print()
    
    # By Environment
    print("=" * (metric_col_width + 3 + len(model_names) * (col_width + 3)))
    print("📌 PERFORMANCE BY ENVIRONMENT:")
    print(sep_line)
    
    # Collect all environments
    all_envs = set()
    for model_name in model_names:
        summary = summaries[model_name]
        all_envs.update(summary.get('by_environment', {}).keys())
    
    for env in sorted(all_envs):
        env_row = f"{env:<{metric_col_width}}"
        for model_name in model_names:
            summary = summaries[model_name]
            env_data = summary.get('by_environment', {}).get(env, {})
            rate = env_data.get('success_rate', 0)
            success = env_data.get('success', 0)
            total = env_data.get('total', 0)
            env_row += f" | {rate:>6.1f}% ({success}/{total}){' ' * (col_width - 18)}"
        print(env_row)
    
    print()
    
    # By Category
    print("=" * (metric_col_width + 3 + len(model_names) * (col_width + 3)))
    print("📂 PERFORMANCE BY CATEGORY:")
    print(sep_line)
    
    # Collect all categories
    all_cats = set()
    for model_name in model_names:
        summary = summaries[model_name]
        all_cats.update(summary.get('by_category', {}).keys())
    
    for cat in sorted(all_cats):
        cat_row = f"{cat:<{metric_col_width}}"
        for model_name in model_names:
            summary = summaries[model_name]
            cat_data = summary.get('by_category', {}).get(cat, {})
            rate = cat_data.get('success_rate', 0)
            success = cat_data.get('success', 0)
            total = cat_data.get('total', 0)
            cat_row += f" | {rate:>6.1f}% ({success}/{total}){' ' * (col_width - 18)}"
        print(cat_row)
    
    print(sep_line)
    print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare execution results between multiple models")
    parser.add_argument("--model1", type=str, default="qwen3-4b-inst-2507-vllm",
                       help="First model name (directory name)")
    parser.add_argument("--model2", type=str, default="phi-2-vllm",
                       help="Second model name (directory name)")
    parser.add_argument("--model3", type=str, default=None,
                       help="Third model name (directory name) - optional")
    parser.add_argument("--model4", type=str, default=None,
                       help="Fourth model name (directory name) - optional")
    args = parser.parse_args()
    
    # Collect all provided models
    models = [args.model1, args.model2]
    if args.model3:
        models.append(args.model3)
    if args.model4:
        models.append(args.model4)
    
    compare_models(*models)
