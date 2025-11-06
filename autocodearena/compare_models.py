#!/usr/bin/env python3
"""
Compare execution results between two models
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

def compare_models(model1_name, model2_name):
    """Compare two models"""
    
    model1_dir = f"data/autocodearena_local/model_answer/{model1_name}"
    model2_dir = f"data/autocodearena_local/model_answer/{model2_name}"
    
    summary1 = load_summary(model1_dir)
    summary2 = load_summary(model2_dir)
    
    if not summary1 or not summary2:
        return
    
    print("\n" + "=" * 100)
    print("🏆 MODEL COMPARISON")
    print("=" * 100)
    print()
    
    # Overall comparison
    print("📊 OVERALL PERFORMANCE:")
    print("-" * 100)
    print(f"{'Metric':<40} | {model1_name:<30} | {model2_name:<30}")
    print("-" * 100)
    
    success_rate1 = summary1.get('success_rate', 0)
    success_rate2 = summary2.get('success_rate', 0)
    successful1 = summary1.get('successful', 0)
    successful2 = summary2.get('successful', 0)
    total1 = summary1.get('total_executions', 0)
    total2 = summary2.get('total_executions', 0)
    
    # Determine winner
    winner = "🥇 WINNER" if success_rate1 > success_rate2 else ("🥇 WINNER" if success_rate2 > success_rate1 else "TIE")
    winner1_mark = "🥇 WINNER" if success_rate1 > success_rate2 else ""
    winner2_mark = "🥇 WINNER" if success_rate2 > success_rate1 else ""
    
    print(f"{'Success Rate (%):':<40} | {success_rate1:>6.1f}% {winner1_mark:<20} | {success_rate2:>6.1f}% {winner2_mark:<20}")
    print(f"{'Successful / Total:':<40} | {successful1:>3}/{total1:<3} {' '*24} | {successful2:>3}/{total2:<3}")
    print(f"{'Failed:':<40} | {summary1.get('failed', 0):>3} {' '*24} | {summary2.get('failed', 0):>3}")
    print()
    
    # By Environment
    print("=" * 100)
    print("📌 PERFORMANCE BY ENVIRONMENT:")
    print("-" * 100)
    print(f"{'Environment':<30} | {model1_name:<25} | {model2_name:<25}")
    print(f"{'':30} | {'Success Rate':<25} | {'Success Rate':<25}")
    print("-" * 100)
    
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
    print("=" * 100)
    print("📂 PERFORMANCE BY CATEGORY:")
    print("-" * 100)
    print(f"{'Category':<30} | {model1_name:<25} | {model2_name:<25}")
    print(f"{'':30} | {'Success Rate':<25} | {'Success Rate':<25}")
    print("-" * 100)
    
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
    print("=" * 100)
    print("🎯 KEY INSIGHTS:")
    print("=" * 100)
    
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
    print("=" * 100)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare execution results between two models")
    parser.add_argument("--model1", type=str, default="qwen3-4b-inst-2507-vllm",
                       help="First model name (directory name)")
    parser.add_argument("--model2", type=str, default="phi-2-vllm",
                       help="Second model name (directory name)")
    args = parser.parse_args()
    
    compare_models(args.model1, args.model2)
