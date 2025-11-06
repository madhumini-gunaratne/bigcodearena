#!/usr/bin/env python3
"""
Simple script to analyze execution results
"""

import json
import os
from collections import defaultdict
from pathlib import Path
import argparse

def analyze_execution_results(model_dir):
    """Analyze execution results from JSONL file"""
    
    results_file = os.path.join(model_dir, "execution_results.jsonl")
    
    if not os.path.exists(results_file):
        print(f"❌ Results file not found: {results_file}")
        return
    
    print(f"📊 Analyzing execution results from: {results_file}\n")
    
    results = []
    with open(results_file, 'r') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    
    print(f"📈 Total executions: {len(results)}\n")
    
    # Calculate success/failure
    successful = 0
    failed = 0
    environment_stats = defaultdict(lambda: {"success": 0, "total": 0})
    category_stats = defaultdict(lambda: {"success": 0, "total": 0})
    
    for result in results:
        stderr = result.get('stderr', '').strip()
        environment = result.get('environment', 'Unknown')
        category = result.get('category', 'Unknown')
        
        # Count total executions per environment and category
        environment_stats[environment]["total"] += 1
        category_stats[category]["total"] += 1
        
        # Check if execution was successful (no stderr or only warnings)
        if not stderr or stderr == "":
            successful += 1
            environment_stats[environment]["success"] += 1
            category_stats[category]["success"] += 1
        else:
            failed += 1
    
    # Calculate success rate for later
    success_rate = (successful / len(results) * 100) if len(results) > 0 else 0
    
    # Print by environment
    print("=" * 80)
    print("📌 Results by Environment:")
    print("=" * 80)
    sorted_envs = sorted(environment_stats.items(), key=lambda x: x[1]["total"], reverse=True)
    for env, stats in sorted_envs:
        if stats["total"] > 0:
            env_success_rate = (stats["success"] / stats["total"] * 100)
            print(f"  {env:30} | Total: {stats['total']:3} | Success: {stats['success']:3} ({env_success_rate:5.1f}%)")
    print()
    
    # Print by category
    print("=" * 80)
    print("📂 Results by Category:")
    print("=" * 80)
    sorted_cats = sorted(category_stats.items(), key=lambda x: x[1]["total"], reverse=True)
    for cat, stats in sorted_cats:
        if stats["total"] > 0:
            cat_success_rate = (stats["success"] / stats["total"] * 100)
            print(f"  {cat:30} | Total: {stats['total']:3} | Success: {stats['success']:3} ({cat_success_rate:5.1f}%)")
    print()
    
    # Count no-code entries
    no_code_count = sum(1 for r in results if r.get('stderr') == 'No code to execute')
    if no_code_count > 0:
        print(f"⚠️  Entries with no extractable code: {no_code_count}")
    
    # Count missing-field entries
    missing_fields = sum(1 for r in results if 'Missing required fields' in r.get('stderr', ''))
    if missing_fields > 0:
        print(f"⚠️  Entries with missing required fields: {missing_fields}")
    print()
    
    # Print final summary
    print("=" * 80)
    print("📊 FINAL SUMMARY:")
    print("=" * 80)
    print(f"📈 Total executions: {len(results)}")
    print(f"✅ Successful executions: {successful} ({success_rate:.1f}%)")
    print(f"❌ Failed executions: {failed} ({100 - success_rate:.1f}%)")
    print("=" * 80)
    print()
    
    # Save summary to file
    summary = {
        "total_executions": len(results),
        "successful": successful,
        "failed": failed,
        "success_rate": success_rate,
        "by_environment": {
            env: {
                "total": stats["total"],
                "success": stats["success"],
                "success_rate": stats["success"] / stats["total"] * 100 if stats["total"] > 0 else 0
            }
            for env, stats in environment_stats.items()
        },
        "by_category": {
            cat: {
                "total": stats["total"],
                "success": stats["success"],
                "success_rate": stats["success"] / stats["total"] * 100 if stats["total"] > 0 else 0
            }
            for cat, stats in category_stats.items()
        }
    }
    
    summary_file = os.path.join(model_dir, "execution_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Summary saved to: {summary_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze execution results")
    parser.add_argument("--model", type=str, default="qwen3-4b-inst-2507-vllm",
                       help="Model name (directory name)")
    args = parser.parse_args()
    
    model_dir = f"data/autocodearena_local/model_answer/{args.model}"
    analyze_execution_results(model_dir)
