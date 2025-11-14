"""
Enrich HuggingFace tasks with GitHub context.

This script:
1. Loads 100 HuggingFace questions (bigcode/autocodearena-v0)
2. Loads 128 GitHub issues from collected_tasks.json
3. Matches them by category and content similarity
4. Creates enriched_tasks.json with both questions + related GitHub issues

Output: enriched_tasks.json
- Each HF task gets 2-3 related GitHub issues as context
- Better instructions for models with real-world examples
"""

import json
import os
from collections import defaultdict
from datetime import datetime
from difflib import SequenceMatcher


def load_hf_questions():
    """Load 100 questions from HuggingFace dataset"""
    try:
        from datasets import load_dataset
    except ImportError:
        print("❌ Error: datasets library not found")
        print("   Install with: pip install datasets")
        return []
    
    print("📥 Loading HuggingFace questions...")
    dataset = load_dataset("bigcode/autocodearena-v0", split="train")
    
    questions = []
    for item in dataset:
        questions.append({
            "uid": item["uid"],
            "instruction": item["instruction"],
            "category": item["category"],
        })
    
    print(f"✅ Loaded {len(questions)} HuggingFace questions")
    return questions


def load_github_issues():
    """Load 128 GitHub issues from collected_tasks.json"""
    github_file = "autocodearena/data/collected_tasks.json"
    
    if not os.path.exists(github_file):
        print(f"❌ Error: {github_file} not found")
        return []
    
    print("📥 Loading GitHub issues...")
    with open(github_file, 'r', encoding='utf-8') as f:
        issues = json.load(f)
    
    print(f"✅ Loaded {len(issues)} GitHub issues")
    return issues


def calculate_similarity(text1, text2):
    """
    Calculate similarity between two texts (0-1).
    Uses SequenceMatcher for simple word-level comparison.
    """
    if not text1 or not text2:
        return 0.0
    
    # Convert to lowercase and take first 200 chars for efficiency
    text1 = text1.lower()[:200]
    text2 = text2.lower()[:200]
    
    return SequenceMatcher(None, text1, text2).ratio()


def find_related_issues(hf_question, github_issues, top_k=3):
    """
    Find top-k related GitHub issues for a HF question.
    
    Match by:
    1. Same category (highest priority)
    2. Content similarity
    """
    
    category = hf_question['category']
    instruction = hf_question['instruction']
    
    # Group issues by category
    issues_by_category = defaultdict(list)
    for issue in github_issues:
        issues_by_category[issue['category']].append(issue)
    
    # Score issues: category match + similarity
    scored_issues = []
    
    # Priority: same category
    if category in issues_by_category:
        for issue in issues_by_category[category]:
            similarity = calculate_similarity(instruction, issue['instruction'])
            # Boost score for same category
            score = similarity * 1.5
            scored_issues.append((score, issue))
    
    # Secondary: other categories (with lower weight)
    for cat, issues_list in issues_by_category.items():
        if cat != category:
            for issue in issues_list:
                similarity = calculate_similarity(instruction, issue['instruction'])
                score = similarity * 0.5  # Lower weight for different category
                scored_issues.append((score, issue))
    
    # Sort by score and return top-k
    scored_issues.sort(reverse=True, key=lambda x: x[0])
    top_issues = [issue for score, issue in scored_issues[:top_k]]
    
    # Convert to display format
    return [
        {
            "title": issue['instruction'].split('\n')[0][:80],  # First line as title
            "source_name": issue['source_name'],
            "source_url": issue['source'],
            "difficulty": issue['difficulty'],
            "language": issue['language'],
            "author": issue['author'],
            "category": issue['category'],
            "uid": issue['uid']
        }
        for issue in top_issues
    ]


def create_enriched_instruction(hf_instruction, related_issues):
    """
    Create enriched instruction combining HF task + GitHub context.
    """
    enriched = hf_instruction
    
    if related_issues:
        enriched += "\n\n---\n\n**Related Real-World Examples from GitHub:**\n"
        for i, issue in enumerate(related_issues, 1):
            enriched += f"\n{i}. **{issue['title']}**\n"
            enriched += f"   - Source: {issue['source_name']}\n"
            enriched += f"   - Difficulty: {issue['difficulty']}\n"
            enriched += f"   - Language: {issue['language']}\n"
            enriched += f"   - Link: {issue['source_url']}\n"
    
    return enriched


def enrich_tasks():
    """Main function to create enriched tasks"""
    
    print("=" * 80)
    print("BigCodeArena - Task Enrichment with GitHub Context")
    print("=" * 80)
    print()
    
    # Load data
    hf_questions = load_hf_questions()
    github_issues = load_github_issues()
    
    if not hf_questions or not github_issues:
        print("❌ Failed to load questions or issues")
        return
    
    print()
    print("🔄 Enriching tasks with GitHub context...")
    print()
    
    # Create enriched tasks
    enriched_tasks = []
    category_stats = defaultdict(lambda: {"count": 0, "avg_related": 0})
    
    for i, hf_task in enumerate(hf_questions, 1):
        # Find related GitHub issues
        related_issues = find_related_issues(hf_task, github_issues, top_k=3)
        
        # Create enriched instruction
        enriched_instruction = create_enriched_instruction(
            hf_task['instruction'],
            related_issues
        )
        
        # Build enriched task
        enriched_task = {
            "uid": hf_task['uid'],
            "instruction": enriched_instruction,
            "category": hf_task['category'],
            "enrichment": {
                "original_instruction": hf_task['instruction'],
                "related_github_issues": related_issues,
                "num_related_issues": len(related_issues),
                "enriched_date": datetime.now().isoformat()
            }
        }
        
        enriched_tasks.append(enriched_task)
        
        # Track stats
        category_stats[hf_task['category']]['count'] += 1
        category_stats[hf_task['category']]['avg_related'] += len(related_issues)
        
        if i % 10 == 0:
            print(f"  ✓ Processed {i}/{len(hf_questions)} tasks")
    
    # Calculate averages
    for cat in category_stats:
        count = category_stats[cat]['count']
        category_stats[cat]['avg_related'] = round(
            category_stats[cat]['avg_related'] / count, 1
        )
    
    print()
    print("=" * 80)
    print("📊 ENRICHMENT SUMMARY")
    print("=" * 80)
    print(f"HuggingFace questions: {len(hf_questions)}")
    print(f"GitHub issues available: {len(github_issues)}")
    print(f"Total enriched tasks: {len(enriched_tasks)}")
    print()
    print("Enrichment by Category:")
    for cat in sorted(category_stats.keys()):
        stats = category_stats[cat]
        print(f"  - {cat:25s}: {stats['count']:3d} tasks, "
              f"avg {stats['avg_related']:.1f} related issues each")
    print()
    
    # Save enriched tasks
    output_file = "autocodearena/data/enriched_tasks.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(enriched_tasks, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Enriched tasks saved to: {output_file}")
    print()
    
    # Show sample
    print("=" * 80)
    print("📋 SAMPLE ENRICHED TASK")
    print("=" * 80)
    print()
    print(f"Task UID: {enriched_tasks[0]['uid']}")
    print(f"Category: {enriched_tasks[0]['category']}")
    print(f"Related GitHub Issues: {enriched_tasks[0]['enrichment']['num_related_issues']}")
    print()
    print("Enriched Instruction (first 500 chars):")
    print("-" * 80)
    print(enriched_tasks[0]['instruction'][:500])
    print("...")
    print()
    
    return enriched_tasks


if __name__ == "__main__":
    try:
        enrich_tasks()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
