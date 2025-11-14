"""
Generate comprehensive documentation for collected tasks.

Creates two documentation files:
1. TASK_DOCUMENTATION.csv - Spreadsheet format (source, category, task, difficulty, language, url)
2. task_metadata.json - JSON format (all metadata for programmatic access)
"""

import json
import csv
import os
from collections import defaultdict
from datetime import datetime


def generate_documentation():
    """Generate all documentation files from collected_tasks.json"""
    
    # Load collected tasks
    tasks_file = 'autocodearena/data/collected_tasks.json'
    
    if not os.path.exists(tasks_file):
        print(f"❌ Error: {tasks_file} not found")
        print("   First run: python fetch_github_tasks.py")
        return
    
    with open(tasks_file, 'r', encoding='utf-8') as f:
        tasks = json.load(f)
    
    print("=" * 80)
    print("BigCodeArena - Task Documentation Generator")
    print("=" * 80)
    print()
    
    # Create output directory if needed
    output_dir = 'autocodearena/data'
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Generate CSV file
    print("📄 Generating CSV documentation...")
    csv_file = os.path.join(output_dir, 'collected_tasks.csv')
    generate_csv(tasks, csv_file)
    print(f"   ✅ Created: {csv_file}")
    
    # 2. Generate JSON metadata file
    print("🔗 Generating JSON metadata...")
    json_file = os.path.join(output_dir, 'task_metadata.json')
    generate_json_metadata(tasks, json_file)
    print(f"   ✅ Created: {json_file}")
    
    # Print summary
    tasks_by_category = defaultdict(list)
    for task in tasks:
        tasks_by_category[task['category']].append(task)
    
    print()
    print("=" * 80)
    print("📊 DOCUMENTATION SUMMARY")
    print("=" * 80)
    print(f"Total tasks documented: {len(tasks)}")
    print()
    print("By Category:")
    for category in sorted(tasks_by_category.keys()):
        count = len(tasks_by_category[category])
        pct = (count / len(tasks)) * 100
        print(f"  - {category:25s}: {count:3d} tasks ({pct:5.1f}%)")
    print()
    print("Output files created in: autocodearena/data/")
    print("=" * 80)
    print()


def generate_csv(tasks, output_file):
    """Generate CSV file with tasks"""
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Task ID',
            'Source Repository',
            'Category',
            'Difficulty',
            'Language',
            'Question/Instruction (Preview)',
            'Full Source URL',
            'Author',
            'Fetched Date'
        ])
        
        for i, task in enumerate(tasks, 1):
            # Truncate instruction to first 150 chars for readability in CSV
            instruction_preview = task['instruction'][:150].replace('\n', ' ').replace('\r', '')
            if len(task['instruction']) > 150:
                instruction_preview += '...'
            
            writer.writerow([
                task['uid'],
                task['source_name'],
                task['category'],
                task['difficulty'],
                task['language'],
                instruction_preview,
                task['source'],
                task['author'],
                task['fetched_date']
            ])


def generate_json_metadata(tasks, output_file):
    """Generate JSON metadata file"""
    
    # Group by category for summary
    tasks_by_category = defaultdict(list)
    for task in tasks:
        tasks_by_category[task['category']].append(task)
    
    # Group by repository for summary
    tasks_by_repo = defaultdict(list)
    for task in tasks:
        tasks_by_repo[task['source_name']].append(task)
    
    # Create metadata document
    metadata = {
        "summary": {
            "total_tasks": len(tasks),
            "collection_date": datetime.now().isoformat(),
            "source": "GitHub Issues API",
            "categories": {
                cat: len(tasks_list)
                for cat, tasks_list in sorted(tasks_by_category.items())
            },
            "repositories": {
                repo: len(tasks_list)
                for repo, tasks_list in sorted(tasks_by_repo.items())
            }
        },
        "category_details": {
            category: {
                "count": len(tasks_list),
                "percentage": round((len(tasks_list) / len(tasks)) * 100, 1),
                "tasks": tasks_list
            }
            for category, tasks_list in sorted(tasks_by_category.items())
        },
        "repository_details": {
            repo: {
                "count": len(tasks_list),
                "tasks": tasks_list
            }
            for repo, tasks_list in sorted(tasks_by_repo.items())
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    try:
        generate_documentation()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
