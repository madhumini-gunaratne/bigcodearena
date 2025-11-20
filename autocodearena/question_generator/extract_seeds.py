"""
Step 1: Extract Seed Questions from AutoCodeArena Dataset

This script loads existing questions from the AutoCodeArena HuggingFace dataset
and saves them organized by category as seed files for question generation.
    
The script creates JSONL files in the output directory, one per category:
    - web_design_questions.jsonl
    - game_development_questions.jsonl
    - problem_solving_questions.jsonl
    - etc.
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

try:
    from datasets import load_dataset
except ImportError:
    print("Error: datasets library not found. Install with: pip install datasets")
    exit(1)


def extract_seeds(output_dir: str = "seeds", dataset_name: str = "bigcode/autocodearena-v0"):
    """
    Extract seed questions from AutoCodeArena dataset and organize by category.
    
    Args:
        output_dir: Directory to save seed JSONL files
        dataset_name: HuggingFace dataset name
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    print(f"Loading dataset from {dataset_name}...")
    
    try:
        # Load the dataset from HuggingFace
        dataset = load_dataset(dataset_name, split="train")
        print(f"✓ Loaded {len(dataset)} questions from dataset")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Make sure you have internet connection and datasets library installed")
        exit(1)
    
    # Organize questions by category
    questions_by_category = defaultdict(list)
    
    print("\nOrganizing questions by category...")
    for item in tqdm(dataset, desc="Processing"):
        uid = item.get("uid", "")
        instruction = item.get("instruction", "")
        category = item.get("category", "uncategorized")
        
        if not instruction or not uid:
            print(f"Skipping incomplete question: {uid}")
            continue
        
        questions_by_category[category].append({
            "uid": uid,
            "instruction": instruction,
            "category": category
        })
    
    # Save each category to a separate JSONL file
    print(f"\nSaving seed files ({len(questions_by_category)} categories)...")
    
    summary = {}
    for category, questions in sorted(questions_by_category.items()):
        # Create filename from category (replace spaces/special chars)
        safe_category = category.replace(" ", "_").replace("/", "_").lower()
        output_file = output_path / f"{safe_category}_questions.jsonl"
        
        # Write JSONL file with readable formatting
        with open(output_file, 'w') as f:
            for q in questions:
                # Compact JSON, one per line (standard JSONL format)
                f.write(json.dumps(q) + '\n')
        
        summary[category] = len(questions)
        print(f"  ✓ {output_file.name}: {len(questions)} questions")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SEED EXTRACTION SUMMARY")
    print("="*60)
    total_questions = sum(summary.values())
    print(f"Total questions extracted: {total_questions}")
    print(f"Categories: {len(summary)}")
    print("\nBreakdown by category:")
    for category, count in sorted(summary.items(), key=lambda x: x[1], reverse=True):
        print(f"  {category}: {count}")
    print("="*60)
    
    return summary


def validate_seeds(seed_dir: str = "seeds"):
    """
    Validate that seed files were created correctly.
    
    Args:
        seed_dir: Directory containing seed JSONL files
    """
    print(f"\nValidating seed files in {seed_dir}...")
    
    seed_path = Path(seed_dir)
    if not seed_path.exists():
        print(f"Error: {seed_dir} does not exist")
        return False
    
    jsonl_files = list(seed_path.glob("*.jsonl"))
    if not jsonl_files:
        print(f"Error: No JSONL files found in {seed_dir}")
        return False
    
    print(f"Found {len(jsonl_files)} seed files")
    
    for jsonl_file in jsonl_files:
        try:
            count = 0
            with open(jsonl_file, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    assert 'uid' in data, f"Missing 'uid' field in {jsonl_file}"
                    assert 'instruction' in data, f"Missing 'instruction' field in {jsonl_file}"
                    assert 'category' in data, f"Missing 'category' field in {jsonl_file}"
                    count += 1
            
            print(f"  ✓ {jsonl_file.name}: {count} valid questions")
        
        except json.JSONDecodeError as e:
            print(f"  ✗ {jsonl_file.name}: JSON decode error - {e}")
            return False
        except AssertionError as e:
            print(f"  ✗ {jsonl_file.name}: {e}")
            return False
    
    print("✓ All seed files are valid!")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Extract seed questions from AutoCodeArena dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="seeds",
        help="Directory to save seed JSONL files (default: seeds/)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="bigcode/autocodearena-v0",
        help="HuggingFace dataset name (default: bigcode/autocodearena-v0)"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate seed files after extraction"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("STEP 1: EXTRACT SEED QUESTIONS")
    print("="*60)
    
    # Extract seeds
    summary = extract_seeds(args.output_dir, args.dataset)
    
    # Validate if requested
    if args.validate or True:  # Always validate
        validate_seeds(args.output_dir)
    
    print("\n✓ Seed extraction complete!")
    print(f"Seed files saved to: {Path(args.output_dir).absolute()}")


if __name__ == "__main__":
    main()
