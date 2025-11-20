#!/usr/bin/env python3
"""
Deduplication script to remove semantically similar questions from generated questions.
Uses sentence transformers to compute semantic similarity and filters based on threshold.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import numpy as np
from sentence_transformers import SentenceTransformer, util


def load_config():
    """Load configuration from generation_config.yaml"""
    import yaml
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


def load_all_generated_questions(results_dir: Path) -> Tuple[List[Dict], Dict[str, List[int]]]:
    """
    Load all generated questions from results directory.
    Returns list of questions and a mapping of category to indices.
    """
    questions = []
    category_indices = defaultdict(list)
    
    for jsonl_file in sorted(results_dir.glob("*_generated.jsonl")):
        file_questions = load_questions_from_file(jsonl_file)
        start_idx = len(questions)
        category = jsonl_file.stem.replace("_generated", "")
        
        for q in file_questions:
            category_indices[category].append(len(questions))
            questions.append(q)
    
    return questions, dict(category_indices)


def compute_similarities(instructions: List[str], model) -> np.ndarray:
    """Compute pairwise semantic similarity between instructions"""
    print(f"Encoding {len(instructions)} instructions...")
    embeddings = model.encode(instructions, show_progress_bar=True)
    
    print("Computing pairwise similarities...")
    similarities = util.pytorch_cos_sim(embeddings, embeddings).cpu().numpy()
    
    return similarities


def find_duplicates(questions: List[Dict], similarities: np.ndarray, threshold: float = 0.85) -> List[Tuple[int, int, float]]:
    """
    Find pairs of similar questions above threshold.
    Returns list of (idx1, idx2, similarity) tuples.
    """
    duplicates = []
    
    for i in range(len(questions)):
        for j in range(i + 1, len(questions)):
            sim = similarities[i][j]
            if sim >= threshold:
                duplicates.append((i, j, float(sim)))
    
    return sorted(duplicates, key=lambda x: x[2], reverse=True)


def mark_duplicates_for_removal(duplicates: List[Tuple[int, int, float]], total_questions: int) -> List[int]:
    """
    Determine which duplicate indices to remove using a greedy approach.
    Keeps the first occurrence, removes subsequent duplicates.
    """
    keep_mask = [True] * total_questions
    removed_indices = []
    
    for idx1, idx2, sim in duplicates:
        if keep_mask[idx1] and keep_mask[idx2]:
            # Keep idx1, remove idx2
            keep_mask[idx2] = False
            removed_indices.append(idx2)
    
    return removed_indices


def save_deduplicated_questions(questions: List[Dict], removed_indices: List[int], 
                               output_dir: Path, category_indices: Dict[str, List[int]]):
    """Save deduplicated questions back to category-specific JSONL files"""
    removed_set = set(removed_indices)
    
    # Group remaining questions by category
    category_questions = defaultdict(list)
    
    for category, indices in category_indices.items():
        for idx in indices:
            if idx not in removed_set:
                category_questions[category].append(questions[idx])
    
    # Save each category
    for category, qs in category_questions.items():
        output_file = output_dir / f"{category}_generated_deduped.jsonl"
        with open(output_file, 'w') as f:
            for q in qs:
                f.write(json.dumps(q) + "\n")
        print(f"Saved {len(qs)} questions to {output_file.name}")


def print_duplicate_samples(duplicates: List[Tuple[int, int, float]], questions: List[Dict], num_samples: int = 5):
    """Print samples of detected duplicates"""
    print(f"\nTop {min(num_samples, len(duplicates))} duplicate pairs:")
    print("-" * 100)
    
    for i, (idx1, idx2, sim) in enumerate(duplicates[:num_samples]):
        print(f"\nDuplicate pair {i+1} (Similarity: {sim:.4f})")
        print(f"\nQuestion 1 (idx {idx1}):")
        print(f"  Category: {questions[idx1]['category']}")
        print(f"  Instruction: {questions[idx1]['instruction'][:150]}...")
        
        print(f"\nQuestion 2 (idx {idx2}):")
        print(f"  Category: {questions[idx2]['category']}")
        print(f"  Instruction: {questions[idx2]['instruction'][:150]}...")
        print("-" * 100)


def main():
    parser = argparse.ArgumentParser(description="Remove duplicate questions from generated results")
    parser.add_argument("--results_dir", type=Path, default=Path(__file__).parent / "results",
                        help="Directory containing generated JSONL files")
    parser.add_argument("--threshold", type=float, default=0.85,
                        help="Similarity threshold for deduplication (0-1)")
    parser.add_argument("--model", type=str, default="all-mpnet-base-v2",
                        help="Sentence transformer model to use")
    parser.add_argument("--show_samples", type=int, default=5,
                        help="Number of duplicate samples to show")
    
    args = parser.parse_args()
    
    # Load questions
    print(f"Loading questions from {args.results_dir}...")
    questions, category_indices = load_all_generated_questions(args.results_dir)
    print(f"Loaded {len(questions)} total questions from {len(category_indices)} categories")
    
    # Load model
    print(f"\nLoading sentence transformer model: {args.model}...")
    model = SentenceTransformer(args.model)
    
    # Extract instructions
    instructions = [q["instruction"] for q in questions]
    
    # Compute similarities
    similarities = compute_similarities(instructions, model)
    
    # Find duplicates
    print(f"\nFinding duplicates with threshold {args.threshold}...")
    duplicates = find_duplicates(questions, similarities, threshold=args.threshold)
    print(f"Found {len(duplicates)} duplicate pairs")
    
    # Show samples
    if duplicates and args.show_samples > 0:
        print_duplicate_samples(duplicates, questions, num_samples=args.show_samples)
    
    # Mark for removal
    removed_indices = mark_duplicates_for_removal(duplicates, len(questions))
    print(f"\nMarking {len(removed_indices)} questions for removal")
    
    # Save deduplicated results
    print(f"\nSaving deduplicated questions...")
    save_deduplicated_questions(questions, removed_indices, args.results_dir, category_indices)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"DEDUPLICATION SUMMARY")
    print(f"{'='*60}")
    print(f"Original questions: {len(questions)}")
    print(f"Removed duplicates: {len(removed_indices)}")
    print(f"Remaining questions: {len(questions) - len(removed_indices)}")
    print(f"Duplicate pairs found: {len(duplicates)}")
    print(f"Similarity threshold: {args.threshold}")
    print(f"Model: {args.model}")
    print(f"Output files: *_generated_deduped.jsonl in {args.results_dir}")


if __name__ == "__main__":
    main()
