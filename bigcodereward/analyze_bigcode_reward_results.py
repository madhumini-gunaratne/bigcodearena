#!/usr/bin/env python3
"""
Unified BigCode Reward Analysis Script

This script combines:
1. Downloading data from bigcode/bigcodereward-experiment-results
2. Analyzing judge model performance (classification metrics)
3. Computing ELO ratings and correlations

The script provides comprehensive analysis of judge model performance
and ELO rankings from the BigCode reward experiment results.
"""

import json
import os
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, classification_report,
    cohen_kappa_score, matthews_corrcoef
)
from pathlib import Path
import logging
from datetime import datetime
from tqdm import tqdm
import random
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset

# Set random seed for reproducibility
random.seed(42)

def setup_logging(output_dir):
    """Setup logging configuration"""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f'bigcode_reward_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return log_file

def download_bigcode_reward_data(output_dir="data"):
    """
    Download both BigCode reward experiment results and BigCode reward datasets from Hugging Face Hub
    
    Args:
        output_dir: Directory to save the downloaded data
        
    Returns:
        bool: True if download successful, False otherwise
    """
    print("📥 Downloading BigCode reward datasets...")
    
    success = True
    
    # Download BigCode reward experiment results (judge model results)
    print("\n1️⃣ Downloading bigcode/bigcodereward-experiment-results...")
    try:
        judge_dataset = load_dataset("bigcode/bigcodereward-experiment-results")
        
        print("✅ Judge results dataset downloaded successfully!")
        print(f"📊 Dataset info: {judge_dataset}")
        
        # Display basic information about the dataset structure
        if hasattr(judge_dataset, 'keys'):
            print(f"📁 Available splits: {list(judge_dataset.keys())}")
            
            # Iterate through each split to show detailed information
            for split_name in judge_dataset.keys():
                split_data = judge_dataset[split_name]
                print(f"  {split_name} split: {len(split_data)} examples")
                
                # Display column names if available
                if hasattr(split_data, 'column_names'):
                    print(f"    Columns: {split_data.column_names}")
                
                # Show the structure of the first example
                if len(split_data) > 0:
                    print(f"    First example keys: {list(split_data[0].keys())}")
        
        # Save the judge dataset to JSONL format
        save_judge_dataset_to_jsonl(judge_dataset, output_dir)
        
    except Exception as e:
        print(f"❌ Error downloading judge results dataset: {e}")
        success = False
    
    # Download BigCode reward dataset (human votes)
    print("\n2️⃣ Downloading bigcode/bigcodereward...")
    try:
        reward_dataset = load_dataset("bigcode/bigcodereward")
        
        print("✅ BigCode reward dataset downloaded successfully!")
        print(f"📊 Dataset info: {reward_dataset}")
        
        # Display basic information about the dataset structure
        if hasattr(reward_dataset, 'keys'):
            print(f"📁 Available splits: {list(reward_dataset.keys())}")
            
            # Iterate through each split to show detailed information
            for split_name in reward_dataset.keys():
                split_data = reward_dataset[split_name]
                print(f"  {split_name} split: {len(split_data)} examples")
                
                # Display column names if available
                if hasattr(split_data, 'column_names'):
                    print(f"    Columns: {split_data.column_names}")
                
                # Show the structure of the first example
                if len(split_data) > 0:
                    print(f"    First example keys: {list(split_data[0].keys())}")
        
        # Save the reward dataset to JSONL format with field filtering
        save_reward_dataset_to_jsonl(reward_dataset, output_dir)
        
    except Exception as e:
        print(f"❌ Error downloading BigCode reward dataset: {e}")
        success = False
    
    return success

def save_judge_dataset_to_jsonl(dataset, output_dir):
    """
    Save judge results dataset to JSONL files in the local filesystem.
    
    Args:
        dataset: The judge dataset object to save
        output_dir: Output directory for saving files
    """
    # Define output directory for the judge results dataset
    dataset_dir = os.path.join(output_dir, "judge_results_dataset")
    os.makedirs(dataset_dir, exist_ok=True)
    
    print(f"💾 Saving judge dataset to JSONL files in '{dataset_dir}' directory...")
    
    # Handle datasets with multiple splits
    if hasattr(dataset, 'keys'):
        for split_name in dataset.keys():
            split_data = dataset[split_name]
            output_file = os.path.join(dataset_dir, f"{split_name}.jsonl")
            
            print(f"  Saving {split_name} split to {output_file}...")
            
            # Write each example as a JSON line
            with open(output_file, 'w', encoding='utf-8') as f:
                for example in split_data:
                    # Convert example to JSON string and write as a line
                    json_line = json.dumps(example, ensure_ascii=False)
                    f.write(json_line + '\n')
            
            print(f"    ✅ Saved {len(split_data)} examples to {output_file}")
    else:
        # Handle datasets without splits - save as single file
        output_file = os.path.join(dataset_dir, "dataset.jsonl")
        print(f"  Saving dataset to {output_file}...")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in dataset:
                # Convert each example to JSON and write as a line
                json_line = json.dumps(example, ensure_ascii=False)
                f.write(json_line + '\n')
        
        print(f"    ✅ Saved {len(dataset)} examples to {output_file}")
    
    print(f"✅ Judge dataset JSONL files saved in '{dataset_dir}' directory!")

def save_reward_dataset_to_jsonl(dataset, output_dir):
    """
    Save BigCode reward dataset to JSONL files with selective field filtering.
    
    This function creates a directory structure and saves each dataset split
    as a separate JSONL file. Large binary fields (screenshots, execution outputs,
    and error messages) are excluded to reduce file size while preserving
    the core evaluation and comparison data.
    
    Args:
        dataset: The reward dataset object to save
        output_dir: Output directory for saving files
    """
    # Define output directory for the BigCode reward dataset
    dataset_dir = os.path.join(output_dir, "rewardbench")
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Define fields to exclude from saving to reduce file size
    # These fields typically contain large binary data or verbose text
    exclude_fields = [
        'execution_screenshot_A', 'execution_output_A', 'execution_error_A',
        'execution_screenshot_B', 'execution_output_B', 'execution_error_B'
    ]
    
    print(f"💾 Saving reward dataset to JSONL files in '{dataset_dir}' directory...")
    print(f"Excluding fields: {exclude_fields}")
    
    # Handle datasets with multiple splits
    if hasattr(dataset, 'keys'):
        for split_name in dataset.keys():
            split_data = dataset[split_name]
            output_file = os.path.join(dataset_dir, f"{split_name}.jsonl")
            
            print(f"  Saving {split_name} split to {output_file}...")
            
            # Write each filtered example as a JSON line
            with open(output_file, 'w', encoding='utf-8') as f:
                for example in split_data:
                    # Filter out excluded fields to reduce file size
                    filtered_example = {k: v for k, v in example.items() if k not in exclude_fields}
                    # Convert filtered example to JSON string and write as a line
                    json_line = json.dumps(filtered_example, ensure_ascii=False)
                    f.write(json_line + '\n')
            
            print(f"    ✅ Saved {len(split_data)} examples to {output_file}")
    else:
        # Handle datasets without splits - save as single file
        output_file = os.path.join(dataset_dir, "dataset.jsonl")
        print(f"  Saving dataset to {output_file}...")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in dataset:
                # Filter out excluded fields to reduce file size
                filtered_example = {k: v for k, v in example.items() if k not in exclude_fields}
                # Convert filtered example to JSON and write as a line
                json_line = json.dumps(filtered_example, ensure_ascii=False)
                f.write(json_line + '\n')
        
        print(f"    ✅ Saved {len(dataset)} examples to {output_file}")
    
    print(f"✅ Reward dataset JSONL files saved in '{dataset_dir}' directory!")

def load_judge_results_from_dataset(data_dir="data/judge_results_dataset"):
    """
    Load judge results from Hugging Face cached data
    
    Args:
        data_dir: Data directory path
        
    Returns:
        Dict: {model_name: {execution_type: [records]}}
    """
    judge_results = {}
    
    if not os.path.exists(data_dir):
        logging.error(f"Data directory does not exist: {data_dir}")
        return judge_results
    
    # Scan all JSONL files
    for file_name in os.listdir(data_dir):
        if not file_name.endswith('.jsonl'):
            continue
            
        # Parse filename to get model name and execution type
        # Format: model_name_execution_type.jsonl
        if '_with_execution.jsonl' in file_name:
            model_name = file_name.replace('_with_execution.jsonl', '')
            execution_type = 'with_execution'
        elif '_without_execution.jsonl' in file_name:
            model_name = file_name.replace('_without_execution.jsonl', '')
            execution_type = 'without_execution'
        else:
            continue
            
        file_path = os.path.join(data_dir, file_name)
        
        logging.info(f"Loading {file_name}...")
        
        # Load JSONL file
        records = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    records.append(data)
                except json.JSONDecodeError as e:
                    logging.warning(f"Error parsing line {line_num} in {file_name}: {e}")
                    continue
        
        if model_name not in judge_results:
            judge_results[model_name] = {}
        
        judge_results[model_name][execution_type] = records
        logging.info(f"Loaded {len(records)} records for {model_name} ({execution_type})")
    
    return judge_results

def load_human_votes_from_dataset(data_dir="data/rewardbench"):
    """
    Load human voting data from rewardbench dataset
    
    Args:
        data_dir: rewardbench data directory
        
    Returns:
        Dict: {chat_session_id: human_vote_info}
    """
    human_votes = {}
    train_file = os.path.join(data_dir, "train.jsonl")
    
    if not os.path.exists(train_file):
        logging.error(f"Rewardbench train file not found: {train_file}")
        return human_votes
    
    logging.info(f"Loading human votes from {train_file}...")
    
    # Since the file is large, we need to read line by line
    count = 0
    vote_conversion_stats = Counter()
    
    with open(train_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(tqdm(f, desc="Loading human votes"), 1):
            try:
                data = json.loads(line.strip())
                
                # Extract chat_session_id and human vote
                chat_session_id = data.get('chat_session_id')
                if not chat_session_id:
                    continue
                
                # Extract human votes from rewardbench data
                # rewardbench uses vote_model_A/vote_model_B format
                human_vote_raw = data.get('human_vote') or data.get('vote_type')
                category_name = data.get('category_name', 'unknown')
                category_id = data.get('category_id')
                
                # Convert vote format: vote_model_A -> vote_left, vote_model_B -> vote_right
                human_vote = None
                if human_vote_raw == 'vote_model_A':
                    human_vote = 'vote_left'
                    vote_conversion_stats['vote_model_A -> vote_left'] += 1
                elif human_vote_raw == 'vote_model_B':
                    human_vote = 'vote_right'
                    vote_conversion_stats['vote_model_B -> vote_right'] += 1
                elif human_vote_raw in ['vote_tie', 'vote_both_bad']:
                    human_vote = human_vote_raw
                    vote_conversion_stats[f'{human_vote_raw} -> {human_vote_raw}'] += 1
                
                if human_vote and human_vote in ['vote_left', 'vote_right', 'vote_tie', 'vote_both_bad']:
                    human_votes[chat_session_id] = {
                        'vote': human_vote,
                        'category_name': category_name,
                        'category_id': category_id,
                        'original_vote': human_vote_raw
                    }
                    count += 1
                
                # No limit on loading count, load all data
                # if count >= 100000:  # Removed limit
                #     break
                    
            except json.JSONDecodeError as e:
                if line_num <= 10:  # Only log errors for first 10 lines
                    logging.warning(f"Error parsing line {line_num}: {e}")
                continue
    
    logging.info(f"Loaded {len(human_votes)} human vote records")
    logging.info(f"Vote conversion statistics: {dict(vote_conversion_stats)}")
    return human_votes

def calculate_classification_metrics(y_true, y_pred, labels=None):
    """Calculate classification metrics"""
    if labels is None:
        labels = ['vote_left', 'vote_right', 'vote_tie', 'vote_both_bad']
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    
    # Macro and micro averages
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average='macro', zero_division=0
    )
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average='micro', zero_division=0
    )
    
    # Other metrics
    kappa = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Classification report
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision_per_class': precision.tolist(),
        'recall_per_class': recall.tolist(),
        'f1_per_class': f1.tolist(),
        'support_per_class': support.tolist(),
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'micro_f1': micro_f1,
        'cohen_kappa': kappa,
        'matthews_corrcoef': mcc,
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'labels': labels
    }

def merge_labels(labels, merge_rule='tie_and_bad'):
    """
    Merge labels
    
    Args:
        labels: Original label list
        merge_rule: Merge rule
            - 'tie_and_bad': Merge vote_tie and vote_both_bad into vote_uncertain
    
    Returns:
        Merged label list
    """
    merged_labels = []
    
    for label in labels:
        if merge_rule == 'tie_and_bad':
            if label in ['vote_tie', 'vote_both_bad']:
                merged_labels.append('vote_uncertain')
            else:
                merged_labels.append(label)
        else:
            merged_labels.append(label)
    
    return merged_labels

def calculate_merged_classification_metrics(y_true, y_pred):
    """Calculate classification metrics after merging labels"""
    # Merge labels
    y_true_merged = merge_labels(y_true, 'tie_and_bad')
    y_pred_merged = merge_labels(y_pred, 'tie_and_bad')
    
    # Define merged labels
    merged_labels = ['vote_left', 'vote_right', 'vote_uncertain']
    
    # Calculate metrics
    return calculate_classification_metrics(y_true_merged, y_pred_merged, merged_labels)

def calculate_agreement_metrics(y_true, y_pred):
    """Calculate agreement metrics"""
    # Exact agreement rate
    exact_agreement = np.mean(y_true == y_pred)
    
    # Agreement rate by category
    labels = ['vote_left', 'vote_right', 'vote_tie', 'vote_both_bad']
    class_agreement = {}
    
    for label in labels:
        mask = (y_true == label)
        if np.sum(mask) > 0:
            class_agreement[label] = np.mean(y_pred[mask] == label)
        else:
            class_agreement[label] = 0.0
    
    # Vote distribution comparison
    true_dist = Counter(y_true)
    pred_dist = Counter(y_pred)
    
    return {
        'exact_agreement': exact_agreement,
        'class_agreement': class_agreement,
        'true_distribution': dict(true_dist),
        'pred_distribution': dict(pred_dist)
    }

def analyze_judge_models(judge_results, human_votes, output_dir):
    """
    Analyze judge model performance
    
    Args:
        judge_results: Judge results data
        human_votes: Human voting data
        output_dir: Output directory
    """
    # Create model judge-specific output directory
    model_judge_output_dir = os.path.join(output_dir, "model_judge")
    os.makedirs(model_judge_output_dir, exist_ok=True)
    
    results = {}
    
    for model_name, model_data in judge_results.items():
        for execution_type, records in model_data.items():
            system_name = f"{model_name} ({execution_type})"
            
            logging.info(f"Processing {system_name}...")
            
            # Match human votes and model judgments
            matched_pairs = []
            classification_data = {}  # Store data for each classification
            
            for record in records:
                chat_session_id = record.get('chat_session_id')
                judge_vote_raw = record.get('judgment')
                
                if not chat_session_id or not judge_vote_raw:
                    continue
                
                # Convert judge model vote format: vote_model_A -> vote_left, vote_model_B -> vote_right
                judge_vote = None
                if judge_vote_raw == 'vote_model_A':
                    judge_vote = 'vote_left'
                elif judge_vote_raw == 'vote_model_B':
                    judge_vote = 'vote_right'
                elif judge_vote_raw in ['vote_tie', 'vote_both_bad']:
                    judge_vote = judge_vote_raw
                
                if chat_session_id in human_votes and judge_vote:
                    human_vote_info = human_votes[chat_session_id]
                    human_vote = human_vote_info['vote']
                    category_name = human_vote_info['category_name']
                    
                    if judge_vote in ['vote_left', 'vote_right', 'vote_tie', 'vote_both_bad']:
                        matched_pairs.append((human_vote, judge_vote))
                        
                        # Statistics by classification
                        if category_name not in classification_data:
                            classification_data[category_name] = []
                        classification_data[category_name].append((human_vote, judge_vote))
            
            if not matched_pairs:
                logging.warning(f"No matched pairs found for {system_name}")
                continue
            
            # Extract labels
            y_true = np.array([pair[0] for pair in matched_pairs])
            y_pred = np.array([pair[1] for pair in matched_pairs])
            
            logging.info(f"Found {len(matched_pairs)} matched pairs for {system_name}")
            
            # Calculate classification metrics
            classification_metrics = calculate_classification_metrics(y_true, y_pred)
            
            # Calculate classification metrics after merging labels
            merged_classification_metrics = calculate_merged_classification_metrics(y_true, y_pred)
            
            # Calculate agreement metrics
            agreement_metrics = calculate_agreement_metrics(y_true, y_pred)
            
            # Calculate metrics for each classification
            by_classification_metrics = {}
            for category_name, category_pairs in classification_data.items():
                if len(category_pairs) > 0:
                    cat_y_true = np.array([pair[0] for pair in category_pairs])
                    cat_y_pred = np.array([pair[1] for pair in category_pairs])
                    
                    # Calculate classification metrics
                    cat_classification_metrics = calculate_classification_metrics(cat_y_true, cat_y_pred)
                    cat_merged_classification_metrics = calculate_merged_classification_metrics(cat_y_true, cat_y_pred)
                    cat_agreement_metrics = calculate_agreement_metrics(cat_y_true, cat_y_pred)
                    
                    by_classification_metrics[category_name] = {
                        'n_samples': len(category_pairs),
                        'classification_metrics': cat_classification_metrics,
                        'merged_classification_metrics': cat_merged_classification_metrics,
                        'agreement_metrics': cat_agreement_metrics,
                        'predictions': cat_y_pred.tolist(),
                        'true_labels': cat_y_true.tolist()
                    }
            
            # Save results
            results[system_name] = {
                'judge_model': model_name,
                'output_mode': execution_type,
                'n_samples': len(matched_pairs),
                'classification_metrics': classification_metrics,
                'merged_classification_metrics': merged_classification_metrics,
                'agreement_metrics': agreement_metrics,
                'predictions': y_pred.tolist(),
                'true_labels': y_true.tolist(),
                'by_classification': by_classification_metrics
            }
            
            # Save detailed results
            safe_model_name = model_name.replace('/', '_').replace('-', '_')
            result_file = os.path.join(model_judge_output_dir, f'{safe_model_name}_{execution_type}_results.json')
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(results[system_name], f, indent=2, ensure_ascii=False)
    
    return results

# ============================================================================
# ELO ANALYSIS FUNCTIONS
# ============================================================================

def compute_online_elo(battles, K=4, SCALE=400, BASE=10, INIT_RATING=1000):
    """
    Compute online ELO ratings
    Parameters are consistent with the original code
    """
    rating = defaultdict(lambda: INIT_RATING)
    for model_a, model_b, vote in battles:
        ra = rating[model_a]
        rb = rating[model_b]
        ea = 1 / (1 + BASE ** ((rb - ra) / SCALE))
        eb = 1 / (1 + BASE ** ((ra - rb) / SCALE))
        if vote == "vote_left":
            sa = 1
        elif vote == "vote_right":
            sa = 0
        else:
            sa = 0.5
        rating[model_a] += K * (sa - ea)
        rating[model_b] += K * (1 - sa - eb)
    return dict(rating)

def bootstrap_confidence_intervals(battles, n_bootstrap=1000, confidence_level=0.9, K=4, SCALE=400,
                                   BASE=10, INIT_RATING=1000):
    """Calculate confidence intervals using Bootstrap method"""
    original_ratings = compute_online_elo(battles, K, SCALE, BASE, INIT_RATING)
    bootstrap_ratings = defaultdict(list)

    for _ in tqdm(range(n_bootstrap), desc="Bootstrap sampling"):
        bootstrap_sample = random.choices(battles, k=len(battles))
        sample_ratings = compute_online_elo(bootstrap_sample, K, SCALE, BASE, INIT_RATING)
        for model, rating in sample_ratings.items():
            bootstrap_ratings[model].append(rating)

    alpha = (1 - confidence_level) / 2
    result = {}

    for model, ratings in bootstrap_ratings.items():
        ratings_sorted = sorted(ratings)
        lower_idx = int(alpha * n_bootstrap)
        upper_idx = int((1 - alpha) * n_bootstrap)

        result[model] = {
            'rating': original_ratings[model],
            'lower_bound': ratings_sorted[lower_idx],
            'upper_bound': ratings_sorted[upper_idx],
            'ci': [ratings_sorted[lower_idx] - original_ratings[model], ratings_sorted[upper_idx] - original_ratings[model]]
        }

    sorted_result = dict(sorted(result.items(), key=lambda x: x[1]['rating'], reverse=True))
    return sorted_result

def load_human_votes_for_elo(data_dir="data/rewardbench"):
    """Load human voting data from rewardbench dataset and convert to battle format"""
    battles = []
    train_file = os.path.join(data_dir, "train.jsonl")
    
    if not os.path.exists(train_file):
        logging.error(f"Rewardbench train file not found: {train_file}")
        return battles
    
    logging.info(f"Loading human votes from {train_file}...")
    
    count = 0
    vote_conversion_stats = defaultdict(int)
    
    with open(train_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(tqdm(f, desc="Loading human votes"), 1):
            try:
                data = json.loads(line.strip())
                
                # Extract basic information
                chat_session_id = data.get('chat_session_id')
                model_a = data.get('model_A')
                model_b = data.get('model_B')
                human_vote_raw = data.get('human_vote')
                
                if not all([chat_session_id, model_a, model_b, human_vote_raw]):
                    continue
                
                # Convert vote format: vote_model_A -> vote_left, vote_model_B -> vote_right
                human_vote = None
                if human_vote_raw == 'vote_model_A':
                    human_vote = 'vote_left'
                    vote_conversion_stats['vote_model_A -> vote_left'] += 1
                elif human_vote_raw == 'vote_model_B':
                    human_vote = 'vote_right'
                    vote_conversion_stats['vote_model_B -> vote_right'] += 1
                elif human_vote_raw in ['vote_tie', 'vote_both_bad']:
                    human_vote = human_vote_raw
                    vote_conversion_stats[f'{human_vote_raw} -> {human_vote_raw}'] += 1
                
                if human_vote and human_vote in ['vote_left', 'vote_right', 'vote_tie', 'vote_both_bad']:
                    battles.append([model_a, model_b, human_vote])
                    count += 1
                    
            except json.JSONDecodeError as e:
                if line_num <= 10:  # Only log errors for first 10 lines
                    logging.warning(f"Error parsing line {line_num}: {e}")
                continue
    
    logging.info(f"Extracted {len(battles)} human vote battles")
    logging.info(f"Vote conversion statistics: {dict(vote_conversion_stats)}")
    return battles

def load_judge_results_for_elo(data_dir="data/judge_results_dataset"):
    """Load judge results from Hugging Face cached data and convert to battle format"""
    judge_battles = {}
    
    if not os.path.exists(data_dir):
        logging.error(f"Data directory does not exist: {data_dir}")
        return judge_battles
    
    # Scan all JSONL files
    for file_name in os.listdir(data_dir):
        if not file_name.endswith('.jsonl'):
            continue
            
        # Parse filename to get model name and execution type
        if '_with_execution.jsonl' in file_name:
            model_name = file_name.replace('_with_execution.jsonl', '')
            execution_type = 'with_execution'
        elif '_without_execution.jsonl' in file_name:
            model_name = file_name.replace('_without_execution.jsonl', '')
            execution_type = 'without_execution'
        else:
            continue
            
        file_path = os.path.join(data_dir, file_name)
        system_name = f"{model_name} ({execution_type})"
        
        logging.info(f"Loading {file_name} for ELO...")
        
        battles = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    
                    # Extract basic information
                    chat_session_id = data.get('chat_session_id')
                    model_a = data.get('model_A')
                    model_b = data.get('model_B')
                    judge_vote_raw = data.get('judgment')
                    
                    if not all([chat_session_id, model_a, model_b, judge_vote_raw]):
                        continue
                    
                    # Convert judge model vote format: vote_model_A -> vote_left, vote_model_B -> vote_right
                    judge_vote = None
                    if judge_vote_raw == 'vote_model_A':
                        judge_vote = 'vote_left'
                    elif judge_vote_raw == 'vote_model_B':
                        judge_vote = 'vote_right'
                    elif judge_vote_raw in ['vote_tie', 'vote_both_bad']:
                        judge_vote = judge_vote_raw
                    
                    if judge_vote and judge_vote in ['vote_left', 'vote_right', 'vote_tie', 'vote_both_bad']:
                        battles.append([model_a, model_b, judge_vote])
                        
                except json.JSONDecodeError as e:
                    logging.warning(f"Error parsing line {line_num} in {file_name}: {e}")
                    continue
        
        judge_battles[system_name] = battles
        logging.info(f"Extracted {len(battles)} battles for {system_name}")
    
    return judge_battles

def calculate_correlation(ratings1, ratings2, method='spearman'):
    """Calculate correlation between two rating systems"""
    # Find common models
    common_models = set(ratings1.keys()) & set(ratings2.keys())
    
    if len(common_models) < 3:
        logging.warning(f"Not enough common models for correlation: {len(common_models)}")
        return None, None, None
    
    # Extract ratings for common models
    scores1 = [ratings1[model]['rating'] if isinstance(ratings1[model], dict) else ratings1[model] 
               for model in common_models]
    scores2 = [ratings2[model]['rating'] if isinstance(ratings2[model], dict) else ratings2[model] 
               for model in common_models]
    
    if method == 'spearman':
        corr, p_value = stats.spearmanr(scores1, scores2)
    elif method == 'pearson':
        corr, p_value = stats.pearsonr(scores1, scores2)
    else:
        raise ValueError(f"Unsupported correlation method: {method}")
    
    return corr, p_value, list(common_models)

def create_correlation_plot(ratings_dict, output_dir, title_prefix=""):
    """Create correlation matrix plot"""
    # Get names of all rating systems
    system_names = list(ratings_dict.keys())
    n_systems = len(system_names)
    
    if n_systems < 2:
        logging.warning("Need at least 2 rating systems for correlation analysis")
        return
    
    # Create correlation matrix
    correlation_matrix = np.zeros((n_systems, n_systems))
    p_value_matrix = np.zeros((n_systems, n_systems))
    
    for i, system1 in enumerate(system_names):
        for j, system2 in enumerate(system_names):
            if i == j:
                correlation_matrix[i, j] = 1.0
                p_value_matrix[i, j] = 0.0
            else:
                corr, p_val, _ = calculate_correlation(ratings_dict[system1], ratings_dict[system2])
                correlation_matrix[i, j] = corr if corr is not None else 0.0
                p_value_matrix[i, j] = p_val if p_val is not None else 1.0
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
    
    sns.heatmap(correlation_matrix, 
                mask=mask,
                annot=True, 
                fmt='.3f',
                xticklabels=system_names,
                yticklabels=system_names,
                cmap='RdBu_r',
                center=0,
                vmin=-1, vmax=1,
                square=True)
    
    plt.title(f'{title_prefix}Correlation Matrix (Spearman)')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save plot
    safe_title = title_prefix.lower().replace(" ", "_").replace("/", "_")
    plt.savefig(os.path.join(output_dir, f'{safe_title}correlation_matrix.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save numerical data
    correlation_data = {
        'system_names': system_names,
        'correlation_matrix': correlation_matrix.tolist(),
        'p_value_matrix': p_value_matrix.tolist()
    }
    
    with open(os.path.join(output_dir, f'{safe_title}correlation_data.json'), 'w') as f:
        json.dump(correlation_data, f, indent=2)
    
    return correlation_matrix, p_value_matrix

def create_scatter_plot(ratings1, ratings2, name1, name2, output_dir):
    """Create scatter plot for two rating systems"""
    # Find common models
    common_models = set(ratings1.keys()) & set(ratings2.keys())
    
    if len(common_models) < 3:
        logging.warning(f"Not enough common models for scatter plot: {len(common_models)}")
        return
    
    # Extract ratings
    scores1 = []
    scores2 = []
    model_names = []
    
    for model in common_models:
        score1 = ratings1[model]['rating'] if isinstance(ratings1[model], dict) else ratings1[model]
        score2 = ratings2[model]['rating'] if isinstance(ratings2[model], dict) else ratings2[model]
        scores1.append(score1)
        scores2.append(score2)
        model_names.append(model)
    
    # Calculate correlation
    corr, p_val, _ = calculate_correlation(ratings1, ratings2)
    
    # Create scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(scores1, scores2, alpha=0.7, s=50)
    
    # Add model name labels
    for i, model in enumerate(model_names):
        plt.annotate(model, (scores1[i], scores2[i]), 
                    xytext=(5, 5), textcoords='offset points', 
                    fontsize=8, alpha=0.7)
    
    # Add trend line
    if len(scores1) > 1:
        z = np.polyfit(scores1, scores2, 1)
        p = np.poly1d(z)
        plt.plot(scores1, p(scores1), "r--", alpha=0.8)
    
    plt.xlabel(f'{name1} ELO Rating')
    plt.ylabel(f'{name2} ELO Rating')
    plt.title(f'{name1} vs {name2}\nSpearman r = {corr:.3f}, p = {p_val:.3f}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    safe_name1 = name1.replace('/', '_').replace(' ', '_')
    safe_name2 = name2.replace('/', '_').replace(' ', '_')
    plt.savefig(os.path.join(output_dir, f'scatter_{safe_name1}_vs_{safe_name2}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def calculate_model_battle_stats(battles):
    """Calculate win/tie/loss statistics for each model"""
    model_stats = defaultdict(lambda: {'wins': 0, 'ties': 0, 'losses': 0, 'total': 0})
    
    for model_a, model_b, vote in battles:
        model_stats[model_a]['total'] += 1
        model_stats[model_b]['total'] += 1
        
        if vote == 'vote_left':  # model_a wins
            model_stats[model_a]['wins'] += 1
            model_stats[model_b]['losses'] += 1
        elif vote == 'vote_right':  # model_b wins
            model_stats[model_b]['wins'] += 1
            model_stats[model_a]['losses'] += 1
        elif vote in ['vote_tie', 'vote_both_bad']:  # tie
            model_stats[model_a]['ties'] += 1
            model_stats[model_b]['ties'] += 1
    
    # Calculate win rates
    for model, stats in model_stats.items():
        if stats['total'] > 0:
            stats['win_rate'] = stats['wins'] / stats['total']
            stats['tie_rate'] = stats['ties'] / stats['total']
            stats['loss_rate'] = stats['losses'] / stats['total']
        else:
            stats['win_rate'] = stats['tie_rate'] = stats['loss_rate'] = 0.0
    
    return dict(model_stats)

# ============================================================================
# REPORT GENERATION FUNCTIONS
# ============================================================================

def generate_judge_evaluation_report(results, output_dir):
    """Generate judge evaluation report"""
    # Create model judge-specific output directory
    model_judge_output_dir = os.path.join(output_dir, "model_judge")
    os.makedirs(model_judge_output_dir, exist_ok=True)
    
    report = {
        'evaluation_timestamp': datetime.now().isoformat(),
        'summary': {
            'total_systems': len(results),
            'systems_evaluated': list(results.keys())
        },
        'performance_ranking': {},
        'merged_performance_ranking': {},
        'detailed_results': results
    }
    
    # Ranking by different metrics - original 4-class classification
    metrics_for_ranking = ['accuracy', 'macro_f1', 'cohen_kappa', 'matthews_corrcoef']
    
    for metric in metrics_for_ranking:
        ranking = sorted(
            results.items(),
            key=lambda x: x[1]['classification_metrics'][metric],
            reverse=True
        )
        
        report['performance_ranking'][metric] = [
            {
                'rank': i + 1,
                'system': system_name,
                'score': data['classification_metrics'][metric],
                'n_samples': data['n_samples']
            }
            for i, (system_name, data) in enumerate(ranking)
        ]
    
    # Ranking by different metrics - 3-class classification after merging labels
    for metric in metrics_for_ranking:
        ranking = sorted(
            results.items(),
            key=lambda x: x[1]['merged_classification_metrics'][metric],
            reverse=True
        )
        
        report['merged_performance_ranking'][metric] = [
            {
                'rank': i + 1,
                'system': system_name,
                'score': data['merged_classification_metrics'][metric],
                'n_samples': data['n_samples']
            }
            for i, (system_name, data) in enumerate(ranking)
        ]
    
    # Save JSON report
    with open(os.path.join(model_judge_output_dir, 'judge_evaluation_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    
    # Generate text report
    generate_judge_text_report(report, model_judge_output_dir)
    
    return report

def generate_judge_text_report(report, output_dir):
    """Generate readable text report for judge evaluation"""
    with open(os.path.join(output_dir, 'judge_evaluation_report.txt'), 'w', encoding='utf-8') as f:
        f.write("Judge Model Evaluation Report (BigCode Reward Dataset)\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Evaluation Date: {report['evaluation_timestamp']}\n")
        f.write(f"Total Systems Evaluated: {report['summary']['total_systems']}\n")
        f.write(f"Systems: {', '.join(report['summary']['systems_evaluated'])}\n\n")
        
        # Performance ranking - using full names
        metric_names = {
            'accuracy': 'ACCURACY',
            'macro_f1': 'MACRO F1-SCORE',
            'cohen_kappa': 'COHEN\'S KAPPA',
            'matthews_corrcoef': 'MATTHEWS CORRELATION COEFFICIENT'
        }
        
        # Original 4-class classification results
        f.write("=" * 60 + "\n")
        f.write("ORIGINAL 4-CLASS EVALUATION (vote_left, vote_right, vote_tie, vote_both_bad)\n")
        f.write("=" * 60 + "\n")
        
        for metric in ['accuracy', 'macro_f1', 'cohen_kappa', 'matthews_corrcoef']:
            f.write(f"\n{metric_names[metric]} Ranking:\n")
            f.write("-" * 50 + "\n")
            
            for rank_data in report['performance_ranking'][metric]:
                f.write(f"{rank_data['rank']:2d}. {rank_data['system']:<35} "
                       f"{rank_data['score']:6.3f} (samples={rank_data['n_samples']})\n")
        
        # Merged 3-class classification results
        f.write("\n\n" + "=" * 60 + "\n")
        f.write("MERGED 3-CLASS EVALUATION (vote_left, vote_right, vote_uncertain)\n")
        f.write("=" * 60 + "\n")
        f.write("Note: vote_tie and vote_both_bad are merged into vote_uncertain\n\n")
        
        for metric in ['accuracy', 'macro_f1', 'cohen_kappa', 'matthews_corrcoef']:
            f.write(f"\n{metric_names[metric]} Ranking (Merged Labels):\n")
            f.write("-" * 50 + "\n")
            
            for rank_data in report['merged_performance_ranking'][metric]:
                f.write(f"{rank_data['rank']:2d}. {rank_data['system']:<35} "
                       f"{rank_data['score']:6.3f} (samples={rank_data['n_samples']})\n")
        
        # Detailed results comparison
        f.write("\n\n" + "=" * 60 + "\n")
        f.write("DETAILED RESULTS COMPARISON\n")
        f.write("=" * 60 + "\n")
        
        for system_name, data in report['detailed_results'].items():
            f.write(f"\n{system_name}:\n")
            f.write("-" * len(system_name) + "\n")
            
            cm = data['classification_metrics']
            merged_cm = data['merged_classification_metrics']
            am = data['agreement_metrics']
            
            f.write(f"Sample Size: {data['n_samples']}\n\n")
            
            # Original 4-class results
            f.write("Original 4-Class Results:\n")
            f.write(f"  Accuracy: {cm['accuracy']:.3f}\n")
            f.write(f"  Macro F1-Score: {cm['macro_f1']:.3f}\n")
            f.write(f"  Cohen's Kappa: {cm['cohen_kappa']:.3f}\n")
            f.write(f"  Matthews Correlation Coefficient: {cm['matthews_corrcoef']:.3f}\n")
            
            # Merged 3-class results
            f.write("\nMerged 3-Class Results:\n")
            f.write(f"  Accuracy: {merged_cm['accuracy']:.3f}\n")
            f.write(f"  Macro F1-Score: {merged_cm['macro_f1']:.3f}\n")
            f.write(f"  Cohen's Kappa: {merged_cm['cohen_kappa']:.3f}\n")
            f.write(f"  Matthews Correlation Coefficient: {merged_cm['matthews_corrcoef']:.3f}\n")
            
            # Performance improvement calculation
            acc_improvement = merged_cm['accuracy'] - cm['accuracy']
            f1_improvement = merged_cm['macro_f1'] - cm['macro_f1']
            kappa_improvement = merged_cm['cohen_kappa'] - cm['cohen_kappa']
            
            f.write(f"\nImprovement with Merged Labels:\n")
            f.write(f"  Accuracy: {acc_improvement:+.3f} ({acc_improvement/cm['accuracy']*100:+.1f}%)\n")
            f.write(f"  Macro F1-Score: {f1_improvement:+.3f} ({f1_improvement/cm['macro_f1']*100:+.1f}%)\n")
            f.write(f"  Cohen's Kappa: {kappa_improvement:+.3f} ({kappa_improvement/cm['cohen_kappa']*100:+.1f}%)\n")
            
            f.write(f"\nExact Agreement Rate: {am['exact_agreement']:.3f}\n")
            
            # Per-class performance for original 4 classes
            f.write("\nOriginal Per-class Performance:\n")
            labels = cm['labels']
            for i, label in enumerate(labels):
                f.write(f"  {label}: Precision={cm['precision_per_class'][i]:.3f}, "
                       f"Recall={cm['recall_per_class'][i]:.3f}, "
                       f"F1-Score={cm['f1_per_class'][i]:.3f}\n")
            
            # Per-class performance for merged labels
            f.write("\nMerged Per-class Performance:\n")
            merged_labels = merged_cm['labels']
            for i, label in enumerate(merged_labels):
                f.write(f"  {label}: Precision={merged_cm['precision_per_class'][i]:.3f}, "
                       f"Recall={merged_cm['recall_per_class'][i]:.3f}, "
                       f"F1-Score={merged_cm['f1_per_class'][i]:.3f}\n")
            
            # Vote distribution information
            f.write("\nVote Distribution Comparison:\n")
            f.write("  Human votes: ")
            human_dist = am['true_distribution']
            for vote_type, count in human_dist.items():
                f.write(f"{vote_type}={count} ")
            f.write("\n")
            
            f.write("  Judge votes: ")
            judge_dist = am['pred_distribution']
            for vote_type, count in judge_dist.items():
                f.write(f"{vote_type}={count} ")
            f.write("\n")
        
        # Evaluation guide - using complete descriptions
        f.write("\n\n" + "=" * 60 + "\n")
        f.write("EVALUATION METRICS GUIDE\n")
        f.write("=" * 60 + "\n")
        f.write("• Accuracy: Overall correctness rate (correct predictions / total predictions)\n")
        f.write("• Macro F1-Score: Average F1-score across all classes (handles class imbalance better)\n")
        f.write("• Cohen's Kappa: Agreement beyond chance level (>0.6 = substantial agreement)\n")
        f.write("• Matthews Correlation Coefficient: Balanced measure considering all confusion matrix elements\n")
        f.write("• Exact Agreement Rate: Direct match rate with human judges\n")
        f.write("• Precision: True positives / (True positives + False positives) - accuracy of positive predictions\n")
        f.write("• Recall: True positives / (True positives + False negatives) - coverage of actual positives\n")
        f.write("• F1-Score: Harmonic mean of Precision and Recall (2 * Precision * Recall / (Precision + Recall))\n")
        
        # Add explanatory summary
        f.write("\n\nKEY FINDINGS SUMMARY\n")
        f.write("=" * 30 + "\n")
        
        # Find best models - original and merged labels
        best_accuracy_orig = max(report['performance_ranking']['accuracy'], key=lambda x: x['score'])
        best_accuracy_merged = max(report['merged_performance_ranking']['accuracy'], key=lambda x: x['score'])
        best_kappa_orig = max(report['performance_ranking']['cohen_kappa'], key=lambda x: x['score'])
        best_kappa_merged = max(report['merged_performance_ranking']['cohen_kappa'], key=lambda x: x['score'])
        
        f.write("Original 4-Class Results:\n")
        f.write(f"• Best Overall Performance: {best_accuracy_orig['system']} (Accuracy: {best_accuracy_orig['score']:.3f})\n")
        f.write(f"• Best Agreement with Humans: {best_kappa_orig['system']} (Cohen's Kappa: {best_kappa_orig['score']:.3f})\n")
        
        f.write("\nMerged 3-Class Results:\n")
        f.write(f"• Best Overall Performance: {best_accuracy_merged['system']} (Accuracy: {best_accuracy_merged['score']:.3f})\n")
        f.write(f"• Best Agreement with Humans: {best_kappa_merged['system']} (Cohen's Kappa: {best_kappa_merged['score']:.3f})\n")
        
        # Analyze with_output vs without_output
        with_output_systems = [s for s in report['summary']['systems_evaluated'] if 'with_execution' in s]
        without_output_systems = [s for s in report['summary']['systems_evaluated'] if 'without_execution' in s]
        
        if with_output_systems and without_output_systems:
            # Original 4-class average
            with_avg_orig = sum(data['classification_metrics']['accuracy'] 
                          for name, data in report['detailed_results'].items() 
                          if 'with_execution' in name) / len(with_output_systems)
            without_avg_orig = sum(data['classification_metrics']['accuracy'] 
                             for name, data in report['detailed_results'].items() 
                             if 'without_execution' in name) / len(without_output_systems)
            
            # Merged 3-class average
            with_avg_merged = sum(data['merged_classification_metrics']['accuracy'] 
                          for name, data in report['detailed_results'].items() 
                          if 'with_execution' in name) / len(with_output_systems)
            without_avg_merged = sum(data['merged_classification_metrics']['accuracy'] 
                             for name, data in report['detailed_results'].items() 
                             if 'without_execution' in name) / len(without_output_systems)
            
            f.write("\nMode Comparison (Original 4-Class):\n")
            f.write(f"• Average Accuracy with Execution: {with_avg_orig:.3f}\n")
            f.write(f"• Average Accuracy without Execution: {without_avg_orig:.3f}\n")
            f.write(f"• Performance Improvement with Execution: {(with_avg_orig - without_avg_orig):.3f} ({((with_avg_orig - without_avg_orig) / without_avg_orig * 100):+.1f}%)\n")
            
            f.write("\nMode Comparison (Merged 3-Class):\n")
            f.write(f"• Average Accuracy with Execution: {with_avg_merged:.3f}\n")
            f.write(f"• Average Accuracy without Execution: {without_avg_merged:.3f}\n")
            f.write(f"• Performance Improvement with Execution: {(with_avg_merged - without_avg_merged):.3f} ({((with_avg_merged - without_avg_merged) / without_avg_merged * 100):+.1f}%)\n")

def generate_elo_report(all_ratings, all_battles, output_dir):
    """Generate comprehensive ELO analysis report"""
    report = {
        'analysis_timestamp': datetime.now().isoformat(),
        'analysis_type': 'elo_from_bigcode_dataset',
        'data_source': 'bigcode/bigcodereward-experiment-results',
        'summary': {},
        'elo_rankings': {},
        'battle_statistics': {},
        'model_battle_stats': {},
        'human_vs_model_correlations': {},
        'mode_comparisons': {},
        'with_execution_correlations': {},
        'without_execution_correlations': {},
        'all_correlations_ranking': {}
    }
    
    # Summary statistics
    report['summary'] = {
        'total_rating_systems': len(all_ratings),
        'systems_analyzed': list(all_ratings.keys())
    }
    
    # ELO rankings
    for system_name, ratings in all_ratings.items():
        sorted_models = sorted(ratings.items(), key=lambda x: x[1]['rating'], reverse=True)
        report['elo_rankings'][system_name] = [
            {
                'rank': i+1,
                'model': model,
                'rating': data['rating'],
                'lower_bound': data['lower_bound'],
                'upper_bound': data['upper_bound']
            }
            for i, (model, data) in enumerate(sorted_models)
        ]
    
    # Battle statistics
    for system_name, battles in all_battles.items():
        report['battle_statistics'][system_name] = {
            'total_battles': len(battles),
            'unique_models': len(set([b[0] for b in battles] + [b[1] for b in battles])),
            'vote_distribution': {}
        }
        
        # Vote distribution
        vote_counts = defaultdict(int)
        for battle in battles:
            vote_counts[battle[2]] += 1
        report['battle_statistics'][system_name]['vote_distribution'] = dict(vote_counts)
    
    # Calculate model win/tie/loss statistics for each system
    for system_name, battles in all_battles.items():
        model_stats = calculate_model_battle_stats(battles)
        report['model_battle_stats'][system_name] = model_stats
    
    # Calculate correlation ranking between human votes and model judgments
    human_vs_model_correlations = []
    if 'Human Votes' in all_ratings:
        human_ratings = all_ratings['Human Votes']
        
        for system_name, model_ratings in all_ratings.items():
            if system_name != 'Human Votes':
                corr, p_val, common_models = calculate_correlation(human_ratings, model_ratings)
                
                if corr is not None:
                    human_vs_model_correlations.append({
                        'model_system': system_name,
                        'correlation': corr,
                        'p_value': p_val,
                        'common_models': len(common_models) if common_models else 0,
                        'significance': 'significant' if p_val and p_val < 0.05 else 'not significant'
                    })
        
        # Sort by correlation coefficient in descending order
        human_vs_model_correlations.sort(key=lambda x: x['correlation'], reverse=True)
    
    report['human_vs_model_correlations'] = human_vs_model_correlations
    
    # Calculate correlation comparison between modes (with_execution vs without_execution)
    mode_correlations = {}
    judge_models = set()
    
    for system_name in all_ratings.keys():
        if '(' in system_name and ')' in system_name:
            parts = system_name.split(' (')
            if len(parts) == 2:
                judge_model = parts[0]
                execution_type = parts[1].rstrip(')')
                judge_models.add(judge_model)
    
    for judge_model in judge_models:
        with_execution_key = f'{judge_model} (with_execution)'
        without_execution_key = f'{judge_model} (without_execution)'
        
        if with_execution_key in all_ratings and without_execution_key in all_ratings:
            corr, p_val, common_models = calculate_correlation(
                all_ratings[with_execution_key], 
                all_ratings[without_execution_key]
            )
            
            mode_correlations[judge_model] = {
                'spearman_correlation': corr,
                'p_value': p_val,
                'common_models': len(common_models) if common_models else 0,
                'significance': 'significant' if p_val and p_val < 0.05 else 'not significant'
            }
    
    report['mode_comparisons'] = mode_correlations
    
    # Save report
    with open(os.path.join(output_dir, 'elo_analysis_report.json'), 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # Generate text report
    generate_elo_text_report(report, output_dir)

def generate_elo_text_report(report, output_dir):
    """Generate readable text report for ELO analysis"""
    with open(os.path.join(output_dir, 'elo_analysis_report.txt'), 'w', encoding='utf-8') as f:
        f.write("ELO Analysis Report (BigCode Reward Dataset)\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Analysis Date: {report['analysis_timestamp']}\n")
        f.write(f"Analysis Type: {report['analysis_type']}\n")
        f.write(f"Data Source: {report['data_source']}\n")
        f.write(f"Total Rating Systems: {report['summary']['total_rating_systems']}\n")
        f.write(f"Systems Analyzed: {', '.join(report['summary']['systems_analyzed'])}\n\n")
        
        # Human votes vs model correlation ranking
        if report.get('human_vs_model_correlations'):
            f.write("Human Votes vs Model Correlation Ranking:\n")
            f.write("=" * 60 + "\n")
            f.write("Rank  Model System                                     Correlation  P-value    Significance     Models\n")
            f.write("-" * 100 + "\n")
            for i, corr_data in enumerate(report['human_vs_model_correlations'], 1):
                f.write(f"{i:4d}  {corr_data['model_system']:<40} "
                       f"{corr_data['correlation']:10.3f}  "
                       f"{corr_data['p_value']:9.3f}  "
                       f"{corr_data['significance']:<15}  "
                       f"{corr_data['common_models']:6d}\n")
        
        # Mode correlation comparison
        if report.get('mode_comparisons'):
            f.write("\n\nMode Comparisons (with_execution vs without_execution):\n")
            f.write("-" * 50 + "\n")
            for judge_model, corr_data in report['mode_comparisons'].items():
                f.write(f"\n{judge_model}:\n")
                f.write(f"  Spearman r = {corr_data['spearman_correlation']:.3f}, "
                       f"p = {corr_data['p_value']:.3f} ({corr_data['significance']})\n")
                f.write(f"  Common models: {corr_data['common_models']}\n")
        
        # with_execution mode Judge Model correlation ranking
        if report.get('with_execution_correlations'):
            f.write("\n\nWith Execution Mode - Judge Model Correlation Ranking:\n")
            f.write("=" * 60 + "\n")
            for i, corr_data in enumerate(report['with_execution_correlations'], 1):
                f.write(f"{i:2d}. {corr_data['pair_name']:<35} "
                       f"r = {corr_data['correlation']:6.3f}, "
                       f"p = {corr_data['p_value']:6.3f} ({corr_data['significance']:<15}), "
                       f"n = {corr_data['common_models']}\n")
        
        # without_execution mode Judge Model correlation ranking
        if report.get('without_execution_correlations'):
            f.write("\n\nWithout Execution Mode - Judge Model Correlation Ranking:\n")
            f.write("=" * 60 + "\n")
            for i, corr_data in enumerate(report['without_execution_correlations'], 1):
                f.write(f"{i:2d}. {corr_data['pair_name']:<35} "
                       f"r = {corr_data['correlation']:6.3f}, "
                       f"p = {corr_data['p_value']:6.3f} ({corr_data['significance']:<15}), "
                       f"n = {corr_data['common_models']}\n")
        
        # All correlation rankings
        if report.get('all_correlations_ranking'):
            f.write("\n\nAll Systems Correlation Ranking (Including Human Votes):\n")
            f.write("=" * 70 + "\n")
            f.write("Rank  System Pair                                              Correlation  P-value    Significance     Models\n")
            f.write("-" * 110 + "\n")
            for i, corr_data in enumerate(report['all_correlations_ranking'], 1):
                f.write(f"{i:4d}  {corr_data['pair_name']:<50} "
                       f"{corr_data['correlation']:10.3f}  "
                       f"{corr_data['p_value']:9.3f}  "
                       f"{corr_data['significance']:<15}  "
                       f"{corr_data['common_models']:6d}\n")
        
        # ELO rankings
        f.write("\n\nELO Rankings:\n")
        f.write("-" * 30 + "\n")
        for system_name, rankings in report['elo_rankings'].items():
            f.write(f"\n{system_name}:\n")
            for rank_data in rankings:
                f.write(f"  {rank_data['rank']:2d}. {rank_data['model']:<25} "
                       f"Rating: {rank_data['rating']:7.1f} "
                       f"[{rank_data['lower_bound']:7.1f}, {rank_data['upper_bound']:7.1f}]\n")
        
        # Battle statistics
        f.write("\n\nBattle Statistics:\n")
        f.write("-" * 25 + "\n")
        for system_name, stats in report['battle_statistics'].items():
            f.write(f"\n{system_name}:\n")
            f.write(f"  Total battles: {stats['total_battles']}\n")
            f.write(f"  Unique models: {stats['unique_models']}\n")
            f.write(f"  Vote distribution: {stats['vote_distribution']}\n")
        
        # Model Win/Tie/Loss Statistics
        if report.get('model_battle_stats'):
            f.write("\n\nModel Win/Tie/Loss Statistics:\n")
            f.write("-" * 35 + "\n")
            for system_name, model_stats in report['model_battle_stats'].items():
                f.write(f"\n{system_name}:\n")
                f.write(f"{'Model':<25} {'Wins':<6} {'Ties':<6} {'Losses':<7} {'Total':<7} {'Win%':<7} {'Tie%':<7} {'Loss%':<7}\n")
                f.write("-" * 80 + "\n")
                
                # Sort by win rate
                sorted_models = sorted(model_stats.items(), key=lambda x: x[1]['win_rate'], reverse=True)
                for model, stats in sorted_models:
                    f.write(f"{model:<25} {stats['wins']:<6} {stats['ties']:<6} {stats['losses']:<7} "
                           f"{stats['total']:<7} {stats['win_rate']*100:<6.1f} {stats['tie_rate']*100:<6.1f} "
                           f"{stats['loss_rate']*100:<6.1f}\n")
        
        # Add ranking explanation
        f.write("\n\nRanking Interpretation:\n")
        f.write("-" * 30 + "\n")
        f.write("• Human vs Model Correlations: Higher values indicate better alignment with human preferences\n")
        f.write("• Higher correlation coefficients indicate better agreement between judge models\n")
        f.write("• Significant correlations (p < 0.05) suggest reliable agreement patterns\n")
        f.write("• 'n' represents the number of common models evaluated by both judges\n")
        f.write("• With Execution: Judges have access to model outputs when making decisions\n")
        f.write("• Without Execution: Judges make decisions based only on prompts and responses\n")
        f.write("• Win/Tie/Loss Statistics: Shows performance of each model in head-to-head comparisons\n")
        f.write("• Win% = Wins / Total battles, higher values indicate better performance\n")

def export_merged_3class_summary_json(results, output_dir):
    """
    Export MERGED 3-CLASS EVALUATION ACCURACY and MACRO F1-SCORE to JSON
    """
    # Create model judge-specific output directory
    model_judge_output_dir = os.path.join(output_dir, "model_judge")
    os.makedirs(model_judge_output_dir, exist_ok=True)
    
    summary_data = {
        'evaluation_timestamp': datetime.now().isoformat(),
        'evaluation_type': 'MERGED_3_CLASS_EVALUATION',
        'description': 'vote_left, vote_right, vote_uncertain (tie+both_bad merged)',
        'data_source': 'bigcode/bigcodereward-experiment-results',
        'systems': {}
    }
    
    for system_name, data in results.items():
        merged_metrics = data['merged_classification_metrics']
        summary_data['systems'][system_name] = {
            'accuracy': merged_metrics['accuracy'],
            'macro_f1_score': merged_metrics['macro_f1'],
            'n_samples': data['n_samples'],
            'judge_model': data['judge_model'],
            'output_mode': data['output_mode']
        }
    
    # Sort by accuracy
    sorted_systems = sorted(
        summary_data['systems'].items(),
        key=lambda x: x[1]['accuracy'],
        reverse=True
    )
    
    # Add ranking information
    summary_data['ranking_by_accuracy'] = []
    summary_data['ranking_by_macro_f1'] = []
    
    for rank, (system_name, metrics) in enumerate(sorted_systems, 1):
        summary_data['ranking_by_accuracy'].append({
            'rank': rank,
            'system': system_name,
            'accuracy': metrics['accuracy'],
            'macro_f1_score': metrics['macro_f1_score'],
            'n_samples': metrics['n_samples']
        })
    
    # Sort by macro_f1
    sorted_by_f1 = sorted(
        summary_data['systems'].items(),
        key=lambda x: x[1]['macro_f1_score'],
        reverse=True
    )
    
    for rank, (system_name, metrics) in enumerate(sorted_by_f1, 1):
        summary_data['ranking_by_macro_f1'].append({
            'rank': rank,
            'system': system_name,
            'accuracy': metrics['accuracy'],
            'macro_f1_score': metrics['macro_f1_score'],
            'n_samples': metrics['n_samples']
        })
    
    # Save JSON file
    json_file = os.path.join(model_judge_output_dir, 'merged_3class_summary.json')
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    logging.info(f"Merged 3-class summary saved to: {json_file}")
    return json_file

# ============================================================================
# MAIN ANALYSIS FUNCTIONS
# ============================================================================

def analyze_elo_from_dataset(judge_data_dir, human_votes_dir, output_dir):
    """
    Analyze ELO ratings and correlations from BigCode reward dataset
    
    Args:
        judge_data_dir: Judge results dataset directory
        human_votes_dir: Human votes dataset directory
        output_dir: Output directory
    """
    # Create ELO-specific output directory
    elo_output_dir = os.path.join(output_dir, "elo")
    os.makedirs(elo_output_dir, exist_ok=True)
    
    logging.info("Starting ELO analysis from BigCode reward dataset")
    logging.info(f"Judge data directory: {judge_data_dir}")
    logging.info(f"Human votes directory: {human_votes_dir}")
    logging.info(f"ELO output directory: {elo_output_dir}")
    
    all_ratings = {}
    all_battles = {}
    
    # Process human voting data
    logging.info("Processing human votes...")
    human_battles = load_human_votes_for_elo(human_votes_dir)
    
    if human_battles:
        human_elo = bootstrap_confidence_intervals(human_battles)
        all_ratings['Human Votes'] = human_elo
        all_battles['Human Votes'] = human_battles
        
        # Save human voting ELO results
        with open(os.path.join(elo_output_dir, 'human_votes_elo.json'), 'w') as f:
            json.dump(human_elo, f, indent=2)
        
        logging.info(f"Successfully processed {len(human_battles)} human vote battles")
    else:
        logging.warning("No human battles found")
    
    # Process judge model data
    logging.info("Processing judge model data...")
    judge_battles_dict = load_judge_results_for_elo(judge_data_dir)
    
    for system_name, battles in judge_battles_dict.items():
        if battles:
            logging.info(f"Computing ELO for {system_name}...")
            elo_results = bootstrap_confidence_intervals(battles)
            all_ratings[system_name] = elo_results
            all_battles[system_name] = battles
            
            # Save ELO results
            safe_filename = system_name.replace('/', '_').replace(' ', '_').replace('(', '').replace(')', '') + '_elo.json'
            with open(os.path.join(elo_output_dir, safe_filename), 'w') as f:
                json.dump(elo_results, f, indent=2)
    
    # Create overall correlation analysis
    if len(all_ratings) >= 2:
        logging.info("Creating overall correlation analysis...")
        create_correlation_plot(all_ratings, elo_output_dir, "Overall ")
        
        # Create pairwise scatter plots
        system_names = list(all_ratings.keys())
        for i in range(len(system_names)):
            for j in range(i+1, len(system_names)):
                name1, name2 = system_names[i], system_names[j]
                create_scatter_plot(all_ratings[name1], all_ratings[name2], 
                                  name1, name2, elo_output_dir)
    
    # Generate comprehensive report
    generate_elo_report(all_ratings, all_battles, elo_output_dir)
    
    logging.info(f"ELO analysis completed. Results saved to {elo_output_dir}")
    return all_ratings, all_battles

def main():
    """Main function"""
    # Configure paths (consistent with original scripts)
    judge_data_dir = "data/judge_results_dataset"
    human_votes_dir = "data/rewardbench"
    output_dir = "analysis_results_bigcode_reward"
    
    print("🚀 Starting BigCode Reward Analysis...")
    print(f"📁 Judge data directory: {judge_data_dir}")
    print(f"👥 Human votes directory: {human_votes_dir}")
    print(f"📊 Output directory: {output_dir}")
    print("=" * 80)
    
    # Setup logging
    setup_logging(output_dir)
    
    # Download data if not exists
    if not os.path.exists(judge_data_dir) or not os.path.exists(human_votes_dir):
        print("📥 Downloading BigCode reward datasets...")
        if not download_bigcode_reward_data("data"):
            print("❌ Failed to download datasets. Exiting.")
            return
    else:
        print(f"✅ Datasets already exist")
        print(f"   📁 Judge data: {judge_data_dir}")
        print(f"   👥 Human votes: {human_votes_dir}")
    
    # Load judge results data
    print("📊 Loading judge results from dataset...")
    judge_results = load_judge_results_from_dataset(judge_data_dir)
    
    if not judge_results:
        print("❌ No judge results loaded. Exiting.")
        return
    
    # Load human voting data
    print("👥 Loading human votes from dataset...")
    human_votes = load_human_votes_from_dataset(human_votes_dir)
    
    if not human_votes:
        print("❌ No human votes loaded. Exiting.")
        return
    
    # Analyze judge models
    print("🔍 Analyzing judge models...")
    results = analyze_judge_models(judge_results, human_votes, output_dir)
    
    if not results:
        print("❌ No results to analyze. Exiting.")
        return
    
    # Generate judge evaluation report
    print("📝 Generating judge evaluation report...")
    generate_judge_evaluation_report(results, output_dir)
    
    # Export summary
    print("📤 Exporting summary...")
    export_merged_3class_summary_json(results, output_dir)
    
    # Analyze ELO ratings
    print("🏆 Analyzing ELO ratings...")
    elo_ratings, elo_battles = analyze_elo_from_dataset(judge_data_dir, human_votes_dir, output_dir)
    
    print("\n" + "=" * 80)
    print("🎉 Analysis completed successfully!")
    print(f"📊 Summary:")
    print(f"   📁 Judge models analyzed: {len(judge_results)}")
    print(f"   🤖 Total systems: {len(results)}")
    print(f"   👥 Human votes loaded: {len(human_votes)}")
    print(f"   🏆 ELO systems analyzed: {len(elo_ratings)}")
    print(f"   💾 Output saved to: {output_dir}")
    
    print(f"\n📊 Key output files:")
    print(f"   📁 Model Judge Results (analysis_results_bigcode_reward/model_judge/):")
    print(f"      - merged_3class_summary.json: MERGED 3-CLASS EVALUATION summary")
    print(f"      - judge_evaluation_report.txt: Complete judge evaluation report")
    print(f"      - judge_evaluation_report.json: Complete judge evaluation JSON")
    print(f"      - *_results.json: Individual model results")
    print(f"   📁 ELO Analysis Results (analysis_results_bigcode_reward/elo/):")
    print(f"      - elo_analysis_report.txt: Complete ELO analysis report")
    print(f"      - elo_analysis_report.json: Complete ELO analysis JSON")
    print(f"      - human_votes_elo.json: Human votes ELO ratings")
    print(f"      - *_elo.json: Individual judge model ELO ratings")
    print(f"      - overall_correlation_matrix.png: Correlation heatmap")
    print(f"      - scatter_*.png: Pairwise scatter plots")

if __name__ == '__main__':
    main()
