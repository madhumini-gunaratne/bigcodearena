#!/usr/bin/env python3
"""
Refactored judge model analyzer - analyze experimental results from Hugging Face downloaded data

This script:
1. Loads Hugging Face downloaded data from data/ directory
2. Analyzes performance of all judge models
3. Generates the same analysis results as the original evaluate_judge_models.py
4. Supports extracting human votes from rewardbench dataset as ground truth labels
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
import yaml
from datasets import load_dataset

def setup_logging(output_dir):
    """Setup logging configuration"""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f'judge_evaluation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return log_file

def load_judge_results_from_local(results_dir="results"):
    """
    Load judge results from local results directory
    
    Args:
        results_dir: Results directory path (contains model subdirectories)
        
    Returns:
        Dict: {model_name: {execution_type: [records]}}
    """
    judge_results = {}
    
    if not os.path.exists(results_dir):
        logging.error(f"Results directory does not exist: {results_dir}")
        logging.info("Please run eval_hf_data.py first to generate judge results")
        return judge_results
    
    # Scan all model directories
    for model_dir in os.listdir(results_dir):
        model_path = os.path.join(results_dir, model_dir)
        
        if not os.path.isdir(model_path):
            continue
        
        logging.info(f"Loading results for model: {model_dir}")
        
        # Check for with_execution and without_execution subdirectories
        for execution_type in ['with_execution', 'without_execution']:
            execution_path = os.path.join(model_path, execution_type)
            
            if not os.path.exists(execution_path):
                continue
            
            # Find JSONL files in this directory
            for file_name in os.listdir(execution_path):
                if not file_name.endswith('.jsonl') or '_failed' in file_name:
                    continue
                
                file_path = os.path.join(execution_path, file_name)
                
                logging.info(f"Loading {file_path}...")
                
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
                
                if model_dir not in judge_results:
                    judge_results[model_dir] = {}
                
                judge_results[model_dir][execution_type] = records
                logging.info(f"Loaded {len(records)} records for {model_dir} ({execution_type})")
    
    return judge_results

def load_human_votes_from_rewardbench(dataset_name="bigcode/bigcodereward", split="train"):
    """
    Load human voting data from HuggingFace dataset
    
    Args:
        dataset_name: HuggingFace dataset name
        split: Dataset split to load
        
    Returns:
        Dict: {chat_session_id: human_vote_info}
    """
    human_votes = {}
    
    logging.info(f"Loading human votes from HuggingFace: {dataset_name} ({split})...")
    
    try:
        dataset = load_dataset(dataset_name, split=split)
        logging.info(f"Loaded {len(dataset)} records from HuggingFace")
    except Exception as e:
        logging.error(f"Error loading dataset from HuggingFace: {e}")
        return human_votes
    
    count = 0
    vote_conversion_stats = Counter()
    
    for idx, example in enumerate(tqdm(dataset, desc="Processing human votes")):
        try:
            data = dict(example)
            
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
            
        except Exception as e:
            if idx < 10:  # Only log errors for first 10 records
                logging.warning(f"Error processing record {idx}: {e}")
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
    os.makedirs(output_dir, exist_ok=True)
    
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
            result_file = os.path.join(output_dir, f'{safe_model_name}_{execution_type}_results.json')
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(results[system_name], f, indent=2, ensure_ascii=False)
    
    return results

def generate_evaluation_report(results, output_dir):
    """Generate evaluation report"""
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
    with open(os.path.join(output_dir, 'judge_evaluation_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    
    # Generate text report
    generate_text_report(report, output_dir)
    
    return report

def generate_text_report(report, output_dir):
    """Generate readable text report"""
    with open(os.path.join(output_dir, 'judge_evaluation_report.txt'), 'w', encoding='utf-8') as f:
        f.write("Judge Model Evaluation Report (From Hugging Face Data)\n")
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

def export_merged_3class_summary_json(results, output_dir):
    """
    Export MERGED 3-CLASS EVALUATION ACCURACY and MACRO F1-SCORE to JSON
    """
    summary_data = {
        'evaluation_timestamp': datetime.now().isoformat(),
        'evaluation_type': 'MERGED_3_CLASS_EVALUATION',
        'description': 'vote_left, vote_right, vote_uncertain (tie+both_bad merged)',
        'data_source': 'Hugging Face cached data',
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
    json_file = os.path.join(output_dir, 'merged_3class_summary.json')
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    logging.info(f"Merged 3-class summary saved to: {json_file}")
    return json_file

def main():
    """Main function"""
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Analyze judge model results')
    parser.add_argument('--results-dir', type=str, default='results', 
                       help='Directory containing judge results (default: results)')
    parser.add_argument('--dataset', type=str, default='bigcode/bigcodereward',
                       help='HuggingFace dataset for human votes (default: bigcode/bigcodereward)')
    parser.add_argument('--split', type=str, default='train',
                       help='Dataset split to use (default: train)')
    parser.add_argument('--output-dir', type=str, default='analysis_results_model_judge',
                       help='Output directory for analysis results (default: analysis_results_model_judge)')
    args = parser.parse_args()
    
    # Configure paths
    results_dir = args.results_dir
    dataset_name = args.dataset
    split = args.split
    output_dir = args.output_dir
    
    print("🚀 Starting judge model evaluation...")
    print(f"📁 Results directory: {results_dir}")
    print(f"👥 Human votes dataset: {dataset_name} ({split})")
    print(f"📊 Output directory: {output_dir}")
    print("=" * 80)
    
    # Setup logging
    setup_logging(output_dir)
    
    # Load judge results data
    print("📊 Loading judge results from local directory...")
    judge_results = load_judge_results_from_local(results_dir)
    
    if not judge_results:
        print("❌ No judge results loaded.")
        print("Please run eval_hf_data.py first to generate judge results")
        return
    
    # Load human voting data
    print("👥 Loading human votes from HuggingFace...")
    human_votes = load_human_votes_from_rewardbench(dataset_name, split)
    
    if not human_votes:
        print("❌ No human votes loaded. Exiting.")
        return
    
    # Analyze judge models
    print("🔍 Analyzing judge models...")
    results = analyze_judge_models(judge_results, human_votes, output_dir)
    
    if not results:
        print("❌ No results to analyze. Exiting.")
        return
    
    # Generate report
    print("📝 Generating evaluation report...")
    generate_evaluation_report(results, output_dir)
    
    # Export summary
    print("📤 Exporting summary...")
    export_merged_3class_summary_json(results, output_dir)
    
    print("\n" + "=" * 80)
    print("🎉 Evaluation completed successfully!")
    print(f"📊 Summary:")
    print(f"   📁 Judge models analyzed: {len(judge_results)}")
    print(f"   🤖 Total systems: {len(results)}")
    print(f"   👥 Human votes loaded: {len(human_votes)}")
    print(f"   💾 Output saved to: {output_dir}")
    
    print(f"\n📊 Key output files:")
    print(f"   - merged_3class_summary.json: MERGED 3-CLASS EVALUATION summary")
    print(f"   - judge_evaluation_report.txt: Complete text report")
    print(f"   - judge_evaluation_report.json: Complete JSON report")

if __name__ == '__main__':
    main()
