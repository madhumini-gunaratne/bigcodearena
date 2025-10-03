#!/usr/bin/env python3
"""
Compute ELO ratings from Hugging Face data

This script:
1. Loads Hugging Face downloaded data from data/ directory
2. Computes ELO ratings for human votes and various judge models
3. Analyzes correlations between different judge models
4. Generates the same analysis results as the original compute_elo_unified.py
"""

import json
import os
from pathlib import Path
from collections import defaultdict
import random
from tqdm import tqdm
from datetime import datetime
import logging
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import yaml
from datasets import load_dataset

random.seed(42)

def setup_logging(output_dir):
    """Setup logging configuration"""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f'elo_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    return log_file

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

def load_human_votes_from_rewardbench(dataset_name="bigcode/bigcodereward", split="train"):
    """Load human voting data from HuggingFace dataset and convert to battle format"""
    battles = []
    
    logging.info(f"Loading human votes from HuggingFace: {dataset_name} ({split})...")
    
    try:
        dataset = load_dataset(dataset_name, split=split)
        logging.info(f"Loaded {len(dataset)} records from HuggingFace")
    except Exception as e:
        logging.error(f"Error loading dataset from HuggingFace: {e}")
        return battles
    
    count = 0
    vote_conversion_stats = defaultdict(int)
    
    for idx, example in enumerate(tqdm(dataset, desc="Processing human votes")):
        try:
            data = dict(example)
                
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
                    
        except Exception as e:
            if idx < 10:  # Only log errors for first 10 records
                logging.warning(f"Error processing record {idx}: {e}")
            continue
    
    logging.info(f"Extracted {len(battles)} human vote battles")
    logging.info(f"Vote conversion statistics: {dict(vote_conversion_stats)}")
    return battles

def load_judge_results_from_local(results_dir="results"):
    """Load judge results from local results directory and convert to battle format"""
    judge_battles = {}
    
    if not os.path.exists(results_dir):
        logging.error(f"Results directory does not exist: {results_dir}")
        logging.info("Please run eval_hf_data.py first to generate judge results")
        return judge_battles
    
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
            
            system_name = f"{model_dir} ({execution_type})"
            battles = []
            
            # Find JSONL files in this directory
            for file_name in os.listdir(execution_path):
                if not file_name.endswith('.jsonl') or '_failed' in file_name:
                    continue
                
                file_path = os.path.join(execution_path, file_name)
                
                logging.info(f"Loading {file_path}...")
                
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

def analyze_elo_from_local_and_hf(results_dir, dataset_name, split, output_dir):
    """
    Analyze ELO ratings and correlations
    
    Args:
        results_dir: Local results directory
        dataset_name: HuggingFace dataset name for human votes
        split: Dataset split to use
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    setup_logging(output_dir)
    
    logging.info("Starting ELO analysis")
    logging.info(f"Results directory: {results_dir}")
    logging.info(f"Human votes dataset: {dataset_name} ({split})")
    
    all_ratings = {}
    all_battles = {}
    
    # Process human voting data
    logging.info("Processing human votes...")
    human_battles = load_human_votes_from_rewardbench(dataset_name, split)
    
    if human_battles:
        human_elo = bootstrap_confidence_intervals(human_battles)
        all_ratings['Human Votes'] = human_elo
        all_battles['Human Votes'] = human_battles
        
        # Save human voting ELO results
        with open(os.path.join(output_dir, 'human_votes_elo.json'), 'w') as f:
            json.dump(human_elo, f, indent=2)
        
        logging.info(f"Successfully processed {len(human_battles)} human vote battles")
    else:
        logging.warning("No human battles found")
    
    # Process judge model data
    logging.info("Processing judge model data...")
    judge_battles_dict = load_judge_results_from_local(results_dir)
    
    for system_name, battles in judge_battles_dict.items():
        if battles:
            logging.info(f"Computing ELO for {system_name}...")
            elo_results = bootstrap_confidence_intervals(battles)
            all_ratings[system_name] = elo_results
            all_battles[system_name] = battles
            
            # Save ELO results
            safe_filename = system_name.replace('/', '_').replace(' ', '_').replace('(', '').replace(')', '') + '_elo.json'
            with open(os.path.join(output_dir, safe_filename), 'w') as f:
                json.dump(elo_results, f, indent=2)
    
    # Create overall correlation analysis
    if len(all_ratings) >= 2:
        logging.info("Creating overall correlation analysis...")
        create_correlation_plot(all_ratings, output_dir, "Overall ")
        
        # Create pairwise scatter plots
        system_names = list(all_ratings.keys())
        for i in range(len(system_names)):
            for j in range(i+1, len(system_names)):
                name1, name2 = system_names[i], system_names[j]
                create_scatter_plot(all_ratings[name1], all_ratings[name2], 
                                  name1, name2, output_dir)
    
    # Generate comprehensive report
    generate_elo_report(all_ratings, all_battles, output_dir)
    
    logging.info(f"Analysis completed. Results saved to {output_dir}")

def generate_elo_report(all_ratings, all_battles, output_dir):
    """Generate comprehensive ELO analysis report"""
    report = {
        'analysis_timestamp': datetime.now().isoformat(),
        'analysis_type': 'elo_from_hf_data',
        'data_source': 'Hugging Face cached data',
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
    
    # Calculate correlation ranking of different judge models under the same mode
    with_execution_systems = [name for name in all_ratings.keys() if 'with_execution' in name]
    without_execution_systems = [name for name in all_ratings.keys() if 'without_execution' in name]
    
    # with_execution mode correlation ranking
    with_execution_correlations = []
    for i in range(len(with_execution_systems)):
        for j in range(i+1, len(with_execution_systems)):
            name1, name2 = with_execution_systems[i], with_execution_systems[j]
            corr, p_val, common_models = calculate_correlation(all_ratings[name1], all_ratings[name2])
            
            if corr is not None:
                model1 = name1.replace(' (with_execution)', '')
                model2 = name2.replace(' (with_execution)', '')
                pair_name = f"{model1} vs {model2}"
                
                with_execution_correlations.append({
                    'pair_name': pair_name,
                    'model1': model1,
                    'model2': model2,
                    'correlation': corr,
                    'p_value': p_val,
                    'common_models': len(common_models) if common_models else 0,
                    'significance': 'significant' if p_val and p_val < 0.05 else 'not significant'
                })
    
    with_execution_correlations.sort(key=lambda x: x['correlation'], reverse=True)
    report['with_execution_correlations'] = with_execution_correlations
    
    # without_execution mode correlation ranking
    without_execution_correlations = []
    for i in range(len(without_execution_systems)):
        for j in range(i+1, len(without_execution_systems)):
            name1, name2 = without_execution_systems[i], without_execution_systems[j]
            corr, p_val, common_models = calculate_correlation(all_ratings[name1], all_ratings[name2])
            
            if corr is not None:
                model1 = name1.replace(' (without_execution)', '')
                model2 = name2.replace(' (without_execution)', '')
                pair_name = f"{model1} vs {model2}"
                
                without_execution_correlations.append({
                    'pair_name': pair_name,
                    'model1': model1,
                    'model2': model2,
                    'correlation': corr,
                    'p_value': p_val,
                    'common_models': len(common_models) if common_models else 0,
                    'significance': 'significant' if p_val and p_val < 0.05 else 'not significant'
                })
    
    without_execution_correlations.sort(key=lambda x: x['correlation'], reverse=True)
    report['without_execution_correlations'] = without_execution_correlations
    
    # All correlation rankings (including human votes)
    all_correlations = []
    system_names = list(all_ratings.keys())
    for i in range(len(system_names)):
        for j in range(i+1, len(system_names)):
            name1, name2 = system_names[i], system_names[j]
            corr, p_val, common_models = calculate_correlation(all_ratings[name1], all_ratings[name2])
            
            if corr is not None:
                all_correlations.append({
                    'pair_name': f"{name1} vs {name2}",
                    'system1': name1,
                    'system2': name2,
                    'correlation': corr,
                    'p_value': p_val,
                    'common_models': len(common_models) if common_models else 0,
                    'significance': 'significant' if p_val and p_val < 0.05 else 'not significant'
                })
    
    all_correlations.sort(key=lambda x: x['correlation'], reverse=True)
    report['all_correlations_ranking'] = all_correlations
    
    # Save report
    with open(os.path.join(output_dir, 'elo_analysis_report.json'), 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # Generate text report
    generate_elo_text_report(report, output_dir)

def generate_elo_text_report(report, output_dir):
    """Generate readable text report"""
    with open(os.path.join(output_dir, 'elo_analysis_report.txt'), 'w', encoding='utf-8') as f:
        f.write("ELO Analysis Report (From Hugging Face Data)\n")
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
        
        # 模型胜负平统计
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

def main():
    """Main function"""
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Compute ELO ratings and correlations')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Directory containing judge results (default: results)')
    parser.add_argument('--dataset', type=str, default='bigcode/bigcodereward',
                       help='HuggingFace dataset for human votes (default: bigcode/bigcodereward)')
    parser.add_argument('--split', type=str, default='train',
                       help='Dataset split to use (default: train)')
    parser.add_argument('--output-dir', type=str, default='analysis_results_elo',
                       help='Output directory for analysis results (default: analysis_results_elo)')
    args = parser.parse_args()
    
    # Configure paths
    results_dir = args.results_dir
    dataset_name = args.dataset
    split = args.split
    output_dir = args.output_dir
    
    print("🚀 Starting ELO analysis...")
    print(f"📁 Results directory: {results_dir}")
    print(f"👥 Human votes dataset: {dataset_name} ({split})")
    print(f"📊 Output directory: {output_dir}")
    print("=" * 80)
    
    # Check if results directory exists
    if not os.path.exists(results_dir):
        print(f"❌ Results directory not found: {results_dir}")
        print("Please run eval_hf_data.py first to generate judge results")
        return
    
    # Run analysis
    analyze_elo_from_local_and_hf(results_dir, dataset_name, split, output_dir)
    print(f"✅ Analysis completed. Results saved to {output_dir}/")
    
    print(f"\n📊 Key output files:")
    print(f"   - elo_analysis_report.json: Complete JSON report")
    print(f"   - elo_analysis_report.txt: Complete text report")
    print(f"   - *_elo.json: Individual ELO results for each system")
    print(f"   - overall_correlation_matrix.png: Correlation heatmap")
    print(f"   - scatter_*.png: Pairwise scatter plots")

if __name__ == '__main__':
    main()
