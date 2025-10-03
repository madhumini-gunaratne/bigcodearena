# BigCodeReward

BigCodeReward is a comprehensive evaluation framework for assessing reward models on code generation tasks. The system evaluates reward models by comparing their judgments against human preferences, using ELO ratings and correlation analysis.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Running Evaluation](#running-evaluation)
- [Analyzing Existing Results](#analyzing-existing-results)
- [Analysis](#analysis)

## Overview

The BigCodeReward evaluation pipeline consists of three main steps:

1. **Judge Evaluation** (`eval_hf_data.py`): Evaluate code comparisons using LLM-as-a-Judge models
2. **Performance Analysis** (`analyze_model_judge_results.py`): Analyze judge model performance against human votes  
3. **ELO Computation** (`analyze_elo.py`): Calculate ELO ratings and correlation analysis

All data is loaded directly from HuggingFace Hub (`bigcode/bigcodereward`), while results are saved locally.

### Alternative: Analyze Pre-computed Results

If you want to analyze existing judge model results without running new evaluations, use:

- **Unified Analysis** (`analyze_bigcode_reward_results.py`): Download and analyze pre-computed results from `bigcode/bigcodereward-experiment-results`

This script combines judge model analysis and ELO computation in one step, using publicly available evaluation results.

### Dataset Format

The HuggingFace dataset contains code comparison battles with execution results:
- `chat_session_id`: Unique identifier for each comparison
- `instruction`: The coding task/question text
- `model_A` / `model_B`: Names of models being compared
- `states`: Contains code, execution outputs, and screenshots for both models
- `human_vote`: Human judgment (vote_model_A, vote_model_B, vote_tie, vote_both_bad)
- `category_name`: Category of the task

## Installation

### 1. Install Python Dependencies

```bash
pip install datasets numpy pandas scikit-learn scipy matplotlib seaborn tqdm pyyaml
pip install litellm openai anthropic pillow
```

This includes all necessary dependencies for:
- Core functionality (datasets, yaml, numpy, pandas, etc.)
- API clients (openai, anthropic, litellm)
- Analysis (scikit-learn, scipy, matplotlib, seaborn)
- Image processing (PIL/Pillow)

### 2. Set Up API Keys

The system uses environment variables for API keys:

```bash
export OPENAI_API_KEY=your-openai-api-key
export ANTHROPIC_API_KEY=your-anthropic-api-key
export AWS_ACCESS_KEY_ID=your-aws-key
export AWS_SECRET_ACCESS_KEY=your-aws-secret
```

## Quick Start

Once installation is complete, run the complete evaluation pipeline:

### Evaluate Judge Models (Full Pipeline)

```bash
# 1. Set API keys (see above)

# 2. Run judge evaluation on the dataset
python eval_hf_data.py --judge-model gpt-4o --workers 8

# 3. Analyze judge model performance
python analyze_model_judge_results.py

# 4. Compute ELO ratings and correlations
python analyze_elo.py
```

### Evaluate Single Judge Model (Recommended for Testing)

```bash
# 1. Set API keys
export OPENAI_API_KEY=your-openai-api-key

# 2. Test with small subset first
python eval_hf_data.py --judge-model gpt-4o --max-records 5 --workers 1

# 3. Run full evaluation with execution context
python eval_hf_data.py --judge-model gpt-4o --workers 8

# 4. Run code-only evaluation (without execution results)
python eval_hf_data.py --judge-model gpt-4o --no-output --workers 8

# 5. Analyze results
python analyze_model_judge_results.py
python analyze_elo.py
```

## Configuration

### 1. Configure Judge Models

Edit `config/judge_model_config.yaml` to add or modify judge models:

```yaml
your-custom-judge:
  model_id: provider/model-name
  api_type: openai  # or litellm, sglang
  context_limit: 128000
  min_request_interval: 1.0
  # Optional: for local models
  api_base: http://localhost:30000/v1
  custom_model_path: /path/to/model
```

**Available API types:**
- `openai`: Standard OpenAI-compatible APIs
- `litellm`: For any litellm-supported models (Claude via Bedrock, etc.)
- `sglang`: For locally deployed SGLang models

### 2. Configure Evaluation Settings

Edit `config/bigcodereward.yaml`:

```yaml
# Dataset configuration
dataset:
  name: bigcode/bigcodereward

# Default judge model
default_judge_model: sonnet35v2

# Evaluation settings
evaluation:
  default_workers: 8
  include_output: true  # Include execution results
  retry_failed: false

# ELO parameters
elo_params:
  K: 4
  SCALE: 400
  BASE: 10
  INIT_RATING: 1000
```

### 3. Supported Judge Models

The system supports various judge models out of the box:

**Claude Models (via AWS Bedrock):**
- `sonnet35v2`: Claude 3.5 Sonnet v2
- `sonnet37v1`: Claude 3.7 Sonnet
- `sonnet4`: Claude 4 Sonnet
- `haiku35v1`: Claude 3.5 Haiku

**OpenAI Models:**
- `gpt-4o`: GPT-4o
- `gpt-4o-mini`: GPT-4o Mini
- `gpt-4.1`: GPT-4.1
- `gpt-4.1-mini`: GPT-4.1 Mini

**Local SGLang Models:**
- `qwen2.5-vl-32b`: Qwen2.5-VL-32B
- `qwen2.5-vl-72b`: Qwen2.5-VL-72B
- `glm-4.5v`: GLM-4.5V
- `gemma-3-27b`: Gemma-3-27B
- `OpenGVLab_InternVL3-38B/78B`: InternVL3
- And more...

## Running Evaluation

### Complete Pipeline

Run the full evaluation pipeline:

```bash
# Step 1: Run judge evaluation
python eval_hf_data.py --judge-model gpt-4o --workers 8

# Step 2: Analyze performance
python analyze_model_judge_results.py

# Step 3: Compute ELO ratings
python analyze_elo.py
```

### Step-by-Step Guide

#### Step 1: Run Judge Evaluation

```bash
# Basic evaluation with execution results
python eval_hf_data.py --judge-model gpt-4o --workers 8

# Code-only evaluation (without execution results)
python eval_hf_data.py --judge-model gpt-4o --no-output --workers 4

# Test with limited records
python eval_hf_data.py --judge-model gpt-4o --max-records 100

# Retry failed evaluations
python eval_hf_data.py --judge-model gpt-4o --retry-failed

# Use custom dataset
python eval_hf_data.py \
    --judge-model gpt-4o \
    --dataset bigcode/bigcodereward \
    --split train

# Specify custom config
python eval_hf_data.py \
    --judge-model gpt-4o \
    --config config/bigcodereward.yaml
```

**Output Structure:**
```
results/
├── {judge_model}/
│   ├── with_execution/
│   │   ├── train-judge_{model}_with_execution.jsonl
│   │   └── jsonl_experiment_progress_{model}_with_execution.json
│   └── without_execution/
│       ├── train-judge_{model}_without_execution.jsonl
│       └── jsonl_experiment_progress_{model}_without_execution.json
└── logs/
```

#### Step 2: Analyze Judge Performance

```bash
# Analyze all judge model results
python analyze_model_judge_results.py
```

This generates:
- Classification metrics (accuracy, F1-score, Cohen's Kappa, etc.)
- Performance rankings across different metrics
- Merged 3-class evaluation (merging tie/both_bad into uncertain)
- Per-category breakdowns

**Output Files:**
- `analysis_results_model_judge/judge_evaluation_report.txt`: Human-readable report
- `analysis_results_model_judge/judge_evaluation_report.json`: Complete JSON report
- `analysis_results_model_judge/merged_3class_summary.json`: Merged 3-class summary
- Individual model results: `{model}_{execution_type}_results.json`

#### Step 3: Compute ELO Ratings

```bash
# Compute ELO ratings and correlations
python analyze_elo.py
```

This generates:
- ELO ratings for all models with confidence intervals
- Correlation analysis between judge models and human votes
- Mode comparisons (with_execution vs without_execution)
- Battle statistics

**Output Files:**
- `analysis_results_elo/elo_analysis_report.txt`: Human-readable ELO report
- `analysis_results_elo/elo_analysis_report.json`: Complete JSON report
- `analysis_results_elo/human_votes_elo.json`: Human vote ELO ratings
- `analysis_results_elo/{model}_elo.json`: Individual model ELO ratings
- `analysis_results_elo/overall_correlation_matrix.png`: Correlation heatmap
- `analysis_results_elo/scatter_*.png`: Pairwise scatter plots

## Analyzing Existing Results

If you want to analyze the existing results from HuggingFace datasets without running new evaluations, use the unified analysis script:

### Quick Analysis

```bash
# Analyze existing BigCode reward datasets
python analyze_bigcode_reward_results.py
```

This script will:
1. **Download datasets** (if not already cached):
   - `bigcode/bigcodereward-experiment-results` - Judge model evaluation results
   - `bigcode/bigcodereward` - Human voting data

2. **Perform comprehensive analysis**:
   - Judge model performance metrics (accuracy, F1-score, Cohen's Kappa, MCC)
   - ELO ratings with confidence intervals
   - Correlation analysis between judges and human votes
   - Battle statistics and win/tie/loss records

3. **Generate reports**:
   - `analysis_results_bigcode_reward/model_judge/` - Judge evaluation reports
   - `analysis_results_bigcode_reward/elo/` - ELO analysis reports
   - Correlation heatmaps and scatter plots

### Output Structure

```
analysis_results_bigcode_reward/
├── model_judge/
│   ├── merged_3class_summary.json        # Summary of merged 3-class evaluation
│   ├── judge_evaluation_report.txt       # Human-readable report
│   ├── judge_evaluation_report.json      # Complete JSON report
│   └── *_results.json                    # Individual model results
├── elo/
│   ├── elo_analysis_report.txt           # ELO analysis report
│   ├── elo_analysis_report.json          # Complete JSON report
│   ├── human_votes_elo.json              # Human vote ELO ratings
│   ├── *_elo.json                        # Individual judge ELO ratings
│   ├── overall_correlation_matrix.png    # Correlation heatmap
│   └── scatter_*.png                     # Pairwise scatter plots
└── logs/                                 # Analysis logs
```

### Data Storage

Downloaded datasets are cached locally in:
- `data/judge_results_dataset/` - Judge model results (JSONL format)
- `data/rewardbench/` - Human voting data (JSONL format)

Once downloaded, subsequent runs will use the cached data without re-downloading.

## Analysis

### Judge Model Evaluation

The system provides comprehensive metrics for evaluating judge models:

**Key Metrics:**
- **Accuracy**: Overall correctness rate
- **Macro F1-Score**: Average F1-score across all classes (handles class imbalance)
- **Cohen's Kappa**: Agreement beyond chance level (>0.6 = substantial agreement)
- **Matthews Correlation Coefficient**: Balanced measure considering all confusion matrix elements

**Evaluation Modes:**

#### With Execution Mode
- Includes code, execution outputs, and screenshots
- Provides comprehensive context for judgment
- Better performance on tasks requiring runtime verification
- Higher computational cost and token usage

#### Without Execution Mode (Code-Only)
- Evaluates based solely on code quality and correctness
- Faster evaluation with lower token usage
- Focus on static code analysis capabilities
- Useful for understanding pure code comprehension abilities

### ELO Rating Analysis

**Features:**
- ELO rating computation with bootstrap confidence intervals
- Correlation analysis (Spearman and Pearson)
- Human vs Model alignment ranking
- Inter-model agreement patterns
- Battle statistics (win/tie/loss records)

**ELO Parameters:**
- K-factor: 4 (learning rate)
- Scale: 400 (rating scale)
- Base: 10 (logistic base)
- Initial Rating: 1000

### Classification Labels

**Original 4-Class:**
- `vote_left`: Model A wins
- `vote_right`: Model B wins
- `vote_tie`: Tie
- `vote_both_bad`: Both solutions are bad

**Merged 3-Class:**
- `vote_left`: Model A wins
- `vote_right`: Model B wins
- `vote_uncertain`: Merged from tie/both_bad

## Advanced Usage

### Custom Judge Models

To add a custom judge model:

1. Edit `config/judge_model_config.yaml`:

```yaml
my-custom-model:
  model_id: provider/my-model
  api_type: openai
  context_limit: 128000
  min_request_interval: 1.0
```

2. Run evaluation:

```bash
python eval_hf_data.py --judge-model my-custom-model --workers 4
```

### Batch Evaluation

For evaluating multiple judge models, create a simple shell script:

```bash
#!/bin/bash
# evaluate_all_judges.sh

MODELS=("gpt-4o" "gpt-4o-mini" "sonnet35v2" "sonnet37v1" "sonnet4")

for model in "${MODELS[@]}"; do
    echo "Evaluating $model with execution..."
    python eval_hf_data.py --judge-model $model --workers 8
    
    echo "Evaluating $model without execution..."
    python eval_hf_data.py --judge-model $model --no-output --workers 8
done

echo "Running analysis..."
python analyze_model_judge_results.py
python analyze_elo.py
```