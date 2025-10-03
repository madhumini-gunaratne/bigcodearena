# AutoCodeArena

AutoCodeArena is a comprehensive evaluation framework for assessing code generation models on real-world coding tasks. The system evaluates models across multiple environments (HTML, React, Python, etc.) and uses LLM-as-a-judge for quality assessment.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Environment Setup](#environment-setup)
- [Configuration](#configuration)
- [Running Evaluation](#running-evaluation)

## Overview

The AutoCodeArena evaluation pipeline consists of three main steps:

1. **Generation** (`gen_answer.py`): Generate code solutions from models
2. **Execution** (`gen_execution.py`): Execute generated code in sandboxed environments
3. **Judgment** (`gen_judgment.py`): Evaluate quality using LLM-as-a-judge

All benchmark questions are loaded from HuggingFace Hub (`bigcode/autocodearena-v0`), while results are saved locally.

### Dataset Format

The HuggingFace dataset contains questions with three fields:
- `uid`: Unique identifier for each question
- `instruction`: The coding task/question text
- `category`: Category of the task (e.g., "web_design", "game_development", "problem_solving")

## Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Docker (Required for Code Execution)

Docker is required to run code in isolated sandboxed environments.

```bash
# Install Docker (Ubuntu/Debian)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add your user to docker group (to run without sudo)
sudo usermod -aG docker $USER
newgrp docker

# Verify installation
docker --version
```

### 3. Build Docker Images

Build all required sandbox environments:

```bash
python build_docker_images.py
```

This will build Docker images for:
- HTML/JavaScript environments
- React applications
- Python runners
- PyGame applications
- Streamlit/Gradio apps
- Vue applications
- And more...

**Note**: This process may take 15-30 minutes depending on your internet connection.

## Quick Start

Once installation is complete, run the complete evaluation pipeline:

### Evaluate All Models (Full Pipeline)

```bash
# 1. Set API keys
export OPENAI_API_KEY=your-openai-api-key
export OPENROUTER_API_KEY=your-openrouter-api-key
export DEEPSEEK_API_KEY=your-deepseek-api-key

# 2. Generate answers from all models in config (loads questions from HuggingFace)
python gen_answer.py

# 3. Execute generated code in sandboxed environments
python gen_execution.py --data_path data/autocodearena/model_answer

# 4. Generate judgments using LLM-as-a-judge
python gen_judgment.py

# 5. Display results and leaderboard
python show_result.py --benchmark autocodearena-v0
```

### Evaluate Single Model (Recommended for Testing)

```bash
# 1. Set API keys
export OPENAI_API_KEY=your-openai-api-key

# 2. Generate answers for one model
# First, add your model to config/gen_answer_config.yaml, then:
python gen_answer.py  # Will only generate for models in config

# Or test with a specific model (if already in api_config.yaml):
# Edit config/gen_answer_config.yaml to include only the model you want

# 3. Execute generated code for specific model
python gen_execution.py \
    --data_path data/autocodearena \
    --model_name gpt-4o-2024-11-20

# 4. Generate judgments (will compare against baseline)
python gen_judgment.py

# 5. Display results
python show_result.py --benchmark autocodearena-v0
```

**Note**: Configure which models to evaluate in `config/gen_answer_config.yaml` before running `gen_answer.py`.

## Environment Setup

### Required API Keys

AutoCodeArena uses environment variables for API keys. Set the required API keys like below:

```bash
export OPENAI_API_KEY=your-openai-api-key
export OPENROUTER_API_KEY=your-openrouter-api-key
export DEEPSEEK_API_KEY=your-deepseek-api-key
```

## Configuration

### 1. Configure Models for Evaluation

Edit `config/gen_answer_config.yaml`:

```yaml
bench_name: autocodearena

# List of models to generate answers
model_list:
  - gpt-4o-mini-2024-07-18
  - gpt-4o-2024-11-20
  - litellm_claude35_sonnet
  - litellm_claude37_sonnet
  - gemini-2.5-pro
  - deepseek-chat-v3.1
  # Add your custom models here
```

### 2. Configure API Endpoints

Edit `config/api_config.yaml` to add or modify model configurations:

```yaml
your-custom-model:
    model: provider/model-name
    endpoints:
        - api_base: https://api.provider.com/v1
          api_key: ${YOUR_API_KEY}  # Use environment variable
    api_type: openai  # or openai_thinking, litellm, etc.
    parallel: 32
    max_tokens: 8192
    temperature: 0.0
```

**Available API types:**
- `openai`: Standard OpenAI-compatible APIs
- `openai_thinking`: For reasoning models (o1, o3, o4, etc.)
- `litellm`: For any litellm-supported models

### 3. Configure Judge Model

Edit `config/autocodearena.yaml`:

```yaml
# Judge model configuration
judge_model: claude37_sonnet  # Change to your preferred judge
temperature: 0.0
max_tokens: 8192
parallel: 32

bench_name: autocodearena

# Baseline model for comparison
baseline_model: gpt-4.1-2025-04-14

# Models to evaluate
model_list:
  - glm-4.5
  - kimi-k2
  - claude-4-sonnet
  # Add models you want to judge
```

## Running Evaluation

### Complete Pipeline

Run the full evaluation pipeline:

```bash
# Step 1: Generate answers from models
python gen_answer.py

# Step 2: Execute generated code
python gen_execution.py --data_path data/autocodearena/model_answer

# Step 3: Generate judgments
python gen_judgment.py
```

### Step-by-Step Guide

#### Step 1: Generate Answers

```bash
# Generate answers for all models in config
python gen_answer.py

# Or specify custom config
python gen_answer.py \
    --config-file config/gen_answer_config.yaml \
    --endpoint-file config/api_config.yaml

# Regenerate empty or failed answers
python gen_answer.py --regenerate-empty --regenerate-no-code

# Use custom dataset
python gen_answer.py --dataset your-org/your-dataset
```

**Output**: `data/autocodearena/model_answer/{model}/generation.jsonl`

#### Step 2: Execute Code

```bash
# Execute all models
python gen_execution.py --data_path data/autocodearena/model_answer

# Execute specific model
python gen_execution.py \
    --data_path data/autocodearena \
    --model_name gpt-4o-2024-11-20

# Execute with custom settings
python gen_execution.py \
    --data_path data/autocodearena \
    --max_workers 20 \
    --timeout 180

# Filter by environment
python gen_execution.py \
    --data_path data/autocodearena \
    --environment "React"

# Overwrite existing results
python gen_execution.py \
    --data_path data/autocodearena \
    --overwrite

# Reclassify environments and rerun
python gen_execution.py \
    --data_path data/autocodearena \
    --reclassify

# Clean up Docker resources before execution
python gen_execution.py --cleanup
```

**Output**: 
- `data/autocodearena/model_answer/{model}/execution_results.jsonl`
- `data/autocodearena/model_answer/{model}/screenshots/`
- `data/autocodearena/model_answer/{model}/visual_outputs/`

#### Step 3: Generate Judgments

```bash
# Generate judgments for all models
python gen_judgment.py

# With custom config
python gen_judgment.py \
    --setting-file config/autocodearena.yaml \
    --endpoint-file config/api_config.yaml

# Without execution results (code-only evaluation)
python gen_judgment.py --no-execution-results

# Demo mode (see one example)
python gen_judgment.py --demo

# Use custom dataset
python gen_judgment.py --dataset your-org/your-dataset
```

**Output**: `data/autocodearena/model_judgment/{judge_model}/{model}.jsonl`

#### Step 4: Display Results

```bash
# Show overall leaderboard
python show_result.py --benchmark autocodearena-v0

# Show results by environment
python show_result.py \
    --benchmark autocodearena-v0 \
    --by env

# Filter by specific categories
python show_result.py \
    --benchmark autocodearena-v0 \
    --topic "web_design" "game_development"

# Export results to JSON
python show_result.py \
    --benchmark autocodearena-v0 \
    --json results.json

# Generate LaTeX table
python show_result.py \
    --benchmark autocodearena-v0 \
    --latex score
```

**Output**: Formatted leaderboard with scores and confidence intervals