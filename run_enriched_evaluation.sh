#!/usr/bin/env bash
set -euo pipefail

NUM_QUESTIONS="${1:-100}"
CONFIG_KEY="${2:-qwen3-4b-inst-2507-vllm}"  # Config key from vllm_config.yaml
VLLM_CONFIG_FILE="autocodearena/config/vllm_config.yaml"
VENV_DIR="$(pwd)/vllm_env"
PYTHON="$VENV_DIR/bin/python"
ENRICHED_TASKS_FILE="autocodearena/data/new data v1.1/tasks_multisource.json"

echo "=========================================="
echo "BigCodeArena Evaluation Pipeline - Enriched"
echo "=========================================="
echo ""
echo "⚙️  Configuration:"
echo "  Questions:    $NUM_QUESTIONS"
echo "  Config Key:   $CONFIG_KEY"
echo "  Config File:  $VLLM_CONFIG_FILE"
echo ""

# Check if vLLM server is running
echo "🔍 Checking vLLM server..."
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ vLLM server is running!"
else
    echo "❌ ERROR: vLLM server not running on localhost:8000"
    echo "   Start it with: bash setup_vllm.sh"
    exit 1
fi

echo ""
echo "=========================================="
echo "Step 1: Generate Answers"
echo "=========================================="
echo ""

cd autocodearena

"$PYTHON" << PYTHON_SCRIPT
import sys
sys.path.insert(0, '..')

import json
import os
import requests
from pathlib import Path
from tqdm import tqdm

# Import from existing utils
from utils.completion import make_config
from sandbox.code_analyzer import extract_code_from_markdown

# Load vLLM configuration from YAML
CONFIG_KEY = "$CONFIG_KEY"
NUM_QUESTIONS = $NUM_QUESTIONS
VLLM_CONFIG_FILE = "$VLLM_CONFIG_FILE"
ENRICHED_TASKS_FILE = "$ENRICHED_TASKS_FILE"

# Adjust path since we're inside autocodearena directory
if not os.path.isabs(VLLM_CONFIG_FILE):
    if not os.path.exists(VLLM_CONFIG_FILE):
        VLLM_CONFIG_FILE = os.path.join("..", VLLM_CONFIG_FILE)

if not os.path.isabs(ENRICHED_TASKS_FILE):
    if not os.path.exists(ENRICHED_TASKS_FILE):
        ENRICHED_TASKS_FILE = os.path.join("..", ENRICHED_TASKS_FILE)

print(f"📋 Loading vLLM configuration from: {VLLM_CONFIG_FILE}")

VLLM_CONFIG = make_config(VLLM_CONFIG_FILE)

# Get model configuration from YAML using config key
if CONFIG_KEY not in VLLM_CONFIG:
    print(f"❌ ERROR: Config key '{CONFIG_KEY}' not found in vLLM config")
    print(f"   Available keys: {list(VLLM_CONFIG.keys())}")
    exit(1)

model_config = VLLM_CONFIG[CONFIG_KEY]
ENDPOINT = model_config.get("endpoint", "http://localhost:8000")
MAX_TOKENS = model_config.get("max_tokens", 8000)
TEMPERATURE = model_config.get("temperature", 0.7)
VLLM_MODEL = model_config.get("model", "microsoft/phi-2")

print(f"✅ Configuration loaded from YAML:")
print(f"   Config Key: {CONFIG_KEY}")
print(f"   vLLM Model: {VLLM_MODEL}")
print(f"   Endpoint: {ENDPOINT}")
print(f"   Max Tokens: {MAX_TOKENS}")
print(f"   Temperature: {TEMPERATURE}")
print()

print(f"📊 Processing first {NUM_QUESTIONS} enriched questions")
print()

# Load enriched questions from file
print(f"📥 Loading enriched tasks from: {ENRICHED_TASKS_FILE}")
with open(ENRICHED_TASKS_FILE, 'r', encoding='utf-8') as f:
    questions = json.load(f)
print(f"✓ Loaded {len(questions)} enriched questions total")
print()

# Limit to N questions
questions = questions[:NUM_QUESTIONS]
print(f"✓ Using first {len(questions)} questions for evaluation")
print(f"✓ Each task includes GitHub context for better answers")
print()

# Create output directory in new data v1.1 folder for each model
# Use "multisource" prefix to separate from original enriched results
output_dir = Path("../autocodearena/data/new data v1.1") / f"multisource-{CONFIG_KEY}"
output_dir.mkdir(parents=True, exist_ok=True)
answer_file = output_dir / "generation.jsonl"

print(f"📁 Output directory: {output_dir}")
print()

# Generate answers
print("🚀 Starting generation...")
print("-" * 80)

answers_generated = 0
errors = 0

with open(answer_file, "w") as fout:
    for i, question in enumerate(tqdm(questions, desc="Generating answers")):
        try:
            uid = question["uid"]
            # Use enriched instruction (includes GitHub context)
            instruction = question["instruction"]
            category = question.get("category", "unknown")
            
            # Truncate instruction if it exceeds token limit (~1500 chars = ~375 tokens buffer)
            # Model limit is 2000 tokens, we reserve 500 for output
            max_instruction_length = 1500
            if len(instruction) > max_instruction_length:
                # Keep original task + truncate resources section
                parts = instruction.split("=" * 80)
                if len(parts) >= 2:
                    # Keep original task + first part of resources
                    instruction = parts[0] + "\n\n" + parts[1][:800]
                else:
                    instruction = instruction[:max_instruction_length]
            
            # Call vLLM API with configuration from YAML
            payload = {
                "model": VLLM_MODEL,
                "prompt": instruction,
                "max_tokens": MAX_TOKENS,
                "temperature": TEMPERATURE,
            }
            
            try:
                response = requests.post(
                    f"{ENDPOINT}/v1/completions",
                    json=payload,
                    timeout=600  # 10 minutes - needed for token generation
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if "choices" in result and len(result["choices"]) > 0:
                        answer_text = result["choices"][0].get("text", "")
                    else:
                        answer_text = ""
                else:
                    answer_text = f"ERROR: {response.status_code}"
                    print(f"  ⚠️  Question {i} returned {response.status_code}")
                    print(f"      Response: {response.text[:200]}")
                    errors += 1
            except Exception as e:
                answer_text = f"ERROR: Exception - {str(e)[:100]}"
                errors += 1
                print(f"  ⚠️  Question {i} raised exception: {str(e)[:100]}")
            
            # Format answer in the expected structure (same as original)
            record = {
                "uid": uid,
                "category": category,
                "instruction": instruction,
                "messages": [
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": {"answer": answer_text}}
                ]
            }
            
            fout.write(json.dumps(record) + "\n")
            answers_generated += 1
            
        except Exception as e:
            errors += 1
            print(f"  ⚠️  Error on question {i}: {str(e)[:100]}")

print()
print("=" * 80)
print(f"✅ Generation complete!")
print(f"   Generated: {answers_generated} answers")
print(f"   Errors: {errors}")
print(f"   Saved to: {answer_file}")
print()

PYTHON_SCRIPT

cd ..

echo ""
echo "✅ Generated enriched answers saved to:"
echo "   autocodearena/data/new data v1.1/$CONFIG_KEY/generation.jsonl"
echo ""
