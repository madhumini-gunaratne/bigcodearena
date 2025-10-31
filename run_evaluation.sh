#!/usr/bin/env bash
set -euo pipefail

NUM_QUESTIONS="${1:-100}"
MODEL="${2:-phi-2-vllm}"
CONFIG_FILE="autocodearena/config/gen_answer_vllm_config.yaml"
ENDPOINT_FILE="autocodearena/config/vllm_config.yaml"

echo "=========================================="
echo "BigCodeArena Evaluation Pipeline"
echo "=========================================="
echo ""
echo "⚙️  Configuration:"
echo "  Questions:    $NUM_QUESTIONS"
echo "  Model:        $MODEL"
echo "  Config:       $CONFIG_FILE"
echo "  Endpoints:    $ENDPOINT_FILE"
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

python3 << PYTHON_SCRIPT
import sys
sys.path.insert(0, '..')

import argparse
import json
import os
import requests
from pathlib import Path
from tqdm import tqdm

# Import from existing utils
from utils.completion import load_questions_from_hf, make_config
from sandbox.code_analyzer import extract_code_from_markdown

# vLLM configuration
MODEL = "$MODEL"
NUM_QUESTIONS = $NUM_QUESTIONS
ENDPOINT = "http://localhost:8000"

print(f"🤖 Using model: {MODEL}")
print(f"📊 Processing first {NUM_QUESTIONS} questions")
print()

# Load questions
print("📥 Loading questions from HuggingFace...")
questions = load_questions_from_hf(repo_id="bigcode/autocodearena-v0")
print(f"✓ Loaded {len(questions)} questions total")

# Limit to N questions
questions = questions[:NUM_QUESTIONS]
print(f"✓ Using first {len(questions)} questions for evaluation")
print()

# Create output directory
output_dir = Path("data/autocodearena_local/model_answer") / MODEL
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
            instruction = question["instruction"]
            
            # Call vLLM API
            payload = {
                "model": "microsoft/phi-2",
                "prompt": instruction,
                "max_tokens": 512,
                "temperature": 0.7,
            }
            
            response = requests.post(
                f"{ENDPOINT}/v1/completions",
                json=payload,
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    answer_text = result["choices"][0].get("text", "")
                else:
                    answer_text = ""
            else:
                answer_text = f"ERROR: {response.status_code}"
                errors += 1
            
            # Format answer in the expected structure
            record = {
                "uid": uid,
                "category": question.get("category", "unknown"),
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
echo "✅ Generated answers saved to:"
echo "   autocodearena/data/autocodearena_local/model_answer/$MODEL/generation.jsonl"
echo ""
