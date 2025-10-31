#!/usr/bin/env bash
set -euo pipefail

# Test script for local vLLM server
VLLM_HOST="${1:-localhost}"
VLLM_PORT="${2:-8000}"
OUTPUT_DIR="test_results"
NUM_EXAMPLES=5

echo "Configuration:"
echo "  vLLM Host:   $VLLM_HOST:$VLLM_PORT"
echo "  Examples:    $NUM_EXAMPLES"
echo "  Output Dir:  $OUTPUT_DIR"
echo ""
echo "⚠️  Make sure vLLM is running first:"
echo "   bash setup_vllm.sh"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ ERROR: Python3 is not installed."
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# ============================================================================
# Step 1: Create test questions
# ============================================================================
echo "Step 1: Creating test questions..."
echo ""

python3 << 'PYTHON_SCRIPT'
import json
from pathlib import Path

output_dir = Path("test_results")
output_dir.mkdir(exist_ok=True)

test_questions = [
    {
        "id": 0,
        "uid": "q1",
        "instruction": "Write a Python function that calculates the factorial of a number"
    },
    {
        "id": 1,
        "uid": "q2",
        "instruction": "Write a Python function that reverses a string"
    },
    {
        "id": 2,
        "uid": "q3",
        "instruction": "Write a Python function that checks if a number is prime"
    },
    {
        "id": 3,
        "uid": "q4",
        "instruction": "Write a Python function that finds the maximum element in a list"
    },
    {
        "id": 4,
        "uid": "q5",
        "instruction": "Write a Python function that sorts a list of tuples by the second element"
    },
]

with open(output_dir / "test_questions.jsonl", "w") as f:
    for q in test_questions:
        f.write(json.dumps(q) + "\n")

print(f"✓ Created {len(test_questions)} test questions")
print()

PYTHON_SCRIPT

# ============================================================================
# Step 2: Generate answers using local vLLM
# ============================================================================
echo "Step 2: Generating answers using local vLLM..."
echo ""

python3 << PYTHON_SCRIPT
import json
import requests
import time
from pathlib import Path

vllm_host = "$VLLM_HOST"
vllm_port = $VLLM_PORT
output_dir = Path("test_results")

vllm_url = f"http://{vllm_host}:{vllm_port}/v1/completions"

print(f"Connecting to vLLM at: {vllm_url}")
print()

# First, check if server is running
try:
    response = requests.get(f"http://{vllm_host}:{vllm_port}/health", timeout=5)
    print(f"✓ vLLM server is running!")
except Exception as e:
    print(f"❌ ERROR: Cannot connect to vLLM at {vllm_host}:{vllm_port}")
    print(f"   Make sure to run: bash setup_vllm.sh")
    print(f"   Error: {e}")
    exit(1)

print()

answers = []
with open(output_dir / "test_questions.jsonl") as f:
    for i, line in enumerate(f):
        question = json.loads(line)
        instruction = question["instruction"]
        uid = question.get("uid", str(i))
        
        print(f"[{i+1}] Generating answer for: {instruction[:60]}...")
        
        try:
            # Call local vLLM server
            payload = {
                "model": "microsoft/phi-2",  # Must specify the model
                "prompt": instruction,
                "max_tokens": 500,
                "temperature": 0.7,
            }
            
            response = requests.post(
                vllm_url,
                json=payload,
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    answer_text = result["choices"][0].get("text", "")
                    print(f"  ✓ Generated {len(answer_text)} characters")
                else:
                    answer_text = str(result)
                    print(f"  ⚠ Unexpected response format")
            else:
                answer_text = f"ERROR: {response.status_code}"
                print(f"  ❌ Failed: {response.status_code}")
                print(f"     {response.text[:200]}")
                
        except requests.exceptions.ConnectionError:
            answer_text = "ERROR: Connection refused"
            print(f"  ❌ Connection error - is vLLM running?")
        except requests.exceptions.Timeout:
            answer_text = "ERROR: Timeout"
            print(f"  ⚠ Request timed out")
        except Exception as e:
            answer_text = f"ERROR: {str(e)}"
            print(f"  ❌ Error: {str(e)[:100]}")
        
        answers.append({
            "uid": uid,
            "question": instruction,
            "answer": answer_text,
        })
        
        time.sleep(0.5)

print()

# Save answers
with open(output_dir / "generated_answers.jsonl", "w") as f:
    for answer in answers:
        f.write(json.dumps(answer) + "\n")

print(f"✓ Generated answers for {len(answers)} questions")
print(f"✓ Saved to {output_dir}/generated_answers.jsonl")
print()

PYTHON_SCRIPT

# ============================================================================
# Step 3: Display Results
# ============================================================================
echo "Step 3: Results"
echo ""

python3 << 'PYTHON_SCRIPT'
import json
from pathlib import Path

output_dir = Path("test_results")

print("Generated Answers:")
print("-" * 80)

with open(output_dir / "generated_answers.jsonl") as f:
    for i, line in enumerate(f, 1):
        data = json.loads(line)
        print(f"\n[{i}] Question: {data['question'][:70]}...")
        answer_preview = data['answer'][:150] if not data['answer'].startswith("ERROR") else data['answer']
        print(f"    Answer preview: {answer_preview}...")
        print()

print("-" * 80)

PYTHON_SCRIPT

echo ""
echo "✅ Test complete!"
echo ""
echo "Results saved in: $OUTPUT_DIR/"
echo "  - test_questions.jsonl     (input questions)"
echo "  - generated_answers.jsonl   (model outputs)"
echo ""
