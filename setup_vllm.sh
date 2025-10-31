#!/usr/bin/env bash
set -euo pipefail

# Setup vLLM locally in the project folder
MODEL="${1:-microsoft/phi-2}"  # Lightweight: 2.7GB, fast
VLLM_PORT="${2:-8000}"
CACHE_DIR="./models_cache"

echo "=========================================="
echo "vLLM Local Setup"
echo "=========================================="
echo ""
echo "Model:      $MODEL"
echo "Port:       $VLLM_PORT"
echo "Cache Dir:  $CACHE_DIR"
echo ""

# Create cache directory
mkdir -p "$CACHE_DIR"

# Check if vLLM is installed
if ! command -v vllm &> /dev/null; then
    echo "⚠️  vLLM not found. Installing..."
    pip install vllm -q
fi

echo ""
echo "Starting vLLM server..."
echo ""
echo "📝 Server will be available at: http://localhost:$VLLM_PORT"
echo ""
echo "Once started, you can test with:"
echo "  curl http://localhost:$VLLM_PORT/v1/completions -H \"Content-Type: application/json\" -d '{\"model\": \"$MODEL\", \"prompt\": \"def hello\", \"max_tokens\": 20}'"
echo ""
echo "Or use the test script: bash test_with_vllm.sh"
echo ""
echo "Press Ctrl+C to stop the server"
echo "=========================================="
echo ""

# Run vLLM server
export HF_HOME="$CACHE_DIR"
vllm serve "$MODEL" --port "$VLLM_PORT" --gpu-memory-utilization 0.4
