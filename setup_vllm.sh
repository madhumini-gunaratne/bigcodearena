#!/usr/bin/env bash
set -euo pipefail

# Setup vLLM locally in the project folder using isolated venv
MODEL="${1:-microsoft/phi-2}"  # Lightweight: 2.7GB, fast
VLLM_PORT="${2:-8000}"
VENV_DIR="./vllm_env"

echo "=========================================="
echo "vLLM Local Setup (Isolated Environment)"
echo "=========================================="
echo ""
echo "Model:      $MODEL"
echo "Port:       $VLLM_PORT"
echo "venv Dir:   $VENV_DIR"
echo ""

# Create venv if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv "$VENV_DIR"
    "$VENV_DIR/bin/pip" install -q vllm
    echo "✅ venv created with vLLM installed"
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

# Run vLLM server using the isolated venv
"$VENV_DIR/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --host 0.0.0.0 \
  --port "$VLLM_PORT" \
  --gpu-memory-utilization 0.4