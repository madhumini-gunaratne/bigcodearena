#!/usr/bin/env bash
set -euo pipefail

# Setup vLLM locally in the project folder using isolated venv
CONFIG_KEY="${1:-phi-2-vllm}"  # Config key from vllm_config.yaml
VLLM_PORT="${2:-8000}"
VENV_DIR="./vllm_env"
CONFIG_FILE="autocodearena/config/vllm_config.yaml"

echo "=========================================="
echo "vLLM Local Setup (Isolated Environment)"
echo "=========================================="
echo ""
echo "Config Key: $CONFIG_KEY"
echo "Port:       $VLLM_PORT"
echo "venv Dir:   $VENV_DIR"
echo ""

# Create venv if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv "$VENV_DIR"
    "$VENV_DIR/bin/pip" install -q vllm pyyaml
    echo "✅ venv created with vLLM installed"
fi

echo ""
echo "Reading configuration from: $CONFIG_FILE"
echo ""

# Read model, max_tokens from vllm_config.yaml
read -r MODEL MAX_TOKENS < <(CONFIG_KEY="$CONFIG_KEY" "$VENV_DIR/bin/python" << 'PYTHON_SCRIPT'
import yaml
import os
import sys

config_file = "autocodearena/config/vllm_config.yaml"
config_key = os.environ.get('CONFIG_KEY', 'phi-2-vllm')

with open(config_file, 'r') as f:
    config = yaml.safe_load(f)

# Check if config key exists
if config_key not in config:
    print(f"ERROR: Config key '{config_key}' not found in {config_file}", file=sys.stderr)
    print("Available keys:", file=sys.stderr)
    for key in config.keys():
        if isinstance(config[key], dict):
            print(f"  - {key}", file=sys.stderr)
    sys.exit(1)

model_config = config[config_key]
model = model_config.get('model')
max_tokens = model_config.get('max_tokens', 1024)

if not model:
    print(f"ERROR: 'model' field missing in config key '{config_key}'", file=sys.stderr)
    sys.exit(1)

print(f"{model} {max_tokens}")
PYTHON_SCRIPT
)

if [ $? -ne 0 ]; then
    exit 1
fi

echo "Starting vLLM server..."
echo ""
echo "📝 Model:              $MODEL"
echo "📝 Max Tokens (config): $MAX_TOKENS"
echo "📝 Server:            http://localhost:$VLLM_PORT"
echo ""

# Set max_model_len to 1.5x max_tokens to allow context buffer
MAX_LEN=$((MAX_TOKENS * 3 / 2))  # This is 1.5x

echo "✅ Using max-model-len: $MAX_LEN (1.5x buffer for input)"
echo ""
echo "Press Ctrl+C to stop the server"
echo "=========================================="
echo ""

# Run vLLM server using the isolated venv
"$VENV_DIR/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --host 0.0.0.0 \
  --port "$VLLM_PORT" \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization 0.4