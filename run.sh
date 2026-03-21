#!/bin/bash
# Voice assistant server launcher

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$SCRIPT_DIR/venv/bin/python3"

# Install deps if needed
if ! $PYTHON -c "import aiohttp" 2>/dev/null; then
    echo "Installing dependencies..."
    $PYTHON -m pip install -r "$SCRIPT_DIR/requirements.txt"
fi

# Start llama.cpp with Qwen 3 4B if not already running
if ! pgrep -f "llama-server.*8080" > /dev/null; then
    echo "Starting llama-server (Qwen 3 4B)..."
    "$HOME/Code/llama.cpp/build/bin/llama-server" \
        -hf ggml-org/Qwen3-4B-GGUF:Q4_K_M \
        --jinja -ngl 99 --port 8080 &
    sleep 5
fi

$PYTHON voice_server.py