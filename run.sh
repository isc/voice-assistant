#!/bin/bash
# Voice assistant server launcher

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$SCRIPT_DIR/venv/bin/python3"

# Install deps if needed
if ! $PYTHON -c "import aiohttp" 2>/dev/null; then
    echo "Installing dependencies..."
    $PYTHON -m pip install -r "$SCRIPT_DIR/requirements.txt"
fi

# Cloud mode: if LLM_API_KEY is set, skip local llama-server
if [ -n "$LLM_API_KEY" ]; then
    echo "Using cloud LLM: ${LLM_MODEL:-gpt-5.4-nano}"
    export LLM_URL="${LLM_URL:-https://api.openai.com/v1/chat/completions}"
    export LLM_MODEL="${LLM_MODEL:-gpt-5.4-nano}"
else
    # Start llama.cpp with Qwen 3 4B if not already running
    if ! pgrep -f "llama-server.*8080" > /dev/null; then
        echo "Starting llama-server (Qwen 3 4B)..."
        "$HOME/Code/llama.cpp/build/bin/llama-server" \
            -hf ggml-org/Qwen3-4B-GGUF:Q4_K_M \
            --jinja -ngl 99 --port 8080 &
        sleep 5
    fi
fi

$PYTHON voice_server.py
