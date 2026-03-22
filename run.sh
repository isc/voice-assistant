#!/bin/bash
# Voice assistant server launcher

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$SCRIPT_DIR/venv/bin/python3"

# Install deps if needed
if ! $PYTHON -c "import aiohttp" 2>/dev/null; then
    echo "Installing dependencies..."
    $PYTHON -m pip install -r "$SCRIPT_DIR/requirements.txt"
fi

# Resolve secrets from keychain if not in env
if [ -z "$HA_TOKEN" ]; then
    export HA_TOKEN=$(security find-generic-password -s "voice-assistant-ha-token" -w 2>/dev/null || true)
fi
if [ -z "$LLM_API_KEY" ]; then
    LLM_API_KEY=$(security find-generic-password -s "voice-assistant-openai-key" -w 2>/dev/null || true)
fi

# Cloud mode: if LLM_API_KEY is available, skip local llama-server
if [ -n "$LLM_API_KEY" ]; then
    export LLM_API_KEY
    export LLM_URL="${LLM_URL:-https://api.openai.com/v1/chat/completions}"
    export LLM_MODEL="${LLM_MODEL:-gpt-5.4-nano}"
    echo "Using cloud LLM: ${LLM_MODEL}"
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

# exec replaces bash with Python so signals (SIGHUP, SIGTERM) reach the server directly
exec $PYTHON voice_server.py
