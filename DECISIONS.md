# Architecture Decisions

This document tracks all architecture and technology decisions made for this project.

## 2026-03-21: Initial stack audit and modernization

### AD-001: Replace webrtcvad with silero-vad-lite
- **Context**: webrtcvad is unmaintained, only 50% true-positive rate
- **Decision**: Use silero-vad-lite (87.7% TPR, zero heavy deps, 16kHz native)
- **Trade-off**: API change (boolean -> float probability), 32ms frames instead of 30ms

### AD-002: Pin aioesphomeapi >= 44.0.0 and aiohttp >= 3.13.3
- **Context**: aioesphomeapi was pinned at >=21.0.0 (24 major versions behind). aiohttp had 8 CVEs (Jan 2026)
- **Decision**: Bump both to current secure versions
- **Trade-off**: Requires Python >= 3.11

### AD-003: Clean requirements.txt
- **Context**: 7 of 12 dependencies were unused (fastapi, uvicorn, websockets, asyncio-mqtt, pydantic, aiofiles, python-multipart). numpy was used but missing.
- **Decision**: Remove unused deps, add numpy

### AD-004: Extract hardcoded config to environment variables
- **Context**: ESP IP, port, password, LLM URL were hardcoded
- **Decision**: Use env vars (ESP_HOST, ESP_PORT, ESP_PASSWORD, ESP_NOISE_PSK, LLAMA_URL, HTTP_PORT) with sensible defaults
- **Trade-off**: None, strictly better

### AD-005: Support noise_psk encryption
- **Context**: ESPHome 2026.1.0 removed password auth. Current ESP runs 2025.6.2 so not urgent, but preparing for the upgrade.
- **Decision**: Support ESP_NOISE_PSK env var, fallback to ESP_PASSWORD

## 2026-03-21: Split STT/LLM pipeline

### AD-006: Replace combined Voxtral STT+LLM with separate Parakeet STT + text LLM
- **Context**: Voxtral Mini 3B via llama.cpp had poor French support (responded in English), no transcript visibility, and audio support in llama.cpp is "highly experimental"
- **Decision**: Split into Parakeet MLX for STT + text-only LLM via llama.cpp
- **Trade-off**: Two models to load instead of one, but each is best-in-class for its task

### AD-007: Use Parakeet TDT 0.6B v3 (via parakeet-mlx) for STT
- **Context**: Compared Whisper Large V3 Turbo vs Parakeet V3. Parakeet wins on French accuracy (6.3% vs 7.4% WER), speed (~0.7s vs 3-5s), RAM (~2GB vs 3-6GB), and zero hallucinations.
- **Decision**: parakeet-mlx with mlx-community/parakeet-tdt-0.6b-v3
- **Trade-off**: Only 25 European languages (vs Whisper's 99+), but French is all we need

### AD-008: Use SmolLM3 3B for text LLM (instead of Qwen 3 8B)
- **Context**: Qwen 3 8B had great tool calling (F1: 0.933) but was slow on M1 (12.8 tok/s, ~6s total). SmolLM3 3B is 2.3x faster (29.5 tok/s, ~1.1s total) with French as a first-class language.
- **Decision**: SmolLM3 3B Q4_K_M via llama.cpp
- **Trade-off**: Less capable for complex reasoning, but for home automation commands this is more than enough. Can always switch back to Qwen 3 8B if needed.

### AD-009: Use /no_think to disable Qwen 3 reasoning mode
- **Context**: Qwen 3 8B defaults to "thinking" mode, putting all output in reasoning_content and returning empty content. This adds latency and wastes tokens for a voice assistant.
- **Decision**: Prefix system prompt with /no_think. Also handle <think> tags in response parsing as fallback.
- **Note**: This decision applies if we switch back to Qwen 3. SmolLM3 does not have this issue.

### AD-010: All code and comments in English
- **Context**: Original code had French comments and docstrings
- **Decision**: English for all code artifacts. User-facing LLM system prompts stay in French (the assistant responds in French to the end user).

### AD-011: Replace Piper TTS with Kokoro-82M (kokoro-onnx) + misaki French G2P
- **Context**: Piper TTS (fr_FR-gilles-low) had poor French accent quality ("québécois et bizarre"). Kokoro-82M has a French voice (ff_siwis) with significantly better pronunciation when paired with explicit French phonemization via misaki/espeak.
- **Decision**: Use kokoro-onnx with misaki EspeakG2P(language='fr-fr') for French phonemization. Resample 24kHz output to 16kHz for ESP compatibility.
- **Trade-off**: Kokoro outputs at 24kHz (requires resampling to 16kHz for ESP). Only 1 French voice available (ff_siwis). Model is larger (~310MB vs ~30MB for Piper). Requires espeak-ng system dependency.

## 2026-03-21: Home Assistant integration

### AD-012: Control Home Assistant via REST API with 6 generic LLM tools
- **Context**: The voice assistant needs to control real home automation devices. User has a local Home Assistant installation.
- **Decision**: Connect to HA REST API (local, no cloud) with a long-lived access token. Expose 6 generic tools to the LLM: turn_on, turn_off, open_cover, close_cover, set_temperature, get_state. Entity resolution uses fuzzy matching on friendly_name (the LLM says "lumière salon", Python matches it to `light.salon`).
- **Trade-off**: 6 tools is the max a small model can handle reliably. No second LLM round-trip after tool execution (function result goes directly to TTS for lower latency). Entity list injected into system prompt for context.

### AD-013: Switch from SmolLM3 3B to Qwen 3 4B for LLM
- **Context**: SmolLM3 3B does not support function/tool calling — it returns text responses instead of tool_calls JSON. Qwen 3 4B has native tool calling support (trained with function calling in its chat template).
- **Decision**: Use Qwen 3 4B Q4_K_M via llama.cpp with --jinja flag. Keep /no_think prefix to disable reasoning mode.
- **Trade-off**: Slightly slower than SmolLM3 (~5s cold, ~1.5s warm vs ~1.3s), but tool calling actually works. Still fits in 16GB RAM alongside Parakeet STT and Kokoro TTS.
- **Note**: Qwen 3 4B requires thinking mode (no `/no_think`) for reliable tool calling. With `/no_think` it outputs tool calls as plain text instead of structured `tool_calls`. Thinking mode adds ~5s latency but is the only way to get reliable function calling.

### AD-014: Entity resolution using HA areas and room parameter
- **Context**: Entities in different rooms can have the same name (e.g., "Plafonnier" in Chambre Charlie and Chambre invités). Flat entity lists caused the LLM to pick the wrong one.
- **Decision**: Fetch HA areas via template API, group entities by room in the system prompt, add `room` parameter to all LLM tools. Entity resolution scopes by room first, falls back to global search.
- **Trade-off**: More template API calls at startup (~1 per area). Slightly larger prompt but better structured for the LLM.
