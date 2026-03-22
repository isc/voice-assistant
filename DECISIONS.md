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

### AD-015: Room groups for multi-room commands
- **Context**: User wants to say "ferme les volets des enfants" or "éteins toutes les lumières" and have it affect multiple rooms.
- **Decision**: Define ROOM_GROUPS in ha_client.py (e.g., "enfants" → [Chambre Zoé, Chambre Charlie], "tout/partout" → all rooms). When a room group is detected, return ALL entities of the target domain across all rooms in the group. LLM passes the group name as the room parameter.
- **Trade-off**: Server-side compensation for the small LLM's inability to reliably make multiple tool calls. Groups are hardcoded (not dynamic from HA).

### AD-016: Stay on Qwen 3 4B Q4_K_M (benchmarked vs 8B)
- **Context**: Benchmarked Qwen 3 8B Q4_K_M on M1 16GB. Warm latency: 3.1s (vs 1.7s for 4B). RAM: ~5GB (vs ~950MB). Tool calling works with both thanks to Python compensation.
- **Decision**: Stay on 4B for lower latency. Revisit if tool calling reliability becomes an issue or if a better model becomes available.
- **Trade-off**: Less capable for complex reasoning, but acceptable for home automation commands.

### AD-017: Conversation history for multi-turn context
- **Context**: Each LLM call was stateless — no memory between turns. Impossible to say "allume la lumière du salon" then "éteins-la".
- **Decision**: Keep last 5 exchanges in memory, expire after 2 min of inactivity. Add French action verb fallback parser for when local LLM responds with text instead of tool calls.
- **Trade-off**: More tokens per request (~200 extra for history). Fallback parser is a workaround for small local models — unnecessary with cloud APIs.

### AD-018: Cloud LLM API support (GPT-5.4 Nano)
- **Context**: Local Qwen 3 4B struggles with multi-turn tool calling (responds with text instead of structured tool_calls). Tested GPT-5.4 Nano via OpenAI API as interim solution.
- **Decision**: Support both local (llama.cpp) and cloud (OpenAI-compatible API) via env vars LLM_URL, LLM_API_KEY, LLM_MODEL. GPT-5.4 Nano: reliable tool calls, pronoun resolution, ~1.3s latency, ~$0.39/month.
- **Trade-off**: Introduces cloud dependency (breaks 100% local goal), but temporary until dedicated hardware (Zotac + 14B local model). Local mode remains the default.

## 2026-03-22: Service management, refactoring, weather

### AD-019: macOS Keychain for secrets
- **Context**: HA token stored in env vars was lost on context compaction. User had to regenerate tokens.
- **Decision**: Store secrets in macOS Keychain (`voice-assistant-ha-token`, `voice-assistant-openai-key`). `run.sh` reads them at launch and exports as env vars. Voice server only reads env vars.
- **Trade-off**: macOS-specific. Secrets survive context compaction and system restarts.

### AD-020: launchd service management with signal-based control
- **Context**: No standardized way to start/stop/restart/reload the server.
- **Decision**: Use launchd plist (`com.voice-assistant.server.plist`) for daemon management. `ctl.sh` wraps launchctl commands. SIGHUP triggers hot-reload of HA entities. SIGTERM triggers graceful shutdown. `run.sh` uses `exec` so signals reach Python directly.
- **Trade-off**: macOS-specific (launchd vs systemd). `ctl.sh` polls log markers instead of sleeping for ready detection.

### AD-021: Refactoring into modules
- **Context**: `voice_server.py` grew to 1315 lines — hard to navigate and maintain.
- **Decision**: Extract into 6 modules: `stt.py`, `tts.py`, `llm.py`, `ha_client.py`, `weather.py`, `web_ui.py`. `voice_server.py` remains the orchestrator (~660 lines).
- **Trade-off**: More files, but each module has a clear responsibility and can be understood in isolation.

### AD-022: Web UI for debugging and text input
- **Context**: No way to visualize exchanges or test without the ESP hardware.
- **Decision**: Web UI at `:8888` with exchange log, debug panel (timings, tool calls), and text input. OS-adaptive light/dark theme via `prefers-color-scheme`. Exchange log persisted to `exchanges.json` to survive restarts.
- **Trade-off**: Adds ~280 lines to `web_ui.py`. JSON persistence is simple but not concurrent-safe (acceptable for single-user).

### AD-023: Weather tool via Open-Meteo
- **Context**: User wants to ask the assistant about weather. Needs to work without API keys (100% local goal).
- **Decision**: Use Open-Meteo API (free, no key). Default to Paris 15e. Geocoding for other cities. Returns structured JSON with current conditions + 5-day forecast. WMO codes mapped to French descriptions.
- **Trade-off**: Requires internet (not truly local), but no API key or account needed. Open-Meteo has generous rate limits.

### AD-024: Second LLM round-trip for natural tool responses
- **Context**: Raw tool results (e.g., weather JSON, "Plafonnier éteint") went directly to TTS — sounded robotic and unnatural.
- **Decision**: After tool execution, send results back to LLM for natural language formulation. The LLM receives tool results and generates a spoken response.
- **Trade-off**: Adds ~0.5-1s latency per response. Worth it for much better TTS output quality.

### AD-025: Multi-round tool loop
- **Context**: GPT-5.4 Nano doesn't always emit all needed tool calls in one shot (e.g., "éteins la lumière et dis-moi la météo" only called turn_off).
- **Decision**: After tool execution, send results back with tools still available (up to 3 rounds). Add system reminder listing already-called functions to nudge the model to handle remaining parts of the request.
- **Trade-off**: Up to 3 LLM round-trips per user query. In practice, 2 rounds handle 99% of cases. Adds ~1s for multi-part requests.
