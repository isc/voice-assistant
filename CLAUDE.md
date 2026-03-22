# Voice Assistant Server

Custom voice assistant replacing Home Assistant's voice pipeline. Runs on ESPHome hardware (ESP32-S3 with wake word), fully local STT/TTS, with cloud or local LLM.

## Architecture

```
ESP (wake word) → audio stream → voice_server.py
                                   ├─ stt.py (Parakeet MLX) → transcript
                                   ├─ llm.py (OpenAI API / llama.cpp) → tool calls or text
                                   │    ├─ ha_client.py → Home Assistant REST API
                                   │    └─ weather.py → Open-Meteo API (free, no key)
                                   ├─ tts.py (Kokoro-82M) → WAV file
                                   └─ web_ui.py → debug UI on :8888
```

## Files

| File | Purpose |
|---|---|
| `voice_server.py` | Main orchestrator: ESP connection, VAD, pipeline, signal handling, function execution |
| `ha_client.py` | Home Assistant client: entity discovery, fuzzy name resolution, room groups, service calls |
| `llm.py` | LLM integration: chat completion, tool definitions, text-to-tool-call fallback parser |
| `timer.py` | Timer manager: in-memory timers with asyncio scheduling, ESP native timer events |
| `tts.py` | Kokoro-82M TTS with French G2P (misaki/espeak), 24kHz→16kHz resampling |
| `stt.py` | Parakeet TDT 0.6B v3 via MLX (Apple Silicon optimized) |
| `weather.py` | Open-Meteo weather API: geocoding, current conditions, 5-day forecast (French) |
| `web_ui.py` | Web UI (HTML + routes), exchange log with persistence (`exchanges.json`) |
| `run.sh` | Launcher: reads secrets from macOS Keychain, selects cloud/local LLM mode |
| `ctl.sh` | Service control: start/stop/restart/reload/status/logs via launchd |
| `com.voice-assistant.server.plist` | launchd service definition |

## Pipeline flow (voice)

1. ESP detects wake word → sends audio stream
2. Silero VAD detects end of speech (31 frames / ~1s silence)
3. STT: Parakeet transcribes audio to French text
4. LLM: sends transcript + system prompt (with datetime, entity list) + conversation history + tool definitions
5. If tool_calls → execute each, loop back to LLM with results + tools (up to 3 rounds) until text response
6. If text response → check fallback parser for French action verbs (local LLM workaround)
7. Final LLM round formulates natural spoken response from tool results
8. TTS: Kokoro generates WAV, played via announcement API with `start_conversation=True`
9. ESP automatically starts listening for follow-up (no wake word needed)
10. If no speech within 5s → conversation ends. If `end_conversation` tool was called → no follow-up.

## Key design patterns

- **Room groups**: `ROOM_GROUPS` in `ha_client.py` maps group names ("enfants", "partout") to lists of HA areas. When a group is detected, all entities of the matching domain are returned.
- **Entity resolution**: fuzzy matching on normalized friendly_name (accents stripped, stopwords removed), scoped by room/area.
- **Multi-round tool loop**: LLM can return tool_calls across multiple rounds (max 3). After executing tools, results are sent back with tools still available so the LLM can make additional calls. A system reminder lists already-called functions to encourage completeness.
- **Conversation history**: last 5 exchanges (10 messages), expires after 2 min inactivity.
- **DateTime injection**: current date/time injected into system prompt so the LLM can answer time/date questions.
- **Continuous conversation**: after TTS playback, ESP starts listening for follow-up without wake word. 500ms audio skip avoids TTS echo. `end_conversation` tool lets the LLM end cleanly on "merci", "bonne nuit", etc.
- **Text fallback parser**: catches when local LLM outputs tool calls as plain text instead of structured JSON (4 patterns: JSON args, quoted string, Python kwargs, French verbs).
- **Secrets**: stored in macOS Keychain (`voice-assistant-ha-token`, `voice-assistant-openai-key`), read by `run.sh` and exported as env vars.

## Configuration (env vars)

| Variable | Default | Description |
|---|---|---|
| `ESP_HOST` | `(required)` | ESPHome device IP |
| `ESP_PORT` | `6053` | ESPHome native API port |
| `ESP_PASSWORD` | (empty) | Legacy password auth |
| `ESP_NOISE_PSK` | (empty) | Noise encryption key (preferred) |
| `LLM_URL` | `http://localhost:8080/v1/chat/completions` | LLM endpoint |
| `LLM_API_KEY` | (empty) | If set, enables cloud mode |
| `LLM_MODEL` | (empty) | Model name for cloud API (e.g. `gpt-5.4-nano`) |
| `HA_URL` | `(required)` | Home Assistant URL |
| `HA_TOKEN` | (empty) | HA long-lived access token |
| `HTTP_PORT` | `8888` | HTTP server port (TTS files + web UI) |

## Service management

```bash
./ctl.sh start      # Start via launchd, waits for ready
./ctl.sh stop       # Graceful SIGTERM
./ctl.sh restart    # Full restart (after code changes), waits for ready
./ctl.sh reload     # SIGHUP → hot-reload HA entities, waits for completion
./ctl.sh status     # Running/not running + PID
./ctl.sh logs       # tail -f
```

## Conventions

- All code and comments in English. French only in user-facing LLM prompts and TTS output.
- No long `sleep` delays — use polling or signal-based waiting.
- Architectural decisions documented in `DECISIONS.md`.
- Never `git commit --amend` — always new commits, squash on merge.
