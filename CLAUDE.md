# Voice Assistant Server

Custom voice assistant replacing Home Assistant's voice pipeline. Runs on ESPHome hardware (ESP32-S3 with wake word), fully local STT/TTS, with cloud or local LLM.

## Architecture

```
ESP (wake word) → audio stream → voice_server.py
                                   ├─ stt.py (Parakeet MLX) → transcript
                                   ├─ llm.py (OpenAI API / llama.cpp) → tool calls or text
                                   │    ├─ ha_client.py → Home Assistant REST API
                                   │    ├─ weather.py → Open-Meteo API (free, no key)
                                   │    ├─ calendar_client.py → Google Calendar API (OAuth2)
                                   │    └─ spotify.py → go-librespot (playback) + Spotify Web API (search)
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
| `calendar_client.py` | Google Calendar client: OAuth2, event queries, event creation, prompt summary |
| `setup_calendar.py` | One-time OAuth setup script for Google Calendar (requires browser) |
| `spotify.py` | Spotify: go-librespot API, Web API search, and the MP3 stream of go-librespot's PCM FIFO served at `/spotify/stream.mp3` |
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
- **Conversation history**: last 5 exchanges (10 messages), expires after 10 min inactivity (`CONVERSATION_TIMEOUT`).
- **DateTime injection**: current date/time injected into system prompt so the LLM can answer time/date questions.
- **Continuous conversation**: after TTS playback, ESP starts listening for follow-up without wake word. 500ms audio skip avoids TTS echo. `end_conversation` tool lets the LLM end cleanly on "merci", "bonne nuit", etc.
- **Text fallback parser**: catches when local LLM outputs tool calls as plain text instead of structured JSON (4 patterns: JSON args, quoted string, Python kwargs, French verbs).
- **Spotify as a radio station**: go-librespot (separate service) plays into a FIFO; `spotify.py` paces it at real time, pads gaps with silence, encodes MP3 and serves `/spotify/stream.mp3` next to `/tts/`. Speakers (ESP or HA) play that URL; one stream, so one speaker at a time.
- **Secrets**: stored in macOS Keychain (`voice-assistant-ha-token`, `voice-assistant-openai-key`), read by `run.sh` and exported as env vars.

## Configuration (env vars)

| Variable | Default | Description |
|---|---|---|
| `ESP_HOST` | (required) | ESPHome device IP |
| `ESP_PORT` | `6053` | ESPHome native API port |
| `ESP_PASSWORD` | (empty) | Legacy password auth |
| `ESP_NOISE_PSK` | (empty) | Noise encryption key (preferred) |
| `LLM_URL` | `http://localhost:8080/v1/chat/completions` | LLM endpoint |
| `LLM_API_KEY` | (empty) | If set, enables cloud mode |
| `LLM_MODEL` | (empty) | Model name for cloud API (e.g. `gpt-5.4-nano`) |
| `LLM_REASONING_EFFORT` | `none` local, unset cloud | Reasoning budget for thinking models. Keep `none` locally — thinking costs ~10s/round and empties `content` |
| `STT_BACKEND` | `auto` | Parakeet backend: `mlx` (Apple), `onnx` (Linux/CUDA via onnx-asr), or `auto` |
| `HA_URL` | (required) | Home Assistant URL |
| `HA_TOKEN` | (empty) | HA long-lived access token |
| `HTTP_PORT` | `8888` | HTTP server port (TTS files + web UI) |
| `TTS_BASE_URL` | `http://<LAN IP>:<HTTP_PORT>/tts/` | URL the ESP fetches TTS files from, ending in `/`. Set it when the ESP reaches `/tts/` through a relay on another network |
| `CALENDAR_CREDENTIALS` | `client_secret.json` | Google OAuth credentials file path |
| `CALENDAR_TOKEN` | `token.json` | Google OAuth token file path |
| `SPOTIFY_CLIENT_ID` / `SPOTIFY_CLIENT_SECRET` | (empty) | Spotify developer app, for search (client credentials). Music is enabled when go-librespot answers |
| `SPOTIFY_LIBRESPOT_URL` | `http://127.0.0.1:3678` | go-librespot API |
| `SPOTIFY_PCM_FIFO` | `/run/go-librespot/pcm` | FIFO go-librespot writes its PCM into |

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
- Never `git commit --amend` — always new commits.

## Commit straight to `main`

No pull request: single maintainer, no reviewer, so a PR is a branch, a push and a merge to reach
the commit that could have been made directly. Commit to `main` and push (from a worktree:
`git push origin HEAD:main`). The commit message then carries the reasoning — what was tried,
measured, ruled out — since there is no PR description to hold it.

The live server is `/opt/voice-assistant` on charras-server, a root-owned checkout of `main`
(`voice-assistant.service`, config in `/etc/voice-assistant.env`): deploy with
`sudo git -C /opt/voice-assistant pull` then `sudo systemctl restart voice-assistant`.
