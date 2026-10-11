# Voice Assistant Server

Custom voice assistant replacing Home Assistant's voice pipeline. Runs on ESPHome hardware (ESP32-S3 with wake word), fully local STT/TTS, with cloud or local LLM.

Production runs on a Linux server ([Deployment](#deployment-and-service-management)); a Mac remains usable for development.

## Architecture

```
ESP (wake word) → audio stream → voice_server.py
                                   ├─ stt.py (Parakeet: onnx-asr on CUDA, MLX on a Mac) → transcript
                                   ├─ llm.py (OpenAI-compatible: Ollama, or cloud) → tool calls or text
                                   │    ├─ ha_client.py → Home Assistant REST API
                                   │    ├─ weather.py → Open-Meteo API (free, no key)
                                   │    ├─ places.py → Photon + Nominatim (OpenStreetMap, free, no key): shops, opening hours
                                   │    ├─ calendar_client.py → Google Calendar API (OAuth2)
                                   │    ├─ spotify.py → go-librespot (playback) + Spotify Web API (search)
                                   │    └─ images.py → Wikipedia / Openverse (free, no key) → cast to a Nest Hub
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
| `stt.py` | Parakeet TDT 0.6B v3: onnx-asr on Linux/CUDA (server) or parakeet-mlx on Apple Silicon (Mac dev), picked by `STT_BACKEND` |
| `weather.py` | Open-Meteo weather API: geocoding, current conditions, 5-day forecast (French) |
| `places.py` | Shops and places from OpenStreetMap, nearest to the home first: address, phone, the week's opening hours read by opening-hours-py |
| `calendar_client.py` | Google Calendar client: OAuth2, event queries, event creation, prompt summary |
| `setup_calendar.py` | One-time OAuth setup script for Google Calendar (requires browser) |
| `spotify.py` | Spotify: go-librespot API, Web API search, and the MP3 stream of go-librespot's PCM FIFO served at `/spotify/stream.mp3` |
| `images.py` | Pictures for the screen: Wikipedia article's lead image, else an Openverse photo; the screen loads the URL itself |
| `recordings.py` | Each voice request kept as WAV + JSON (transcript, reply, tool calls, timings) for 90 days, root-only |
| `web_ui.py` | Web UI (HTML + routes), exchange log with persistence (`exchanges.json`) |
| `requirements-linux.txt` / `requirements-mac.txt` | Base deps (`requirements.txt`) plus the STT backend: onnx-asr (server) or parakeet-mlx (Mac) |
| `run.sh` | Mac dev launcher: sources `.env`, reads secrets from the macOS Keychain, starts llama-server unless a cloud key is set |
| `ctl.sh` | Mac dev service control via launchd; `./ctl.sh install` generates the (gitignored) plist |

## Pipeline flow (voice)

1. ESP detects wake word → sends audio stream
2. Silero VAD detects end of speech (31 frames / ~1s silence)
3. STT: Parakeet transcribes audio to French text
4. LLM: sends a static system prompt (instructions, family, entity list) + tool definitions + conversation history + a short context message (date/time, timers, music) + the transcript
5. If tool_calls → execute each, loop back to LLM with results + tools (up to 3 rounds) until text response
6. If text response → check the fallback parser for a tool call written as text (`fn({...})`)
7. Final LLM round formulates natural spoken response from tool results
8. TTS: Kokoro generates WAV, played via announcement API with `start_conversation=True`
9. ESP automatically starts listening for follow-up (no wake word needed)
10. If no speech within 5s → conversation ends. If `end_conversation` tool was called → no follow-up.

## Key design patterns

- **Sites** (`sites.py`): one server for several homes, each with its satellite, Home Assistant, TTS URL, conversation, timers and last speaker. The ESP handlers are bound to their site, and a voice request activates it; the other state (`ha_client`, `conversation_history`…) is read from the active site. One conversation at a time: a satellite that wakes while another site is mid-conversation gets an error. Text requests take an optional `"site"`.
- **Room groups**: `room_groups` in `config.local.json`, shared by all sites (matched loosely: « Chambre Zoé » finds « Chambre de Zoé »), maps group names ("enfants") to lists of HA areas; "tout", "toute la maison" and "partout" are built in and mean every room. For a group, all entities of the matching domain are returned (only the named one for a generic name like "lumières").
- **Entity resolution**: fuzzy matching on normalized friendly_name (accents stripped, stopwords removed), scoped by room/area.
- **Multi-round tool loop**: LLM can return tool_calls across multiple rounds (max 3). After executing tools, results are sent back with tools still available so the LLM can make additional calls; a call already run this turn reuses its result, and a round whose only call is `end_conversation` gets no second round. Tool calls and follow-up flags of a request live in a per-request object (`_Request`, ContextVar), so concurrent requests stay apart.
- **Conversation history**: per site, last 5 exchanges with their tool calls and results (capped), expires after 10 min inactivity (`CONVERSATION_TIMEOUT`). Dry runs keep a history of their own.
- **Static system prompt**: date/time, timers and music go in a context message just before the request, not in the system prompt, so that Ollama reuses its prompt cache (a new minute re-read ~3.6K tokens, ~1.9 s). The server pins the model with a native `keep_alive: -1` after each request.
- **Continuous conversation**: after TTS playback, ESP starts listening for follow-up without wake word. 500ms audio skip avoids TTS echo. `end_conversation` tool lets the LLM end cleanly on "merci", "bonne nuit", etc.
- **Text fallback parser**: catches when the LLM outputs tool calls as plain text instead of structured JSON (3 patterns: JSON args, quoted string, Python kwargs; never in dry runs).
- **Spotify as a radio station**: go-librespot (separate service) plays into a FIFO; `spotify.py` paces it at real time, pads gaps with silence, encodes MP3 and serves `/spotify/stream.mp3` next to `/tts/`. Speakers (ESP or HA) play that URL; one stream, so one speaker at a time.
- **Locks lock, never unlock**: a `lock` tool and state reads, no unlock tool. Only the tools offered this turn run, and `HAClient.call_service` refuses `lock.unlock` and `lock.open` (AD-041).
- **Web UI on Tailscale only**: port 8888 also answers on the LAN, where speakers fetch `/tts/` and `/spotify/stream.mp3`; everything else (UI, API, recordings) answers only localhost and the tailnet (`web_ui.lan_sees_only_speaker_paths`).
- **Secrets**: the server only reads env vars: `/etc/voice-assistant.env` on the server, the Keychain (`voice-assistant-ha-token`, `voice-assistant-openai-key`) through `run.sh` on a Mac.

## Configuration (env vars)

| Variable | Default | Description |
|---|---|---|
| `SITES` | (empty) | Comma-separated homes served by this server (`paris,charras`). Each then reads the satellite and Home Assistant settings below with its name as prefix (`PARIS_HA_URL`, `CHARRAS_ESP_HOST`…); `ESP_PORT`, `ESP_PASSWORD`, `ESP_NOISE_PSK` may stay unprefixed, shared. Empty: one site, unprefixed settings |
| `ESP_HOST` | (empty) | ESPHome device IP or `name.local` (resolved by mDNS); empty for a site without satellite |
| `ESP_PORT` | `6053` | ESPHome native API port |
| `ESP_PASSWORD` | (empty) | Legacy password auth |
| `ESP_NOISE_PSK` | (empty) | Noise encryption key (preferred) |
| `LLM_URL` | `http://localhost:8080/v1/chat/completions` (llama-server) | OpenAI-compatible endpoint. Server: Ollama, `http://localhost:11434/v1/chat/completions` |
| `LLM_API_KEY` | (empty) | If set, enables cloud mode (bearer auth, provider's default reasoning). Empty on the server |
| `LLM_MODEL` | (empty) | Model name: `gemma4:12b-8k` on the server's Ollama (gemma4:12b with an 8192-token context, see INSTALL.md), e.g. `gpt-5.4-nano` in the cloud; empty lets llama-server use its only model |
| `LLM_REASONING_EFFORT` | `none` local, unset cloud | Reasoning budget for thinking models. Keep `none` locally — thinking costs ~10s/round and empties `content` |
| `STT_BACKEND` | `auto` | Parakeet backend: `onnx` (Linux/CUDA via onnx-asr, set on the server), `mlx` (Apple), or `auto` (MLX if installed, else ONNX) |
| `HA_URL` | (required) | Home Assistant URL |
| `HA_TOKEN` | (empty) | HA long-lived access token |
| `HTTP_PORT` | `8888` | HTTP server port (TTS files + web UI) |
| `TTS_BASE_URL` | `http://<LAN IP>:<HTTP_PORT>/tts/` | URL the ESP fetches TTS files from, ending in `/`. Set it when the ESP reaches `/tts/` through a relay on another network |
| `CALENDAR_CREDENTIALS` | `client_secret.json` | Google OAuth credentials file path |
| `CALENDAR_TOKEN` | `token.json` | Google OAuth token file path |
| `SPOTIFY_CLIENT_ID` / `SPOTIFY_CLIENT_SECRET` | (empty) | Spotify developer app, for search (client credentials). Music is enabled when go-librespot answers |
| `SPOTIFY_LIBRESPOT_URL` | `http://127.0.0.1:3678` | go-librespot API |
| `SPOTIFY_PCM_FIFO` | `/run/go-librespot/pcm` | FIFO go-librespot writes its PCM into |
| `RECORDINGS_DAYS` | `90` | Days voice requests are kept in `recordings/` (one folder per day, mode 700); `0` turns recording off |

## Deployment and service management

The live server is charras-server: `/opt/voice-assistant`, a root-owned checkout of `main` with
its own `venv`, run by `voice-assistant.service` (systemd, `/etc/systemd/system/`) with config and
secrets in `/etc/voice-assistant.env`. Beside it: Ollama in Docker (container `ollama`, port
11434) and `go-librespot.service` for Spotify. Host setup (CUDA libraries, locale, Ollama model):
home-infra `charras/SETUP.md` §7–8b.

```bash
sudo git -C /opt/voice-assistant pull     # deploy main
sudo systemctl restart voice-assistant    # after code or env changes
sudo systemctl reload voice-assistant     # SIGHUP → hot-reload HA entities
sudo systemctl stop voice-assistant       # SIGTERM → graceful shutdown
systemctl status voice-assistant
journalctl -u voice-assistant -f          # logs
```

`systemctl restart` returns before the models are loaded: the server is up once the journal says
`Server running` (`Fatal error` otherwise), or `curl -s localhost:8888/health` answers 200.

On a Mac (dev), `./run.sh` runs the server in the foreground, and `ctl.sh` wraps launchd:

```bash
./ctl.sh install    # once: generates the plist and registers it with launchd
./ctl.sh start      # waits for ready
./ctl.sh stop       # graceful SIGTERM
./ctl.sh restart    # after code changes, waits for ready
./ctl.sh reload     # SIGHUP → hot-reload HA entities, waits for completion
./ctl.sh status     # running/not running + PID
./ctl.sh logs       # tail -f /tmp/voice-assistant.log
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

Then deploy it to the live server: see [Deployment](#deployment-and-service-management).
