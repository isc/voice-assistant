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
- **Update 2026-10-10**: « tout », « toute la maison » and « partout » are built in; `config.local.json` keeps only real subsets like « enfants » (entries for those three words in an older file are dropped at load). A room list cannot reach a device HA has in no room: in Paris, « lumières partout » found 12 lights with the old list and finds 15 now, with Entrée, Bibliothèque and Couloir. A generic name (« les lumières ») takes every device of its kind, in a room or not; a sweep over several domains (« éteins tout ») keeps to devices in a room, which leaves alone the loudness and cross-fade switches of the Paris Sonos, itself in no room. Also, a generic name now keeps a group to its own domain: « éteins les lumières des enfants » (turn_off hints light, switch and media_player) used to switch off every switch and speaker in those rooms too.

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

### AD-026: Continuous conversation via announcement API
- **Context**: User had to say the wake word between every turn, breaking conversational flow.
- **Decision**: After TTS generation, deliver audio via `send_voice_assistant_announcement_await_response(start_conversation=True)` instead of event-based TTS_END. The ESP automatically starts a new pipeline without wake word. 500ms audio skip at follow-up start to avoid TTS echo pickup. 5s timeout for follow-up silence (extends to 30s once speech detected). `end_conversation` LLM tool for clean closure ("merci", "bonne nuit") — skips follow-up and clears history.
- **Trade-off**: Slightly more complex pipeline flow. The announcement API replaces event-based TTS delivery. ESP stays in active mode between turns (LEDs on, mic open).

### AD-027: Server-side timer manager with ESP native timer events
- **Context**: User wants timers ("mets un timer de 5 minutes") and alarms ("réveille-moi à 7h"). ESPHome supports timers natively via `send_voice_assistant_timer_event()` (feature flag `VoiceAssistantFeature.TIMERS`), handling LED animations and sounds on the device.
- **Decision**: Server-side `TimerManager` in `timer.py` using asyncio tasks. Three LLM tools: `set_timer` (duration), `set_alarm` (absolute time), `cancel_timer`. Timer lifecycle sends events to ESP: STARTED, UPDATED (tick every 1s), CANCELLED, FINISHED. On FINISHED, additionally play a TTS announcement via the announcement API with `start_conversation=True` for follow-up.
- **Trade-off**: Timers are in-memory only — lost on server restart. Acceptable for kitchen timers; persistent alarms would need disk storage. The 1s tick loop is lightweight but adds continuous traffic to the ESP while timers are active.

### AD-028: Partial cover position support
- **Context**: Users want to say "ouvre les volets à 50%" instead of just fully open/close.
- **Decision**: Add `set_cover_position` tool with a `position` parameter (0-100). Update `ha_client.py` to call `cover.set_cover_position` and report the position in state descriptions when partially open/closed.
- **Trade-off**: One more tool for the LLM to handle. Stays within budget since it's in the same domain as open/close.

### AD-029: Automatic ESP reconnection with ReconnectLogic
- **Context**: Server blocked on startup if ESP was unavailable, and crashed on disconnect with no recovery.
- **Decision**: Use aioesphomeapi's `ReconnectLogic` for automatic reconnection with exponential backoff. Server starts even if ESP is offline and connects when available.
- **Trade-off**: Slightly more complex connection lifecycle, but eliminates manual restart on transient network issues.

### AD-030: Dynamic plist generation and .env for configuration
- **Context**: `com.voice-assistant.server.plist` contained hardcoded home directory paths. ESP/HA IP addresses were hardcoded as defaults in `voice_server.py`. Both problematic for public repo.
- **Decision**: Generate plist dynamically in `ctl.sh install` using `$SCRIPT_DIR`. Move IP/port config to `.env` file (gitignored), sourced by `run.sh`. Secrets remain in macOS Keychain.
- **Trade-off**: Requires `./ctl.sh install` on first setup instead of just copying the plist.

## 2026-03-23: Google Calendar integration

### AD-031: Google Calendar via OAuth 2.0 with granular scope
- **Context**: Google Calendar API requires OAuth 2.0 for personal accounts. Multiple scopes available from full access (`calendar`) to narrow event-level access (`calendar.events`).
- **Decision**: Use `calendar.events` scope (read+write events only). One-time OAuth setup via `setup_calendar.py` (requires browser). Refresh token handles automatic renewal. Token stored in `token.json` (file-based, gitignored) — standard approach for google-auth-oauthlib.
- **Trade-off**: File-based token is less secure than Keychain, but it's what Google's Python libraries use natively. Acceptable for a personal home server.

### AD-032: Single primary calendar for v1
- **Context**: Roadmap mentions multi-calendar with voice identification, but voice ID is not yet implemented.
- **Decision**: v1 uses only the `primary` calendar. Multi-calendar support will come with voice identification.
- **Trade-off**: Only one person's calendar is accessible for now.

### AD-033: No calendar injection in system prompt
- **Context**: Initially planned to inject a calendar summary into the system prompt. Testing showed GPT-5.4 Nano systematically calls the `query_calendar` tool regardless, making the injected summary redundant.
- **Decision**: No prompt injection. The LLM calls `query_calendar` on demand, which returns fresh data directly from the Google API.
- **Trade-off**: Every calendar question costs one tool round-trip (~1.5s), but the data is always fresh and the prompt stays lean.

## 2026-10-10: ESP on another site's network

### AD-034: `TTS_BASE_URL` overrides the announced TTS URL
- **Context**: Testing whether one GPU server (Charras) can serve an ESP at another site (Paris). The server connects out to the ESP, but the ESP downloads each reply WAV from the URL the server announces, built from the server's LAN address. Both sites use the same `192.168.1.0/24`, so that address names a different machine on the ESP's side, and no routing can tell the two apart.
- **Decision**: An optional `TTS_BASE_URL` env var replaces the computed URL, pointing at a relay on the ESP's LAN that forwards `/tts/` to the server. The native API needs no change: `ESP_HOST`/`ESP_PORT` already accept a forwarded port, and audio travels over that TCP connection (no UDP).
- **Trade-off**: One more setting, empty by default, so the single-LAN setup is unchanged. Rejected: site-to-site routing (impossible with overlapping subnets) and a second server instance per site (doubles STT/TTS VRAM beside the LLM).

## 2026-10-10: Web radio playback

### AD-035: Web radio playback, direct-to-ESP by default, HA fallback by room
- **Context**: User wants to play French web radios ("mets France Inter") with volume follow-ups ("plus fort", "moins fort"). The ESP32-S3 running `speaker.media_player` can decode MP3/FLAC/OPUS/WAV HTTP streams natively — no HA round-trip needed if the target is the ESP itself.
- **Decision**: Four tools — `play_radio(station, room)`, `stop_media(room)`, `set_volume(level, room)`, `change_volume(direction, room)`. Station names are fuzzy-resolved against a hardcoded `RADIO_STATIONS` dict in `voice_server.py` (18 French stations: Radio France family, RTL, Europe 1, NRJ, Skyrock, TSF Jazz, etc.).
  - Target resolution: `room` specified + matches an HA `media_player` entity → route via HA. Otherwise → the ESP that received the command, via `aioesphomeapi.APIClient.media_player_command()` (with `announcement=False` so the stream uses the default pipeline, not the TTS ducking pipeline).
  - The ESP's `media_player` entity key is discovered once in `on_connect` via `list_entities_services()` and cached in `device_media_keys[host]`.
  - `_last_media_target` (either `("esp", host)` or `("ha", entity_id)`) is remembered so "plus fort" after "mets france inter dans le salon" keeps targeting the right speaker.
  - Tools are registered unconditionally (present even if HA is offline), since ESP direct playback doesn't need HA.
  - After `play_radio`, `set_volume` or `change_volume`, the conversation ends (no follow-up listening): the open mic would otherwise pick the radio up and send it to STT as a request — from the ESP itself, or from a HA speaker that may share its room (the server has no way to know). Skipping follow-ups whenever something is playing would need the server to track media player states, which it does not do yet.
  - Station lookup returns nothing when a fuzzy match is ambiguous ("musique classique" hits two stations) or names no station ("la radio"), so the LLM asks instead of playing a random one.
- **Trade-off**: Hardcoded station URLs need manual updates when they move: by 2026-10-10, five of the first eighteen were already dead (France Bleu renamed ici, RMC, three NRJ-group streams) and were replaced after checking each returns `audio/mpeg`. The Voice PE decodes MP3, not AAC, so only MP3 streams are listed. Volume step for `change_volume` is whatever the ESP is configured for. A French fallback parser entry catches "plus fort"/"moins fort"/"monte le son"/"baisse le son" for local LLMs that emit text instead of structured tool calls.

### AD-036: No transcription when the VAD heard no speech
- **Context**: Parakeet hallucinates text on recordings where nobody spoke (a follow-up window that closed on silence, a wake-word request that hit the 30 s timeout), and the LLM answers it — once an apology for a request nobody made, right after the radio was stopped.
- **Decision**: `process_voice_pipeline` skips STT when the VAD never flagged speech in the recording, and takes the existing empty-transcript path (follow-up: end quietly; wake word: error LED).
- **Also**: no `STT_VAD_END` for such a recording. ESPHome 2025.6.2 handles `STT_VAD_END` by moving to `STOP_MICROPHONE` (then `AWAITING_RESPONSE` on its next loop) and ignores a `RUN_END` that arrives in `STOP_MICROPHONE`; with STT skipped, the `RUN_END` followed within a millisecond and the ESP stayed stuck "thinking", LEDs blinking. While the ESP is still `STREAMING_MICROPHONE`, `RUN_END` takes it back to idle.
- **Trade-off**: Speech too quiet for Silero VAD (threshold 0.5) is now dropped instead of transcribed; so far the VAD has caught every real request, and a hallucinated answer is worse than a red LED.

## 2026-10-10: Spotify

### AD-037: Spotify through go-librespot, streamed to the speakers like a radio
- **Context**: The family wants Spotify by voice ("mets du Daft Punk", "ma playlist du matin") on the Voice PE as well as on the Google Home / Sonos. The Voice PE is no Spotify Connect device. Spotify Connect through the Web API reaches Sonos, but a Google Cast device only shows up in the device list after a phone has woken it, so a voice command would often find no target. Music Assistant would cover every speaker, but it reaches the Voice PE only through HA, and the ESP accepts a single voice-assistant subscriber: with HA's ESPHome entry enabled, HA and this server would compete for it. The Paris HAOS VM also has 1 vCPU / 2 GB.
- **Decision**: go-librespot (a headless Spotify Connect player, separate systemd service on charras, configured in home-infra `charras/go-librespot`) writes raw PCM into a FIFO. `spotify.py` reads the FIFO at real-time pace (it writes as fast as it is read), pads the gaps (pause, track loading) with silence so that speakers never starve, encodes MP3 with ffmpeg while someone listens, and serves the stream at `/spotify/stream.mp3`, next to `/tts/` so that the Paris relay serves both. Any speaker plays that URL as it plays a radio: the ESP directly, an HA speaker through `media_player.play_media`.
  - Two tools: `play_music(query, kind, shuffle, room)` with kind in artist / track / album / playlist / liked, and `music_control(action)` for pause / resume / next / previous / current.
  - While music plays, the system prompt says what and on which speaker, so that "mets la musique sur la Google Home" moves it (`music_control` resume with a room, same position) instead of starting something else: without that line, gemma4:12b picked Liked Songs. With no room, media commands keep going to the last speaker used, which the family finds natural with a single Voice PE.
  - Playlists are looked up in the user's library first (go-librespot's `/library/playlists`, fuzzy on the spoken name), then in the catalog. Liked Songs is the `spotify:user:<username>:collection` context.
  - Catalog search uses a Spotify developer app of our own with the client-credentials flow (no user, no redirect). go-librespot's `/token` hands out its session token, but that token belongs to Spotify's desktop client ID, shared by every librespot install: on 2026-10-10 the Web API answered it with a standing 429 (Retry-After 30–60 s, renewed at each try).
  - The speaker is pointed at the stream first, and go-librespot starts once it has connected, so the first seconds are not lost. Starting the music on another speaker stops the previous one, if it is still on the stream. Music that plays to nobody for 5 s pauses go-librespot: that covers radio or stop on the Spotify speaker as well as a speaker stopped from elsewhere, with no Spotify case in the radio and stop paths. `music_control` acts on go-librespot's single queue without resolving a speaker.
  - At startup, the server picks up the HA speaker a previous run left on the stream or on a radio (`media_content_id` is the stream URL or a station's, state playing or paused) as the last media target, and as the Spotify speaker for the stream. Without it, a restart (a deploy) made media commands with no room go to the Voice PE while the music went on on the Sonos: on 2026-10-10 a `play_music` from a garbled request started a second copy of the stream on the Voice PE itself.
- **Measured**: ESP connected to the stream 130 ms after the command (through the Paris relay), Paris Google Home 1.3 s, playing after ~8 s of Cast buffering; Paris Sonos Roam SL plays it ~5 s after the command (it opens the URL twice first). go-librespot's position advances at exactly real time. 192 kbps MP3, 24 kB/s per listener.
- **Trade-off**: One go-librespot is one queue: one music at a time, on one speaker. Pause and skip take effect after the speaker's buffer (a second on the ESP, several on Cast). Search depends on a developer app whose owner must stay Premium (Spotify rule since 2026-02). Music started from a phone onto "Assistant vocal" plays to nobody until a speaker is pointed at the stream (ROADMAP follow-up).

## 2026-10-10: Pictures on the screen

### AD-038: Web pictures cast to the Nest Hub, from Wikipedia then Openverse
- **Context**: Both sites have a Google Nest Hub Max (Paris names it « Google Home »). Its software is locked — secure boot, no root or Linux port — so the only way in is Google Cast, and Cast shows media URLs: video, audio, pictures. The family wants « montre-moi la tour Eiffel », « à quoi ressemble un okapi ? ».
- **Decision**: a `show_image(query)` tool. `images.py` finds a picture and checks with a HEAD that its URL answers with an image (and which type); the server casts that URL with `media_player.play_media` and the screen loads it straight from Wikimedia or Flickr.
  - **Wikipedia first**: the lead image of the top French Wikipedia hit, 1280 px (the screen is 1280×800), kept only when every word of the query is in the article's title — « coucher de soleil sur la mer » otherwise lands on a painting's article. **Openverse otherwise**: free-licence photos, mostly Flickr at ~1024 px, searched with the same French keywords. Both are free with no key (Openverse anonymous quota: 20/min, 200/day). Measured on 2026-10-10, six spoken requests: the right picture each time, found and checked in 0.4–0.7 s.
  - « Une autre » calls the tool again with the same query; pictures already shown in the conversation are skipped, so the second one comes from Openverse. The whole search gives up after 8 s, as the voice device waits on it.
  - Screens are found, not configured: HA has no "has a display" attribute (a Nest Hub Max reports the same supported features as a Google Home speaker), so `ha_client` asks the template API for media players whose device model contains « Hub ». No screen, no tool. One screen per site, so the tool takes no room; a second screen would bring one.
  - Not downloaded by the server: nothing to store or clean up, no detour through the Paris relay (which only forwards `/tts/` and `/spotify/stream.mp3`), and the screen only needs internet access.
- **Measured** on the Paris Nest Hub Max, 2026-10-10: `play_media` answered in 1.2 s; 1.7 s after the call the Default Media Receiver had loaded the Wikimedia thumbnail (HA then reports `paused`, content type `video`) and the picture was on screen.
- **Trade-off**: Cast plays one thing at a time: a picture sent to a Nest that plays the Spotify stream or a radio stops it. Long phrases find nothing on Openverse (« un chien qui court sur la plage »: 0 results), hence the « quelques mots » in the tool description.

## 2026-10-10: Production on a Linux server

### AD-039: charras-server under systemd; Keychain and launchd kept for Mac dev
- **Context**: AD-019 (Keychain) and AD-020 (launchd) were made on the M1 MacBook, which travels and cannot serve a fixed site. The pipeline has run on charras-server (Ubuntu 24.04, NVIDIA GPU) since July 2026 (journal from 2026-07-23), STT on onnx-asr with CUDA since the backend split of 2026-06-28, while the docs and the startup banner still described the Mac.
- **Decision**: Production is `/opt/voice-assistant`, a root-owned checkout of `main`, run by `voice-assistant.service` (systemd) with config and secrets in `/etc/voice-assistant.env` (root, 0600) as its `EnvironmentFile`. `ExecReload` sends SIGHUP, so AD-020's signal handling carries over. LLM: `gemma4:12b` on Ollama in Docker; `STT_BACKEND=onnx`. `run.sh`, `ctl.sh` and the Keychain stay, for development on the Mac only.
- **Also**: the startup banner is gone. It still named MLX and Qwen 3 4B, and as plain `print`s to a pipe it only reached the journal at shutdown. The startup log lines already name the STT backend and the TTS; the LLM line now gives the model next to the URL.
- **Trade-off**: Two ways to run the server. `systemctl restart` returns before the models are loaded, where `ctl.sh` waited for the "Server running" log line: readiness is read from the journal or `/health`.

## 2026-10-10: Questions are not claimed actions

### AD-040: The empty-claim challenge skips questions
- **Context**: Since 2026-07-23 (`e78fe9f`), a reply that claims an action (« j'ai allumé », « je lance ») with no function call gets challenged once: the model is told nothing was done and asked to call the functions. On 2026-10-10, music near the Voice PE garbled a request into « …une picture emplie de ville… ». gemma4:12b rightly asked « Souhaitez-vous que je lance de la musique ou que j'affiche une image d'une ville ? ». The « je lance » matched, the challenge fired, and the model then played music and cast a city picture that nobody had asked for.
- **Decision**: `claims_completed_action` only looks at sentences that do not end with « ? »: a question offers an action, it does not claim one. « J'ai éteint les lumières. Autre chose ? » is still a claim.
- **Trade-off**: A model that wraps a false claim in a question (« j'ai fermé les volets, autre chose ? » as one sentence) is no longer challenged. Replayed on the 50 replies of the exchange log (2026-10-10): of the 31 the old check matched, none stops matching.

## 2026-10-10: Door locks

### AD-041: Lock by voice, never unlock
- **Context**: Paris added a Nuki Smart Lock Pro to HA (Matter, `lock.smart_lock_pro`, in no room); Charras has a Yale, « Porte d'entrée ». The server knew neither: `lock` was not a supported domain. Anyone within earshot of the Voice PE can talk to it, a radio or a TV included, and the server cannot tell voices apart (see the voice ID notes in ROADMAP).
- **Decision**: A `lock` tool, offered only where HA has a lock, and `get_state` reads its state. No unlock tool: its description tells the model to answer that unlocking takes the key or the app. « porte », « serrure » and « verrou » are generic names for the lock domain, so that « est-ce que la porte est verrouillée ? » reads the lock and not the alarm-sensor switches named « Bas Porte … » at Charras. A generic name of another domain resolves to nothing: « ferme la porte » sent as `close_cover` closes no shutter and sends no cover service to the lock.
  - `execute_function` refused no function name. Any name it did not know went to HA as a service, so a model answering « déverrouille la porte » with an invented `unlock(entity="porte")` would have called `lock.unlock`. Three locks now, from the outside in: the tool loop and the text fallback run only the tools offered this turn; `execute_function` refuses a name it does not know; `HAClient.call_service`, through which every HA write goes, refuses `lock.unlock` and `lock.open` whatever route led there.
  - The lock call is awaited, where other service calls are fired and forgotten (see the comment in `execute_function`): someone leaving the house on « porte verrouillée » must hear about it when HA could not reach the lock.
- **Measured** on gemma4:12b, 2026-10-10. First live try, « Tu peux fermer la porte ? » right after a « verrouille la porte » in the conversation: no call, and the answer meant for unlocking (« le déverrouillage se fait avec la clé ou via l'application »), 5 times out of 5 in replay; fine with no history. The first description said « si on demande d'ouvrir ou de déverrouiller, réponds que… sans appeler de fonction ». The tool description now says to lock at every request, even a door locked earlier, and that no function opens or unlocks a door. Dropping that last sentence is worse: « déverrouille la porte » then called `lock` and answered « j'ai déverrouillé la porte », 3 out of 3. With both: 65/65 over 13 phrasings (lock, re-lock, state, unlock, open, a shutter), with and without history.
- **Trade-off**: Unlocking stays in the phone app. « Ferme la porte » means locking only when the model picks `lock`; the guard keeps it from moving a shutter instead.

## 2026-10-10: Recordings of voice requests

### AD-042: Keep each voice request 90 days, root-only; web UI on Tailscale only
- **Context**: Nothing kept the audio: `stt.py` wrote a temporary WAV and deleted it after transcription, and the exchange log holds the last 50 exchanges as text. On 2026-10-10 a request drowned by music came out as song lyrics, and nobody could hear what had actually been said. The ROADMAP's "STT through the LLM" also needs real French command recordings to compare engines, and there were none.
- **Decision**: `recordings.py` writes each voice request as a 16 kHz WAV at the start of the pipeline, and a JSON beside it at the end: wake word or follow-up, whether the VAD heard speech, transcript, reply, tool calls, timings, error. One folder per day under `recordings/`; folders older than `RECORDINGS_DAYS` (90) are deleted at each new recording; `0` turns it off. Requests the VAD found silent are kept too, flagged: they are where too-quiet speech would hide (AD-036). The JSON files are the long-term log. The exchange log keeps its 50 entries, now including failed requests ("[Rien compris]", LLM or pipeline errors), each pointing at its WAV, which the web UI plays ("Ecouter la demande"); a follow-up nobody spoke in is left out of it, as every conversation ends that way.
  - The folder is mode 700, for root, which the server runs as. It does not keep the recordings from Ivan or a Claude session (passwordless sudo); it keeps them from the services charras runs under other users (apps under `ivan`, the wiki, nginx workers), some reachable from the network.
  - Port 8888 answered on the LAN with no authentication: the web UI, transcripts (`/api/exchanges`), commands (`/api/send`, `/test`), now recordings. A middleware lets the LAN reach only what speakers fetch (`/tts/`, `/spotify/stream.mp3`, and `/health`); the rest answers localhost and Tailscale (100.64.0.0/10) only, as home-infra already documented ("the web UI stays on Tailscale"). Checked before: no HA configuration calls port 8888, and `test_e2e.py` uses localhost. The Paris relay arrives as localhost through its SSH tunnel; its nginx only forwards `/tts/` and the Spotify stream.
  - What the address check cannot do: tell root from another local user. A service on charras can still fetch `http://localhost:8888/recordings/…`, as it could already reach the API.
- **Trade-off**: The family's voices, children's included, sit on charras for 90 days, and a follow-up window records whatever is said around the satellite. About 32 kB per second of audio: well under 1 GB a month at the current use. A recording that fails to save is logged and never fails the request.

## 2026-10-11: Several homes from one server

### AD-043: Sites, each with its satellite and Home Assistant
- **Context**: Since 2026-10-10 the single server on charras served the Paris satellite through an SSH tunnel (AD-034), configured for one ESP and one Home Assistant per process. The family goes to Charras for the Toussaint holidays; switching `/etc/voice-assistant.env` by hand at each trip, or running a second instance (another ~4 GB of STT/TTS beside `gemma4:12b`), were both ruled out. There is one Voice PE, which travels with the family.
- **Decision**: `sites.py` reads `SITES=paris,charras` and, per site, the satellite and HA settings prefixed with its name (`PARIS_HA_URL`, `CHARRAS_ESP_HOST`…); `ESP_PORT`, `ESP_PASSWORD` and `ESP_NOISE_PSK` may be shared, unprefixed. Without `SITES`, the unprefixed variables describe one site, as before.
  - Per site: Home Assistant client (entities, screens, home location), TTS URL announced to its speakers (Paris through the relay, Charras on the LAN), conversation history, timers, last speaker for follow-ups, pictures already shown. Shared: models, LLM, calendar, family, room groups (matched loosely, so « Chambre Zoé » finds Charras's « Chambre de Zoé »), the single Spotify stream (its speaker remembers its site).
  - The ESP handlers are bound to their site with `functools.partial`: aioesphomeapi's callbacks do not say which device speaks, and the old `current_device` was simply the last one connected. A voice request activates its site; the rest of the code reads `ha_client`, `conversation_history`… from the active site through properties, which kept the change small. Text requests from the web UI or `/test` take an optional `"site"`.
  - One conversation at a time: the recording and VAD state is not per site. A satellite that wakes while another site is mid-conversation gets an error answer (`handle_start` returning `None`), and its audio is ignored.
  - Timers fire on the satellite of the site they were set at, with that site's TTS URL. `/health` reports each site; its verdict needs one reachable Home Assistant, not all: the other home may be closed for the season.
  - The travelling Voice PE: Paris reaches it through the tunnel (`127.0.0.1:16053`), Charras as `home-assistant-voice-09d3f1.local`, which aioesphomeapi resolves by mDNS whatever the DHCP address. Each site keeps trying; only the one where the device sits connects.
- **Measured**: against both real Home Assistants, without executing anything (dry run, read-only): Paris 29 entities and its Google Home screen, Charras 68 entities and its dining-room display; « montre-moi un okapi » at Charras targets Charras's screen; each site keeps its own conversation history.
- **Trade-off**: Two people speaking to two homes at the same moment: the second waits for an error. The site that is not connected retries its satellite in the background (log lines, no cost).

### AD-044: An 8192-token context for gemma4:12b
- **Context**: Testing the two sites on 2026-10-11, every request at Charras came back empty. Ollama 0.30.11 gives gemma4:12b a 4096-token context on this 16 GB card (the model takes 262144), and Charras's prompt (68 devices, the tool definitions, then tool results) reached 3605 tokens on the first round and 4089 on the next: Ollama truncated it (`truncated = 1`), the completion stopped after 7 tokens with `finish_reason: length`, and the reply was empty. When it did not come back empty, the truncated system prompt made the model invent calls (« un chat roux », an example from the `show_image` description). Paris's prompt is 2051 tokens, which is why it went unnoticed. The summer tests at Charras ran with fewer tools.
- **Decision**: a derived model, `gemma4:12b-8k` (`FROM gemma4:12b`, `PARAMETER num_ctx 8192`), named in `LLM_MODEL`. It shares the weights; the OpenAI-compatible endpoint has no way to pass `num_ctx` per request, and `OLLAMA_CONTEXT_LENGTH` on the container would mean recreating it.
- **Measured**: 8.4 GB of VRAM loaded, against 8.1 GB at 4096; the card at 12.6 of 16.3 GB with the voice server. Charras: 4 out of 4 correct after the change (prompts up to 4228 tokens), 0 out of 6 before.
- **Trade-off**: The model name now carries a setting, so pulling a new gemma means recreating the derived model. A prompt can still outgrow 8192 with a much larger house or a long tool loop: the retry on `finish_reason: length` (raising `max_tokens`) cannot help when the prompt itself fills the context.

## 2026-10-11: Code audit

### AD-045: What the overnight audit changed, and why
- **Context**: Much of the code was written with weaker models. Six read-only reviews (pipeline, LLM loop, Home Assistant, Spotify, calendar/timers/web, hygiene) replayed real requests on both houses' entity lists, ran ~60 dry runs and read the journal; each finding was checked before being fixed. The commits of 2026-10-11 carry the details and measurements.
- **Decision**, the main lines:
  - Entity resolution no longer actuates lookalikes (fuzzy cutoff 0.75, no lone candidate under another name, rooms said by name, no switch in a « tout » sweep, roof windows left out of shutter sweeps, cover kind by name before device_class). Devices are re-read every 5 minutes.
  - LLM: a static system prompt (Ollama's cache: 0.6-0.9 s a round instead of 2.5 s at each new minute), the model pinned in memory, history with the tool calls, per-request state, no French-verb fallback, no « already called » reminder, no second round after a goodbye.
  - Voice pipeline: no follow-up while the satellite plays its own media (the likely cause of the 2026-10-10 reboot), aioesphomeapi timeouts actually caught, Kokoro on the GPU with its models out of /tmp, alarms right across the clock change.
  - Spotify: no false « playing », music kept through a restart, silence not streamed forever. Web UI on Tailscale with a site selector; exchange log 0600 and atomic. Lint clean, versions pinned.
- **Measured**: test_e2e 19/19 in dry run after the changes; 133 unit tests.
- **Left for later** (ROADMAP): timers lost on restart, alarms more than a day ahead, duplicate device names, sensor-search synonyms, a VAD minimum speech length, the web API without CSRF/Host checks.

## 2026-10-11: Shops and their opening hours

### AD-046: OpenStreetMap for places, opening hours computed for the LLM
- **Context**: « Est-ce que le magasin Altermundi de la rue du Commerce il est ouvert le dimanche ? » got « je ne peux pas vérifier les horaires ». OpenStreetMap has the shop, with `opening_hours: Mo-Sa 10:00-19:30`.
- **Decision**: a `find_place` tool, offered everywhere like the weather (`places.py`).
  - **Photon** (komoot's OSM geocoder) finds the places, biased to the home's coordinates from Home Assistant with `zoom=16`, `location_bias_scale=0.1`. Nominatim's own search ranks by importance, not distance: « Altermundi » gave five shops across Paris before the one 700 m away. Streets, stations, bus stops and Vélib' stations named after a place are dropped, on our side: Photon's `osm_tag=!highway` filter reranks, and put a stadium before the Leclerc supermarket.
  - **Nominatim's `/lookup`** gives the three nearest places' tags (`opening_hours`, `phone`) in one request: Photon returns no tags.
  - **opening-hours-py** turns `opening_hours` into the next seven days in French and « fermé, ouvre demain à 10:00 » for now, with the country's public holidays (`PH off`). The 12B model reads a week instead of OSM's syntax. The time zone is the home's and the country Photon's: guessed from coordinates, the library loads a 110 MB map and keeps it.
  - Google Places would know more shops and fresher hours, but needs a key, billing, and sends each request to Google; OSM fits the free, no-key pattern of the weather and pictures.
- **Measured**: Photon + Nominatim ~0.3–0.6 s. Dry runs: 7 out of 7 phrasings called `find_place` with a good query (« Altermundi rue du Commerce », « pharmacie », « Monoprix »…), weather and lights unchanged. Real runs: 7 out of 7 answers true to the data (« Non, le magasin Altermundi de la rue du Commerce est fermé aujourd'hui. Il rouvrira demain à 10h. »), 2.0–3.3 s.
- **Trade-off**: OSM's hours can be stale (Altermundi's were last checked in 2024) and many small shops have none: the answer is then « je ne connais pas leurs horaires ». A place far enough to be in another time zone gets the home's. Three places with a week each is ~700 tokens of the 8192-token context.

## 2026-10-11: Dry runs apart from the family's conversation

### AD-047: A dry run has a conversation of its own, and reset clears only that one
- **Context**: On 2026-10-10 at Paris, dry runs testing the new `lock` tool ended with « Verrouille la porte d'entrée » → « C'est fait… ». `/api/dry-run` then shared the voice pipeline's conversation, so a real « Tu peux fermer la porte ? » a minute later was answered with that exchange as context: no tool call, an unrelated refusal. Tool calls do not run in a dry run, so the history claimed a lock that never happened. 05deca3 gave dry runs a history of their own, but a dry run still refreshed the live conversation's clock (a test every few minutes kept a stale family conversation alive past its 10 minutes), expired it and handed it a new id, cleared the pictures already shown; its own history never expired; and `/api/reset-conversation`, called by test_e2e between tests, wiped the family's history mid-conversation.
- **Decision**: a conversation is one object (`Conversation` in sites.py: history, id, last request time, pictures shown), and each site has two: the family's and the dry runs'. `process_with_llm` picks one at the start and touches only that one, with the same expiry after `CONVERSATION_TIMEOUT`, so multi-turn tests (« et demain ? ») keep working. Before, the dry-run history was a field beside the live state, and every per-conversation field needed a twin. `/api/reset-conversation` clears the dry-run conversation only; its one caller is test_e2e, and the live conversation ends by itself (`end_conversation`, 10 minutes of silence). The exchange log (`exchanges.json`) and `recordings/` never held dry runs: `handle_dry_run` adds no entry, and only the voice pipeline records.
- **Trade-off**: There is no endpoint to cut the family's conversation short any more; restart the server if it is ever needed. Tests still share the live server's model and Home Assistant entity list, which is what they are meant to exercise; this replaces ROADMAP's « isolated test server instance ».

## 2026-10-11: The web UI from a phone, without Tailscale

### AD-048: Through Funnel at /assistant/, behind a key link
- **Context**: The PWA (installable since 2fb5f6b) answered only on the tailnet (AD-042), over plain HTTP: installing it on a phone meant installing Tailscale there first, and even then Chrome offers no real install outside HTTPS. The other apps on charras (kiné, Par chœur, wiki) are public through Tailscale Funnel behind a link with a key, with nothing to install.
- **Decision**: Funnel publishes port 8888 at `https://charras-server.tailcef78d.ts.net/assistant/`; `web_ui.access_control` asks those requests for a key.
  - Funnel's proxy runs on this machine, so its requests come from loopback, which AD-042 trusted outright: published as it was, the UI would have handed the house to the internet. They are told apart by the headers the proxy adds (`X-Forwarded-For`, `Tailscale-Funnel-Request`) and need the `WEB_UI_KEY` cookie, set by `?k=<key>` (1 year, Secure, HttpOnly, SameSite=Lax, which keeps other sites from posting commands with it) and followed by a redirect to the page without the key. Empty key: Funnel gets 401 everywhere. Without those headers, AD-042 holds: the LAN gets the speaker paths, localhost and the tailnet everything.
  - The manifest and `static/` (the empty shell and its icons) answer without the cookie: Chrome fetches the manifest without credentials, and Google's WebAPK server downloads the icons.
  - A path, not the root of charras-server or a name of its own: with its service worker and manifest scoped to `/assistant/`, the installed app claims only that folder, where a `/`-scoped one would take the whole host on Android (home-infra SETUP §17, §20). Funnel strips `/assistant` before forwarding, so every URL in `static/` is relative and the app works at `/` on the tailnet as well. The cookie has no Path: the browser scopes it to the folder it sees.
- **Measured** through the public address (176.58.90.145, from charras): without the key, 401 on the page, the API, `/health`, `/tts/`, `/recordings/`, `sw.js`; manifest and icons 200; the link → 302 → page 200, cookie scoped to `/assistant/`, API 200, `/wiki/` still 401. Headless Chromium: service worker scope `/assistant/`, `Page.getInstallabilityErrors` empty, 50 exchanges shown, no failed request. The other Funnel paths answered as before.
- **Trade-off**: Anyone holding the link commands the house and hears the recordings, from anywhere; rotating the key (edit `/etc/voice-assistant.env`, restart) locks every device out until it opens the new link. Apps on the same origin (wiki, kiné…) could send requests with the cookie; they are ours. `…/assistant` without its final slash loads the page with broken relative URLs: Funnel forwards both as `/`, so the server cannot redirect it.
