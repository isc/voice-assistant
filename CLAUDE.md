# Voice Assistant Server

Custom voice assistant replacing Home Assistant's voice pipeline. Runs on ESPHome hardware (ESP32-S3 with wake word), fully local STT/TTS, with cloud or local LLM.

Production runs on a Linux server ([Deployment](#deployment-and-service-management)); a Mac remains usable for development.

## Architecture

```
ESP (wake word) ─┐
Phone (web app) ─┴→ audio stream → voice_server.py / web_ui.py
                                   ├─ stt.py (Parakeet: onnx-asr on CUDA, MLX on a Mac) → transcript
                                   ├─ llm.py (OpenAI-compatible: Ollama, or cloud) → tool calls or text
                                   │    ├─ ha_client.py → Home Assistant REST API
                                   │    ├─ weather.py → Open-Meteo API (free, no key)
                                   │    ├─ places.py → Photon + Nominatim (OpenStreetMap, free, no key): shops, opening hours
                                   │    ├─ calendar_client.py → Google Calendar API (OAuth2)
                                   │    ├─ spotify.py → go-librespot (playback) + Spotify Web API (search)
                                   │    └─ images.py → Wikipedia / Openverse (free, no key) → the web app, or cast to a Nest Hub
                                   ├─ tts.py (Kokoro-82M) → FLAC file
                                   ├─ web_ui.py → the web app on :8888 (static/index.html)
                                   └─ push.py → Web Push: a timer set from a phone rings there
```

## Files

| File | Purpose |
|---|---|
| `voice_server.py` | Main orchestrator: ESP connection, VAD, pipeline, signal handling, function execution |
| `ha_client.py` | Home Assistant client: entity discovery (a `select` only when it sets a heating's mode, AD-071, with its setpoint sensor, AD-075; an alarm, its maker's switches never, AD-083), fuzzy name resolution, room groups, service calls, an alarm armed or disarmed until it says so (AD-083) |
| `llm.py` | LLM integration: chat completion, tool definitions, text-to-tool-call fallback parser |
| `timer.py` | Timer manager: in-memory timers with asyncio scheduling, ESP native timer events; each remembers the browser that set it and whether its satellite shows it (AD-070) |
| `push.py` | Web Push to the web app's browsers (RFC 8291 encryption, RFC 8292 VAPID, over `cryptography`): the server's key and each browser's subscription in `push.json` (600, gitignored, made at first start) |
| `tts.py` | Kokoro-82M TTS with French G2P (misaki/espeak), 48 kHz mono FLAC through ffmpeg (the Voice PE decodes no WAV) |
| `stt.py` | Parakeet TDT 0.6B v3: onnx-asr on Linux/CUDA (server) or parakeet-mlx on Apple Silicon (Mac dev), picked by `STT_BACKEND` |
| `onnx_gpu.py` | ONNX Runtime on the GPU shared with Ollama: one thread for STT and TTS runs, CUDA arenas that give their memory back after each run (AD-053), the CPU when CUDA fails (AD-055) |
| `weather.py` | Open-Meteo weather API: geocoding, current conditions, 5-day forecast (French) |
| `places.py` | Shops and places from OpenStreetMap, nearest to the home first: address, phone, the week's opening hours read by opening-hours-py; an event's place as its address (`address_of`, AD-072) |
| `calendar_client.py` | Google Calendar client: OAuth2, event queries, event creation, and changes to an event in place (`update_event`: the conversation's own first, AD-072), prompt summary |
| `setup_calendar.py` | One-time OAuth setup script for Google Calendar (requires browser) |
| `spotify.py` | Spotify: go-librespot API, Web API search, and the MP3 stream of go-librespot's PCM FIFO served at `/spotify/stream.mp3` |
| `images.py` | Pictures for the web app or a screen: Wikipedia article's lead image, else an Openverse photo; the phone or the screen loads the URL itself |
| `recordings.py` | Each voice request kept as WAV + JSON (transcript, reply, tool calls, timings) for 90 days, root-only |
| `web_ui.py` | Web app routes: the phone as a satellite (`/api/listen`, a WebSocket), typed requests, shortcuts run again without the LLM (`/api/redo`, AD-065) and offered unless already in effect by HA's state (`/api/in-effect`, AD-066), a door unlocked or an alarm disarmed by a tap on the button its request shows (`/api/confirm`, AD-068, AD-083), timers (which are the asking browser's) and music, push subscriptions (`/api/push`, AD-070), exchange log with persistence (`exchanges.json`) |
| `static/index.html` | The web app, a PWA: voice and typed requests, this phone's conversation (AD-060) with the pictures it asked for (AD-069) and the buttons that unlock or disarm (AD-068, AD-083), timers (a chime when its own reach 0, AD-070) and music now, shortcuts, the history and technical details on demand |
| `static/sw.js` | The PWA's service worker: the shell offline, and a timer's push shown as a notification (AD-070) |
| `vad.py` | End of speech by Silero VAD, for the satellite's recordings and the phone's |
| `requirements-linux.txt` / `requirements-mac.txt` | Base deps (`requirements.txt`) plus the STT backend: onnx-asr (server) or parakeet-mlx (Mac) |
| `run.sh` | Mac dev launcher: sources `.env`, reads secrets from the macOS Keychain, starts llama-server unless a cloud key is set |
| `ctl.sh` | Mac dev service control via launchd; `./ctl.sh install` generates the (gitignored) plist |
| `test_e2e.py` / `test_hard_cases.py` | Dry runs (AD-047) judged on their tool calls: 23 everyday requests; the requests DECISIONS.md records a model getting wrong, several runs each (AD-081). `VOICE_SERVER` points them at a test server |
| `test_scenarios.py` | Conversations replayed with the real LLM against fake homes whose device states each scenario sets, the server's own code running every call (AD-084): what a dry run's « OK » cannot show, a speaker offline, a state read. The homes' devices come from the live server's `/api/snapshot` (this machine only) |
| `review/` | The nightly review (AD-074): Claude Code reads the day's interactions, fixes the obvious bugs (pushed and deployed), emails Ivan proposals that open a Desktop session; then runs the model-backed tests and emails a turn that regressed (AD-085, `review/README.md`) |
| `firmware/README.md` | How the Voice PE firmware is built and flashed (our build of 26.9.0 with ESPHome 2026.9.1, AD-054) |

## Pipeline flow (voice)

1. ESP detects wake word → sends audio stream
2. Silero VAD detects end of speech (31 frames / ~1s silence)
3. STT: Parakeet transcribes audio to French text
4. LLM: sends a static system prompt (instructions, family, entity list) + tool definitions + conversation history + a short context message (date/time, timers, music) + the transcript
5. If tool_calls → execute each, loop back to LLM with results + tools (up to 3 rounds) until text response
6. If text response → check the fallback parser for a tool call written as text (`fn({...})`)
7. Final LLM round formulates natural spoken response from tool results
   - Steps 4-7 past 5 s: « Un petit instant. » meanwhile, by the satellite's media player, which keeps the run open, or the app's `wait` message (AD-079)
8. TTS: Kokoro generates FLAC (ffmpeg), played via announcement API with `start_conversation=True`
9. ESP automatically starts listening for follow-up (no wake word needed)
10. If no speech within 5s → conversation ends. If `end_conversation` tool was called → no follow-up.

## Key design patterns

- **Sites** (`sites.py`): one server for several homes, each with its satellite, Home Assistant, TTS URL, conversation, timers and last speaker. The ESP handlers are bound to their site, and a voice request activates it; the other state (`ha_client`, `conversation`…) is read from the active site. One conversation at a time: a satellite that wakes while another site is mid-conversation gets an error. Text requests take an optional `"site"`; without one, a request through Funnel from a home's internet address (`PUBLIC_HOST`) serves that home. `lock`, `unlock`, `arm_alarm`, `disarm_alarm`, `set_heating`, `get_state` and `get_sensor` may name another home (`home`, or its name in the entity, room or query): the call runs serving that home (`_at`), the request's own restored after (AD-068, AD-071, AD-083).
- **Room groups**: `room_groups` in `config.local.json`, shared by all sites (matched loosely: « Chambre Zoé » finds « Chambre de Zoé »), maps group names ("enfants") to lists of HA areas; "tout", "toute la maison" and "partout" are built in and mean every room, for an action only when the request says so (tout, tous, partout, la maison…); with no room, a generic name (« les volets ») gets a question, which room (AD-058). For a group, all entities of the matching domain are returned (only the named one for a generic name like "lumières").
- **Entity resolution**: fuzzy matching on normalized friendly_name (accents stripped, stopwords removed), scoped by room/area.
- **Multi-round tool loop**: LLM can return tool_calls across multiple rounds (max 3). After executing tools, results are sent back with tools still available so the LLM can make additional calls; a call already run this turn reuses its result, and a round whose only call is `end_conversation` gets no second round. Tool calls and follow-up flags of a request live in a per-request object (`_Request`, ContextVar), so concurrent requests stay apart.
- **Conversation history**: per site and per device (the satellite, each browser of the web app by its id, AD-061), last 5 exchanges with their tool calls and results (capped), expires after 10 min inactivity (`CONVERSATION_TIMEOUT`). Dry runs keep a conversation of their own (`Site.dry_run`) and never read, extend or end the live one; `/api/reset-conversation` clears only theirs (AD-047). Their log lines start with `[dry run]`, and the nightly review's journal leaves them out (AD-085).
- **GPU memory**: the server shares charras's top 16 GB card with Ollama, which keeps ~10.5 GB of it (`LLAMA_ARG_SWA_FULL=1`; other GPU jobs go to the bottom card, home-infra `charras/SETUP.md` §23). STT and TTS run on one thread (`onnx_gpu.run`), and the arenas of Kokoro and Parakeet's encoder shrink after each run: the voice server stays at ~4.2 GB, 4.9 GB during a run. Left to grow, the arenas reached 7.2 GB and the card ran out (AD-053). A model that finds no room at startup loads on the CPU (STT 5×, TTS 10× slower) with an ERROR in the log and `degraded` in `/health`, and the server tries CUDA again every minute between requests (AD-055).
- **Static system prompt**: date/time, timers and music go in a context message just before the request, not in the system prompt, so that Ollama reuses its prompt cache (a new minute re-read ~3.6K tokens, ~1.9 s). The server pins the model with a native `keep_alive: -1` at each request. A wake word, or the phone's microphone, has the model read the request's system prompt and history while the request is spoken (`warm_llm`), loading it first if a job had unloaded it, as ddmr's OCR does: a first request then waited 7 to 15 s for the LLM (AD-073).
- **The phone as a satellite** (AD-059): the web app streams its microphone (16 kHz PCM, AudioWorklet) over `/api/listen`; the server runs the satellite's VAD, STT, LLM and TTS, and the phone plays the reply then listens again for a follow-up. Browser noise suppression off: it garbled Parakeet's input. WebSocket handshakes must carry this server's `Origin` (`cross_site_guard`), which AD-056's JSON rule does not cover. Typed requests (`/api/send`) get no TTS. Tool results saying nothing was done are flagged `failed` in the log (`_NOTHING_DONE`), and the app shows them. A picture asked from the app shows in its conversation (`image` on the call in the log), and on the home's Nest Hub only when the request names the screen or its room (`show_image`'s `room`); the satellite's go to the Nest Hub (AD-069).
- **Timers on the phone** (AD-070): a timer remembers the browser that set it (`Timer.client`); when it rings, the server pushes to that browser alone (Web Push, `push.py`), and the page, if shown, chimes as the card reaches 0. The satellite shows and rings it too, unless the request came through Funnel from another connection than the home's (`_Request.address`, checked when the timer is set): then it is the phone's alone (`Timer.satellite` false), and only that browser sees it (`get_timers(client)`: prompt, cancel, the strip). Push needs HTTPS (Funnel) and, on an iPhone, the app installed; the bell on the card or the settings' switch asks.
- **Continuous conversation**: after TTS playback, ESP starts listening for follow-up without wake word. 500ms audio skip avoids TTS echo. `end_conversation` tool lets the LLM end cleanly on "merci", "bonne nuit", etc.
- **Text fallback parser**: catches when the LLM outputs tool calls as plain text instead of structured JSON (3 patterns: JSON args, quoted string, Python kwargs; never in dry runs).
- **Ducking**: from wake word to transcription, the HA speakers playing at the site are turned down to a fifth of their volume (the Voice PE ducks only its own speaker), and back up once the request is heard; the media tools wait for it, so that « monte le son » starts from their real volume (AD-049).
- **Spotify as a radio station**: go-librespot (separate service) plays into a FIFO; `spotify.py` paces it at real time, pads gaps with silence, encodes MP3 and serves `/spotify/stream.mp3` next to `/tts/`. Speakers (ESP or HA) play that URL; one stream, so one speaker at a time. Search results are ranked again on the request's words, number and key (« do mineur » = « C minor »; « numéro 2 » the piece's own number, not « Book 2 »), from up to three searches (as said, as the catalog writes it, composer first), a track the account liked first among equals (AD-088), and a screen gets the title, artist and cover of what was asked, not updated at each track (AD-057). A track asked over, go-librespot goes on to ones it chooses; meanwhile the context message names the track asked, so that « remets le prélude » finds it again once its turn has left the history (AD-087).
- **The heating is a mode** (AD-071): Charras's Ökofen boiler is three HA template `select`s (Radiateurs, Plancher, Plancher 2: Off, Auto, Confort, Réduit; home-infra `charras/oekofen/`), listed as « chauffage », which with no room means all three. `set_heating` sets them (`select.select_option`), awaited as a lock is, from either home; « allume » is Auto, the boiler's program, held in code when the request names no mode (`asks_heating_on`, AD-082). Each circuit's setpoint is a sensor that names its select (`circuit`, `HAClient.setpoints`): `get_state`, and `get_sensor` on « chauffage », say it with the mode (AD-075).
- **Locks lock; unlocking takes a tap in the web app**: `lock` locks. `unlock` only proposes: from the web app it puts a « Déverrouiller » button on its call (`confirm` in the log), which `/api/confirm` honours once, for 2 minutes, from the browser that asked; from the satellite it refuses, in dry runs too. The model never unlocks: `HAClient.call_service` refuses `lock.unlock` and `lock.open`, and `HAClient.unlock` is called by `confirm` alone (AD-041, AD-068).
- **The alarm arms by voice; disarming takes a tap, the same way** (AD-083): an `alarm_control_panel` is « alarme », the home's (Charras's Somfy, through SomfyProtect2MQTT). `get_state` says its state in French. `arm_alarm` arms it in a mode its `supported_features` offers (absence, nuit…; none guessed), from disarmed or up to a mode that watches the whole house, never down (that is a disarm first), and waits until it says it is armed (`HAClient.arm`, up to 15 s). `disarm_alarm` proposes a « Désarmer » button as `unlock` does, and `HAClient.disarm`, called by `confirm` alone, waits until it is disarmed. `call_service` refuses `alarm_disarm` and `alarm_trigger`, and the switches its maker made (its siren, its sensors' settings) are never entities. Another home's alarm is reached by naming it, never guessed for a home that has none.
- **Web UI on the tailnet, or the internet with a key**: port 8888 also answers on the LAN, where speakers fetch `/tts/` and `/spotify/stream.mp3`; everything else (UI, API, recordings) answers localhost and the tailnet. Tailscale Funnel publishes it at `https://charras-server.tailcef78d.ts.net/assistant/`, where a request (told apart by the proxy's headers) needs the `WEB_UI_KEY` cookie, dropped by `?k=<key>` (`web_ui.access_control`, AD-048). Hence relative URLs in `static/`: the app lives at `/` and at `/assistant/`. The microphone needs HTTPS, so voice works through Funnel only. Pages of other sites open in a browser there cannot use it: POSTs take a JSON body only, and the Host must be an IP address, `localhost`, the machine's name or a `*.ts.net` name (`web_ui.cross_site_guard`, AD-056).
- **Secrets**: the server only reads env vars: `/etc/voice-assistant.env` on the server, the Keychain (`voice-assistant-ha-token`, `voice-assistant-openai-key`) through `run.sh` on a Mac. Its Web Push key it makes itself, in `push.json` (AD-070).

## Configuration (env vars)

| Variable | Default | Description |
|---|---|---|
| `SITES` | (empty) | Comma-separated homes served by this server (`paris,charras`). Each then reads the satellite and Home Assistant settings below with its name as prefix (`PARIS_HA_URL`, `CHARRAS_ESP_HOST`…); `ESP_PORT`, `ESP_PASSWORD`, `ESP_NOISE_PSK`, `WAKE_WORDS` may stay unprefixed, shared. Empty: one site, unprefixed settings |
| `ESP_HOST` | (empty) | ESPHome device IP or `name.local` (resolved by mDNS); empty for a site without satellite |
| `ESP_PORT` | `6053` | ESPHome native API port |
| `ESP_PASSWORD` | (empty) | Legacy password auth |
| `ESP_NOISE_PSK` | (empty) | Noise encryption key (preferred) |
| `WAKE_WORDS` | (empty) | Satellite wake words to keep active, by id (`okay_nabu,hey_jarvis`). Only this server can set them (AD-051); it does at each connection, which keeps them past a reflash or a new satellite. Empty leaves the satellite's own |
| `LLM_URL` | `http://localhost:8080/v1/chat/completions` (llama-server) | OpenAI-compatible endpoint. Server: Ollama, `http://localhost:11434/v1/chat/completions` |
| `LLM_API_KEY` | (empty) | If set, enables cloud mode (bearer auth, provider's default reasoning). Empty on the server |
| `LLM_MODEL` | (empty) | Model name: `gemma4:12b-8k` on the server's Ollama (gemma4:12b with an 8192-token context, see INSTALL.md), e.g. `gpt-5.4-nano` in the cloud; empty lets llama-server use its only model |
| `LLM_REASONING_EFFORT` | `none` local, unset cloud | Reasoning budget for thinking models. Keep `none` locally — thinking costs ~10s/round and empties `content` |
| `STT_BACKEND` | `auto` | Parakeet backend: `onnx` (Linux/CUDA via onnx-asr, set on the server), `mlx` (Apple), or `auto` (MLX if installed, else ONNX) |
| `HA_URL` | (required) | Home Assistant URL |
| `HA_TOKEN` | (empty) | HA long-lived access token |
| `HTTP_PORT` | `8888` | HTTP server port (TTS files + web UI) |
| `TTS_BASE_URL` | `http://<LAN IP>:<HTTP_PORT>/tts/` | URL the ESP fetches TTS files from, ending in `/`. Set it when the ESP reaches `/tts/` through a relay on another network |
| `PUBLIC_HOST` | (empty) | Per site: a name resolving to the home's internet address (its Freebox's `….fbxos.fr`). Through Funnel, a phone on that home's Wi-Fi gets it without choosing (its IPv4, or its IPv6 /64) |
| `CALENDAR_CREDENTIALS` | `client_secret.json` | Google OAuth credentials file path |
| `CALENDAR_TOKEN` | `token.json` | Google OAuth token file path |
| `SPOTIFY_CLIENT_ID` / `SPOTIFY_CLIENT_SECRET` | (empty) | Spotify developer app, for search (client credentials). Music is enabled when go-librespot answers |
| `SPOTIFY_LIBRESPOT_URL` | `http://127.0.0.1:3678` | go-librespot API |
| `SPOTIFY_PCM_FIFO` | `/run/go-librespot/pcm` | FIFO go-librespot writes its PCM into |
| `WEB_UI_KEY` | (empty) | Key of the web UI's link through Tailscale Funnel (`/assistant/?k=<key>`); empty, Funnel gets 401 everywhere |
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
`Server running` (`Fatal error` otherwise), or `curl -s localhost:8888/health` answers 200. There, `status: degraded` with `device: cpu` under `stt` or `tts`: that model runs on the CPU, the card was full (AD-055).

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

## Tests

Run them with the server's venv, on the server (`/opt/voice-assistant/venv/bin/python`).

- `test_unit.py`: about 3 s, no network, no model. Before every commit: `.githooks/pre-commit` runs it, once
  a clone has `git config core.hooksPath .githooks` (set on charras).
- A change to what the model reads (system prompt, tool descriptions, tool results) or to how a call finds its
  device: `test_scenarios.py` (fake homes, every call run) and `test_hard_cases.py` (dry runs on the live server),
  10 runs of the cases concerned, before and after. « Before » is a worktree at `main` running this checkout's
  `test_scenarios.py`; the counts go into the decision in `DECISIONS.md`. A case DECISIONS.md records a model
  getting wrong becomes a scenario, or a hard case when a dry run's « OK » cannot mislead it.
- Each night, after its review, `review/nightly.py` runs both, 3 times, against the code deployed, and emails Ivan
  a turn that lost 2 passes or its last one (AD-085). Not on each commit: they take the GPU the family talks to
  by day, and a turn's count moves by about 2 of 10 from run to run. A turn that moves so is listed in
  `UNSTABLE` (`review/nightly.py`), flagged only when its fall holds two nights.

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
