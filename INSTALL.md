# Installation

## Prerequisites

- Python 3.11+
- macOS (Apple Silicon, MLX STT) or Linux with an NVIDIA GPU (ONNX STT, see below)
- espeak-ng (`brew install espeak-ng` / `apt install espeak-ng`)
- Home Assistant instance (optional, for device control; also gives the home's location and time zone)
- ESPHome device with wake word support (ESP32-S3, e.g. Home Assistant Voice PE)
- An OpenAI-compatible LLM endpoint: a cloud API, or Ollama / llama.cpp locally

## Install

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Configure

```bash
cp .env.example .env
# Edit .env with your ESP and Home Assistant IPs
```

Store secrets in macOS Keychain:

```bash
security add-generic-password -a "$USER" -s voice-assistant-ha-token -w "YOUR_HA_TOKEN"
security add-generic-password -a "$USER" -s voice-assistant-openai-key -w "YOUR_OPENAI_KEY"  # optional, for cloud LLM
```

All configuration via environment variables. See [CLAUDE.md](CLAUDE.md#configuration-env-vars) for the full list.

## Run

```bash
# Direct launch (reads secrets from Keychain, sources .env)
./run.sh

# Or via launchd service
./ctl.sh install   # one-time setup: generates plist, registers with launchd
./ctl.sh start
./ctl.sh status
./ctl.sh logs
```

## Linux server

```bash
pip install -r requirements-linux.txt   # adds onnx-asr for the STT
```

Set `STT_BACKEND=onnx`, point `LLM_URL` at the local model (e.g. Ollama's
`http://localhost:11434/v1/chat/completions` with `LLM_MODEL=gemma4:12b`), and run
`voice_server.py` as a systemd service with the same variables in an `EnvironmentFile`
(`ExecReload=/bin/kill -HUP $MAINPID` gives the hot reload of Home Assistant entities).
`run.sh`, `ctl.sh` and the Keychain are macOS only.

## Google Calendar (optional)

Run the setup script (requires a browser):

```bash
python setup_calendar.py
```

It guides you through creating a Google Cloud project, enabling the Calendar API, and authorizing access. On completion, it saves the token to `token.json`. The server will automatically detect it on next start.

## Spotify (optional)

Needs Spotify Premium, `ffmpeg`, and two pieces:

1. **[go-librespot](https://github.com/devgianlu/go-librespot)**, a headless Spotify Connect
   player, on the same host as the server. It plays into a FIFO that the server streams to the
   speakers as MP3. Minimal `config.yml`:

   ```yaml
   device_name: Assistant vocal
   audio_backend: pipe
   audio_output_pipe: /run/go-librespot/pcm   # create it with mkfifo before starting (see below)
   audio_output_pipe_format: s16le
   external_volume: true                       # the speakers own the volume
   credentials:
     type: device_auth                         # pair once with the code it logs, at spotify.com/pair
   server:
     enabled: true
     address: 127.0.0.1
     port: 3678
   ```

   Start it with the FIFO's read end held open, e.g. `sh -c 'exec 3<>/run/go-librespot/pcm; exec go-librespot …'`:
   otherwise a restart of the voice server breaks the pipe, and go-librespot closes its output
   for good (it then reports playing while writing nothing).

2. **A Spotify developer app** (developer.spotify.com/dashboard, Web API) for search: set
   `SPOTIFY_CLIENT_ID` and `SPOTIFY_CLIENT_SECRET`. The session token go-librespot hands out
   is rate-limited on the Web API, so search needs an app of its own.

Music is enabled when go-librespot answers at startup. Speakers fetch the stream at
`/spotify/stream.mp3`, next to `/tts/`: a relay set up for `TTS_BASE_URL` must serve it too.

## Web UI

Open `http://localhost:8888/` for the debug interface (exchange log, timings, text input).
