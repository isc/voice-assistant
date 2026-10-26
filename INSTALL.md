# Installation

The production setup is a Linux server with an NVIDIA GPU (Ubuntu 24.04, the LLM served by
Ollama, the server run by systemd). A Mac with Apple Silicon works for development.

## Prerequisites

- Python 3.11+
- Linux with an NVIDIA GPU (STT through onnx-asr on CUDA), or macOS on Apple Silicon (STT through MLX)
- espeak-ng (`apt install espeak-ng` / `brew install espeak-ng`)
- Home Assistant instance (optional, for device control; also gives the home's location and time zone)
- ESPHome device with wake word support (ESP32-S3, e.g. Home Assistant Voice PE)
- An OpenAI-compatible LLM endpoint: Ollama or llama.cpp locally, or a cloud API

All configuration goes through environment variables, listed in
[CLAUDE.md](CLAUDE.md#configuration-env-vars); [.env.example](.env.example) is the sample.

## Linux server

### Install

```bash
sudo apt install python3-venv espeak-ng
sudo locale-gen fr_FR.UTF-8                           # dates are spoken in French
sudo git clone https://github.com/isc/voice-assistant /opt/voice-assistant
cd /opt/voice-assistant
sudo python3 -m venv venv
sudo venv/bin/pip install -r requirements-linux.txt -c constraints-linux.txt   # adds onnx-asr for the STT
```

`constraints-linux.txt` pins the versions charras runs (`pip freeze`, 2026-10-11): requirements are
unpinned, and a fresh install would otherwise pull versions this code never ran with
(aioesphomeapi 46, kokoro-onnx 0.6 on that date). Refresh it with `venv/bin/pip freeze` after an upgrade.

The STT then runs on the CPU, with the `onnxruntime` that kokoro-onnx pulls in. For the GPU, swap
it for the CUDA build and its CUDA 13 libraries:

```bash
sudo venv/bin/pip uninstall -y onnxruntime
sudo venv/bin/pip install onnxruntime-gpu nvidia-cuda-runtime nvidia-cublas nvidia-cufft nvidia-curand nvidia-cudnn-cu13
NV=/opt/voice-assistant/venv/lib/python3.12/site-packages/nvidia
printf '%s\n' "$NV/cu13/lib" "$NV/cudnn/lib" | sudo tee /etc/ld.so.conf.d/voice-assistant-cuda.conf
sudo ldconfig
```

Both builds install the same `onnxruntime` module, so a later `pip install -r requirements-linux.txt`
brings the CPU one back and the STT silently drops to the CPU; restore the GPU one with
`sudo venv/bin/pip install --force-reinstall --no-deps onnxruntime-gpu`. The startup log lists the
ONNX execution providers.

The LLM, Ollama in Docker. With several cards, replace `--gpus all` by the voice server's card,
`--gpus device=<UUID>` from `nvidia-smi -L`, or the model spreads over all of them (charras: home-infra
`charras/SETUP.md` §7 and §23):

```bash
docker run -d --name ollama --gpus all --restart unless-stopped \
  -p 11434:11434 -v /opt/ollama:/root/.ollama ollama/ollama
docker exec ollama ollama pull gemma4:12b
# Ollama gives it a 4096-token context on a 16 GB card; the prompt of a house with ~70 devices
# outgrows it. A derived model with 8192 shares the weights (+0.3 GB of VRAM):
docker exec ollama sh -c 'printf "FROM gemma4:12b\nPARAMETER num_ctx 8192\n" > /tmp/Modelfile-8k && ollama create gemma4:12b-8k -f /tmp/Modelfile-8k'
```

### Configure

Config and secrets go in `/etc/voice-assistant.env`, readable by root only: fill in the ESP and
Home Assistant settings and uncomment the Ollama lines.

```bash
sudo install -m 600 .env.example /etc/voice-assistant.env
sudoedit /etc/voice-assistant.env
```

And the unit, `/etc/systemd/system/voice-assistant.service`:

```ini
[Unit]
Description=Voice Assistant orchestrator (STT/LLM/TTS + Home Assistant)
After=network-online.target docker.service
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=/opt/voice-assistant
EnvironmentFile=/etc/voice-assistant.env
ExecStart=/opt/voice-assistant/venv/bin/python voice_server.py
ExecReload=/bin/kill -HUP $MAINPID
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now voice-assistant
```

Deploying, reloading and logs: see [CLAUDE.md](CLAUDE.md#deployment-and-service-management).

## Mac (development)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-mac.txt   # adds parakeet-mlx for the STT
cp .env.example .env                  # ESP and Home Assistant addresses
```

Secrets go in the macOS Keychain, where `run.sh` reads them:

```bash
security add-generic-password -a "$USER" -s voice-assistant-ha-token -w "YOUR_HA_TOKEN"
security add-generic-password -a "$USER" -s voice-assistant-openai-key -w "YOUR_OPENAI_KEY"  # optional, for cloud LLM
```

Without a cloud key, `run.sh` also starts a local llama-server. Run the server in the foreground
with `./run.sh`, or as a launchd service with `./ctl.sh install` (once), then the `ctl.sh`
commands in [CLAUDE.md](CLAUDE.md#deployment-and-service-management).

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
From a phone, without Tailscale: Tailscale Funnel publishes it at a path of its own, behind a key
given once per device (AD-048).

```bash
# once: the key, in the service's environment
echo "WEB_UI_KEY=$(python3 -c 'import secrets; print(secrets.token_urlsafe(24))')" | sudo tee -a /etc/voice-assistant.env >/dev/null
sudo systemctl restart voice-assistant
# funnel, never serve: on charras `serve` turns Funnel off for the whole node (home-infra SETUP §10)
sudo tailscale funnel --bg --set-path=/assistant http://127.0.0.1:8888
# the link to open on the phone, then « Installer l'application »
echo "https://charras-server.tailcef78d.ts.net/assistant/?k=$(sudo sed -n 's/^WEB_UI_KEY=//p' /etc/voice-assistant.env)"
```

A new key (edit the line, restart) locks out every device until it opens the new link.
