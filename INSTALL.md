# Installation

## Prerequisites

- Python 3.11+
- macOS (Apple Silicon) for MLX-based STT
- espeak-ng (`brew install espeak-ng`)
- Home Assistant instance (optional, for device control)
- ESPHome device with wake word support (ESP32-S3)

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

## Google Calendar (optional)

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Create a project (or select existing)
3. Enable **Google Calendar API**
4. Go to **Credentials** > **Create Credentials** > **OAuth client ID**
5. Choose **Desktop App** as application type
6. Download the JSON and save it as `client_secret.json` in the project directory

Then run the setup script (requires a browser):

```bash
python setup_calendar.py
```

This opens a browser for Google sign-in, then saves the token to `token.json`. The server will automatically detect it on next start.

## Web UI

Open `http://localhost:8888/` for the debug interface (exchange log, timings, text input).
