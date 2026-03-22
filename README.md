# Voice Assistant Server

Custom voice assistant replacing Home Assistant's voice pipeline. Runs on ESPHome hardware (ESP32-S3 with wake word), fully local STT/TTS, with cloud or local LLM.

See [CLAUDE.md](CLAUDE.md) for architecture details and [DECISIONS.md](DECISIONS.md) for design decisions.

## Quick start

### Prerequisites

- Python 3.11+
- macOS (Apple Silicon) for MLX-based STT
- espeak-ng (`brew install espeak-ng`)
- Home Assistant instance (optional, for device control)

### Install

```bash
pip install -r requirements.txt
```

### Store secrets in macOS Keychain

```bash
security add-generic-password -a "$USER" -s voice-assistant-ha-token -w "YOUR_HA_TOKEN"
security add-generic-password -a "$USER" -s voice-assistant-openai-key -w "YOUR_OPENAI_KEY"  # optional, for cloud LLM
```

### Run

```bash
# Direct launch (reads secrets from Keychain)
./run.sh

# Or via launchd service
./ctl.sh install   # one-time setup
./ctl.sh start
./ctl.sh status
./ctl.sh logs
```

### Web UI

Open `http://localhost:8888/` for the debug interface (exchange log, timings, text input).

## Configuration

All configuration via environment variables. See [CLAUDE.md](CLAUDE.md#configuration-env-vars) for the full list.

Key variables: `ESP_HOST`, `HA_URL`, `HA_TOKEN`, `LLM_URL`, `LLM_API_KEY`, `LLM_MODEL`.
