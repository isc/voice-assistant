# Voice Assistant

Custom voice assistant replacing Home Assistant's voice pipeline. Runs on ESPHome hardware (ESP32-S3 with wake word), fully local STT/TTS, with cloud or local LLM.

## Features

### Voice pipeline
- Wake word detection via ESPHome, automatic end-of-speech detection (Silero VAD)
- Local STT: Parakeet TDT 0.6B v3 via MLX (Apple Silicon optimized, ~0.7s)
- Local TTS: Kokoro-82M with French phonemization (misaki/espeak)
- Continuous conversation: follow-up without repeating the wake word, with automatic timeout and clean closure ("merci", "bonne nuit")

### Home automation
- Home Assistant control via REST API: lights, covers, switches, climate
- Fuzzy entity name resolution with room scoping ("la lumière du salon")
- Room groups for multi-room commands ("éteins tout", "ferme les volets des enfants")
- Partial cover positioning ("ouvre les volets à 50%")

### Timers and alarms
- Voice-controlled timers and alarms with ESP native timer events (LED animations, sounds)
- TTS announcement when timer finishes, with follow-up conversation

### Weather
- Current conditions and 5-day forecast via Open-Meteo (free, no API key)
- Geocoding for any city, defaults to configured location

### LLM
- Cloud mode: OpenAI-compatible API (GPT-5.4 Nano tested)
- Local mode: llama.cpp with Qwen 3 4B (tool calling via thinking mode)
- Multi-round tool loop for complex requests ("éteins la lumière et dis-moi la météo")
- Text fallback parser for local models that output tool calls as plain text

### Infrastructure
- Automatic ESP reconnection with exponential backoff
- launchd service with signal-based control (start/stop/restart/reload)
- Hot-reload of Home Assistant entities via SIGHUP
- Web debug UI with exchange log, timings, and text input
- Secrets in macOS Keychain

## Architecture

```
ESP (wake word) → audio stream → voice_server.py
                                   ├─ stt.py (Parakeet MLX) → transcript
                                   ├─ llm.py (OpenAI API / llama.cpp) → tool calls or text
                                   │    ├─ ha_client.py → Home Assistant REST API
                                   │    ├─ timer.py → in-memory timers with ESP events
                                   │    └─ weather.py → Open-Meteo API
                                   ├─ tts.py (Kokoro-82M) → WAV audio
                                   └─ web_ui.py → debug UI on :8888
```

## Getting started

See [INSTALL.md](INSTALL.md) for setup instructions.

## Documentation

- [INSTALL.md](INSTALL.md) — installation and configuration
- [CLAUDE.md](CLAUDE.md) — architecture details, pipeline flow, design patterns
- [DECISIONS.md](DECISIONS.md) — architecture decision log
