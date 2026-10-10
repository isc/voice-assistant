# Voice Assistant

Custom voice assistant replacing Home Assistant's voice pipeline. Runs on ESPHome hardware (ESP32-S3 with wake word), fully local STT/TTS, with cloud or local LLM.

## Features

### Voice pipeline
- Wake word detection via ESPHome, automatic end-of-speech detection (Silero VAD); no transcription when nobody spoke
- Local STT: Parakeet TDT 0.6B v3, via onnx-asr on Linux/CUDA or MLX on Apple Silicon
- Local TTS: Kokoro-82M with French phonemization (misaki/espeak), sentence by sentence
- Continuous conversation: follow-ups without the wake word, closed cleanly on "merci", "bonne nuit"; context kept 10 minutes

### Home automation
- Home Assistant control via REST API: lights, switches, covers, climate, entity states
- Door locks: locked by voice ("ferme la porte à clé") and read ("la porte est verrouillée ?"), never unlocked
- Fuzzy entity resolution scoped by room ("la lumière du salon"), room groups ("éteins tout", "ferme les volets des enfants")
- Roller shutters and roof windows told apart, partial positions ("ouvre les volets à 50%")
- Sensor search by keyword ("quelle est la température de la piscine ?")
- Never reports an action that no function call performed

### Media
- French web radios (France Inter, FIP, RTL, TSF Jazz…) with volume control ("plus fort", "mets le son à 40")
- Spotify: artists, tracks, albums, the user's playlists, Liked Songs ("mets du Daft Punk", "mets ma playlist du matin"), with pause, skip, "c'est quoi cette chanson ?"
- On the voice device's own speaker or any Home Assistant speaker (Google Cast, Sonos…); "mets la musique sur la Sonos" moves what is playing
- Pictures from the web on a Google Nest Hub ("montre-moi la tour Eiffel", "une autre"): Wikipedia, else Openverse photos (free, no API key)

### Timers, calendar, weather
- Timers and alarms with ESP native timer events (LED ring, sound) and a spoken announcement
- Google Calendar: query events and create new ones
- Weather and 5-day forecast via Open-Meteo (free, no API key), at home (location from Home Assistant) or in any city

### LLM
- Any OpenAI-compatible endpoint: cloud (GPT-5.4 Nano tested) or local (Ollama / llama.cpp; Gemma 4 12B in use)
- Multi-round tool loop, concurrent tool calls ("éteins la lumière et dis-moi la météo")
- Text fallback parser for local models that output tool calls as plain text

### Infrastructure
- Linux server in production (systemd, CUDA, Ollama); macOS for development (launchd, secrets in Keychain)
- One server can serve a voice device at another site: `TTS_BASE_URL` points it at a relay
- Automatic ESP reconnection; hot reload of Home Assistant entities via SIGHUP
- Time zone taken from Home Assistant, so the server itself may run in UTC
- Web debug UI with exchange log, timings and text input; `/health` endpoint

## Architecture

```
ESP (wake word) → audio stream → voice_server.py
                                   ├─ stt.py (Parakeet, ONNX or MLX) → transcript
                                   ├─ llm.py (OpenAI-compatible API) → tool calls or text
                                   │    ├─ ha_client.py → Home Assistant REST API
                                   │    ├─ timer.py → in-memory timers with ESP events
                                   │    ├─ weather.py → Open-Meteo API
                                   │    ├─ calendar_client.py → Google Calendar API
                                   │    ├─ spotify.py → go-librespot + Spotify Web API
                                   │    └─ images.py → Wikipedia / Openverse, cast to a Nest Hub
                                   ├─ tts.py (Kokoro-82M) → WAV audio
                                   └─ web_ui.py → debug UI on :8888
Speakers fetch /tts/*.wav, radio streams and /spotify/stream.mp3 over HTTP.
```

## Getting started

See [INSTALL.md](INSTALL.md) for setup instructions.

## Documentation

- [INSTALL.md](INSTALL.md) — installation and configuration
- [CLAUDE.md](CLAUDE.md) — architecture details, pipeline flow, design patterns
- [DECISIONS.md](DECISIONS.md) — architecture decision log
