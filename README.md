# Voice Assistant Server Python

Serveur Python qui remplace Home Assistant pour le pipeline vocal des devices ESPHome Voice Assistant.

Utilise `aioesphomeapi` pour une intégration native complète.

## Installation

```bash
cd voice-server-python
pip install -r requirements.txt
```

## Configuration

1. **Modifiez l'IP de votre ESP** dans `voice_server.py`:
```python
await self.connect_to_device('192.168.1.100', 'your-esp-password')
```

2. **Configurez le mot de passe** (même que dans votre config ESPHome):
```python
'password': 'your-esp-password'
```

## Pipeline Voice Assistant

Le serveur gère automatiquement tous les événements ESPHome:

### Événements reçus de l'ESP:
- `VOICE_ASSISTANT_RUN_START` (1) - Pipeline démarré
- `VOICE_ASSISTANT_WAKE_WORD_START/END` (9/10) - Wake word détecté
- `VOICE_ASSISTANT_STT_VAD_START/END` (11/12) - Détection de voix
- Audio brut pour traitement

### Événements envoyés à l'ESP:
- `VOICE_ASSISTANT_STT_START/END` (3/4) - STT en cours/terminé
- `VOICE_ASSISTANT_INTENT_START/END` (5/6) - Intent en cours/terminé
- `VOICE_ASSISTANT_TTS_START/END` (7/8) - TTS en cours/terminé
- `VOICE_ASSISTANT_TTS_STREAM_START/END` (98/99) - Streaming audio
- `VOICE_ASSISTANT_RUN_END` (2) - Pipeline terminé

## Services à implémenter

### 1. Speech-to-Text (`speech_to_text()`)

Remplacez le placeholder par votre service:

```python
# OpenAI Whisper
import openai
transcription = await openai.Audio.atranscribe("whisper-1", audio_bytes)

# Google Speech-to-Text
from google.cloud import speech
client = speech.SpeechClient()
response = client.recognize(config=config, audio=audio)

# Vosk (local)
import vosk
rec = vosk.KaldiRecognizer(model, 16000)
rec.AcceptWaveform(audio_bytes)
```

### 2. LLM + MCP (`process_with_llm()`)

Intégrez votre stack:

```python
# OpenAI
response = await openai.ChatCompletion.acreate(
    model="gpt-4",
    messages=[{"role": "user", "content": text}]
)

# Claude
import anthropic
client = anthropic.AsyncAnthropic()
response = await client.messages.create(
    model="claude-3-sonnet-20240229",
    messages=[{"role": "user", "content": text}]
)

# MCP Servers
# Connecter vos serveurs MCP ici
```

### 3. Text-to-Speech (`text_to_speech()`)

```python
# ElevenLabs
from elevenlabs import generate, save
audio = generate(text=text, voice="Bella")

# Google Text-to-Speech
from google.cloud import texttospeech
client = texttospeech.TextToSpeechClient()
response = client.synthesize_speech(input=input_text, voice=voice, audio_config=audio_config)

# Azure Speech
import azure.cognitiveservices.speech as speechsdk
synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)
result = synthesizer.speak_text_async(text).get()
```

## Démarrage

```bash
python voice_server.py
```

## Logs

Le serveur affiche tous les événements:

```
🚀 Voice Assistant Server démarré
✅ Connecté à 192.168.1.100
🎤 Voice assistant supporté (flags: 15)
📨 Événement voice assistant: 9 - wake_word_start
🔊 Wake word détecté: 'okay nabu'
📨 Événement voice assistant: 11 - vad_start
👂 Début d'écoute (VAD start)
📨 Événement voice assistant: 12 - vad_end
⏹️ Fin d'écoute (VAD end) - traitement audio en cours...
🎵 Audio reçu: 48000 bytes
🔄 Démarrage pipeline vocal complet
🎤 STT: Traitement audio...
📝 Transcription: "Allume la lumière du salon"
🧠 LLM: Traitement de "Allume la lumière du salon"
🤖 Réponse LLM: "J'allume la lumière du salon."
🗣️ TTS: Génération audio pour "J'allume la lumière du salon."
🔊 Audio généré: 96000 bytes
📤 Envoi audio à l'ESP: 96000 bytes
🎵 Audio envoyé (simulation)
✅ Pipeline terminé avec succès
```

## Debugging

Activez le debug dans la config:
```python
'debug': True
```

Pour voir tous les messages protobuf:
```python
logging.getLogger('aioesphomeapi').setLevel(logging.DEBUG)
```

## Architecture

```
ESP32 Voice Device  ←→  Python Server (aioesphomeapi)  ←→  Vos Services
                                                             ├─ STT Service
                                                             ├─ LLM + MCP
                                                             └─ TTS Service
```

Le serveur agit comme un "faux Home Assistant" spécialisé uniquement dans le voice assistant.