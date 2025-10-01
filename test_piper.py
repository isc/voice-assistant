#!/usr/bin/env python3
"""Test Piper TTS isolément"""

from piper import PiperVoice
import wave
import numpy as np

print("🔊 Test Piper TTS")

# Charger le modèle
print("📦 Chargement modèle fr_FR-gilles-low...")
voice = PiperVoice.load(
    '/tmp/piper_models/fr_FR-gilles-low.onnx',
    '/tmp/piper_models/fr_FR-gilles-low.onnx.json'
)

print(f"✅ Modèle chargé: {voice.config.sample_rate}Hz")

# Texte à synthétiser
text = "Bonjour, ceci est un test de Piper TTS en français."
print(f"📝 Texte: {text}")

# Synthétiser
print("🔊 Synthèse en cours...")
all_audio = []

for i, audio_chunk in enumerate(voice.synthesize(text), 1):
    print(f"  📦 Chunk {i}: {len(audio_chunk.audio_float_array)} samples")
    all_audio.append(audio_chunk.audio_float_array)

# Concaténer
audio_float = np.concatenate(all_audio)
print(f"✅ Audio total: {len(audio_float)} samples ({len(audio_float)/voice.config.sample_rate:.2f}s)")

# Convertir en int16
audio_int16 = (audio_float * 32767).astype(np.int16)

# Sauvegarder
output_file = "/tmp/test_piper_output.wav"
with wave.open(output_file, 'wb') as wav_file:
    wav_file.setnchannels(1)
    wav_file.setsampwidth(2)
    wav_file.setframerate(voice.config.sample_rate)
    wav_file.writeframes(audio_int16.tobytes())

import os
size = os.path.getsize(output_file)
print(f"✅ WAV créé: {output_file} ({size} bytes)")
print(f"🎵 Joue avec: afplay {output_file}")
