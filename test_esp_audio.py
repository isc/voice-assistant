#!/usr/bin/env python3
"""Test envoi audio direct à l'ESP"""

import asyncio
from aioesphomeapi import APIClient, VoiceAssistantEventType
import wave
import time

async def test_audio_stream():
    print("🔌 Connexion à l'ESP...")

    api = APIClient("192.168.1.195", 6053, "", client_info="test-audio")
    await api.connect(login=True)

    device_info = await api.device_info()
    print(f"✅ Connecté: {device_info.name}")
    print(f"🎤 Feature flags: {device_info.voice_assistant_feature_flags} (0b{bin(device_info.voice_assistant_feature_flags)[2:]})")

    # Lire le fichier WAV de test
    wav_path = "/tmp/test_piper_output.wav"
    print(f"📦 Lecture {wav_path}...")

    with wave.open(wav_path, 'rb') as wav_file:
        print(f"🎵 Format: {wav_file.getframerate()}Hz, {wav_file.getnchannels()}ch, {wav_file.getsampwidth()*8}-bit")
        audio_data = wav_file.readframes(wav_file.getnframes())

    print(f"📦 Audio: {len(audio_data)} bytes")

    # Envoyer TTS_STREAM_START
    print("📤 Envoi TTS_STREAM_START...")
    api.send_voice_assistant_event(VoiceAssistantEventType.VOICE_ASSISTANT_TTS_STREAM_START, {})
    await asyncio.sleep(0.5)

    # Stream par chunks
    print("📤 Streaming audio...")
    chunk_size = 1024
    chunk_count = 0

    for i in range(0, len(audio_data), chunk_size):
        chunk = audio_data[i:i + chunk_size]
        api.send_voice_assistant_audio(chunk)
        chunk_count += 1

        # Throttle
        chunk_duration = len(chunk) / 2 / 16000
        await asyncio.sleep(chunk_duration)

        if chunk_count % 20 == 0:
            print(f"  📊 {chunk_count} chunks...")

    print(f"✅ {chunk_count} chunks envoyés")

    # Envoyer TTS_STREAM_END
    print("📤 Envoi TTS_STREAM_END...")
    api.send_voice_assistant_event(VoiceAssistantEventType.VOICE_ASSISTANT_TTS_STREAM_END, {})

    # Attendre
    print("⏱️  Attente 5s...")
    await asyncio.sleep(5)

    print("✅ Test terminé")
    await api.disconnect()

if __name__ == "__main__":
    asyncio.run(test_audio_stream())
