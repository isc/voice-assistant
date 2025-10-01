#!/usr/bin/env python3
"""Test TTS simple vers ESP - envoie juste "Bonjour" """

import asyncio
from piper import PiperVoice
from aioesphomeapi import APIClient, VoiceAssistantEventType
import wave
import io
import numpy as np

async def test():
    print("🔊 Test TTS simple")

    # 1. Charger Piper
    print("📦 Chargement Piper...")
    voice = PiperVoice.load(
        '/tmp/piper_models/fr_FR-gilles-low.onnx',
        '/tmp/piper_models/fr_FR-gilles-low.onnx.json'
    )

    # 2. Générer audio
    print("🎤 Génération: 'Bonjour'")
    all_audio = []
    for audio_chunk in voice.synthesize("Bonjour"):
        all_audio.append(audio_chunk.audio_float_array)

    audio_float = np.concatenate(all_audio)
    audio_int16 = (audio_float * 32767).astype(np.int16)

    print(f"✅ {len(audio_float)} samples générés")

    # 3. Créer WAV
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(audio_int16.tobytes())
    wav_buffer.seek(0)

    # 4. Connexion ESP
    print("🔌 Connexion ESP...")
    api = APIClient("192.168.1.195", 6053, "", client_info="test-simple")
    await api.connect(login=True)
    device_info = await api.device_info()
    print(f"✅ Connecté: {device_info.name}")

    # 5. Stream audio
    print("📤 TTS_STREAM_START")
    api.send_voice_assistant_event(VoiceAssistantEventType.VOICE_ASSISTANT_TTS_STREAM_START, {})
    await asyncio.sleep(0.5)

    print("📤 Streaming audio...")
    with wave.open(wav_buffer, 'rb') as wav_file:
        samples_per_chunk = 512
        chunk_count = 0

        while True:
            chunk = wav_file.readframes(samples_per_chunk)
            if not chunk:
                break

            api.send_voice_assistant_audio(chunk)
            chunk_count += 1

            # Throttle 90%
            samples_in_chunk = len(chunk) // 2
            seconds_in_chunk = samples_in_chunk / 16000
            await asyncio.sleep(seconds_in_chunk * 0.9)

    print(f"✅ {chunk_count} chunks envoyés")

    print("📤 TTS_STREAM_END")
    api.send_voice_assistant_event(VoiceAssistantEventType.VOICE_ASSISTANT_TTS_STREAM_END, {})

    print("⏱️  Attente 5s...")
    await asyncio.sleep(5)

    print("✅ Terminé")
    await api.disconnect()

if __name__ == "__main__":
    asyncio.run(test())
