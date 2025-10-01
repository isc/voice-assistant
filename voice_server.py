#!/usr/bin/env python3
"""
Custom Voice Assistant Server for ESPHome
Remplace Home Assistant pour le pipeline vocal
"""

import asyncio
import logging
import sys
from typing import Dict, Any, Optional
import json
import time
from pathlib import Path
import aiohttp
from aiohttp import web
import subprocess
import os
import tempfile
import socket
import wave

import aioesphomeapi
from aioesphomeapi import (
    APIClient,
    VoiceAssistantEventType,
    VoiceAssistantAudioSettings,
    VoiceAssistantCommand,
)

# Configuration du logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class VoiceAssistantServer:
    """
    Serveur voice assistant personnalisé pour ESPHome
    Gère le pipeline STT → LLM → TTS
    """

    def __init__(self):
        self.devices: Dict[str, APIClient] = {}
        self.config = {"port": 6053, "password": "", "debug": True}

        # État du pipeline vocal
        self.conversation_id = None
        self.current_device = None
        self.audio_buffer = bytearray()  # Buffer pour accumuler l'audio
        self.is_recording = False
        self.recording_task = None  # Task pour timeout d'enregistrement
        self.last_audio_time = 0  # Timestamp du dernier audio reçu

        # API TTS intégrée (Piper)
        self.tts_engine = None

    async def init_tts_engine(self):
        """Initialiser Piper TTS (16KHz natif, optimisé pour ESP)"""
        logger.info("🔊 Initialisation Piper TTS...")

        try:
            from piper import PiperVoice
            import json
            import urllib.request

            # Modèle français 16KHz NATIF (pas de resampling nécessaire !)
            voice_name = "fr_FR-gilles-low"
            models_dir = Path("/tmp/piper_models")
            models_dir.mkdir(exist_ok=True)

            model_path = models_dir / "fr_FR-gilles-low.onnx"
            config_path = models_dir / "fr_FR-gilles-low.onnx.json"

            # Télécharger le modèle si nécessaire
            if not model_path.exists() or not config_path.exists():
                logger.info("⏬ Téléchargement du modèle Piper (16KHz natif)...")

                base_url = f"https://huggingface.co/rhasspy/piper-voices/resolve/main/fr/fr_FR/gilles/low"

                # Télécharge modèle
                if not model_path.exists():
                    logger.info(f"  📥 Téléchargement {voice_name}.onnx (~30MB)...")
                    urllib.request.urlretrieve(
                        f"{base_url}/fr_FR-gilles-low.onnx",
                        model_path
                    )
                    logger.info(f"  ✅ Modèle téléchargé")

                # Télécharge config
                if not config_path.exists():
                    logger.info(f"  📥 Téléchargement config...")
                    urllib.request.urlretrieve(
                        f"{base_url}/fr_FR-gilles-low.onnx.json",
                        config_path
                    )
                    logger.info(f"  ✅ Config téléchargée")
            else:
                logger.info("📦 Modèle Piper déjà en cache")

            logger.info(f"📦 Modèle: {model_path}")
            logger.info(f"📦 Config: {config_path}")

            # Charge le modèle
            self.tts_engine = PiperVoice.load(str(model_path), str(config_path), use_cuda=False)

            logger.info(f"✅ Piper TTS prêt (sample_rate: {self.tts_engine.config.sample_rate}Hz)")

            # Test rapide avec l'API correcte (synthesize() retourne des AudioChunk)
            logger.info("🧪 Test TTS...")
            import numpy as np
            test_audio = []
            for chunk in self.tts_engine.synthesize("Bonjour"):
                test_audio.append(chunk.audio_float_array)
            test_samples = np.concatenate(test_audio)
            logger.info(f"✅ Test réussi: {len(test_samples)} samples générés @ {self.tts_engine.config.sample_rate}Hz")

            # Vérifier le sample rate
            if self.tts_engine.config.sample_rate != 16000:
                logger.error(f"❌ ERREUR: Modèle génère en {self.tts_engine.config.sample_rate}Hz au lieu de 16KHz")
                raise ValueError(f"Sample rate incorrect: {self.tts_engine.config.sample_rate}Hz")

        except ImportError:
            logger.error("❌ Piper TTS n'est pas installé")
            logger.info("💡 Installation: pip install piper-tts")
            raise
        except Exception as e:
            logger.error(f"❌ Erreur initialisation Piper: {e}")
            import traceback
            traceback.print_exc()
            raise

    async def start(self):
        """Démarrer le serveur et se connecter aux devices"""
        logger.info("🚀 Voice Assistant Server démarré")
        logger.info(f"📡 Configuration: port {self.config['port']}")

        # Initialiser Piper TTS
        await self.init_tts_engine()

        # Connexion à un device spécifique
        await self.connect_to_device("", self.config["password"])

    async def connect_to_device(self, host: str, password: str):
        """Se connecter à un device ESPHome spécifique"""
        try:
            logger.info(f"🔌 Connexion à {host}...")

            # Créer le client API
            api = APIClient(
                host, self.config["port"], password, client_info="voice-server-python"
            )

            # Connexion
            await api.connect(login=True)
            logger.info(f"✅ Connecté à {host}")

            # Récupérer les infos du device
            device_info = await api.device_info()
            logger.info(
                f"📱 Device: {device_info.name} (v{device_info.esphome_version})"
            )

            # Vérifier support voice assistant
            if device_info.voice_assistant_feature_flags:
                logger.info(
                    f"🎤 Voice assistant supporté (flags: {device_info.voice_assistant_feature_flags})"
                )
            else:
                logger.warning("⚠️  Voice assistant peut ne pas être supporté")

            self.devices[host] = api
            self.current_device = host

            # Configurer les handlers voice assistant
            await self.setup_voice_assistant(api, host)

        except Exception as e:
            logger.error(f"❌ Erreur connexion {host}: {e}")
            logger.info("💡 Vérifiez que:")
            logger.info("   - L'ESP est allumé et connecté au WiFi")
            logger.info("   - L'IP est correcte")
            logger.info("   - Le password correspond à la config ESPHome")

    async def setup_voice_assistant(self, api: APIClient, device_host: str):
        """Configuration du voice assistant"""
        logger.info(f"🎤 Configuration voice assistant pour {device_host}")

        try:
            # S'abonner aux événements voice assistant avec la nouvelle API
            unsubscribe = api.subscribe_voice_assistant(
                handle_start=self.handle_voice_assistant_start,
                handle_stop=self.handle_voice_assistant_stop,
                handle_audio=self.handle_voice_assistant_audio,
                handle_announcement_finished=self.handle_announcement_finished,
            )
            self.va_unsubscribe = unsubscribe
            logger.info("✅ Abonnement voice assistant réussi")

        except Exception as e:
            logger.error(f"❌ Erreur setup voice assistant: {e}")

    async def handle_voice_assistant_start(
        self,
        conversation_id: str,
        flags: int,
        audio_settings: Any,
        wake_word_phrase: str | None,
    ) -> int | None:
        """
        Handler appelé quand le voice assistant démarre
        Retourne un ID de conversation si besoin
        """
        logger.info(f"🚀 Voice Assistant START")
        logger.info(f"   Conversation ID: {conversation_id}")
        logger.info(f"   Flags: {flags}")
        logger.info(f"   Wake word: {wake_word_phrase}")
        logger.info(f"   Audio settings: {audio_settings}")

        self.conversation_id = conversation_id

        # Réinitialiser le buffer audio pour une nouvelle session
        self.audio_buffer = bytearray()
        self.is_recording = True
        self.vad_speech_frames = 0
        self.vad_silence_frames = 0
        self.vad_has_speech = False

        # Dire à l'ESP de commencer à enregistrer et envoyer l'audio
        if self.current_device:
            api = self.devices[self.current_device]
            try:
                api.send_voice_assistant_event(
                    VoiceAssistantEventType.VOICE_ASSISTANT_STT_START, {}
                )
                logger.info("📤 Commande STT_START envoyée - ESP va envoyer l'audio")

                # Démarrer une tâche de timeout max (sécurité)
                self.recording_task = asyncio.create_task(
                    self.monitor_recording_timeout(api)
                )

            except Exception as e:
                logger.error(f"❌ Erreur envoi STT_START: {e}")

        # Retourner 0 pour indiquer succès
        return 0

    async def monitor_recording_timeout(self, api: APIClient):
        """
        Timeout de sécurité : arrête l'enregistrement après 30 secondes max
        """
        max_recording_time = 30.0

        logger.info(f"⏱️  Timeout sécurité: {max_recording_time}s")
        await asyncio.sleep(max_recording_time)

        if self.is_recording:
            logger.warning(f"⏰ TIMEOUT SÉCURITÉ atteint ({max_recording_time}s)")
            await self.stop_recording(api, "timeout")

    async def stop_recording(self, api: APIClient, reason: str):
        """
        Arrêter l'enregistrement et traiter l'audio
        """
        if not self.is_recording:
            return

        self.is_recording = False
        audio_data = bytes(self.audio_buffer)

        logger.info(f"🛑 Arrêt enregistrement ({reason}): {len(audio_data)} bytes")

        try:
            # Envoyer STT_VAD_END - LEDs passent en orange
            api.send_voice_assistant_event(
                VoiceAssistantEventType.VOICE_ASSISTANT_STT_VAD_END, {}
            )
            logger.info("📤 STT_VAD_END envoyé - LEDs orange (réflexion)")

            # Traiter le pipeline (STT_END sera envoyé après transcription)
            if len(audio_data) > 0:
                await self.process_voice_pipeline(api, audio_data)

        except Exception as e:
            logger.error(f"❌ Erreur arrêt enregistrement: {e}")

    async def handle_voice_assistant_stop(self, abort: bool):
        """
        Handler appelé quand le voice assistant s'arrête
        - abort=False: arrêt normal (bouton relâché)
        - abort=True: interruption (timeout, erreur, ou VAD)
        Dans les deux cas, on traite l'audio si on en a reçu
        """
        logger.info(f"⏹️  Voice Assistant STOP (abort={abort})")

        if self.is_recording and len(self.audio_buffer) > 0:
            # L'enregistrement est terminé, on traite l'audio complet
            logger.info(f"🎵 Audio complet reçu: {len(self.audio_buffer)} bytes")
            self.is_recording = False

            # Traiter le pipeline avec tout l'audio (même si abort=True)
            if self.current_device:
                api = self.devices[self.current_device]
                await self.process_voice_pipeline(api, bytes(self.audio_buffer))
        else:
            logger.info("⏭️  Pas d'audio à traiter")

    async def handle_announcement_finished(self, announcement_finished):
        """
        Gestionnaire pour l'événement announcement_finished
        L'ESP envoie ce message quand il a fini de jouer l'audio TTS streamé
        """
        logger.info("🔔 Announcement finished reçu de l'ESP")

    async def handle_voice_assistant_event(self, event: Any):
        """
        Gestionnaire des événements voice assistant

        Events possibles:
        - VOICE_ASSISTANT_ERROR = 0
        - VOICE_ASSISTANT_RUN_START = 1
        - VOICE_ASSISTANT_RUN_END = 2
        - VOICE_ASSISTANT_STT_START = 3
        - VOICE_ASSISTANT_STT_END = 4
        - VOICE_ASSISTANT_INTENT_START = 5
        - VOICE_ASSISTANT_INTENT_END = 6
        - VOICE_ASSISTANT_TTS_START = 7
        - VOICE_ASSISTANT_TTS_END = 8
        - VOICE_ASSISTANT_WAKE_WORD_START = 9
        - VOICE_ASSISTANT_WAKE_WORD_END = 10
        - VOICE_ASSISTANT_STT_VAD_START = 11
        - VOICE_ASSISTANT_STT_VAD_END = 12
        - VOICE_ASSISTANT_TTS_STREAM_START = 98
        - VOICE_ASSISTANT_TTS_STREAM_END = 99
        """

        event_type = event.event_type
        event_data = getattr(event, "data", {})

        logger.info(f"📨 Événement voice assistant: {event_type} - {event_data}")

        # === ÉVÉNEMENTS REÇUS DE L'ESP ===

        if event_type == VoiceAssistantEventType.VOICE_ASSISTANT_RUN_START:
            await self.handle_run_start(event)

        elif event_type == VoiceAssistantEventType.VOICE_ASSISTANT_WAKE_WORD_START:
            await self.handle_wake_word_start(event)

        elif event_type == VoiceAssistantEventType.VOICE_ASSISTANT_WAKE_WORD_END:
            await self.handle_wake_word_end(event)

        elif event_type == VoiceAssistantEventType.VOICE_ASSISTANT_STT_VAD_START:
            await self.handle_vad_start(event)

        elif event_type == VoiceAssistantEventType.VOICE_ASSISTANT_STT_VAD_END:
            await self.handle_vad_end(event)

        elif event_type == VoiceAssistantEventType.VOICE_ASSISTANT_ERROR:
            await self.handle_voice_error(event)

        else:
            logger.debug(f"🔍 Événement non géré: {event_type}")

    async def handle_voice_assistant_audio(self, audio_bytes: bytes):
        """
        Gestionnaire de l'audio reçu de l'ESP avec détection VAD
        """
        if self.is_recording:
            # Accumuler l'audio dans le buffer
            self.audio_buffer.extend(audio_bytes)
            self.last_audio_time = time.time()

            # Analyser l'audio avec WebRTC VAD
            await self.analyze_audio_vad(audio_bytes)

            if len(self.audio_buffer) % 10240 == 0 or len(self.audio_buffer) < 10240:
                logger.info(f"🎵 Audio: {len(self.audio_buffer)} bytes")
        else:
            logger.debug("⚠️  Audio reçu après enregistrement (normal)")

    async def analyze_audio_vad(self, audio_bytes: bytes):
        """
        Analyser l'audio avec WebRTC VAD pour détecter la fin de parole
        """
        import webrtcvad

        if not hasattr(self, 'vad'):
            self.vad = webrtcvad.Vad(2)  # Agressivité 0-3 (2 = moyen)

        # WebRTC VAD analyse par frames de 10/20/30ms
        # Audio 16kHz, 16-bit = 320 bytes pour 10ms
        frame_duration_ms = 30  # 30ms
        frame_size = int(16000 * 2 * frame_duration_ms / 1000)  # 960 bytes

        # Traiter l'audio par frames
        for i in range(0, len(audio_bytes), frame_size):
            frame = audio_bytes[i:i + frame_size]
            if len(frame) != frame_size:
                continue  # Frame incomplète, ignorer

            try:
                is_speech = self.vad.is_speech(frame, 16000)

                if is_speech:
                    self.vad_speech_frames += 1
                    self.vad_silence_frames = 0
                    if not self.vad_has_speech:
                        self.vad_has_speech = True
                        logger.info("🗣️  Parole détectée")
                else:
                    if self.vad_has_speech:
                        self.vad_silence_frames += 1

                # Arrêter après 1 seconde de silence (33 frames de 30ms)
                if self.vad_has_speech and self.vad_silence_frames >= 33:
                    logger.info(f"🤫 Silence détecté après parole ({self.vad_silence_frames} frames)")
                    if self.current_device and self.is_recording:
                        api = self.devices[self.current_device]
                        await self.stop_recording(api, "VAD silence")
                    return

            except Exception as e:
                logger.debug(f"Erreur VAD: {e}")
                continue

    # === HANDLERS SPÉCIFIQUES ===

    async def handle_run_start(self, event):
        """Pipeline démarré sur l'ESP"""
        logger.info("🚀 Pipeline démarré")
        self.conversation_id = getattr(
            event, "conversation_id", f"conv_{int(time.time())}"
        )

    async def handle_wake_word_start(self, event):
        """Wake word détecté"""
        wake_word = getattr(event, "wake_word_phrase", "unknown")
        logger.info(f"🔊 Wake word détecté: '{wake_word}'")

    async def handle_wake_word_end(self, event):
        """Fin du wake word"""
        logger.info("⏹️ Wake word terminé")

    async def handle_vad_start(self, event):
        """Début de détection vocale"""
        logger.info("👂 Début d'écoute (VAD start)")

    async def handle_vad_end(self, event):
        """Fin de détection vocale - l'audio va arriver"""
        logger.info("⏹️ Fin d'écoute (VAD end) - traitement audio en cours...")

    async def handle_voice_error(self, event):
        """Erreur voice assistant"""
        error_code = getattr(event, "code", "unknown")
        error_message = getattr(event, "message", "Unknown error")
        logger.error(f"💥 Erreur voice assistant: {error_code} - {error_message}")

    # === PIPELINE PRINCIPAL ===

    async def process_voice_pipeline_async(self, api: APIClient, audio_bytes: bytes):
        """
        Version async du pipeline qui ne bloque pas
        """
        await self.process_voice_pipeline(api, audio_bytes)

    async def process_voice_pipeline(self, api: APIClient, audio_bytes: bytes):
        """
        Pipeline principal: Audio → Voxtral (STT+LLM) → TTS
        """
        logger.info("🔄 Démarrage pipeline vocal complet")

        try:
            # 1. Voxtral - STT + LLM en une seule passe
            response_text = await self.process_audio_with_voxtral(api, audio_bytes)
            if not response_text:
                await self.send_error_to_device(api, "Voxtral processing failed")
                return

            # Envoyer STT_END (Voxtral ne retourne que la réponse LLM, pas la transcription)
            api.send_voice_assistant_event(
                VoiceAssistantEventType.VOICE_ASSISTANT_STT_END, {"text": "(audio transcrit)"}
            )
            logger.info("📤 STT_END envoyé")

            # 2. TTS - Text to Speech (streaming direct vers ESP)
            await self.text_to_speech(api, response_text)

            logger.info("✅ Pipeline terminé avec succès")

        except Exception as e:
            logger.error(f"💥 Erreur pipeline: {e}")
            await self.send_error_to_device(api, f"Pipeline error: {e}")

    async def process_audio_with_voxtral(self, api: APIClient, audio_bytes: bytes) -> Optional[str]:
        """
        STT + LLM en une seule passe avec Voxtral
        Traite l'audio et retourne directement la réponse de l'assistant
        """
        logger.info(f"🎤🧠 Voxtral: Envoi de {len(audio_bytes)} bytes...")

        try:
            import base64
            import wave
            import tempfile
            import os

            # URL OpenAI-compatible de llama.cpp
            llama_url = "http://localhost:8080/v1/chat/completions"

            # Créer un fichier WAV temporaire
            temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            with wave.open(temp_wav.name, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(16000)  # 16kHz
                wav_file.writeframes(audio_bytes)

            logger.info(f"📁 Audio WAV: {temp_wav.name} ({os.path.getsize(temp_wav.name)} bytes)")

            # Lire et encoder l'audio en base64
            with open(temp_wav.name, 'rb') as f:
                audio_data = f.read()

            audio_b64 = base64.b64encode(audio_data).decode('utf-8')
            logger.info(f"📦 Base64: {len(audio_b64)} caractères")

            # Préparer la requête OpenAI-compatible pour assistant vocal
            payload = {
                "model": "voxtral",
                "messages": [
                    {
                        "role": "system",
                        "content": "Tu es un assistant vocal pour la maison connectée. Réponds de manière concise et naturelle en français. Tu peux contrôler les lumières, donner la météo, et répondre aux questions."
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_audio",
                                "input_audio": {
                                    "data": audio_b64,
                                    "format": "wav"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 256,
                "temperature": 0.7
            }

            # Envoyer à llama.cpp
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    llama_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        assistant_response = result['choices'][0]['message']['content'].strip()
                        logger.info(f'🤖 Réponse Voxtral: "{assistant_response}"')

                        # Nettoyer
                        os.unlink(temp_wav.name)

                        return assistant_response if assistant_response else None
                    else:
                        error_text = await response.text()
                        logger.error(f"❌ Erreur Voxtral {response.status}: {error_text}")
                        os.unlink(temp_wav.name)
                        return None

        except Exception as e:
            logger.error(f"❌ Erreur STT: {e}")
            import traceback
            traceback.print_exc()
            # Fallback sur simulation
            return "Allume la lumière du salon"

    async def process_with_llm(self, api: APIClient, text: str) -> Optional[str]:
        """
        Traitement LLM + MCP
        À remplacer par votre stack
        """
        logger.info(f'🧠 LLM: Traitement de "{text}"')

        # Signaler début intent - LEDs continuent en mode réflexion
        api.send_voice_assistant_event(
            VoiceAssistantEventType.VOICE_ASSISTANT_INTENT_START, {}
        )
        logger.info("📤 INTENT_START envoyé")

        # TODO: Implémenter votre LLM + serveurs MCP
        # Exemples:
        # - OpenAI GPT
        # - Claude
        # - Ollama (local)
        # - Serveurs MCP pour actions

        # Simulation pour test
        await asyncio.sleep(2)

        if "lumière" in text.lower():
            response = "J'allume la lumière du salon."
        elif "météo" in text.lower():
            response = "Il fait beau aujourd'hui, 22 degrés."
        else:
            response = f"J'ai entendu : {text}. Comment puis-je vous aider ?"

        logger.info(f'🤖 Réponse LLM: "{response}"')

        # Signaler fin intent
        api.send_voice_assistant_event(
            VoiceAssistantEventType.VOICE_ASSISTANT_INTENT_END, {}
        )
        logger.info("📤 INTENT_END envoyé")

        return response

    async def text_to_speech(self, api: APIClient, text: str):
        """
        Text-to-Speech avec Piper - Streaming direct vers ESP (16KHz natif)
        Réplique exactement l'implémentation de Home Assistant
        """
        logger.info(f'🗣️ TTS: Génération audio pour "{text}"')

        # TTS_START
        api.send_voice_assistant_event(
            VoiceAssistantEventType.VOICE_ASSISTANT_TTS_START, {"text": text}
        )
        logger.info(f"📤 TTS_START envoyé")

        # Streamer l'audio directement vers l'ESP
        await self._stream_tts_to_esp(api, text)

        logger.info("✅ TTS streaming terminé")

    async def _stream_tts_to_esp(self, api: APIClient, text: str):
        """
        Stream l'audio TTS directement à l'ESP chunk par chunk
        Format: 16KHz, 16-bit, mono (natif avec fr_FR-gilles-low)
        Réplique exactement l'implémentation de Home Assistant
        """
        logger.info("🔊 Génération et streaming TTS avec Piper...")

        start_time = time.time()

        try:
            import numpy as np

            # Synthétise tout l'audio en mémoire AVANT TTS_STREAM_START
            # (évite timeout ESP qui attend audio après TTS_STREAM_START)
            logger.info(f"🔊 Synthétise: '{text[:80]}...'")

            all_audio = []
            for audio_chunk in self.tts_engine.synthesize(text):
                all_audio.append(audio_chunk.audio_float_array)

            # Concaténer
            audio_float = np.concatenate(all_audio)
            sample_rate = 16000  # fr_FR-gilles-low est 16KHz natif

            logger.info(f"🎵 Audio total: {len(audio_float)} samples")

            # TTS_STREAM_START juste avant de commencer à envoyer
            api.send_voice_assistant_event(
                VoiceAssistantEventType.VOICE_ASSISTANT_TTS_STREAM_START, {}
            )
            logger.info("📤 TTS_STREAM_START - Début streaming audio")

            # Convertir float32 [-1, 1] → int16 (PCM 16-bit)
            audio_int16 = (audio_float * 32767).astype(np.int16)

            # Convertir en bytes (PCM brut, sans en-tête WAV)
            audio_bytes = audio_int16.tobytes()

            logger.info(f"🎵 Audio PCM brut: {len(audio_bytes)} bytes ({len(audio_int16)} samples)")

            # Stream par chunks de 512 samples = 1024 bytes (comme HA)
            chunk_size = 1024  # 512 samples × 2 bytes/sample
            chunk_count = 0

            for i in range(0, len(audio_bytes), chunk_size):
                chunk = audio_bytes[i:i + chunk_size]

                # Envoyer à l'ESP
                api.send_voice_assistant_audio(chunk)
                chunk_count += 1

                # Throttle: 90% de la durée du chunk (comme HA)
                samples_in_chunk = len(chunk) // 2  # 2 bytes par sample (16-bit)
                seconds_in_chunk = samples_in_chunk / sample_rate
                await asyncio.sleep(seconds_in_chunk * 0.9)

                if chunk_count % 50 == 0:
                    logger.debug(f"📊 {chunk_count} chunks streamés")

            elapsed = time.time() - start_time
            logger.info(f"✅ {chunk_count} chunks streamés en {elapsed:.2f}s")

        except Exception as e:
            logger.error(f"❌ Erreur streaming TTS: {e}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            # TTS_STREAM_END
            api.send_voice_assistant_event(
                VoiceAssistantEventType.VOICE_ASSISTANT_TTS_STREAM_END, {}
            )
            logger.info("📤 TTS_STREAM_END - Fin streaming audio")

            # TTS_END (signal que TTS est terminé)
            api.send_voice_assistant_event(
                VoiceAssistantEventType.VOICE_ASSISTANT_TTS_END, {}
            )
            logger.info("📤 TTS_END - TTS terminé")

            # Calculer la durée réelle de l'audio pour attendre la fin de lecture
            total_samples = len(audio_int16)
            audio_duration = total_samples / sample_rate

            # Attendre que l'ESP finisse de jouer l'audio
            # Le streaming a pris environ 90% du temps réel grâce au throttling,
            # donc on attend les 10% restants + un petit buffer
            remaining_time = audio_duration * 0.15  # 15% pour être sûr
            logger.info(f"⏱️  Attente fin de lecture audio: {remaining_time:.2f}s")
            await asyncio.sleep(remaining_time)

            # Signaler FIN du pipeline complet - LEDs retournent en mode idle
            api.send_voice_assistant_event(
                VoiceAssistantEventType.VOICE_ASSISTANT_RUN_END, {}
            )
            logger.info("📤 RUN_END envoyé - Pipeline terminé, LEDs en mode idle")

    async def send_error_to_device(self, api: APIClient, error_message: str):
        """Envoyer une erreur à l'ESP"""
        logger.error(f"📤 Envoi erreur à l'ESP: {error_message}")

        try:
            # send_voice_assistant_event est synchrone, pas async
            api.send_voice_assistant_event(
                VoiceAssistantEventType.VOICE_ASSISTANT_ERROR,
                {"code": "server_error", "message": error_message}
            )
            logger.info("✅ Événement VOICE_ASSISTANT_ERROR envoyé à l'ESP")
        except Exception as e:
            logger.error(f"❌ Erreur envoi erreur: {e}")


async def main():
    """Point d'entrée principal"""
    print("🎯 Custom Voice Assistant Server for ESPHome (Python)")
    print("📋 Remplace Home Assistant pour le pipeline vocal")
    print("🔊 TTS: Piper (16KHz natif, streaming)")
    print("🎤 STT+LLM: Voxtral\n")

    server = VoiceAssistantServer()

    try:
        await server.start()

        # Garder le serveur en vie
        logger.info("🔄 Serveur en fonctionnement - Ctrl+C pour arrêter")
        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        logger.info("\n👋 Arrêt du serveur...")
    except Exception as e:
        logger.error(f"💥 Erreur fatale: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
