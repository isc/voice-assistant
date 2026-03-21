#!/usr/bin/env python3
"""
Custom Voice Assistant Server for ESPHome
Remplace Home Assistant pour le pipeline vocal
"""

# ./build/bin/llama-server --jinja -fa on -hf ggml-org/Voxtral-Mini-3B-2507-GGUF

import asyncio
import logging
import os
import sys
from typing import Dict, Any, Optional
import time
from pathlib import Path
import aiohttp
from aiohttp import web
import socket
import tempfile
import wave
from aioesphomeapi import (
    APIClient,
    VoiceAssistantEventType,
)

# Configuration du logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration via variables d'environnement
ESP_HOST = os.environ.get("ESP_HOST", "")
ESP_PORT = int(os.environ.get("ESP_PORT", "6053"))
ESP_PASSWORD = os.environ.get("ESP_PASSWORD", "")
ESP_NOISE_PSK = os.environ.get("ESP_NOISE_PSK", "")
LLAMA_URL = os.environ.get("LLAMA_URL", "http://localhost:8080/v1/chat/completions")
HTTP_PORT = int(os.environ.get("HTTP_PORT", "8888"))


class VoiceAssistantServer:
    """
    Serveur voice assistant personnalisé pour ESPHome
    Gère le pipeline STT → LLM → TTS
    """

    def __init__(self):
        self.devices: Dict[str, APIClient] = {}

        # État du pipeline vocal
        self.conversation_id = None
        self.current_device = None
        self.audio_buffer = bytearray()  # Buffer pour accumuler l'audio
        self.is_recording = False
        self.recording_task = None  # Task pour timeout d'enregistrement
        self.last_audio_time = 0  # Timestamp du dernier audio reçu

        # API TTS intégrée (Piper)
        self.tts_engine = None

        # Serveur HTTP pour héberger les fichiers TTS
        self.http_server = None
        self.tts_dir = Path(tempfile.gettempdir()) / "voice_assistant_tts"
        self.tts_dir.mkdir(exist_ok=True)
        logger.info(f"📁 Dossier TTS: {self.tts_dir}")

    async def init_tts_engine(self):
        """Initialiser Piper TTS (16KHz natif, optimisé pour ESP)"""
        logger.info("🔊 Initialisation Piper TTS...")

        try:
            from piper import PiperVoice
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

                if not model_path.exists():
                    logger.info(f"  📥 Téléchargement {voice_name}.onnx (~30MB)...")
                    urllib.request.urlretrieve(
                        f"{base_url}/fr_FR-gilles-low.onnx", model_path
                    )
                    logger.info(f"  ✅ Modèle téléchargé")

                if not config_path.exists():
                    logger.info(f"  📥 Téléchargement config...")
                    urllib.request.urlretrieve(
                        f"{base_url}/fr_FR-gilles-low.onnx.json", config_path
                    )
                    logger.info(f"  ✅ Config téléchargée")
            else:
                logger.info("📦 Modèle Piper déjà en cache")

            logger.info(f"📦 Modèle: {model_path}")
            logger.info(f"📦 Config: {config_path}")

            # Charge le modèle
            self.tts_engine = PiperVoice.load(
                str(model_path), str(config_path), use_cuda=False
            )

        except ImportError:
            logger.error("❌ Piper TTS n'est pas installé")
            logger.info("💡 Installation: pip install piper-tts")
            raise
        except Exception as e:
            logger.error(f"❌ Erreur initialisation Piper: {e}")
            import traceback

            traceback.print_exc()
            raise

    async def start_http_server(self):
        """Démarrer le serveur HTTP pour héberger les fichiers TTS"""
        app = web.Application()
        app.router.add_static("/tts/", self.tts_dir, show_index=True)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", HTTP_PORT)
        await site.start()

        # Obtenir l'IP locale
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
        finally:
            s.close()

        logger.info(
            f"🌐 Serveur HTTP TTS démarré sur http://{local_ip}:{HTTP_PORT}/tts/"
        )
        self.http_base_url = f"http://{local_ip}:{HTTP_PORT}/tts/"

    async def start(self):
        """Démarrer le serveur et se connecter aux devices"""
        logger.info("🚀 Voice Assistant Server démarré")
        logger.info(f"📡 ESP: {ESP_HOST}:{ESP_PORT} | LLM: {LLAMA_URL} | HTTP: {HTTP_PORT}")

        await self.init_tts_engine()
        await self.start_http_server()
        await self.connect_to_device(ESP_HOST)

    async def connect_to_device(self, host: str):
        """Se connecter à un device ESPHome spécifique"""
        try:
            logger.info(f"🔌 Connexion à {host}...")

            # Noise encryption (recommandé) ou password (legacy)
            if ESP_NOISE_PSK:
                logger.info("🔐 Authentification: Noise encryption")
                api = APIClient(
                    host, ESP_PORT, None, noise_psk=ESP_NOISE_PSK,
                    client_info="voice-server-python",
                )
            else:
                api = APIClient(
                    host, ESP_PORT, ESP_PASSWORD,
                    client_info="voice-server-python",
                )

            await api.connect(login=True)
            logger.info(f"✅ Connecté à {host}")

            device_info = await api.device_info()
            logger.info(
                f"📱 Device: {device_info.name} (v{device_info.esphome_version})"
            )

            if device_info.voice_assistant_feature_flags:
                logger.info(
                    f"🎤 Voice assistant supporté (flags: {device_info.voice_assistant_feature_flags})"
                )
            else:
                logger.warning("⚠️  Voice assistant peut ne pas être supporté")

            self.devices[host] = api
            self.current_device = host

            await self.setup_voice_assistant(api, host)

        except Exception as e:
            logger.error(f"❌ Erreur connexion {host}: {e}")
            logger.info("💡 Vérifiez que:")
            logger.info("   - L'ESP est allumé et connecté au WiFi")
            logger.info("   - L'IP est correcte")
            logger.info("   - ESP_PASSWORD ou ESP_NOISE_PSK est configuré")

    async def setup_voice_assistant(self, api: APIClient, device_host: str):
        """Configuration du voice assistant"""
        logger.info(f"🎤 Configuration voice assistant pour {device_host}")

        try:
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
        L'ESP envoie ce message quand il a fini de jouer l'audio TTS
        """
        logger.info("🔔 Announcement finished reçu de l'ESP")

        # Signaler FIN du pipeline complet - LEDs retournent en mode idle
        if self.current_device:
            api = self.devices[self.current_device]
            api.send_voice_assistant_event(
                VoiceAssistantEventType.VOICE_ASSISTANT_RUN_END, {}
            )
            logger.info("📤 RUN_END envoyé - Pipeline terminé, LEDs en mode idle")

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
        Analyser l'audio avec Silero VAD pour détecter la fin de parole
        """
        from silero_vad_lite import SileroVAD
        import array

        if not hasattr(self, "vad"):
            self.vad = SileroVAD(16000)

        # Silero VAD analyse par frames de 32ms (512 samples à 16kHz)
        # Input : float32 [-1, 1], soit 512 samples = 1024 bytes en int16
        frame_samples = 512  # 32ms à 16kHz
        frame_size = frame_samples * 2  # 1024 bytes en int16

        # Traiter l'audio par frames
        for i in range(0, len(audio_bytes), frame_size):
            frame = audio_bytes[i : i + frame_size]
            if len(frame) != frame_size:
                continue  # Frame incomplète, ignorer

            try:
                # Convertir int16 PCM → float32 [-1, 1] pour Silero
                int16_samples = array.array("h")
                int16_samples.frombytes(frame)
                float_samples = array.array(
                    "f", [s / 32768.0 for s in int16_samples]
                )

                speech_prob = self.vad.process(float_samples)
                is_speech = speech_prob > 0.5

                if is_speech:
                    self.vad_speech_frames += 1
                    self.vad_silence_frames = 0
                    if not self.vad_has_speech:
                        self.vad_has_speech = True
                        logger.info("🗣️  Parole détectée")
                else:
                    if self.vad_has_speech:
                        self.vad_silence_frames += 1

                # Arrêter après ~1 seconde de silence (31 frames de 32ms)
                if self.vad_has_speech and self.vad_silence_frames >= 31:
                    logger.info(
                        f"🤫 Silence détecté après parole ({self.vad_silence_frames} frames)"
                    )
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
                VoiceAssistantEventType.VOICE_ASSISTANT_STT_END,
                {"text": "(audio transcrit)"},
            )
            logger.info("📤 STT_END envoyé")

            # 2. TTS - Text to Speech (streaming direct vers ESP)
            await self.text_to_speech(api, response_text)

            logger.info("✅ Pipeline terminé avec succès")

        except Exception as e:
            logger.error(f"💥 Erreur pipeline: {e}")
            await self.send_error_to_device(api, f"Pipeline error: {e}")

    # === FUNCTION CALLING ===

    def get_available_functions(self) -> list:
        """
        Retourne la liste des fonctions disponibles pour le function calling
        """
        return [
            {
                "name": "close_shutters",
                "description": "Fermer les volets de la maison",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "room": {
                            "type": "string",
                            "description": "La pièce où fermer les volets (salon, chambre, etc.)",
                        }
                    },
                    "required": ["room"],
                },
            }
        ]

    async def execute_function(self, function_name: str, arguments: dict) -> str:
        """
        Exécute une fonction appelée par le LLM
        """
        if function_name == "close_shutters":
            return await self.close_shutters(**arguments)
        else:
            return f"Fonction inconnue: {function_name}"

    async def close_shutters(self, room: str) -> str:
        """
        Fermer les volets d'une pièce (pour le moment juste du logging)
        """
        logger.info(f"🪟 FONCTION APPELÉE: close_shutters(room={room})")
        logger.info(f"   → Simulation: fermeture des volets du {room}")
        return f"Les volets du {room} ont été fermés"

    async def process_audio_with_voxtral(
        self, api: APIClient, audio_bytes: bytes
    ) -> Optional[str]:
        """
        STT + LLM en une seule passe avec Voxtral
        Traite l'audio et retourne directement la réponse de l'assistant
        """
        logger.info(f"🎤🧠 Voxtral: Envoi de {len(audio_bytes)} bytes...")

        try:
            import base64
            import json

            llama_url = LLAMA_URL

            # Créer un fichier WAV temporaire
            temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            with wave.open(temp_wav.name, "wb") as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(16000)  # 16kHz
                wav_file.writeframes(audio_bytes)

            logger.info(
                f"📁 Audio WAV: {temp_wav.name} ({os.path.getsize(temp_wav.name)} bytes)"
            )

            # Lire et encoder l'audio en base64
            with open(temp_wav.name, "rb") as f:
                audio_data = f.read()

            audio_b64 = base64.b64encode(audio_data).decode("utf-8")
            logger.info(f"📦 Base64: {len(audio_b64)} caractères")

            # Préparer la requête OpenAI-compatible pour assistant vocal avec function calling
            payload = {
                "model": "voxtral",
                "messages": [
                    {
                        "role": "system",
                        "content": "Tu es un assistant vocal pour la maison connectée. Réponds de manière concise et naturelle en français. Tu peux contrôler les lumières, les volets, donner la météo, et répondre aux questions. Utilise les fonctions disponibles quand c'est approprié.",
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_audio",
                                "input_audio": {"data": audio_b64, "format": "wav"},
                            }
                        ],
                    },
                ],
                "tools": [
                    {"type": "function", "function": func}
                    for func in self.get_available_functions()
                ],
                "max_tokens": 256,
                "temperature": 0.7,
            }

            # Envoyer à llama.cpp
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    llama_url, json=payload, timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        message = result["choices"][0]["message"]

                        # Vérifier si le LLM veut appeler une fonction
                        if "tool_calls" in message and message["tool_calls"]:
                            tool_call = message["tool_calls"][0]
                            function_name = tool_call["function"]["name"]
                            function_args = json.loads(
                                tool_call["function"]["arguments"]
                            )

                            logger.info(
                                f"🔧 Function call: {function_name}({function_args})"
                            )

                            # Exécuter la fonction (pas de deuxième appel LLM)
                            try:
                                function_result = await self.execute_function(
                                    function_name, function_args
                                )
                                logger.info(f"✅ Résultat fonction: {function_result}")

                                # Voxtral a déjà généré une réponse appropriée dans content
                                # On l'utilise directement
                                assistant_response = message.get(
                                    "content", "C'est fait"
                                )
                                if assistant_response:
                                    assistant_response = assistant_response.strip()
                                else:
                                    assistant_response = "C'est fait"

                                logger.info(
                                    f'🤖 Réponse Voxtral (avec fonction): "{assistant_response}"'
                                )
                                os.unlink(temp_wav.name)
                                return assistant_response

                            except Exception as e:
                                logger.error(f"❌ Erreur exécution fonction: {e}")
                                os.unlink(temp_wav.name)
                                return f"Désolé, une erreur s'est produite lors de l'exécution de {function_name}"
                        else:
                            # Pas de function call, réponse directe
                            assistant_response = message["content"].strip()
                            logger.info(f'🤖 Réponse Voxtral: "{assistant_response}"')
                            os.unlink(temp_wav.name)
                            return assistant_response if assistant_response else None
                    else:
                        error_text = await response.text()
                        logger.error(
                            f"❌ Erreur Voxtral {response.status}: {error_text}"
                        )
                        os.unlink(temp_wav.name)
                        return None

        except Exception as e:
            logger.error(f"❌ Erreur STT: {e}")
            import traceback

            traceback.print_exc()
            # Fallback sur simulation
            return "Allume la lumière du salon"

    async def text_to_speech(self, api: APIClient, text: str):
        """
        Text-to-Speech avec Piper - Génération fichier WAV + URL
        """
        logger.info(f'🗣️ TTS: Génération audio pour "{text}"')

        # TTS_START
        api.send_voice_assistant_event(
            VoiceAssistantEventType.VOICE_ASSISTANT_TTS_START, {"text": text}
        )
        logger.info(f"📤 TTS_START envoyé")

        try:
            import numpy as np
            import hashlib

            # Créer un fichier unique pour cet audio
            filename = f"tts_{int(time.time())}_{hashlib.md5(text.encode()).hexdigest()[:8]}.wav"
            output_path = self.tts_dir / filename

            logger.info(f"🔊 Synthétise: '{text[:80]}...'")

            # Générer l'audio avec Piper
            all_audio = []
            for audio_chunk in self.tts_engine.synthesize(text):
                all_audio.append(audio_chunk.audio_float_array)

            # Concaténer
            audio_float = np.concatenate(all_audio)
            sample_rate = 16000  # fr_FR-gilles-low est 16KHz natif

            logger.info(f"🎵 Audio total: {len(audio_float)} samples")

            # Convertir float32 [-1, 1] → int16 (PCM 16-bit)
            audio_int16 = (audio_float * 32767).astype(np.int16)

            # Sauvegarder comme fichier WAV
            with wave.open(str(output_path), "wb") as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)  # 16kHz
                wav_file.writeframes(audio_int16.tobytes())

            logger.info(
                f"✅ TTS généré: {output_path} ({output_path.stat().st_size} bytes)"
            )

            # URL accessible par l'ESP
            audio_url = f"{self.http_base_url}{filename}"

            api.send_voice_assistant_event(
                VoiceAssistantEventType.VOICE_ASSISTANT_TTS_END, {"url": audio_url}
            )
            logger.info(f"📤 TTS_END envoyé avec URL: {audio_url}")

        except Exception as e:
            logger.error(f"❌ Erreur TTS: {e}")
            import traceback

            traceback.print_exc()
            raise

    async def send_error_to_device(self, api: APIClient, error_message: str):
        """Envoyer une erreur à l'ESP"""
        logger.error(f"📤 Envoi erreur à l'ESP: {error_message}")

        try:
            # send_voice_assistant_event est synchrone, pas async
            api.send_voice_assistant_event(
                VoiceAssistantEventType.VOICE_ASSISTANT_ERROR,
                {"code": "server_error", "message": error_message},
            )
            logger.info("✅ Événement VOICE_ASSISTANT_ERROR envoyé à l'ESP")
        except Exception as e:
            logger.error(f"❌ Erreur envoi erreur: {e}")


async def main():
    """Point d'entrée principal"""
    print("🎯 Custom Voice Assistant Server for ESPHome (Python)")
    print("📋 Remplace Home Assistant pour le pipeline vocal")
    print("🔊 TTS: Piper (16KHz natif, fichier WAV + URL)")
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
