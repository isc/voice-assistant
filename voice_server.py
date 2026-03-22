#!/usr/bin/env python3
"""
Custom Voice Assistant Server for ESPHome
Replaces Home Assistant for the voice pipeline
"""

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

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration via environment variables
ESP_HOST = os.environ.get("ESP_HOST", "")
ESP_PORT = int(os.environ.get("ESP_PORT", "6053"))
ESP_PASSWORD = os.environ.get("ESP_PASSWORD", "")
ESP_NOISE_PSK = os.environ.get("ESP_NOISE_PSK", "")
LLM_URL = os.environ.get("LLM_URL", "http://localhost:8080/v1/chat/completions")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "")  # e.g. "gpt-5.4-nano" — empty = local llama.cpp
HTTP_PORT = int(os.environ.get("HTTP_PORT", "8888"))
HA_URL = os.environ.get("HA_URL", "http://localhost:8123")
HA_TOKEN = os.environ.get("HA_TOKEN", "")


class VoiceAssistantServer:
    """
    Custom voice assistant server for ESPHome
    Handles the STT -> LLM -> TTS pipeline
    """

    def __init__(self):
        self.devices: Dict[str, APIClient] = {}

        # Voice pipeline state
        self.conversation_id = None
        self.current_device = None
        self.audio_buffer = bytearray()
        self.is_recording = False
        self.recording_task = None
        self.last_audio_time = 0

        # Kokoro TTS engine + French G2P
        self.tts_engine = None
        self.tts_g2p = None

        # Home Assistant client
        self.ha_client = None

        # Conversation history for multi-turn context
        self.conversation_history = []
        self.last_interaction_time = 0
        self.CONVERSATION_TIMEOUT = 120  # seconds — forget context after 2 min of silence

        # HTTP server for serving TTS audio files
        self.http_server = None
        self.tts_dir = Path(tempfile.gettempdir()) / "voice_assistant_tts"
        self.tts_dir.mkdir(exist_ok=True)
        logger.info(f"TTS directory: {self.tts_dir}")

    async def init_tts_engine(self):
        """Initialize Kokoro TTS with French G2P via misaki/espeak"""
        logger.info("Initializing Kokoro TTS...")

        try:
            import kokoro_onnx
            from misaki.espeak import EspeakG2P

            models_dir = Path("/tmp/kokoro_models")
            models_dir.mkdir(exist_ok=True)

            model_path = models_dir / "kokoro-v1.0.onnx"
            voices_path = models_dir / "voices-v1.0.bin"

            if not model_path.exists() or not voices_path.exists():
                logger.info("Downloading Kokoro models...")
                import urllib.request

                if not model_path.exists():
                    logger.info("  Downloading kokoro-v1.0.onnx (~310MB)...")
                    urllib.request.urlretrieve(
                        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
                        model_path,
                    )
                if not voices_path.exists():
                    logger.info("  Downloading voices-v1.0.bin (~27MB)...")
                    urllib.request.urlretrieve(
                        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
                        voices_path,
                    )
            else:
                logger.info("Kokoro models already cached")

            self.tts_engine = kokoro_onnx.Kokoro(str(model_path), str(voices_path))
            self.tts_g2p = EspeakG2P(language="fr-fr")
            logger.info("Kokoro TTS ready (voice: ff_siwis, French G2P via espeak)")

        except ImportError as e:
            logger.error(f"Kokoro TTS dependencies missing: {e}")
            logger.info("Install with: pip install kokoro-onnx misaki")
            raise
        except Exception as e:
            logger.error(f"Kokoro init error: {e}")
            import traceback
            traceback.print_exc()
            raise

    async def start_http_server(self):
        """Start HTTP server to host TTS audio files"""
        app = web.Application()
        app.router.add_post("/test", self.handle_test_prompt)
        app.router.add_static("/tts/", self.tts_dir, show_index=True)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", HTTP_PORT)
        await site.start()

        # Get local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
        finally:
            s.close()

        logger.info(f"HTTP TTS server started on http://{local_ip}:{HTTP_PORT}/tts/")
        logger.info(f"Test endpoint: curl -X POST http://{local_ip}:{HTTP_PORT}/test -d '{{\"text\": \"allume la lumière\"}}'")
        self.http_base_url = f"http://{local_ip}:{HTTP_PORT}/tts/"

    async def handle_test_prompt(self, request):
        """Test endpoint: bypass ESP/STT, send text directly to LLM -> HA -> TTS"""
        import json

        try:
            body = await request.json()
            text = body.get("text", "")
        except Exception:
            text = (await request.text()).strip()

        if not text:
            return web.json_response({"error": "missing 'text' field"}, status=400)

        logger.info(f"[TEST] Input: \"{text}\"")

        # Run LLM (no ESP api needed for tool execution)
        response_text = await self.process_with_llm(None, text)
        if not response_text:
            return web.json_response({"error": "LLM returned no response"}, status=500)

        # Generate TTS audio
        tts_url = await self.text_to_speech_file(response_text)

        result = {"input": text, "response": response_text, "tts_url": tts_url}
        logger.info(f"[TEST] Result: {json.dumps(result, ensure_ascii=False)}")
        return web.json_response(result)

    async def start(self):
        """Start the server and connect to devices"""
        logger.info("Voice Assistant Server started")
        logger.info(f"ESP: {ESP_HOST}:{ESP_PORT} | LLM: {LLM_URL} | HTTP: {HTTP_PORT}")

        await self.init_stt_engine()
        await self.init_tts_engine()
        await self.init_ha_client()
        await self.start_http_server()
        await self.connect_to_device(ESP_HOST)

    async def init_ha_client(self):
        """Connect to Home Assistant and discover entities."""
        from ha_client import HAClient

        if not HA_TOKEN:
            logger.info("No HA_TOKEN configured, device control disabled")
            return

        self.ha_client = HAClient(HA_URL, HA_TOKEN)
        if await self.ha_client.connect():
            logger.info(f"Home Assistant: {len(self.ha_client.entities)} entities discovered")
        else:
            logger.warning("Home Assistant not reachable, device control disabled")
            self.ha_client = None

    async def connect_to_device(self, host: str):
        """Connect to a specific ESPHome device"""
        try:
            logger.info(f"Connecting to {host}...")

            # Noise encryption (recommended) or password (legacy)
            if ESP_NOISE_PSK:
                logger.info("Auth: Noise encryption")
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
            logger.info(f"Connected to {host}")

            device_info = await api.device_info()
            logger.info(
                f"Device: {device_info.name} (v{device_info.esphome_version})"
            )

            if device_info.voice_assistant_feature_flags:
                logger.info(
                    f"Voice assistant supported (flags: {device_info.voice_assistant_feature_flags})"
                )
            else:
                logger.warning("Voice assistant may not be supported")

            self.devices[host] = api
            self.current_device = host

            await self.setup_voice_assistant(api, host)

        except Exception as e:
            logger.error(f"Connection error {host}: {e}")
            logger.info("Check that:")
            logger.info("   - The ESP is powered on and connected to WiFi")
            logger.info("   - The IP address is correct")
            logger.info("   - ESP_PASSWORD or ESP_NOISE_PSK is configured")

    async def setup_voice_assistant(self, api: APIClient, device_host: str):
        """Set up voice assistant subscription"""
        logger.info(f"Setting up voice assistant for {device_host}")

        try:
            unsubscribe = api.subscribe_voice_assistant(
                handle_start=self.handle_voice_assistant_start,
                handle_stop=self.handle_voice_assistant_stop,
                handle_audio=self.handle_voice_assistant_audio,
                handle_announcement_finished=self.handle_announcement_finished,
            )
            self.va_unsubscribe = unsubscribe
            logger.info("Voice assistant subscription successful")

        except Exception as e:
            logger.error(f"Voice assistant setup error: {e}")

    async def handle_voice_assistant_start(
        self,
        conversation_id: str,
        flags: int,
        audio_settings: Any,
        wake_word_phrase: str | None,
    ) -> int | None:
        """Called when the voice assistant starts"""
        logger.info(f"Voice Assistant START")
        logger.info(f"   Conversation ID: {conversation_id}")
        logger.info(f"   Flags: {flags}")
        logger.info(f"   Wake word: {wake_word_phrase}")
        logger.info(f"   Audio settings: {audio_settings}")

        self.conversation_id = conversation_id

        # Reset audio buffer for new session
        self.audio_buffer = bytearray()
        self.is_recording = True
        self.vad_speech_frames = 0
        self.vad_silence_frames = 0
        self.vad_has_speech = False

        # Tell the ESP to start recording and sending audio
        if self.current_device:
            api = self.devices[self.current_device]
            try:
                api.send_voice_assistant_event(
                    VoiceAssistantEventType.VOICE_ASSISTANT_STT_START, {}
                )
                logger.info("STT_START sent - ESP will send audio")

                # Start safety timeout task
                self.recording_task = asyncio.create_task(
                    self.monitor_recording_timeout(api)
                )

            except Exception as e:
                logger.error(f"Error sending STT_START: {e}")

        return 0

    async def monitor_recording_timeout(self, api: APIClient):
        """Safety timeout: stop recording after 30 seconds max"""
        max_recording_time = 30.0

        logger.info(f"Safety timeout: {max_recording_time}s")
        await asyncio.sleep(max_recording_time)

        if self.is_recording:
            logger.warning(f"SAFETY TIMEOUT reached ({max_recording_time}s)")
            await self.stop_recording(api, "timeout")

    async def stop_recording(self, api: APIClient, reason: str):
        """Stop recording and process the audio"""
        if not self.is_recording:
            return

        self.is_recording = False
        audio_data = bytes(self.audio_buffer)

        logger.info(f"Recording stopped ({reason}): {len(audio_data)} bytes")

        try:
            # Send STT_VAD_END - LEDs turn orange
            api.send_voice_assistant_event(
                VoiceAssistantEventType.VOICE_ASSISTANT_STT_VAD_END, {}
            )
            logger.info("STT_VAD_END sent - LEDs orange (thinking)")

            if len(audio_data) > 0:
                await self.process_voice_pipeline(api, audio_data)

        except Exception as e:
            logger.error(f"Error stopping recording: {e}")

    async def handle_voice_assistant_stop(self, abort: bool):
        """
        Called when the voice assistant stops
        - abort=False: normal stop (button released)
        - abort=True: interrupted (timeout, error, or VAD)
        """
        logger.info(f"Voice Assistant STOP (abort={abort})")

        if self.is_recording and len(self.audio_buffer) > 0:
            logger.info(f"Complete audio received: {len(self.audio_buffer)} bytes")
            self.is_recording = False

            if self.current_device:
                api = self.devices[self.current_device]
                await self.process_voice_pipeline(api, bytes(self.audio_buffer))
        else:
            logger.info("No audio to process")

    async def handle_announcement_finished(self, announcement_finished):
        """Called when the ESP finishes playing TTS audio"""
        logger.info("Announcement finished received from ESP")

        if self.current_device:
            api = self.devices[self.current_device]
            api.send_voice_assistant_event(
                VoiceAssistantEventType.VOICE_ASSISTANT_RUN_END, {}
            )
            logger.info("RUN_END sent - Pipeline complete, LEDs idle")

    async def handle_voice_assistant_audio(self, audio_bytes: bytes):
        """Handle audio received from the ESP with VAD detection"""
        if self.is_recording:
            self.audio_buffer.extend(audio_bytes)
            self.last_audio_time = time.time()

            await self.analyze_audio_vad(audio_bytes)

            if len(self.audio_buffer) % 10240 == 0 or len(self.audio_buffer) < 10240:
                logger.info(f"Audio: {len(self.audio_buffer)} bytes")
        else:
            logger.debug("Audio received after recording stopped (normal)")

    async def analyze_audio_vad(self, audio_bytes: bytes):
        """Analyze audio with Silero VAD to detect end of speech"""
        from silero_vad_lite import SileroVAD
        import array

        if not hasattr(self, "vad"):
            self.vad = SileroVAD(16000)

        # Silero VAD uses 32ms frames (512 samples at 16kHz)
        # Input: float32 [-1, 1], 512 samples = 1024 bytes as int16
        frame_samples = 512  # 32ms at 16kHz
        frame_size = frame_samples * 2  # 1024 bytes as int16

        for i in range(0, len(audio_bytes), frame_size):
            frame = audio_bytes[i : i + frame_size]
            if len(frame) != frame_size:
                continue

            try:
                # Convert int16 PCM -> float32 [-1, 1] for Silero
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
                        logger.info("Speech detected")
                else:
                    if self.vad_has_speech:
                        self.vad_silence_frames += 1

                # Stop after ~1 second of silence (31 frames of 32ms)
                if self.vad_has_speech and self.vad_silence_frames >= 31:
                    logger.info(
                        f"Silence detected after speech ({self.vad_silence_frames} frames)"
                    )
                    if self.current_device and self.is_recording:
                        api = self.devices[self.current_device]
                        await self.stop_recording(api, "VAD silence")
                    return

            except Exception as e:
                logger.debug(f"VAD error: {e}")
                continue

    # === EVENT HANDLERS ===

    async def handle_run_start(self, event):
        """Pipeline started on the ESP"""
        logger.info("Pipeline started")
        self.conversation_id = getattr(
            event, "conversation_id", f"conv_{int(time.time())}"
        )

    async def handle_wake_word_start(self, event):
        """Wake word detected"""
        wake_word = getattr(event, "wake_word_phrase", "unknown")
        logger.info(f"Wake word detected: '{wake_word}'")

    async def handle_wake_word_end(self, event):
        """Wake word ended"""
        logger.info("Wake word ended")

    async def handle_vad_start(self, event):
        """Voice activity detection started"""
        logger.info("Listening started (VAD start)")

    async def handle_vad_end(self, event):
        """Voice activity detection ended"""
        logger.info("Listening ended (VAD end) - processing audio...")

    async def handle_voice_error(self, event):
        """Voice assistant error"""
        error_code = getattr(event, "code", "unknown")
        error_message = getattr(event, "message", "Unknown error")
        logger.error(f"Voice assistant error: {error_code} - {error_message}")

    # === MAIN PIPELINE ===

    async def process_voice_pipeline(self, api: APIClient, audio_bytes: bytes):
        """
        Main pipeline: Audio -> STT (Parakeet) -> LLM (SmolLM3) -> TTS (Kokoro)
        """
        logger.info("Starting full voice pipeline")

        try:
            # 1. STT - Speech to Text with Parakeet
            transcript = await self.speech_to_text(audio_bytes)
            if not transcript:
                await self.send_error_to_device(api, "STT failed")
                return

            api.send_voice_assistant_event(
                VoiceAssistantEventType.VOICE_ASSISTANT_STT_END,
                {"text": transcript},
            )
            logger.info(f'STT_END sent: "{transcript}"')

            # 2. LLM - Process text with Qwen
            response_text = await self.process_with_llm(api, transcript)
            if not response_text:
                await self.send_error_to_device(api, "LLM processing failed")
                return

            # 3. TTS - Text to Speech with Piper
            await self.text_to_speech(api, response_text)

            logger.info("Pipeline completed successfully")

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            await self.send_error_to_device(api, f"Pipeline error: {e}")

    # === FUNCTION CALLING ===

    def get_available_functions(self) -> list:
        """Return available functions for LLM function calling"""
        if not self.ha_client:
            return []

        room_param = {"type": "string", "description": "Pièce ou groupe (ex: chambre Charlie, salon, enfants, partout)"}

        return [
            {
                "name": "turn_on",
                "description": "Allumer un appareil (lumière, prise, etc.)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entity": {"type": "string", "description": "Nom de l'appareil"},
                        "room": room_param,
                        "brightness": {"type": "integer", "description": "Luminosité 0-100, seulement pour les lumières"},
                    },
                    "required": ["entity"],
                },
            },
            {
                "name": "turn_off",
                "description": "Éteindre un appareil (lumière, prise, etc.)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entity": {"type": "string", "description": "Nom de l'appareil"},
                        "room": room_param,
                    },
                    "required": ["entity"],
                },
            },
            {
                "name": "open_cover",
                "description": "Ouvrir un volet ou un store",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entity": {"type": "string", "description": "Nom du volet"},
                        "room": room_param,
                    },
                    "required": ["entity"],
                },
            },
            {
                "name": "close_cover",
                "description": "Fermer un volet ou un store",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entity": {"type": "string", "description": "Nom du volet"},
                        "room": room_param,
                    },
                    "required": ["entity"],
                },
            },
            {
                "name": "set_temperature",
                "description": "Régler la température d'un thermostat",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entity": {"type": "string", "description": "Nom du thermostat"},
                        "room": room_param,
                        "temperature": {"type": "number", "description": "Température en degrés Celsius"},
                    },
                    "required": ["entity", "temperature"],
                },
            },
            {
                "name": "get_state",
                "description": "Obtenir l'état actuel d'un appareil (allumé/éteint, température, ouvert/fermé)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entity": {"type": "string", "description": "Nom de l'appareil"},
                        "room": room_param,
                    },
                    "required": ["entity"],
                },
            },
        ]

    async def execute_function(self, function_name: str, arguments: dict) -> str:
        """Execute a function by calling Home Assistant"""
        if not self.ha_client:
            return "Home Assistant non disponible"

        entity_name = arguments.get("entity", "")
        room = arguments.get("room")

        # Domain hints per tool
        domain_map = {
            "turn_on": ["light", "switch", "media_player"],
            "turn_off": ["light", "switch", "media_player"],
            "open_cover": ["cover"],
            "close_cover": ["cover"],
            "set_temperature": ["climate"],
            "get_state": None,
        }

        domains = domain_map.get(function_name)

        # Resolve entity name(s) — may return multiple for room groups
        entity_ids = self.ha_client.resolve_all_entities(entity_name, room=room, domain_hints=domains)
        if not entity_ids:
            return f"Appareil {entity_name} non trouvé"

        # get_state is a read operation (use first entity only)
        if function_name == "get_state":
            state_data = await self.ha_client.get_entity_state(entity_ids[0])
            if state_data:
                return self.ha_client.format_state_for_speech(entity_ids[0], state_data)
            return f"Impossible de lire l'état de {entity_name}"

        # Service calls — loop over all resolved entities
        service = {
            "turn_on": "turn_on",
            "turn_off": "turn_off",
            "open_cover": "open_cover",
            "close_cover": "close_cover",
            "set_temperature": "set_temperature",
        }.get(function_name, function_name)

        extra = {}
        if function_name == "set_temperature" and "temperature" in arguments:
            extra["temperature"] = arguments["temperature"]
        if function_name == "turn_on" and "brightness" in arguments:
            extra["brightness_pct"] = arguments["brightness"]

        for entity_id in entity_ids:
            domain = entity_id.split(".")[0]
            await self.ha_client.call_service(domain, service, entity_id, **extra)

        # Compact response for multi-entity commands
        if len(entity_ids) > 1:
            whole_house = room and room.lower().strip() in ("tout", "toute la maison", "partout")
            if whole_house:
                action_map = {
                    "close_cover": "Tous les volets en cours de fermeture",
                    "open_cover": "Tous les volets en cours d'ouverture",
                    "turn_on": "Toutes les lumières allumées",
                    "turn_off": "Toutes les lumières éteintes",
                }
            else:
                room_label = room or "la maison"
                action_map = {
                    "close_cover": f"Volets {room_label} en cours de fermeture",
                    "open_cover": f"Volets {room_label} en cours d'ouverture",
                    "turn_on": f"Lumières {room_label} allumées",
                    "turn_off": f"Lumières {room_label} éteintes",
                }
            return action_map.get(function_name, "C'est fait")

        return self.ha_client._build_response(
            entity_ids[0].split(".")[0], service,
            self.ha_client.entities.get(entity_ids[0], {}).get("friendly_name", entity_ids[0]),
            extra,
        )

    async def init_stt_engine(self):
        """Initialize Parakeet MLX STT model"""
        import parakeet_mlx

        logger.info("Loading Parakeet STT model...")
        self.stt_model = parakeet_mlx.from_pretrained(
            "mlx-community/parakeet-tdt-0.6b-v3"
        )
        logger.info("Parakeet STT model loaded")

    async def speech_to_text(self, audio_bytes: bytes) -> Optional[str]:
        """STT with Parakeet MLX (local, Apple Silicon optimized)"""
        logger.info(f"STT Parakeet: {len(audio_bytes)} bytes...")

        try:
            # Create temporary WAV file
            temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            with wave.open(temp_wav.name, "wb") as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(16000)  # 16kHz
                wav_file.writeframes(audio_bytes)

            logger.info(f"Audio WAV: {temp_wav.name} ({os.path.getsize(temp_wav.name)} bytes)")

            # Transcribe with Parakeet (run in executor to avoid blocking)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self.stt_model.transcribe, temp_wav.name
            )
            transcript = result.text.strip()

            os.unlink(temp_wav.name)

            logger.info(f'Transcript: "{transcript}"')
            return transcript if transcript else None

        except Exception as e:
            logger.error(f"STT error: {e}")
            import traceback
            traceback.print_exc()
            return None

    _TOOL_NAMES = {"turn_on", "turn_off", "open_cover", "close_cover", "set_temperature", "get_state"}

    def _parse_text_tool_call(self, content: str) -> tuple[Optional[str], Optional[dict]]:
        """Parse tool calls that Qwen outputs as plain text instead of structured tool_calls.
        Handles: fn({"entity":"X","room":"Y"}), fn("X"), fn(entity="X", room="Y")
        """
        import re
        import json

        # Pattern 1: fn({"key": "val", ...})
        m = re.match(r'(\w+)\s*\(\s*(\{.*\})\s*\)', content, re.DOTALL)
        if m and m.group(1) in self._TOOL_NAMES:
            try:
                return m.group(1), json.loads(m.group(2))
            except json.JSONDecodeError:
                pass

        # Pattern 2: fn("value")
        m = re.match(r'(\w+)\s*\(\s*"([^"]+)"\s*\)', content)
        if m and m.group(1) in self._TOOL_NAMES:
            return m.group(1), {"entity": m.group(2)}

        # Pattern 3: fn(entity="X", room="Y") — Python-style kwargs
        m = re.match(r'(\w+)\s*\((.+)\)', content, re.DOTALL)
        if m and m.group(1) in self._TOOL_NAMES:
            args = {}
            for kv in re.findall(r'(\w+)\s*=\s*"([^"]*)"', m.group(2)):
                args[kv[0]] = kv[1]
            if args:
                return m.group(1), args

        # Pattern 4: French natural language action (e.g. "Éteins les appliques salon.")
        # Only triggered when conversation history exists (follow-up command)
        if self.conversation_history:
            action_map = {
                "allume": "turn_on", "rallume": "turn_on",
                "éteins": "turn_off", "eteins": "turn_off",
                "ferme": "close_cover", "ouvre": "open_cover",
            }
            lower = content.lower().rstrip(".")
            for verb, func in action_map.items():
                if lower.startswith(verb):
                    entity_part = lower[len(verb):].strip()
                    # Strip articles
                    for article in ("les ", "le ", "la ", "l'", "l'"):
                        if entity_part.startswith(article):
                            entity_part = entity_part[len(article):]
                            break
                    if entity_part:
                        logger.info(f"French action fallback: '{content}' -> {func}('{entity_part}')")
                        return func, {"entity": entity_part}

        return None, None

    async def process_with_llm(self, api: APIClient, text: str) -> Optional[str]:
        """LLM via OpenAI-compatible API (local llama.cpp or cloud provider)"""
        import json
        import time

        logger.info(f'LLM input: "{text}"')

        is_local = not LLM_API_KEY

        try:
            # Build system prompt with available entities
            system_prompt = (
                "Tu es un assistant vocal pour la maison connectée. "
                "Réponds en français, en 1-2 phrases courtes maximum. "
                "Pas de markdown. Parle naturellement comme à l'oral. "
                "Quand on te demande de contrôler un appareil (même avec un pronom comme 'la' ou 'les'), utilise TOUJOURS les fonctions disponibles. "
                "Passe le paramètre room quand la pièce est mentionnée. "
                "Utilise les noms de groupes tels quels (ex: room='enfants'), ne les décompose pas."
            )
            # Qwen-specific: disable thinking for lower latency
            if is_local:
                system_prompt = "/no_think " + system_prompt
            if self.ha_client:
                entity_list = self.ha_client.get_entity_list_for_prompt()
                if entity_list:
                    system_prompt += f"\nAppareils disponibles: {entity_list}"

            tools = self.get_available_functions()

            # Expire old conversation history
            now = time.time()
            if now - self.last_interaction_time > self.CONVERSATION_TIMEOUT:
                if self.conversation_history:
                    logger.info("Conversation history expired, starting fresh")
                self.conversation_history = []
            self.last_interaction_time = now

            # Build messages: system + history + current user message
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(self.conversation_history)
            messages.append({"role": "user", "content": text})

            max_tokens_key = "max_completion_tokens" if LLM_API_KEY else "max_tokens"
            payload = {
                "messages": messages,
                max_tokens_key: 150,
                "temperature": 0.3,
            }
            if LLM_MODEL:
                payload["model"] = LLM_MODEL
            if tools:
                payload["tools"] = [
                    {"type": "function", "function": func} for func in tools
                ]

            tool_names = [t["function"]["name"] for t in payload.get("tools", [])]
            logger.info(f'LLM prompt: [system] ...voice assistant... [user] "{text}"')
            if tool_names:
                logger.info(f"Tools: {tool_names}")

            headers = {"Content-Type": "application/json"}
            if LLM_API_KEY:
                headers["Authorization"] = f"Bearer {LLM_API_KEY}"

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    LLM_URL, json=payload, headers=headers,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"LLM raw response: {json.dumps(result, ensure_ascii=False, indent=2)}")
                        message = result["choices"][0]["message"]

                        # Helper to save exchange to conversation history
                        def save_to_history(response_text):
                            self.conversation_history.append({"role": "user", "content": text})
                            self.conversation_history.append({"role": "assistant", "content": response_text})
                            # Keep last 5 exchanges (10 messages) to limit prompt size
                            if len(self.conversation_history) > 10:
                                self.conversation_history = self.conversation_history[-10:]

                        # Check if the LLM wants to call a function
                        if "tool_calls" in message and message["tool_calls"]:
                            tool_call = message["tool_calls"][0]
                            function_name = tool_call["function"]["name"]
                            function_args = json.loads(
                                tool_call["function"]["arguments"]
                            )

                            logger.info(f"Function call: {function_name}({function_args})")

                            try:
                                function_result = await self.execute_function(
                                    function_name, function_args
                                )
                                logger.info(f"Function result: {function_result}")

                                save_to_history(function_result)
                                return function_result

                            except Exception as e:
                                logger.error(f"Function execution error: {e}")
                                return "Désolé, une erreur s'est produite"
                        else:
                            # Direct response
                            content = message.get("content", "")
                            # Qwen 3 in thinking mode may wrap in <think>...</think>
                            if is_local and "</think>" in content:
                                content = content.split("</think>")[-1]
                            content = content.strip()

                            # Fallback: Qwen sometimes puts tool calls as text
                            fn_name, fn_args = self._parse_text_tool_call(content)
                            if fn_name and fn_args:
                                try:
                                    logger.info(f"Function call (text fallback): {fn_name}({fn_args})")
                                    result = await self.execute_function(fn_name, fn_args)
                                    logger.info(f"Function result: {result}")
                                    save_to_history(result)
                                    return result
                                except Exception as e:
                                    logger.warning(f"Failed to execute text tool call: {e}")

                            logger.info(f'LLM response: "{content}"')
                            if content:
                                save_to_history(content)
                            return content if content else None
                    else:
                        error_text = await response.text()
                        logger.error(f"LLM error {response.status}: {error_text}")
                        return None

        except Exception as e:
            logger.error(f"LLM error: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def text_to_speech_file(self, text: str) -> str:
        """Generate a TTS WAV file and return its URL."""
        import numpy as np
        import hashlib
        import re

        filename = f"tts_{int(time.time())}_{hashlib.md5(text.encode()).hexdigest()[:8]}.wav"
        output_path = self.tts_dir / filename

        # Strip markdown artifacts before TTS
        clean_text = re.sub(r'\*+', '', text)
        clean_text = re.sub(r'#+\s*', '', clean_text)
        clean_text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', clean_text)
        clean_text = re.sub(r'`([^`]+)`', r'\1', clean_text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()

        logger.info(f"Synthesizing: '{clean_text[:80]}...'")

        # French phonemization via espeak
        phonemes = self.tts_g2p(clean_text)
        logger.info(f"Phonemes: {phonemes}")

        # Generate audio with Kokoro (runs sync, use executor)
        loop = asyncio.get_event_loop()
        audio_24k, sr = await loop.run_in_executor(
            None,
            lambda: self.tts_engine.create(
                phonemes, voice="ff_siwis", speed=1.0, lang="fr-fr", is_phonemes=True
            ),
        )

        logger.info(f"Kokoro output: {len(audio_24k)} samples at {sr}Hz")

        # Resample 24kHz -> 16kHz (ESP expects 16kHz)
        target_sr = 16000
        num_samples_16k = int(len(audio_24k) * target_sr / sr)
        indices = np.linspace(0, len(audio_24k) - 1, num_samples_16k)
        audio_16k = np.interp(indices, np.arange(len(audio_24k)), audio_24k).astype(np.float32)

        # Convert float32 [-1, 1] -> int16 (PCM 16-bit)
        audio_int16 = (np.clip(audio_16k, -1.0, 1.0) * 32767).astype(np.int16)

        # Save as WAV file
        with wave.open(str(output_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(target_sr)
            wav_file.writeframes(audio_int16.tobytes())

        logger.info(f"TTS generated: {output_path} ({output_path.stat().st_size} bytes)")
        return f"{self.http_base_url}{filename}"

    async def text_to_speech(self, api: APIClient, text: str):
        """Text-to-Speech with ESP events (for live pipeline)"""
        logger.info(f'TTS: generating audio for "{text}"')

        api.send_voice_assistant_event(
            VoiceAssistantEventType.VOICE_ASSISTANT_TTS_START, {"text": text}
        )
        logger.info("TTS_START sent")

        try:
            audio_url = await self.text_to_speech_file(text)
            api.send_voice_assistant_event(
                VoiceAssistantEventType.VOICE_ASSISTANT_TTS_END, {"url": audio_url}
            )
            logger.info(f"TTS_END sent with URL: {audio_url}")

        except Exception as e:
            logger.error(f"TTS error: {e}")
            import traceback
            traceback.print_exc()
            raise

    async def send_error_to_device(self, api: APIClient, error_message: str):
        """Send an error event to the ESP"""
        logger.error(f"Sending error to ESP: {error_message}")

        try:
            api.send_voice_assistant_event(
                VoiceAssistantEventType.VOICE_ASSISTANT_ERROR,
                {"code": "server_error", "message": error_message},
            )
            logger.info("VOICE_ASSISTANT_ERROR event sent to ESP")
        except Exception as e:
            logger.error(f"Error sending error: {e}")


async def main():
    """Main entry point"""
    print("Custom Voice Assistant Server for ESPHome (Python)")
    print("Replaces Home Assistant for voice pipeline")
    print("STT: Parakeet MLX | LLM: Qwen 3 4B (llama.cpp) | TTS: Kokoro\n")

    server = VoiceAssistantServer()

    try:
        await server.start()

        logger.info("Server running - Ctrl+C to stop")
        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        logger.info("\nShutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
