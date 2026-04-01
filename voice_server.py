#!/usr/bin/env python3
"""
Custom Voice Assistant Server for ESPHome
Replaces Home Assistant for the voice pipeline
"""

import asyncio
import datetime
import locale
import logging
import os
import signal
import socket
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

from aioesphomeapi import (
    APIClient,
    VoiceAssistantEventType,
)
from aioesphomeapi.reconnect_logic import ReconnectLogic
from aiohttp import web

from ha_client import LOCAL_CONFIG

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

try:
    locale.setlocale(locale.LC_TIME, "fr_FR.UTF-8")
except locale.Error:
    pass  # Fall back to system locale

# Configuration via environment variables
ESP_HOST = os.environ.get("ESP_HOST", "")
ESP_PORT = int(os.environ.get("ESP_PORT", "6053"))
ESP_PASSWORD = os.environ.get("ESP_PASSWORD", "")
ESP_NOISE_PSK = os.environ.get("ESP_NOISE_PSK", "")
LLM_URL = os.environ.get("LLM_URL", "http://localhost:8080/v1/chat/completions")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "")  # e.g. "gpt-5.4-nano" — empty = local llama.cpp
HTTP_PORT = int(os.environ.get("HTTP_PORT", "8888"))
HA_URL = os.environ.get("HA_URL", "")
HA_TOKEN = os.environ.get("HA_TOKEN", "")


def _format_duration(total_seconds: int) -> str:
    """Format seconds into a French human-readable duration."""
    if total_seconds >= 3600:
        h = total_seconds // 3600
        m = (total_seconds % 3600) // 60
        if m:
            return f"{h}h{m:02d}"
        return f"{h} heure{'s' if h > 1 else ''}"
    if total_seconds >= 60:
        m = total_seconds // 60
        s = total_seconds % 60
        if s:
            return f"{m} minute{'s' if m > 1 else ''} {s}s"
        return f"{m} minute{'s' if m > 1 else ''}"
    return f"{total_seconds} seconde{'s' if total_seconds > 1 else ''}"


def _format_family_for_prompt(family: list) -> str:
    """Format family member list with ages for injection into system prompt."""
    today = datetime.date.today()
    parts = []
    role_fr = {"parent": "parent", "child": "enfant"}
    for m in family:
        age = today.year - int(m["birth_date"][:4])
        # Adjust if birthday hasn't occurred yet this year
        bday_this_year = datetime.date(today.year, int(m["birth_date"][5:7]), int(m["birth_date"][8:10]))
        if bday_this_year > today:
            age -= 1
        parts.append(f"{m['name']} ({role_fr.get(m['role'], m['role'])}, {age} ans)")
    return "Membres de la famille: " + ", ".join(parts)


class VoiceAssistantServer:
    """
    Custom voice assistant server for ESPHome
    Handles the STT -> LLM -> TTS pipeline
    """

    def __init__(self):
        self.devices: Dict[str, APIClient] = {}
        self._reconnect_logic: Optional[ReconnectLogic] = None

        # Voice pipeline state
        self.conversation_id = None
        self.current_device = None
        self.audio_buffer = bytearray()
        self.is_recording = False
        self.recording_task = None
        self.last_audio_time = 0

        # Sub-engines (initialized in start())
        self.tts = None  # tts.KokoroTTS
        self.stt = None  # stt.ParakeetSTT

        # Home Assistant client
        self.ha_client = None

        # Google Calendar client
        self.calendar_client = None

        # Timer manager
        from timer import TimerManager

        self.timer_manager = TimerManager()

        # Conversation history for multi-turn context
        self.conversation_history = []
        self.conversation_id = None  # Groups exchanges in the same conversation
        self.last_interaction_time = 0
        self.CONVERSATION_TIMEOUT = 120  # seconds — forget context after 2 min of silence

        # Continuous conversation
        self.is_followup = False  # True when pipeline was started without wake word
        self._end_conversation = False  # Set by end_conversation tool

        # Exchange log for web UI (persisted to disk)
        from web_ui import ExchangeLog

        self.exchange_log = ExchangeLog(Path(__file__).parent / "exchanges.json")
        self._last_tool_calls = []

        # HTTP server for serving TTS audio files
        self.http_server = None
        self.tts_dir = Path(tempfile.gettempdir()) / "voice_assistant_tts"
        self.tts_dir.mkdir(exist_ok=True)
        logger.info(f"TTS directory: {self.tts_dir}")

    # === INIT ===

    async def start(self):
        """Start the server and connect to devices"""
        logger.info("Voice Assistant Server started")
        logger.info(f"ESP: {ESP_HOST}:{ESP_PORT} | LLM: {LLM_URL} | HTTP: {HTTP_PORT}")

        from stt import ParakeetSTT
        from tts import KokoroTTS

        self.stt = ParakeetSTT()
        await self.stt.init()

        self.tts = KokoroTTS()
        await self.tts.init()

        await self.init_ha_client()
        await self.init_calendar_client()
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

    async def init_calendar_client(self):
        """Connect to Google Calendar API."""
        from calendar_client import CalendarClient

        project_dir = Path(__file__).parent
        credentials_path = project_dir / os.environ.get("CALENDAR_CREDENTIALS", "client_secret.json")
        token_path = project_dir / os.environ.get("CALENDAR_TOKEN", "token.json")

        if not token_path.exists():
            logger.info("No calendar token found (run setup_calendar.py), calendar disabled")
            return

        self.calendar_client = CalendarClient(credentials_path, token_path)
        if await self.calendar_client.connect():
            logger.info("Google Calendar connected")
        else:
            logger.warning("Google Calendar not available, calendar disabled")
            self.calendar_client = None

    async def start_http_server(self):
        """Start HTTP server to host TTS audio files and web UI"""
        from web_ui import setup_routes

        app = web.Application()
        setup_routes(app, self)
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
        logger.info(
            f'Test endpoint: curl -X POST http://{local_ip}:{HTTP_PORT}/test -d \'{{"text": "allume la lumière"}}\''
        )
        self.http_base_url = f"http://{local_ip}:{HTTP_PORT}/tts/"

    # === ESP CONNECTION ===

    async def connect_to_device(self, host: str):
        """Connect to an ESPHome device with automatic reconnection."""
        logger.info(f"Connecting to {host}...")

        if ESP_NOISE_PSK:
            logger.info("Auth: Noise encryption")
            api = APIClient(
                host,
                ESP_PORT,
                None,
                noise_psk=ESP_NOISE_PSK,
                client_info="voice-server-python",
            )
        else:
            api = APIClient(
                host,
                ESP_PORT,
                ESP_PASSWORD,
                client_info="voice-server-python",
            )

        async def on_connect() -> None:
            device_info = await api.device_info()
            logger.info(f"Connected to {host}: {device_info.name} (v{device_info.esphome_version})")

            if device_info.voice_assistant_feature_flags:
                logger.info(f"Voice assistant supported (flags: {device_info.voice_assistant_feature_flags})")
            else:
                logger.warning("Voice assistant may not be supported")

            self.devices[host] = api
            self.current_device = host
            await self.setup_voice_assistant(api, host)

        async def on_disconnect(expected_disconnect: bool) -> None:
            reason = "expected" if expected_disconnect else "unexpected"
            logger.warning(f"ESP {host} disconnected ({reason}), will reconnect automatically")
            self.devices.pop(host, None)
            if self.current_device == host:
                self.current_device = None

        self._reconnect_logic = ReconnectLogic(
            client=api,
            on_connect=on_connect,
            on_disconnect=on_disconnect,
            name=host,
        )
        await self._reconnect_logic.start()

    async def setup_voice_assistant(self, api: APIClient, device_host: str):
        """Set up voice assistant subscription"""
        logger.info(f"Setting up voice assistant for {device_host}")

        try:
            unsubscribe = api.subscribe_voice_assistant(
                handle_start=self.handle_voice_assistant_start,
                handle_stop=self.handle_voice_assistant_stop,
                handle_audio=self.handle_voice_assistant_audio,
            )
            self.va_unsubscribe = unsubscribe
            logger.info("Voice assistant subscription successful")

        except Exception as e:
            logger.error(f"Voice assistant setup error: {e}")

    # === VOICE ASSISTANT EVENT HANDLERS ===

    async def handle_voice_assistant_start(
        self,
        conversation_id: str,
        flags: int,
        audio_settings: Any,
        wake_word_phrase: str | None,
    ) -> int | None:
        """Called when the voice assistant starts"""
        logger.info("Voice Assistant START")
        logger.info(f"   Conversation ID: {conversation_id}")
        logger.info(f"   Flags: {flags}")
        logger.info(f"   Wake word: {wake_word_phrase}")
        logger.info(f"   Audio settings: {audio_settings}")

        self.is_followup = wake_word_phrase is None
        # Only overwrite conversation_id for new conversations (wake word detected).
        # Follow-ups keep the server-assigned ID so exchanges group correctly.
        if not self.is_followup:
            self.conversation_id = None

        # Cancel any previous recording timeout
        if self.recording_task and not self.recording_task.done():
            self.recording_task.cancel()

        self.audio_buffer = bytearray()
        self.is_recording = True
        self.vad_speech_frames = 0
        self.vad_silence_frames = 0
        self.vad_has_speech = False
        # In follow-up mode, skip initial audio to avoid picking up TTS echo
        self.skip_audio_until = time.time() + 0.5 if self.is_followup else 0

        if self.current_device:
            api = self.devices[self.current_device]
            try:
                api.send_voice_assistant_event(VoiceAssistantEventType.VOICE_ASSISTANT_STT_START, {})
                logger.info(f"STT_START sent - {'follow-up' if self.is_followup else 'wake word'} mode")
                self.recording_task = asyncio.create_task(self.monitor_recording_timeout(api))
            except Exception as e:
                logger.error(f"Error sending STT_START: {e}")

        return 0

    async def monitor_recording_timeout(self, api: APIClient):
        """Safety timeout. In follow-up mode, wait 5s for speech to start,
        then extend to 30s once speech is detected."""
        if self.is_followup:
            # Short timeout: end if nobody speaks
            await asyncio.sleep(5.0)
            if self.is_recording and not self.vad_has_speech:
                logger.info("No speech in follow-up - stopping")
                await self.stop_recording(api, "no speech")
                return
            # Speech detected: switch to normal timeout for the rest
            remaining = 25.0  # 30s total max
            logger.info("Speech detected in follow-up, extending timeout to 30s")
            await asyncio.sleep(remaining)
        else:
            await asyncio.sleep(30.0)

        if self.is_recording:
            logger.warning("SAFETY TIMEOUT reached (30s)")
            await self.stop_recording(api, "timeout")

    async def stop_recording(self, api: APIClient, reason: str):
        """Stop recording and process the audio"""
        if not self.is_recording:
            return

        self.is_recording = False
        audio_data = bytes(self.audio_buffer)
        logger.info(f"Recording stopped ({reason}): {len(audio_data)} bytes")

        try:
            api.send_voice_assistant_event(VoiceAssistantEventType.VOICE_ASSISTANT_STT_VAD_END, {})
            logger.info("STT_VAD_END sent - LEDs orange (thinking)")

            if len(audio_data) > 0:
                await self.process_voice_pipeline(api, audio_data)

        except Exception as e:
            logger.error(f"Error stopping recording: {e}")

    async def handle_voice_assistant_stop(self, abort: bool):
        """Called when the voice assistant stops"""
        logger.info(f"Voice Assistant STOP (abort={abort})")

        if self.is_recording and len(self.audio_buffer) > 0:
            logger.info(f"Complete audio received: {len(self.audio_buffer)} bytes")
            self.is_recording = False

            if self.current_device:
                api = self.devices[self.current_device]
                await self.process_voice_pipeline(api, bytes(self.audio_buffer))
        else:
            logger.info("No audio to process")

    async def _play_tts_and_continue(self, api: APIClient, audio_url: str):
        """Play TTS audio via announcement API and start follow-up listening.
        Uses send_voice_assistant_announcement_await_response which both plays
        the audio and tells the ESP to start a new pipeline when done.

        Note: with start_conversation=True, ESPHome firmware (v2025.6.2) only sends
        VoiceAssistantAnnounceFinished after the full conversation cycle completes, not
        after the audio ends. We use a short timeout to cancel the pending Future before
        the ESP sends its late response (~25-37s), which would otherwise crash the API
        connection and trigger an ESP soft-reset (red LEDs).
        """
        try:
            logger.info(f"Playing TTS and requesting follow-up: {audio_url}")
            result = await api.send_voice_assistant_announcement_await_response(
                media_id=audio_url,
                timeout=8.0,
                text="",
                start_conversation=True,
            )
            logger.info(f"TTS playback done, follow-up started (success={result.success})")
        except TimeoutError:
            logger.info("TTS playing (follow-up starting, no announce confirmation)")
        except Exception as e:
            logger.warning(f"TTS announcement error: {e}")

    async def handle_voice_assistant_audio(self, audio_bytes: bytes):
        """Handle audio received from the ESP with VAD detection"""
        if self.is_recording:
            # Skip early audio in follow-up mode to avoid TTS echo
            if self.skip_audio_until and time.time() < self.skip_audio_until:
                return

            self.audio_buffer.extend(audio_bytes)
            self.last_audio_time = time.time()

            await self.analyze_audio_vad(audio_bytes)

            if len(self.audio_buffer) % 10240 == 0 or len(self.audio_buffer) < 10240:
                logger.info(f"Audio: {len(self.audio_buffer)} bytes")
        else:
            logger.debug("Audio received after recording stopped (normal)")

    async def analyze_audio_vad(self, audio_bytes: bytes):
        """Analyze audio with Silero VAD to detect end of speech"""
        import array

        from silero_vad_lite import SileroVAD

        if not hasattr(self, "vad"):
            self.vad = SileroVAD(16000)

        frame_samples = 512  # 32ms at 16kHz
        frame_size = frame_samples * 2  # 1024 bytes as int16

        for i in range(0, len(audio_bytes), frame_size):
            frame = audio_bytes[i : i + frame_size]
            if len(frame) != frame_size:
                continue

            try:
                int16_samples = array.array("h")
                int16_samples.frombytes(frame)
                float_samples = array.array("f", [s / 32768.0 for s in int16_samples])

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

                if self.vad_has_speech and self.vad_silence_frames >= 31:
                    logger.info(f"Silence detected after speech ({self.vad_silence_frames} frames)")
                    if self.current_device and self.is_recording:
                        api = self.devices[self.current_device]
                        await self.stop_recording(api, "VAD silence")
                    return

            except Exception as e:
                logger.debug(f"VAD error: {e}")
                continue

    # === MAIN PIPELINE ===

    async def process_voice_pipeline(self, api: APIClient, audio_bytes: bytes):
        """Main pipeline: Audio -> STT -> LLM -> TTS"""
        logger.info("Starting full voice pipeline")
        self._end_conversation = False

        try:
            timings = {}

            # 1. STT
            t0 = time.time()
            transcript = await self.stt.transcribe(audio_bytes)
            timings["stt"] = round(time.time() - t0, 2)
            if not transcript:
                if self.is_followup:
                    # Normal: no speech during follow-up, end conversation quietly
                    logger.info("No speech during follow-up - ending conversation")
                    api.send_voice_assistant_event(VoiceAssistantEventType.VOICE_ASSISTANT_RUN_END, {})
                else:
                    await self.send_error_to_device(api, "STT failed")
                return

            api.send_voice_assistant_event(
                VoiceAssistantEventType.VOICE_ASSISTANT_STT_END,
                {"text": transcript},
            )
            logger.info(f'STT_END sent: "{transcript}"')

            # 2. LLM
            t0 = time.time()
            response_text = await self.process_with_llm(api, transcript)
            timings["llm"] = round(time.time() - t0, 2)
            # Capture conversation_id now — TTS triggers ESP follow-up which overwrites it
            conversation_id = self.conversation_id
            if not response_text:
                self.exchange_log.add(
                    "voice",
                    transcript,
                    transcript,
                    "[Erreur LLM: pas de réponse]",
                    timings=timings,
                    conversation_id=conversation_id,
                )
                await self.send_error_to_device(api, "LLM processing failed")
                return

            # 3. TTS
            t0 = time.time()
            try:
                tts_timings = await self.text_to_speech(api, response_text)
                timings["tts_gen"] = tts_timings["gen"]
                timings["tts_play"] = tts_timings["play"]
                timings["tts"] = round(time.time() - t0, 2)
            except Exception as e:
                logger.error(f"TTS failed, but exchange will still be logged: {e}")
                await self.send_error_to_device(api, f"TTS error: {e}")
            timings["total"] = round(timings["stt"] + timings["llm"] + timings.get("tts_gen", 0), 2)

            self.exchange_log.add(
                "voice",
                transcript,
                transcript,
                response_text,
                timings=timings,
                tool_calls=self._last_tool_calls,
                conversation_id=conversation_id,
            )
            self._last_tool_calls = []
            logger.info(
                f"Pipeline: STT={timings['stt']}s LLM={timings['llm']}s "
                f"TTS={timings.get('tts_gen', '?')}s+{timings.get('tts_play', '?')}s "
                f"total={timings['total']}s"
            )

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            await self.send_error_to_device(api, f"Pipeline error: {e}")

    # === FUNCTION CALLING ===

    async def execute_function(self, function_name: str, arguments: dict) -> str:
        """Execute a function call from the LLM."""
        if function_name == "end_conversation":
            self._end_conversation = True
            return "Conversation terminée"

        # Timers — standalone, doesn't need HA
        if function_name == "set_timer":
            duration_minutes = arguments.get("duration_minutes", 0)
            total_seconds = int(duration_minutes * 60)
            label = arguments.get("label")
            timer = self.timer_manager.start_timer(total_seconds, label, self._handle_timer_event)
            if label:
                return f"Timer {label} de {_format_duration(total_seconds)} lancé"
            return f"Timer de {_format_duration(total_seconds)} lancé"

        if function_name == "set_alarm":
            time_str = arguments.get("time", "")
            label = arguments.get("label")
            timer = self.timer_manager.start_alarm(time_str, label, self._handle_timer_event)
            return f"Alarme programmée pour {time_str}"

        if function_name == "cancel_timer":
            label = arguments.get("label")
            timer = self.timer_manager.cancel_timer(name=label)
            if timer:
                from aioesphomeapi import VoiceAssistantTimerEventType

                await self._handle_timer_event(VoiceAssistantTimerEventType.VOICE_ASSISTANT_TIMER_CANCELLED, timer)
                if timer.is_alarm and timer.target_time:
                    desc = f"l'alarme pour {timer.target_time}"
                elif timer.name:
                    desc = f'le timer "{timer.name}"'
                else:
                    desc = f"le timer de {_format_duration(timer.total_seconds)}"
                return f"Annulé: {desc}"
            return "Aucun timer actif"

        # Weather — standalone, doesn't need HA
        if function_name == "get_weather":
            from weather import get_weather

            return await get_weather(arguments.get("location", ""))

        # Calendar — standalone, no HA dependency
        if function_name == "query_calendar":
            if not self.calendar_client:
                return "Agenda non disponible"
            return await self.calendar_client.query_events(
                start_date=arguments.get("start_date", ""),
                end_date=arguments.get("end_date"),
                search=arguments.get("search"),
            )

        if function_name == "create_event":
            if not self.calendar_client:
                return "Agenda non disponible"
            return await self.calendar_client.create_event(
                title=arguments.get("title", ""),
                start_datetime=arguments.get("start_datetime", ""),
                duration_minutes=arguments.get("duration_minutes", 60),
            )

        if not self.ha_client:
            return "Home Assistant non disponible"

        entity_name = arguments.get("entity", "")
        room = arguments.get("room")

        domain_map = {
            "turn_on": ["light", "switch", "media_player"],
            "turn_off": ["light", "switch", "media_player"],
            "open_cover": ["cover"],
            "close_cover": ["cover"],
            "set_cover_position": ["cover"],
            "set_temperature": ["climate"],
            "get_state": None,
        }
        domains = domain_map.get(function_name)

        entity_ids = self.ha_client.resolve_all_entities(entity_name, room=room, domain_hints=domains)
        if not entity_ids:
            return f"Appareil {entity_name} non trouvé"

        if function_name == "get_state":
            state_data = await self.ha_client.get_entity_state(entity_ids[0])
            if state_data:
                return self.ha_client.format_state_for_speech(entity_ids[0], state_data)
            return f"Impossible de lire l'état de {entity_name}"

        service = {
            "turn_on": "turn_on",
            "turn_off": "turn_off",
            "open_cover": "open_cover",
            "close_cover": "close_cover",
            "set_cover_position": "set_cover_position",
            "set_temperature": "set_temperature",
        }.get(function_name, function_name)

        extra = {}
        if function_name == "set_temperature" and "temperature" in arguments:
            extra["temperature"] = arguments["temperature"]
        if function_name == "turn_on" and "brightness" in arguments:
            extra["brightness"] = round(arguments["brightness"] * 255 / 100)
        if function_name == "set_cover_position" and "position" in arguments:
            extra["position"] = arguments["position"]

        for entity_id in entity_ids:
            domain = entity_id.split(".")[0]
            await self.ha_client.call_service(domain, service, entity_id, **extra)

        if len(entity_ids) > 1:
            whole_house = room and room.lower().strip() in (
                "tout",
                "toute la maison",
                "partout",
            )
            if whole_house:
                action_map = {
                    "close_cover": "Tous les volets en cours de fermeture",
                    "open_cover": "Tous les volets en cours d'ouverture",
                    "set_cover_position": f"Tous les volets réglés à {arguments.get('position', '?')}%",
                    "turn_on": "Toutes les lumières allumées",
                    "turn_off": "Toutes les lumières éteintes",
                }
            else:
                room_label = room or "la maison"
                action_map = {
                    "close_cover": f"Volets {room_label} en cours de fermeture",
                    "open_cover": f"Volets {room_label} en cours d'ouverture",
                    "set_cover_position": f"Volets {room_label} réglés à {arguments.get('position', '?')}%",
                    "turn_on": f"Lumières {room_label} allumées",
                    "turn_off": f"Lumières {room_label} éteintes",
                }
            return action_map.get(function_name, "C'est fait")

        return self.ha_client._build_response(
            entity_ids[0].split(".")[0],
            service,
            self.ha_client.entities.get(entity_ids[0], {}).get("friendly_name", entity_ids[0]),
            extra,
        )

    # === TIMER EVENTS ===

    async def _handle_timer_event(self, event_type, timer):
        """Forward timer events to the ESP device. On FINISHED, also play TTS announcement."""
        api = self.devices.get(self.current_device) if self.current_device else None
        if api:
            try:
                api.send_voice_assistant_timer_event(
                    event_type,
                    timer.id,
                    timer.name,
                    timer.total_seconds,
                    timer.seconds_left,
                    timer.is_active,
                )
            except Exception as e:
                logger.warning(f"Failed to send timer event to ESP: {e}")

        from aioesphomeapi import VoiceAssistantTimerEventType

        if event_type == VoiceAssistantTimerEventType.VOICE_ASSISTANT_TIMER_FINISHED and api:
            await self._announce_timer_finished(api, timer)

    async def _announce_timer_finished(self, api, timer):
        """Generate TTS and play announcement when a timer finishes."""
        if timer.name:
            text = f"Le timer {timer.name} est terminé !"
        else:
            text = f"Le timer de {_format_duration(timer.total_seconds)} est terminé !"

        try:
            audio_url = await self.text_to_speech_file(text)
            logger.info(f"Timer finished announcement: {text}")
            await api.send_voice_assistant_announcement_await_response(
                media_id=audio_url,
                timeout=8.0,
                text="",
                start_conversation=True,
            )
        except Exception as e:
            logger.error(f"Timer announcement error: {e}")

    # === LLM ===

    async def process_with_llm(self, api: Optional[APIClient], text: str, dry_run: bool = False) -> Optional[str]:
        """LLM via OpenAI-compatible API (local llama.cpp or cloud provider)"""
        from llm import chat_completion, get_tool_definitions, parse_text_tool_call

        logger.info(f'LLM input: "{text}"')

        is_local = not LLM_API_KEY

        # Build system prompt
        now_str = time.strftime("%A %d %B %Y, %H:%M")

        system_prompt = (
            "Tu es un assistant vocal pour la maison connectée. "
            "Réponds en français, en 1-2 phrases courtes maximum. "
            "Pas de markdown. Parle naturellement comme à l'oral. "
            "Quand on te demande de contrôler un appareil "
            "(même avec un pronom comme 'la' ou 'les'), "
            "utilise TOUJOURS les fonctions disponibles. "
            "Passe le paramètre room quand la pièce est mentionnée. "
            "Utilise les noms de groupes tels quels "
            "(ex: room='enfants'), ne les décompose pas. "
            "Si la demande couvre plusieurs actions ou sujets "
            "(ex: contrôler un appareil ET demander la météo), "
            "appelle TOUTES les fonctions nécessaires "
            "en parallèle dans ta réponse.\n"
            f"Date et heure actuelles: {now_str}."
        )
        family = LOCAL_CONFIG.get("family", [])
        if family:
            system_prompt += "\n" + _format_family_for_prompt(family)
        if is_local:
            system_prompt = "/no_think " + system_prompt
        if self.ha_client:
            entity_list = self.ha_client.get_entity_list_for_prompt()
            if entity_list:
                system_prompt += f"\nAppareils disponibles: {entity_list}"

        from timer import format_timers_for_prompt

        timers_info = format_timers_for_prompt(self.timer_manager.get_timers())
        if timers_info:
            system_prompt += f"\n{timers_info}"

        tools = get_tool_definitions(self.ha_client, self.calendar_client)

        # Expire old conversation history
        now = time.time()
        if now - self.last_interaction_time > self.CONVERSATION_TIMEOUT:
            if self.conversation_history:
                logger.info("Conversation history expired, starting fresh")
            self.conversation_history = []
            self.conversation_id = None
        self.last_interaction_time = now

        # Assign conversation ID for first exchange in a conversation
        if not self.conversation_id:
            self.conversation_id = int(now)

        # Build messages
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": text})

        message = await chat_completion(
            LLM_URL,
            LLM_API_KEY,
            LLM_MODEL,
            messages,
            tools,
        )
        if not message:
            return None

        def save_to_history(response_text):
            self.conversation_history.append({"role": "user", "content": text})
            self.conversation_history.append({"role": "assistant", "content": response_text})
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]

        # Tool call loop: keep calling LLM until it returns text (max 3 rounds)
        import json

        self._last_tool_calls = []
        max_rounds = 3
        for round_num in range(max_rounds):
            if "tool_calls" not in message or not message["tool_calls"]:
                break

            tool_messages = []
            for tool_call in message["tool_calls"]:
                function_name = tool_call["function"]["name"]
                function_args = json.loads(tool_call["function"]["arguments"])
                logger.info(f"Function call: {function_name}({function_args})")
                tc_entry = {"function": function_name, "args": function_args}
                if dry_run:
                    result = "OK"
                    tc_entry["result"] = result
                    logger.info(f"Dry run: {function_name}({function_args})")
                else:
                    try:
                        result = await self.execute_function(function_name, function_args)
                        logger.info(f"Function result: {result}")
                        tc_entry["result"] = result
                    except Exception as e:
                        logger.error(f"Function execution error: {e}")
                        result = f"Erreur: {e}"
                        tc_entry["error"] = str(e)
                self._last_tool_calls.append(tc_entry)
                tool_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": result,
                    }
                )

            # Follow-up LLM call with tools still available for additional calls
            messages.append(message)  # assistant message with tool_calls
            messages.extend(tool_messages)
            # Remind the model to handle all parts of the original request
            called_names = [tc["function"] for tc in self._last_tool_calls]
            messages.append(
                {
                    "role": "system",
                    "content": (
                        f"Fonctions déjà appelées: {', '.join(called_names)}. "
                        "Si la demande de l'utilisateur comporte d'autres parties non traitées, "
                        "appelle les fonctions nécessaires MAINTENANT avant de répondre."
                    ),
                }
            )
            message = await chat_completion(
                LLM_URL,
                LLM_API_KEY,
                LLM_MODEL,
                messages,
                tools=tools,
            )
            if not message:
                # Fallback: join raw results if follow-up call fails
                response = ". ".join(tc.get("result", "") for tc in self._last_tool_calls if tc.get("result"))
                save_to_history(response)
                return response

        if self._last_tool_calls:
            # LLM returned text after tool execution
            response = (message.get("content") or "").strip()
            if not response:
                response = ". ".join(tc.get("result", "") for tc in self._last_tool_calls if tc.get("result"))
            save_to_history(response)
            return response

        # Direct response (possibly with text-encoded tool call)
        content = message.get("content", "")
        if is_local and "</think>" in content:
            content = content.split("</think>")[-1]
        content = content.strip()

        fn_name, fn_args = parse_text_tool_call(content, bool(self.conversation_history))
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

    # === TTS ===

    async def text_to_speech_file(self, text: str) -> str:
        """Generate a TTS WAV file and return its URL."""
        return await self.tts.synthesize_to_file(text, self.tts_dir, self.http_base_url)

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentences for chunked TTS generation."""
        import re

        # Split on sentence-ending punctuation followed by space
        parts = re.split(r"(?<=[.!?;])\s+", text)
        sentences = [s.strip() for s in parts if s.strip()]
        # Don't split very short texts
        if len(sentences) <= 1:
            return [text]
        return sentences

    async def text_to_speech(self, api: APIClient, text: str) -> dict:
        """Generate TTS sentence-by-sentence with pipelined playback.
        Each sentence is played as soon as generated while the next one
        generates in parallel, reducing perceived latency.
        Returns timing dict with 'gen' (generation) and 'play' (playback) durations.
        """
        logger.info(f'TTS: generating audio for "{text}"')

        api.send_voice_assistant_event(VoiceAssistantEventType.VOICE_ASSISTANT_TTS_START, {"text": text})

        gen_time = 0.0
        play_time = 0.0

        try:
            # Don't split sentences for end_conversation — multiple announcements
            # with start_conversation=False cause the ESP to miss AnnounceFinished.
            if self._end_conversation:
                sentences = [text]
            else:
                sentences = self._split_sentences(text)
            logger.info(f"TTS: {len(sentences)} sentence(s) to generate")

            # Generate first sentence
            t0 = time.time()
            next_url = await self.text_to_speech_file(sentences[0])
            gen_time += time.time() - t0

            # End pipeline before playing announcements
            api.send_voice_assistant_event(VoiceAssistantEventType.VOICE_ASSISTANT_RUN_END, {})

            for i in range(len(sentences)):
                current_url = next_url
                is_last = i == len(sentences) - 1

                # Pipeline: start generating next sentence while playing current
                gen_task = None
                if not is_last:
                    gen_task = asyncio.create_task(self.text_to_speech_file(sentences[i + 1]))

                # Play current sentence
                logger.info(f"TTS: playing sentence {i + 1}/{len(sentences)}")
                t_play = time.time()

                if is_last and not self._end_conversation:
                    await self._play_tts_and_continue(api, current_url)
                elif is_last and self._end_conversation:
                    logger.info("End of conversation - no follow-up")
                    try:
                        await api.send_voice_assistant_announcement_await_response(
                            media_id=current_url,
                            timeout=30.0,
                            text="",
                            start_conversation=False,
                        )
                    except TimeoutError:
                        logger.warning("End-of-conversation announcement timed out")
                    self.conversation_history = []
                    self.conversation_id = None
                    logger.info("Conversation ended, history cleared")
                else:
                    await api.send_voice_assistant_announcement_await_response(
                        media_id=current_url,
                        timeout=30.0,
                        text="",
                        start_conversation=False,
                    )

                play_time += time.time() - t_play

                # Await next sentence generation (should already be done during playback)
                if gen_task:
                    t0 = time.time()
                    next_url = await gen_task
                    extra_gen = time.time() - t0
                    gen_time += extra_gen
                    if extra_gen > 0.05:
                        logger.info(f"TTS: waited {extra_gen:.2f}s for next sentence generation")

            timings = {"gen": round(gen_time, 2), "play": round(play_time, 2)}
            logger.info(f"TTS done: gen={timings['gen']}s play={timings['play']}s")
            return timings

        except Exception as e:
            logger.error(f"TTS error: {e}")
            import traceback

            traceback.print_exc()
            raise

    async def send_error_to_device(self, api: APIClient, error_message: str):
        """Send an error event and end the pipeline."""
        logger.error(f"Sending error to ESP: {error_message}")
        try:
            api.send_voice_assistant_event(
                VoiceAssistantEventType.VOICE_ASSISTANT_ERROR,
                {"code": "server_error", "message": error_message},
            )
            api.send_voice_assistant_event(VoiceAssistantEventType.VOICE_ASSISTANT_RUN_END, {})
            logger.info("ERROR + RUN_END sent to ESP")
        except Exception as e:
            logger.error(f"Error sending error: {e}")


async def main():
    """Main entry point"""
    print("Custom Voice Assistant Server for ESPHome (Python)")
    print("Replaces Home Assistant for voice pipeline")
    print("STT: Parakeet MLX | LLM: Qwen 3 4B (llama.cpp) | TTS: Kokoro\n")

    server = VoiceAssistantServer()
    reload_event = asyncio.Event()
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def handle_sighup():
        logger.info("SIGHUP received — scheduling reload")
        reload_event.set()

    def handle_sigterm():
        logger.info("SIGTERM received — shutting down")
        stop_event.set()

    loop.add_signal_handler(signal.SIGHUP, handle_sighup)
    loop.add_signal_handler(signal.SIGTERM, handle_sigterm)

    try:
        await server.start()
        logger.info("Server running — send SIGHUP to reload, SIGTERM to stop")

        while not stop_event.is_set():
            reload_task = asyncio.create_task(reload_event.wait())
            stop_task = asyncio.create_task(stop_event.wait())
            done, _ = await asyncio.wait({reload_task, stop_task}, return_when=asyncio.FIRST_COMPLETED)
            for t in {reload_task, stop_task} - done:
                t.cancel()

            if reload_event.is_set():
                reload_event.clear()
                logger.info("Reloading...")
                await server.init_ha_client()
                logger.info("Reload complete")

    except KeyboardInterrupt:
        logger.info("\nShutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
    finally:
        if server._reconnect_logic:
            await server._reconnect_logic.stop()
        if server.calendar_client:
            await server.calendar_client.close()

    logger.info("Server stopped")


if __name__ == "__main__":
    asyncio.run(main())
