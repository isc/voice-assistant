#!/usr/bin/env python3
"""
Custom Voice Assistant Server for ESPHome
Replaces Home Assistant for the voice pipeline
"""

import asyncio
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
from aiohttp import web

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

        # Sub-engines (initialized in start())
        self.tts = None  # tts.KokoroTTS
        self.stt = None  # stt.ParakeetSTT

        # Home Assistant client
        self.ha_client = None

        # Conversation history for multi-turn context
        self.conversation_history = []
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
        """Connect to a specific ESPHome device"""
        try:
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

            await api.connect(login=True)
            logger.info(f"Connected to {host}")

            device_info = await api.device_info()
            logger.info(f"Device: {device_info.name} (v{device_info.esphome_version})")

            if device_info.voice_assistant_feature_flags:
                logger.info(f"Voice assistant supported (flags: {device_info.voice_assistant_feature_flags})")
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

        self.conversation_id = conversation_id
        self.is_followup = wake_word_phrase is None

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
        """
        try:
            logger.info(f"Playing TTS and requesting follow-up: {audio_url}")
            result = await api.send_voice_assistant_announcement_await_response(
                media_id=audio_url,
                timeout=30.0,
                text="",
                start_conversation=True,
            )
            logger.info(f"TTS playback done, follow-up started (success={result.success})")
        except TimeoutError:
            logger.warning("TTS announcement timed out")
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
            if not response_text:
                await self.send_error_to_device(api, "LLM processing failed")
                return

            # 3. TTS
            t0 = time.time()
            tts_timings = await self.text_to_speech(api, response_text)
            timings["tts_gen"] = tts_timings["gen"]
            timings["tts_play"] = tts_timings["play"]
            timings["tts"] = round(time.time() - t0, 2)
            timings["total"] = round(timings["stt"] + timings["llm"] + timings["tts_gen"], 2)

            self.exchange_log.add(
                "voice",
                transcript,
                transcript,
                response_text,
                timings=timings,
                tool_calls=self._last_tool_calls,
            )
            self._last_tool_calls = []
            logger.info(
                f"Pipeline completed: STT={timings['stt']}s LLM={timings['llm']}s TTS gen={timings['tts_gen']}s play={timings['tts_play']}s total={timings['total']}s"
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

        # Weather — standalone, doesn't need HA
        if function_name == "get_weather":
            from weather import get_weather

            return await get_weather(arguments.get("location", ""))

        if not self.ha_client:
            return "Home Assistant non disponible"

        entity_name = arguments.get("entity", "")
        room = arguments.get("room")

        domain_map = {
            "turn_on": ["light", "switch", "media_player"],
            "turn_off": ["light", "switch", "media_player"],
            "open_cover": ["cover"],
            "close_cover": ["cover"],
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
            "set_temperature": "set_temperature",
        }.get(function_name, function_name)

        extra = {}
        if function_name == "set_temperature" and "temperature" in arguments:
            extra["temperature"] = arguments["temperature"]
        if function_name == "turn_on" and "brightness" in arguments:
            extra["brightness"] = round(arguments["brightness"] * 255 / 100)

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
            entity_ids[0].split(".")[0],
            service,
            self.ha_client.entities.get(entity_ids[0], {}).get("friendly_name", entity_ids[0]),
            extra,
        )

    # === LLM ===

    async def process_with_llm(self, api: Optional[APIClient], text: str) -> Optional[str]:
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
            "Quand on te demande de contrôler un appareil (même avec un pronom comme 'la' ou 'les'), utilise TOUJOURS les fonctions disponibles. "
            "Passe le paramètre room quand la pièce est mentionnée. "
            "Utilise les noms de groupes tels quels (ex: room='enfants'), ne les décompose pas. "
            "Si la demande couvre plusieurs actions ou sujets (ex: contrôler un appareil ET demander la météo), appelle TOUTES les fonctions nécessaires en parallèle dans ta réponse.\n"
            f"Date et heure actuelles: {now_str}."
        )
        if is_local:
            system_prompt = "/no_think " + system_prompt
        if self.ha_client:
            entity_list = self.ha_client.get_entity_list_for_prompt()
            if entity_list:
                system_prompt += f"\nAppareils disponibles: {entity_list}"

        tools = get_tool_definitions(self.ha_client)

        # Expire old conversation history
        now = time.time()
        if now - self.last_interaction_time > self.CONVERSATION_TIMEOUT:
            if self.conversation_history:
                logger.info("Conversation history expired, starting fresh")
            self.conversation_history = []
        self.last_interaction_time = now

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
                    await api.send_voice_assistant_announcement_await_response(
                        media_id=current_url,
                        timeout=30.0,
                        text="",
                        start_conversation=False,
                    )
                    self.conversation_history = []
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
                logger.info("Reloading Home Assistant entities...")
                await server.init_ha_client()
                logger.info("Reload complete")

    except KeyboardInterrupt:
        logger.info("\nShutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

    logger.info("Server stopped")


if __name__ == "__main__":
    asyncio.run(main())
