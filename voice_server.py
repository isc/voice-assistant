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
LLAMA_URL = os.environ.get("LLAMA_URL", "http://localhost:8080/v1/chat/completions")
HTTP_PORT = int(os.environ.get("HTTP_PORT", "8888"))


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

        # Piper TTS engine
        self.tts_engine = None

        # HTTP server for serving TTS audio files
        self.http_server = None
        self.tts_dir = Path(tempfile.gettempdir()) / "voice_assistant_tts"
        self.tts_dir.mkdir(exist_ok=True)
        logger.info(f"TTS directory: {self.tts_dir}")

    async def init_tts_engine(self):
        """Initialize Piper TTS (native 16KHz, optimized for ESP)"""
        logger.info("Initializing Piper TTS...")

        try:
            from piper import PiperVoice
            import urllib.request

            # French 16KHz native model (no resampling needed)
            voice_name = "fr_FR-gilles-low"
            models_dir = Path("/tmp/piper_models")
            models_dir.mkdir(exist_ok=True)

            model_path = models_dir / "fr_FR-gilles-low.onnx"
            config_path = models_dir / "fr_FR-gilles-low.onnx.json"

            if not model_path.exists() or not config_path.exists():
                logger.info("Downloading Piper model (native 16KHz)...")

                base_url = f"https://huggingface.co/rhasspy/piper-voices/resolve/main/fr/fr_FR/gilles/low"

                if not model_path.exists():
                    logger.info(f"  Downloading {voice_name}.onnx (~30MB)...")
                    urllib.request.urlretrieve(
                        f"{base_url}/fr_FR-gilles-low.onnx", model_path
                    )
                    logger.info(f"  Model downloaded")

                if not config_path.exists():
                    logger.info(f"  Downloading config...")
                    urllib.request.urlretrieve(
                        f"{base_url}/fr_FR-gilles-low.onnx.json", config_path
                    )
                    logger.info(f"  Config downloaded")
            else:
                logger.info("Piper model already cached")

            logger.info(f"Model: {model_path}")
            logger.info(f"Config: {config_path}")

            self.tts_engine = PiperVoice.load(
                str(model_path), str(config_path), use_cuda=False
            )

        except ImportError:
            logger.error("Piper TTS is not installed")
            logger.info("Install with: pip install piper-tts")
            raise
        except Exception as e:
            logger.error(f"Piper init error: {e}")
            import traceback

            traceback.print_exc()
            raise

    async def start_http_server(self):
        """Start HTTP server to host TTS audio files"""
        app = web.Application()
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
        self.http_base_url = f"http://{local_ip}:{HTTP_PORT}/tts/"

    async def start(self):
        """Start the server and connect to devices"""
        logger.info("Voice Assistant Server started")
        logger.info(f"ESP: {ESP_HOST}:{ESP_PORT} | LLM: {LLAMA_URL} | HTTP: {HTTP_PORT}")

        await self.init_stt_engine()
        await self.init_tts_engine()
        await self.start_http_server()
        await self.connect_to_device(ESP_HOST)

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
        Main pipeline: Audio -> STT (Parakeet) -> LLM (Qwen) -> TTS (Piper)
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
        return [
            {
                "name": "close_shutters",
                "description": "Close the shutters in a room",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "room": {
                            "type": "string",
                            "description": "The room where to close the shutters (living room, bedroom, etc.)",
                        }
                    },
                    "required": ["room"],
                },
            }
        ]

    async def execute_function(self, function_name: str, arguments: dict) -> str:
        """Execute a function called by the LLM"""
        if function_name == "close_shutters":
            return await self.close_shutters(**arguments)
        else:
            return f"Unknown function: {function_name}"

    async def close_shutters(self, room: str) -> str:
        """Close the shutters in a room (simulation for now)"""
        logger.info(f"FUNCTION CALLED: close_shutters(room={room})")
        logger.info(f"   Simulation: closing shutters in {room}")
        return f"Shutters in {room} have been closed"

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

    async def process_with_llm(self, api: APIClient, text: str) -> Optional[str]:
        """LLM with Qwen via llama.cpp (OpenAI-compatible API)"""
        import json

        logger.info(f'LLM input: "{text}"')

        try:
            payload = {
                "messages": [
                    {
                        "role": "system",
                        "content": "/no_think Tu es un assistant vocal pour la maison connectée. Réponds de manière concise et naturelle en français. Tu peux contrôler les lumières, les volets, donner la météo, et répondre aux questions. Utilise les fonctions disponibles quand c'est approprié.",
                    },
                    {
                        "role": "user",
                        "content": text,
                    },
                ],
                "tools": [
                    {"type": "function", "function": func}
                    for func in self.get_available_functions()
                ],
                "max_tokens": 256,
                "temperature": 0.7,
            }

            tool_names = [t["function"]["name"] for t in payload["tools"]]
            logger.info(f'LLM prompt: [system] ...voice assistant... [user] "{text}"')
            logger.info(f"Tools: {tool_names}")

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    LLAMA_URL, json=payload, timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"LLM raw response: {json.dumps(result, ensure_ascii=False, indent=2)}")
                        message = result["choices"][0]["message"]

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

                                assistant_response = message.get("content", "C'est fait")
                                if assistant_response:
                                    assistant_response = assistant_response.strip()
                                else:
                                    assistant_response = "C'est fait"

                                logger.info(f'LLM response (with function): "{assistant_response}"')
                                return assistant_response

                            except Exception as e:
                                logger.error(f"Function execution error: {e}")
                                return "Désolé, une erreur s'est produite"
                        else:
                            # Direct response
                            content = message.get("content", "")
                            # Qwen 3 in thinking mode may wrap in <think>...</think>
                            if "</think>" in content:
                                content = content.split("</think>")[-1]
                            assistant_response = content.strip()
                            logger.info(f'LLM response: "{assistant_response}"')
                            return assistant_response if assistant_response else None
                    else:
                        error_text = await response.text()
                        logger.error(f"LLM error {response.status}: {error_text}")
                        return None

        except Exception as e:
            logger.error(f"LLM error: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def text_to_speech(self, api: APIClient, text: str):
        """Text-to-Speech with Piper - WAV file generation + URL"""
        logger.info(f'TTS: generating audio for "{text}"')

        api.send_voice_assistant_event(
            VoiceAssistantEventType.VOICE_ASSISTANT_TTS_START, {"text": text}
        )
        logger.info(f"TTS_START sent")

        try:
            import numpy as np
            import hashlib

            filename = f"tts_{int(time.time())}_{hashlib.md5(text.encode()).hexdigest()[:8]}.wav"
            output_path = self.tts_dir / filename

            logger.info(f"Synthesizing: '{text[:80]}...'")

            # Generate audio with Piper
            all_audio = []
            for audio_chunk in self.tts_engine.synthesize(text):
                all_audio.append(audio_chunk.audio_float_array)

            audio_float = np.concatenate(all_audio)
            sample_rate = 16000  # fr_FR-gilles-low is native 16KHz

            logger.info(f"Audio total: {len(audio_float)} samples")

            # Convert float32 [-1, 1] -> int16 (PCM 16-bit)
            audio_int16 = (audio_float * 32767).astype(np.int16)

            # Save as WAV file
            with wave.open(str(output_path), "wb") as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)  # 16kHz
                wav_file.writeframes(audio_int16.tobytes())

            logger.info(
                f"TTS generated: {output_path} ({output_path.stat().st_size} bytes)"
            )

            # URL accessible by the ESP
            audio_url = f"{self.http_base_url}{filename}"

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
    print("STT: Parakeet MLX | LLM: Qwen 3 (llama.cpp) | TTS: Piper\n")

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
