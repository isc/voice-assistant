"""Speech-to-Text using Parakeet MLX (Apple Silicon optimized)."""

import asyncio
import logging
import os
import tempfile
import wave
from typing import Optional

logger = logging.getLogger(__name__)


class ParakeetSTT:
    """Parakeet TDT 0.6B v3 STT engine via MLX."""

    def __init__(self):
        self.model = None

    async def init(self):
        """Load the Parakeet MLX model."""
        import parakeet_mlx

        logger.info("Loading Parakeet STT model...")
        self.model = parakeet_mlx.from_pretrained("mlx-community/parakeet-tdt-0.6b-v3")
        logger.info("Parakeet STT model loaded")

    async def transcribe(self, audio_bytes: bytes) -> Optional[str]:
        """Transcribe raw 16kHz 16-bit PCM audio to text."""
        logger.info(f"STT Parakeet: {len(audio_bytes)} bytes...")

        try:
            temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            with wave.open(temp_wav.name, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                wav_file.writeframes(audio_bytes)

            logger.info(f"Audio WAV: {temp_wav.name} ({os.path.getsize(temp_wav.name)} bytes)")

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self.model.transcribe, temp_wav.name)
            transcript = result.text.strip()

            os.unlink(temp_wav.name)

            logger.info(f'Transcript: "{transcript}"')
            return transcript if transcript else None

        except Exception as e:
            logger.error(f"STT error: {e}")
            import traceback

            traceback.print_exc()
            return None
