"""Speech-to-Text using Parakeet TDT 0.6B v3 (multilingual, 25 EU languages incl. French).

Backend selectable via the STT_BACKEND env var:
  - 'mlx'  : Apple Silicon, via parakeet-mlx          (dev on the Mac)
  - 'onnx' : Linux/CUDA, via onnx-asr                  (the dedicated server)
  - 'auto' (default): MLX if importable, else onnx-asr.
"""

import asyncio
import logging
import os
import tempfile
import wave
from typing import Optional

logger = logging.getLogger(__name__)


class ParakeetSTT:
    """Parakeet TDT 0.6B v3 STT engine (MLX or ONNX backend)."""

    MODEL_MLX = "mlx-community/parakeet-tdt-0.6b-v3"
    MODEL_ONNX = "nemo-parakeet-tdt-0.6b-v3"

    def __init__(self):
        self.model = None
        self.backend = os.environ.get("STT_BACKEND", "auto").lower()

    async def init(self):
        """Load the Parakeet model for the selected backend."""
        if self.backend == "auto":
            try:
                import parakeet_mlx  # noqa: F401

                self.backend = "mlx"
            except ImportError:
                self.backend = "onnx"

        logger.info(f"Loading Parakeet STT model (backend={self.backend})...")
        if self.backend == "mlx":
            import parakeet_mlx

            self.model = parakeet_mlx.from_pretrained(self.MODEL_MLX)
        elif self.backend == "onnx":
            import onnx_asr

            self.model = onnx_asr.load_model(self.MODEL_ONNX)
        else:
            raise ValueError(f"Unknown STT_BACKEND '{self.backend}' (expected mlx|onnx|auto)")
        logger.info("Parakeet STT model loaded")

    def _transcribe_sync(self, wav_path: str) -> str:
        if self.backend == "mlx":
            return self.model.transcribe(wav_path).text
        return self.model.recognize(wav_path)  # onnx-asr

    async def transcribe(self, audio_bytes: bytes) -> Optional[str]:
        """Transcribe raw 16kHz 16-bit PCM audio to text."""
        logger.info(f"STT Parakeet ({self.backend}): {len(audio_bytes)} bytes...")

        try:
            temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            with wave.open(temp_wav.name, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                wav_file.writeframes(audio_bytes)

            loop = asyncio.get_event_loop()
            transcript = (await loop.run_in_executor(None, self._transcribe_sync, temp_wav.name)).strip()

            os.unlink(temp_wav.name)

            logger.info(f'Transcript: "{transcript}"')
            return transcript if transcript else None

        except Exception as e:
            logger.error(f"STT error: {e}")
            import traceback

            traceback.print_exc()
            return None
