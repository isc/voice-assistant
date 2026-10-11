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

import numpy as np

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
            import onnxruntime

            # Use the GPU (CUDA EP) when onnxruntime-gpu is installed, else CPU.
            # Generic: same code on a CPU-only box and a CUDA server.
            providers = ["CPUExecutionProvider"]
            if "CUDAExecutionProvider" in onnxruntime.get_available_providers():
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            logger.info(f"ONNX STT execution providers: {providers}")
            self.model = onnx_asr.load_model(self.MODEL_ONNX, providers=providers)
        else:
            raise ValueError(f"Unknown STT_BACKEND '{self.backend}' (expected mlx|onnx|auto)")
        logger.info("Parakeet STT model loaded")

    def _transcribe_sync(self, audio_bytes: bytes) -> str:
        if self.backend != "mlx":
            # onnx-asr takes the samples: no temporary WAV of the family's voice in /tmp (left
            # behind when recognition failed), no disk write per request
            return self.model.recognize(np.frombuffer(audio_bytes, np.int16).astype(np.float32) / 32768)
        temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        try:
            with wave.open(temp_wav.name, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                wav_file.writeframes(audio_bytes)
            return self.model.transcribe(temp_wav.name).text
        finally:
            os.unlink(temp_wav.name)

    async def transcribe(self, audio_bytes: bytes) -> Optional[str]:
        """Transcribe raw 16kHz 16-bit PCM audio to text.

        Errors are raised, not turned into « nothing heard »: the pipeline logs them with the
        recording, where a CUDA failure used to look like silence."""
        logger.info(f"STT Parakeet ({self.backend}): {len(audio_bytes)} bytes...")
        loop = asyncio.get_running_loop()
        transcript = (await loop.run_in_executor(None, self._transcribe_sync, audio_bytes)).strip()
        logger.info(f'Transcript: "{transcript}"')
        return transcript or None
