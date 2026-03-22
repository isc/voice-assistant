"""Text-to-Speech using Kokoro-82M with French G2P via misaki/espeak."""

import asyncio
import hashlib
import logging
import re
import time
import wave
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

MODELS_DIR = Path("/tmp/kokoro_models")
MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"


class KokoroTTS:
    """Kokoro-82M TTS engine with French phonemization."""

    def __init__(self):
        self.engine = None
        self.g2p = None

    async def init(self):
        """Initialize Kokoro TTS and French G2P."""
        logger.info("Initializing Kokoro TTS...")
        import kokoro_onnx
        from misaki.espeak import EspeakG2P

        MODELS_DIR.mkdir(exist_ok=True)
        model_path = MODELS_DIR / "kokoro-v1.0.onnx"
        voices_path = MODELS_DIR / "voices-v1.0.bin"

        if not model_path.exists() or not voices_path.exists():
            logger.info("Downloading Kokoro models...")
            import urllib.request

            if not model_path.exists():
                logger.info("  Downloading kokoro-v1.0.onnx (~310MB)...")
                urllib.request.urlretrieve(MODEL_URL, model_path)
            if not voices_path.exists():
                logger.info("  Downloading voices-v1.0.bin (~27MB)...")
                urllib.request.urlretrieve(VOICES_URL, voices_path)
        else:
            logger.info("Kokoro models already cached")

        self.engine = kokoro_onnx.Kokoro(str(model_path), str(voices_path))
        self.g2p = EspeakG2P(language="fr-fr")
        logger.info("Kokoro TTS ready (voice: ff_siwis, French G2P via espeak)")

    async def synthesize_to_file(self, text: str, output_dir: Path, base_url: str) -> str:
        """Generate a 16kHz WAV file and return its URL."""
        filename = f"tts_{int(time.time())}_{hashlib.md5(text.encode()).hexdigest()[:8]}.wav"
        output_path = output_dir / filename

        # Strip markdown artifacts before TTS
        clean_text = re.sub(r"\*+", "", text)
        clean_text = re.sub(r"#+\s*", "", clean_text)
        clean_text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", clean_text)
        clean_text = re.sub(r"`([^`]+)`", r"\1", clean_text)
        clean_text = re.sub(r"\s+", " ", clean_text).strip()

        logger.info(f"Synthesizing: '{clean_text[:80]}...'")

        # French phonemization via espeak
        phonemes = self.g2p(clean_text)
        logger.info(f"Phonemes: {phonemes}")

        # Generate audio with Kokoro (runs sync, use executor)
        loop = asyncio.get_event_loop()
        audio_24k, sr = await loop.run_in_executor(
            None,
            lambda: self.engine.create(phonemes, voice="ff_siwis", speed=1.0, lang="fr-fr", is_phonemes=True),
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
        return f"{base_url}{filename}"
