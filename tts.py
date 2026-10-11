"""Text-to-Speech using Kokoro-82M with French G2P via misaki/espeak."""

import asyncio
import hashlib
import logging
import re
import time
from pathlib import Path

import numpy as np

import onnx_gpu

logger = logging.getLogger(__name__)

# Next to the code, not in /tmp: /tmp is emptied at boot, and the server could not start without GitHub
MODELS_DIR = Path(__file__).parent / "models"
# Replies are fetched once by the speaker, right after being written
_REPLY_MAX_AGE = 600  # seconds
MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"

# Clock times as the LLM writes them: "17h34", "7h", "17:34". espeak reads the "h" as the letter.
_CLOCK_TIME = re.compile(r"\b([01]?\d|2[0-3])(?:h([0-5]\d)?|:([0-5]\d))(?![\w:])")
# Hours and minutes are feminine: "une heure", "vingt et une heures", "dix heures trente et une"
_FEMININE = {1: "une", 21: "vingt et une", 31: "trente et une", 41: "quarante et une", 51: "cinquante et une"}


def spell_times(text: str) -> str:
    """Spell clock times out for speech: "17h34" -> "17 heures 34", "1h" -> "une heure"."""

    def spell(match: re.Match) -> str:
        hour = int(match.group(1))
        minute = int(match.group(2) or match.group(3) or 0)
        words = f"{_FEMININE.get(hour, hour)} heure{'s' if hour > 1 else ''}"
        return f"{words} {_FEMININE.get(minute, minute)}" if minute else words

    return _CLOCK_TIME.sub(spell, text)


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

        self.engine = self._engine(kokoro_onnx, model_path, voices_path)
        self.g2p = EspeakG2P(language="fr-fr")
        logger.info("Kokoro TTS ready (voice: ff_siwis, French G2P via espeak)")

    @staticmethod
    def _engine(kokoro_onnx, model_path: Path, voices_path: Path):
        """Kokoro on CUDA when onnxruntime has it: kokoro-onnx 0.5 looks for the GPU package under a
        name find_spec never finds, and ran on the CPU (0.18 x real time)."""
        import onnxruntime

        session = onnxruntime.InferenceSession(
            str(model_path), onnx_gpu.session_options(), providers=onnx_gpu.providers()
        )
        logger.info(f"Kokoro providers: {session.get_providers()}")
        return kokoro_onnx.Kokoro.from_session(onnx_gpu.shrinking(session), str(voices_path))

    async def synthesize_to_file(self, text: str, output_dir: Path, base_url: str) -> str:
        """Generate a 48 kHz FLAC file and return its URL."""
        filename = f"tts_{int(time.time())}_{hashlib.md5(text.encode()).hexdigest()[:8]}.flac"
        output_path = output_dir / filename
        # Older replies: the speaker fetched them long ago, and they kept every answer since boot
        for old in output_dir.glob("tts_*"):
            if time.time() - old.stat().st_mtime > _REPLY_MAX_AGE:
                old.unlink(missing_ok=True)

        # Strip markdown artifacts before TTS
        clean_text = re.sub(r"\*+", "", text)
        clean_text = re.sub(r"#+\s*", "", clean_text)
        clean_text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", clean_text)
        clean_text = re.sub(r"`([^`]+)`", r"\1", clean_text)
        clean_text = re.sub(r"\s+", " ", clean_text).strip()
        clean_text = spell_times(clean_text)

        logger.info(f"Synthesizing: '{clean_text[:80]}...'")

        # French phonemization via espeak (misaki EspeakG2P returns (phonemes, tokens))
        phonemes = self.g2p(clean_text)
        if isinstance(phonemes, tuple):
            phonemes = phonemes[0]
        logger.info(f"Phonemes: {phonemes}")

        # Generate audio with Kokoro (runs sync, on the models' thread)
        audio_24k, sr = await onnx_gpu.run(
            lambda: self.engine.create(phonemes, voice="ff_siwis", speed=1.0, lang="fr-fr", is_phonemes=True)
        )

        logger.info(f"Kokoro output: {len(audio_24k)} samples at {sr}Hz")

        # Convert float32 [-1, 1] -> int16 (PCM 16-bit)
        audio_int16 = (np.clip(audio_24k, -1.0, 1.0) * 32767).astype(np.int16)

        # FLAC: the Voice PE's firmware decodes FLAC and MP3 since 26.x (ESPHome 2026.6), no longer WAV.
        # 48 kHz mono, its announcement format: ffmpeg resamples here, so the satellite runs no resampler
        # beside its wake word models (a 16 kHz reply stuttered). ffmpeg writes to a file so that
        # STREAMINFO holds the length (flac_seconds); ~35 ms.
        ffmpeg = await asyncio.create_subprocess_exec(
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "s16le", "-ar", str(sr), "-ac", "1", "-i", "pipe:0",
            "-ar", "48000", "-c:a", "flac", str(output_path),
            stdin=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )  # fmt: skip
        _, error = await ffmpeg.communicate(audio_int16.tobytes())
        if ffmpeg.returncode:
            raise RuntimeError(f"ffmpeg: {error.decode().strip()}")

        logger.info(f"TTS generated: {output_path} ({output_path.stat().st_size} bytes)")
        return f"{base_url}{filename}"


def flac_seconds(path: Path) -> float:
    """A FLAC file's length, from its STREAMINFO: sample rate (20 bits), then after channels and
    bits per sample (8), the total samples (36)."""
    with open(path, "rb") as f:
        head = f.read(26)  # "fLaC", the block header (4), STREAMINFO from its 11th byte
    if len(head) < 26 or head[:4] != b"fLaC":
        raise ValueError(f"not a FLAC file: {path}")
    info = int.from_bytes(head[18:26], "big")
    return (info & (1 << 36) - 1) / (info >> 44)
