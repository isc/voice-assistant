"""Text-to-Speech: Phonon 40M French (Gradium), voice Noémie, on the CPU (AD-103)."""

import asyncio
import hashlib
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Gated on Hugging Face (CC BY-NC 4.0): downloaded at the first start into its cache, by an account that
# accepted its conditions (`hf auth login`); later starts read the cache, network or not. Pinned, with
# the voice: a new revision is a new voice, to hear before it speaks to the family
MODEL = "hf://gradium/phonon-40m-fr@d91086970206b88efd2c88ca971d0628b5968f0f"
VOICE = "Noémie"
VOICE_CATALOG = "hf://gradium/phonon-voices@da6c9ec0d47be8df5c87c730a3aa97187f3229c3"
# A 4 s reply in 0.23 s on charras's i5-14400F, its first audio after 28 ms; 0.54 s on one thread
THREADS = 4
# Replies are fetched once by the speaker, right after being written
_REPLY_MAX_AGE = 600  # seconds

# Clock times as the LLM writes them: "17h34", "7h", "17:34". Phonon's normalizer leaves them as written.
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


class PhononTTS:
    """Phonon on the CPU: no ONNX Runtime, no room taken on the card the LLM needs."""

    def __init__(self):
        self.engine = None
        self._executor = ThreadPoolExecutor(1, thread_name_prefix="tts")  # one synthesis at a time, off the loop

    async def init(self):
        logger.info("Initializing Phonon TTS...")
        self.engine = await asyncio.get_running_loop().run_in_executor(self._executor, self._load)
        logger.info(f"Phonon TTS ready (cpu, {THREADS} threads, voice: {VOICE})")

    @staticmethod
    def _load():
        import ptts

        ptts.set_num_threads(THREADS)
        # The CPU said: "auto" would take a CUDA build's card, the LLM's
        engine = ptts.TTS(MODEL, device="cpu", lang="fr", quant="q8_0", voice=VOICE, voice_catalog=VOICE_CATALOG)
        engine.synth("Bonjour.")  # the voice found, the first run's allocations made
        return engine

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

        # FLAC: the Voice PE's firmware decodes FLAC and MP3 since 26.x (ESPHome 2026.6), no longer WAV.
        # 48 kHz mono, its announcement format: ffmpeg resamples here, so the satellite runs no resampler
        # beside its wake word models (a 16 kHz reply stuttered). ffmpeg writes to a file so that
        # STREAMINFO holds the length (flac_seconds). Started during the synthesis: ~35 ms less.
        sr = self.engine.sample_rate
        ffmpeg = await asyncio.create_subprocess_exec(
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "s16le", "-ar", str(sr), "-ac", "1", "-i", "pipe:0",
            "-ar", "48000", "-c:a", "flac", str(output_path),
            stdin=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )  # fmt: skip
        t0 = time.time()
        try:
            audio = await asyncio.get_running_loop().run_in_executor(self._executor, self.engine.synth, clean_text)
        except BaseException:
            ffmpeg.kill()
            await ffmpeg.wait()
            raise
        logger.info(f"Phonon output: {len(audio) / sr:.1f} s of speech in {time.time() - t0:.2f} s")

        # Convert float32 [-1, 1] -> int16 (PCM 16-bit)
        audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
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
