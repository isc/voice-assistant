"""Voice requests kept as WAV + JSON, to hear what went wrong and replay them through other STT engines."""

import json
import logging
import os
import shutil
import wave
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

RECORDINGS_DIR = Path(__file__).parent / "recordings"
# Days kept; 0 turns recording off
RECORDINGS_DAYS = int(os.environ.get("RECORDINGS_DAYS", "90"))


class Recordings:
    """One folder per day, each request a WAV and a JSON of what became of it. Folders past retention are deleted."""

    def __init__(self, directory: Path = RECORDINGS_DIR, days: int = RECORDINGS_DAYS):
        self.directory = directory
        self.days = days
        if days:
            directory.mkdir(parents=True, exist_ok=True)
            # The family's voices: for root only, the user the server runs as
            os.chmod(directory, 0o700)
            logger.info(f"Recording voice requests in {directory}, kept {days} days")

    def save(self, audio: bytes) -> Optional[str]:
        """Write a request's 16 kHz 16-bit mono audio; returns its path under the directory, None when off."""
        if not self.days:
            return None
        now = datetime.now()
        recording = f"{now:%Y-%m-%d}/{now:%H%M%S}_{now.microsecond // 1000:03d}.wav"
        # A recording that fails must never fail the request
        try:
            self.path(recording).parent.mkdir(exist_ok=True)
            with wave.open(str(self.path(recording)), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(audio)
            self._purge()
        except OSError as e:
            logger.warning(f"Recording not saved: {e}")
            return None
        return recording

    def describe(self, recording: Optional[str], details: dict) -> None:
        """Write what became of the request beside its audio."""
        if not recording:
            return
        try:
            self.path(recording).with_suffix(".json").write_text(
                json.dumps(details, ensure_ascii=False, indent=1, default=str)
            )
        except OSError as e:
            logger.warning(f"Recording details not saved: {e}")

    def path(self, recording: str) -> Path:
        return self.directory / recording

    def _purge(self) -> None:
        oldest = (date.today() - timedelta(days=self.days)).isoformat()
        for day in self.directory.iterdir():
            if day.is_dir() and day.name < oldest:
                shutil.rmtree(day)
                logger.info(f"Recordings of {day.name} deleted ({self.days}-day retention)")
