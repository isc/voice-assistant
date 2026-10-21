#!/usr/bin/env python3
"""A day of the assistant's interactions, gathered for the nightly review (review/README.md).

Run as root: the recordings and the exchange log are root-only. Writes into OUT_DIR, chowned to OWNER:
  interactions.json  each request of the day: the exchange log's entry, merged with its recording's details
  journal.log        the server's journal for the day, without the HTTP and Spotify streaming noise
  audio/             the day's WAVs, for a second transcription by Whisper (deleted after the review)

  sudo review/collect.py 2026-10-21 /path/to/out ivan
"""
import json
import shutil
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

LIVE = Path("/opt/voice-assistant")
# Lines that say nothing of a request: each speaker's fetch, the stream's listeners
NOISE = ("GET /", "POST /", "Spotify: listener", "Spotify: stream", "Phonemes:", "Kokoro output:")


def interactions(day: str) -> list:
    recordings = {}
    for details in sorted((LIVE / "recordings" / day).glob("*.json")):
        try:
            recordings[f"{day}/{details.stem}.wav"] = json.loads(details.read_text())
        except ValueError:
            pass
    try:
        log = json.loads((LIVE / "exchanges.json").read_text())
    except (OSError, ValueError):
        log = []
    merged = []
    for entry in log:
        if not str(entry.get("timestamp", "")).startswith(day):
            continue
        # The recording has VAD and wake word details the log lacks
        merged.append({**recordings.pop(entry.get("audio"), {}), **entry})
    # The log keeps only its last 50 exchanges: older voice requests of the day are in the recordings alone
    for audio, details in recordings.items():
        merged.append({"source": "voice", "audio": audio, **details})
    return sorted(merged, key=lambda e: e.get("timestamp") or e.get("audio") or "")


def journal(day: str) -> str:
    until = (date.fromisoformat(day) + timedelta(days=1)).isoformat()
    lines = subprocess.run(
        ["journalctl", "-u", "voice-assistant", "--since", day, "--until", until, "--no-pager", "-o", "cat"],
        capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    return "\n".join(line for line in lines if not any(noise in line for noise in NOISE)) + "\n"


def main():
    day, out, owner = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
    (out / "audio").mkdir(parents=True, exist_ok=True)
    (out / "interactions.json").write_text(json.dumps(interactions(day), ensure_ascii=False, indent=1, default=str))
    (out / "journal.log").write_text(journal(day))
    for wav in (LIVE / "recordings" / day).glob("*.wav"):
        shutil.copy(wav, out / "audio" / wav.name)
    for path in [out, *out.rglob("*")]:
        shutil.chown(path, owner, owner)
        path.chmod(0o700 if path.is_dir() else 0o600)


if __name__ == "__main__":
    main()
