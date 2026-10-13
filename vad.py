"""End of speech in a request being recorded, by Silero VAD: the satellite's and the web app's."""

import array
import logging

from silero_vad_lite import SileroVAD

logger = logging.getLogger(__name__)

FRAME_BYTES = 512 * 2  # 32 ms at 16 kHz, int16
# How long a request is listened to: for speech to start after a wake word or a tap, after a reply
# (the follow-up), and in all
WAIT_FOR_SPEECH = 8.0  # a false wake kept the mic open 30 s, recording the room
WAIT_FOR_FOLLOW_UP = 5.0
MAX_SECONDS = 30.0
_SPEECH = 0.5  # a frame scored above it is speech
_SILENCE_FRAMES = 31  # ~1 s of silence after speech ends the request


class SpeechEnd:
    """Reads a 16 kHz 16-bit mono recording as it grows: has someone spoken, and stopped?"""

    def __init__(self):
        # Silero's state carries from one window to the next, and has no reset: kept from the
        # previous request, it scored a clear request 0.26 instead of 0.99 (red LED)
        self.vad = SileroVAD(16000)
        self.offset = 0  # bytes of the recording read
        self.heard = False  # speech, at least one frame
        self.silent_frames = 0  # in a row, since speech
        self.ended = False

    def read(self, recording: bytes | bytearray) -> bool:
        """Read the recording's new frames; True once speech has ended.

        Frames are cut from the whole recording, not from each chunk: the satellite's chunks have no
        fixed size since ESPHome 2026.6, and a chunk's leftover would be lost."""
        while not self.ended and self.offset + FRAME_BYTES <= len(recording):
            frame = array.array("h", recording[self.offset : self.offset + FRAME_BYTES])
            self.offset += FRAME_BYTES
            try:
                is_speech = self.vad.process(array.array("f", [s / 32768.0 for s in frame])) > _SPEECH
            except Exception as e:
                logger.debug(f"VAD error: {e}")
                continue
            if is_speech:
                self.silent_frames = 0
                if not self.heard:
                    self.heard = True
                    logger.info("Speech detected")
            elif self.heard:
                self.silent_frames += 1
                if self.silent_frames >= _SILENCE_FRAMES:
                    logger.info(f"Silence detected after speech ({self.silent_frames} frames)")
                    self.ended = True
        return self.ended
