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
# A burst is speech from its 4th frame (128 ms): a click or a breath scored 1 to 3 and was answered, the
# shortest request heard (« Stop. ») 6 (AD-092)
_MIN_SPEECH_FRAMES = 4
_SILENCE_FRAMES = 31  # ~1 s of silence after speech ends the request


class SpeechEnd:
    """Reads a 16 kHz 16-bit mono recording as it grows: has someone spoken, and stopped?"""

    def __init__(self):
        # Silero's state carries from one window to the next, and has no reset: kept from the
        # previous request, it scored a clear request 0.26 instead of 0.99 (red LED)
        self.vad = SileroVAD(16000)
        self.offset = 0  # bytes of the recording read
        self.speech_frames = 0  # in the burst
        self.silent_frames = 0  # in a row, since the burst's last speech frame
        self.ended = False

    @property
    def heard(self) -> bool:
        """Speech: a burst of at least _MIN_SPEECH_FRAMES."""
        return self.speech_frames >= _MIN_SPEECH_FRAMES

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
                self.speech_frames += 1
                if self.speech_frames == _MIN_SPEECH_FRAMES:
                    logger.info("Speech detected")
            elif self.speech_frames:
                self.silent_frames += 1
                if self.silent_frames >= _SILENCE_FRAMES:
                    if self.heard:
                        logger.info(f"Silence detected after speech ({self.silent_frames} frames)")
                        self.ended = True
                    else:  # too short to be speech: forgotten, and the wait for speech goes on
                        logger.info(f"Too short for speech ({self.speech_frames} frames)")
                        self.speech_frames = self.silent_frames = 0
        return self.ended
