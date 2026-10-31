"""Spotify music: go-librespot plays into a FIFO, served to the speakers as an MP3 stream.

go-librespot is a Spotify Connect player running as its own service (home-infra
charras/go-librespot). It writes raw PCM into a FIFO as fast as the FIFO is read,
so SpotifyStream reads it 10 s ahead of the speakers, which get those 10 s at once as
they join (sent at real-time pace, a Google Home stalled 20 s at every start), fills
go-librespot's gaps (pause, track loading) with silence so that speakers never starve,
encodes MP3 with ffmpeg and fans it out to every listener. A speaker plays the stream
URL like a radio station. go-librespot's events tell where each track starts on the
stream: the track told is the one the speakers play, 10 s behind go-librespot.

Search goes through the public Web API with an app token of our own (client
credentials): go-librespot's session token belongs to Spotify's shared desktop client
ID, which the Web API answers with a standing 429. The user's playlists and Liked
Songs come from go-librespot itself.

Spotify ranks results by popularity more than by the words asked: « prélude n° 2 de Bach
en do mineur » came first as the prelude in C major. The results are ranked again here on
the words of the request, its numbers and its key, said in French or written in English
or German (« do mineur », « C minor », « c-Moll »).
"""

import asyncio
import functools
import logging
import os
import re
import time
from collections import Counter, deque
from typing import Any, Callable, NamedTuple, Optional, Sequence

import aiohttp
from aiohttp import web

from ha_client import STOPWORDS, compact, fold, normalize

logger = logging.getLogger(__name__)

LIBRESPOT_URL = os.environ.get("SPOTIFY_LIBRESPOT_URL", "http://127.0.0.1:3678")
PCM_FIFO = os.environ.get("SPOTIFY_PCM_FIFO", "/run/go-librespot/pcm")
WEB_API = "https://api.spotify.com/v1"
# A Spotify developer app (developer.spotify.com), used for search only
CLIENT_ID = os.environ.get("SPOTIFY_CLIENT_ID", "")
CLIENT_SECRET = os.environ.get("SPOTIFY_CLIENT_SECRET", "")
# A Web API call's two tries: a search takes under a second, so a first one past 2 s has hung (2026-10-20)
_WEB_TIMEOUTS = (aiohttp.ClientTimeout(total=2), aiohttp.ClientTimeout(total=5))
# go-librespot answered in 22 ms (2026-10-30), asking Spotify: the liked tracks only steer a choice, not worth a wait
_LIKED_TIMEOUT = aiohttp.ClientTimeout(total=0.5)

# go-librespot's PCM: 44.1 kHz, stereo, s16le
_FRAME = 4
_BYTES_PER_SECOND = 44100 * _FRAME
_TICK = 0.05


def _pcm_bytes(seconds: float) -> int:
    """That much PCM, in whole frames: a split frame would garble the rest of the stream."""
    return int(seconds * _BYTES_PER_SECOND) // _FRAME * _FRAME


_CHUNK = _pcm_bytes(_TICK)
# The encoder's MP3: 192 kbit/s
_MP3_BYTES_PER_SECOND = 192_000 // 8
# The FIFO is read this far ahead of what the speakers play, and a speaker that joins gets it at once. Sent at
# real-time pace, the living room's Google Home played 1 s of it, stalled, and played again only once it held about
# 22 s, at every start; given 6 s at once or more, it played within 1.7 s and never stalled (2026-10-31, AD-091)
_HEAD_START = 10.0
# Read at up to this much a tick, the event loop never held long
_MOST_READ = 1.0
# The FIFO holds 0.37 s, which go-librespot refills at once: a head start is read at this tick. At 50 ms, it took 1.4 s,
# and the Google Home that joined 0.15 s after it was loaded, with 1.3 s, stalled 12.7 s (2026-10-31)
_FILL_TICK = 0.005
# When go-librespot sends nothing (paused from elsewhere, a track slow to come), silence keeps the speakers this far
# ahead: they never starve
_FLOOR = 2.0
# New music waits this long for its speaker to join, and once it has joined, this long at most for its head start
_JOIN_WAIT = 10.0
_FILL_WAIT = 2.0
# A speaker that left finds the stream where it left it this long: the Sonos opens it, closes it and opens it again
_REJOIN_WAIT = 3.0
# Music read while nobody listens this long gets paused: its speaker moved to a radio,
# was stopped, here or from elsewhere, and the account would stay "playing" on this device
_IDLE_PAUSE = 5.0
# A listener that has received nothing for this long was dropped as too slow
_LISTENER_TIMEOUT = 5.0
# After a restart, speakers come back slowly (the Paris Sonos: 49 s): music is not paused before
_STARTUP_GRACE = 60.0
# Paused music still sent silence to its speaker forever (86 MB/h through the Paris tunnel)
_SILENCE_STOP = 600.0


class SpotifyStream:
    """MP3 stream of go-librespot's FIFO, shared by every listener, read a head start ahead of them (AD-091).

    Its clock is the audio encoded, `position`. The speakers play `heard`, a head start behind: the MP3 in between
    is kept, and a speaker that joins gets it at once. When go-librespot changes what it plays (switch, then
    restart), what it had sent of the old music is dropped, and heard waits at the new music's start until its
    speaker joins with the head start read."""

    def __init__(self, fifo_path: str = PCM_FIFO, on_idle: Optional[Callable] = None):
        self.fifo_path = fifo_path
        self.on_idle = on_idle
        self.on_silence: Optional[Callable] = None  # listeners hearing only silence for _SILENCE_STOP
        self._tasks: set[asyncio.Task] = set()
        self.listeners: set[asyncio.Queue] = set()
        self.joins = 0  # listeners that ever joined: tells a new connection from an old one
        self._joined = asyncio.Condition()
        self._fd: Optional[int] = None
        self._pending = b""  # bytes of an incomplete frame, kept for the next read
        self._eof_ticks = 0
        self._encoder: Optional[asyncio.subprocess.Process] = None
        self.position = 0.0  # seconds of audio encoded: the stream's clock
        self.heard = 0.0  # where on that clock the speakers play
        self._silences: deque[list[float]] = deque(maxlen=16)  # [start, end] of silence added in go-librespot's gaps
        self._backlog = bytearray()  # the MP3 from heard to position
        self.switching = False  # go-librespot is changing what it plays: what it sends meanwhile is dropped
        self._filled = asyncio.Event()  # the head start is read
        self._hurry = False  # it is being read
        self._held_until: Optional[float] = None  # heard waits for a speaker to join, until then at most
        self._joined_at: Optional[float] = None  # when the first one joined since
        self._unheard = self._silent = 0.0
        self._grace_until = 0.0

    async def run(self) -> None:
        """Pump PCM from the FIFO into the encoder, forever."""
        deadline = previous = time.monotonic()
        self._grace_until = deadline + _STARTUP_GRACE
        while True:
            if self._hurry:
                await asyncio.sleep(_FILL_TICK)
                deadline = time.monotonic()
            else:
                deadline += _TICK
                delay = deadline - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
                elif delay < -1:
                    deadline = time.monotonic()  # the event loop stalled: resync rather than catch up tick by tick
            now = time.monotonic()
            # An error here killed the stream silently for the life of the process
            try:
                await self._tick(now, now - previous)
            except Exception:
                logger.exception("Spotify stream error")
                self._stop_encoder()
                await asyncio.sleep(1)
            previous = now

    async def _tick(self, now: float, elapsed: float) -> None:
        self._hurry = False
        if self.switching:
            self._read()  # the old music's: dropped
            return
        if self._held_until is not None:
            if self._joined_at is not None and (self._lead() >= _HEAD_START or now - self._joined_at >= _FILL_WAIT):
                self._held_until = None  # its speaker is here, with its head start
            elif now >= self._held_until:
                self._held_until = None
                self._stop_encoder()  # nobody came
        elif self._encoder and not self.listeners:
            self._hold(now, _REJOIN_WAIT)
        held = self._held_until is not None
        if not held and not self.listeners:
            self._alone(now, elapsed, self._read())
            return
        self._unheard = 0.0
        if not held:
            self.heard = min(self.heard + elapsed, self.position)
        # Past the head start, at real-time pace: waiting for its speaker, new music keeps all it read
        wanted = min(max(self.heard + _HEAD_START - self.position, elapsed), _MOST_READ)
        pcm = self._read(_pcm_bytes(wanted))
        self._hurry = bool(pcm) and self._lead() + len(pcm) / _BYTES_PER_SECOND < _HEAD_START
        if not held:
            pcm = self._fill_gap(pcm, elapsed)
        if pcm:
            await self._encode(pcm)
        if self._lead() >= _HEAD_START:
            self._filled.set()

    def _alone(self, now: float, elapsed: float, pcm: bytes) -> None:
        """Music read while nobody listens: paused after a while."""
        self._unheard = self._unheard + elapsed if pcm else 0.0
        if self._unheard >= _IDLE_PAUSE and self.on_idle and now >= self._grace_until:
            logger.info("Spotify: playing to nobody, pausing")
            self._unheard = 0.0
            self._spawn(self.on_idle())

    def _fill_gap(self, pcm: bytes, elapsed: float) -> bytes:
        """Silence when go-librespot sends nothing and the speakers are about to run out."""
        self._silent = 0.0 if pcm else self._silent + elapsed
        if self._silent >= _SILENCE_STOP and self.on_silence:
            logger.info(f"Spotify: silence for {_SILENCE_STOP:.0f} s, stopping its speaker")
            self._silent = 0.0
            self._spawn(self.on_silence())
        if pcm or (short := _FLOOR - self._lead()) <= 0:
            return pcm
        silence = bytes(_pcm_bytes(short))
        end = self.position + len(silence) / _BYTES_PER_SECOND
        if self._silences and self._silences[-1][1] == self.position:
            self._silences[-1][1] = end
        else:
            self._silences.append([self.position, end])
        return silence

    def silence(self, start: float, end: float) -> float:
        """Seconds of silence added between two points of the stream's clock: no track's."""
        return sum(max(0.0, min(end, b) - max(start, a)) for a, b in self._silences)

    def _lead(self) -> float:
        return self.position - self.heard

    def _hold(self, now: float, wait: float) -> None:
        """heard waits for a speaker to join, `wait` at most."""
        self._held_until = now + wait
        self._joined_at = None

    def _drop(self) -> None:
        """Nobody will hear what was read ahead."""
        self._backlog.clear()
        self._filled.clear()
        self.heard = self.position

    async def head_start(self, timeout: float = _FILL_WAIT) -> None:
        """Wait until the head start is read, `timeout` at most: a speaker loaded then gets it at once."""
        try:
            await asyncio.wait_for(self._filled.wait(), timeout)
        except TimeoutError:
            pass

    def switch(self) -> None:
        """go-librespot is changing what it plays: what it sent of the old music, and sends until restart(), is
        dropped."""
        self.switching = True
        self._drop()

    def restart(self) -> None:
        """go-librespot plays the new music from here: heard waits at its start for its speaker."""
        self.switching = False
        self._drop()
        self._hold(time.monotonic(), _JOIN_WAIT)

    def _spawn(self, coroutine) -> None:
        """Run a callback in the background, kept referenced and its failure logged."""
        task = asyncio.ensure_future(coroutine)
        self._tasks.add(task)
        task.add_done_callback(self._task_done)

    def _task_done(self, task: asyncio.Task) -> None:
        self._tasks.discard(task)
        if not task.cancelled() and task.exception():
            logger.error(f"Spotify: {task.exception()}")

    def _read(self, size: int = _CHUNK) -> bytes:
        """Read up to `size` bytes of whole frames from the FIFO, without blocking."""
        if self._fd is None and not self._open():
            return b""
        data = self._pending
        while len(data) < size:
            try:
                more = os.read(self._fd, size - len(data))
            except BlockingIOError:  # go-librespot attached but behind
                break
            if not more:
                break
            data += more
        if not data:
            self._eof_ticks += 1
            # No writer: make sure the FIFO is still the one at the path (recreated at boot)
            if self._eof_ticks % 40 == 0 and not self._same_fifo():
                os.close(self._fd)
                self._fd = None
            return b""
        self._eof_ticks = 0
        cut = len(data) - len(data) % _FRAME
        self._pending = data[cut:]
        return data[:cut]

    def _open(self) -> bool:
        try:
            self._fd = os.open(self.fifo_path, os.O_RDONLY | os.O_NONBLOCK)
        except OSError:
            return False
        self._pending = b""
        logger.info(f"Spotify: reading {self.fifo_path}")
        return True

    def _same_fifo(self) -> bool:
        try:
            return os.stat(self.fifo_path).st_ino == os.fstat(self._fd).st_ino
        except OSError:
            return False

    async def _encode(self, pcm: bytes) -> None:
        if self._encoder is None:
            self._encoder = await asyncio.create_subprocess_exec(
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                # No probing of raw PCM: the first MP3 byte came 1.2 s late, 0.05 s without
                "-probesize", "32", "-analyzeduration", "0",
                "-f", "s16le", "-ar", "44100", "-ac", "2", "-i", "pipe:0",
                "-c:a", "libmp3lame", "-b:a", "192k", "-f", "mp3", "-flush_packets", "1", "pipe:1",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
            )  # fmt: skip
            self._spawn(self._fan_out(self._encoder))
            logger.info("Spotify: encoder started")
        self.position += len(pcm) / _BYTES_PER_SECOND
        try:
            self._encoder.stdin.write(pcm)
            await self._encoder.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            logger.warning("Spotify: encoder exited, restarting")
            self._encoder = None
            self._drop()

    async def _fan_out(self, encoder: asyncio.subprocess.Process) -> None:
        while chunk := await encoder.stdout.read(65536):
            if encoder is not self._encoder:
                continue  # stopped: what it still had is nobody's
            self._keep(chunk)
            for queue in list(self.listeners):
                try:
                    queue.put_nowait(chunk)
                except asyncio.QueueFull:
                    self.listeners.discard(queue)  # too slow: its handler times out and closes

    def _keep(self, mp3: bytes) -> None:
        self._backlog += mp3
        self._trim()

    def _trim(self) -> None:
        """The backlog from heard on."""
        if (excess := len(self._backlog) - int(self._lead() * _MP3_BYTES_PER_SECOND)) > 0:
            del self._backlog[:excess]

    def _stop_encoder(self) -> None:
        if self._encoder is None:
            return
        self._encoder.stdin.close()  # ffmpeg flushes and exits; _fan_out ends with it
        self._encoder = None
        self._drop()
        logger.info("Spotify: encoder stopped")

    async def wait_for_join(self, joins_before: int, timeout: float) -> bool:
        """Wait until a listener connects after `joins_before` was read from self.joins."""
        async with self._joined:
            try:
                await asyncio.wait_for(self._joined.wait_for(lambda: self.joins != joins_before), timeout)
            except TimeoutError:
                return False
        return True

    async def handle(self, request: web.Request) -> web.StreamResponse:
        """GET /spotify/stream.mp3: the endless stream, from what the speakers play now on."""
        response = web.StreamResponse(headers={"Content-Type": "audio/mpeg", "Cache-Control": "no-store"})
        await response.prepare(request)
        if request.method == "HEAD":
            return response
        queue: asyncio.Queue = asyncio.Queue(maxsize=256)
        self._trim()
        if self._backlog:
            queue.put_nowait(bytes(self._backlog))
        self.listeners.add(queue)
        if self._held_until is not None and self._joined_at is None:
            self._joined_at = time.monotonic()
        async with self._joined:
            self.joins += 1
            self._joined.notify_all()
        ahead = len(self._backlog) / _MP3_BYTES_PER_SECOND
        logger.info(f"Spotify: listener {request.remote} joined ({len(self.listeners)} listening, {ahead:.1f} s ahead)")
        try:
            while True:
                await response.write(await asyncio.wait_for(queue.get(), _LISTENER_TIMEOUT))
        except (ConnectionResetError, TimeoutError):
            pass
        finally:
            self.listeners.discard(queue)
            logger.info(f"Spotify: listener {request.remote} left ({len(self.listeners)} listening)")
        return response


def _artists(names: Sequence[str]) -> str:
    return ", ".join(names[:2])


def _artist_names(item: dict) -> list[str]:
    return [a["name"] for a in item.get("artists", [])]


def _current(status: dict) -> Optional[dict]:
    """go-librespot's current track, None when there is none."""
    track = status.get("track")
    return None if not track or status.get("stopped") else track


def _said(name: str, artist_names: Sequence[str]) -> str:
    """A track as told to the LLM: AD-087 compares what was asked with what plays by it."""
    return f"« {name} » de {_artists(artist_names)}"


def _describe(kind: str, item: dict) -> str:
    """How the LLM is told what is playing."""
    if kind == "artist":
        return item["name"]
    if kind == "track":
        return _said(item["name"], _artist_names(item))
    if kind == "album":
        return f"l'album « {item['name']} » de {_artists(_artist_names(item))}"
    return f"la playlist « {item['name']} »"


def screen_metadata(title: str, artists: Sequence[str] = (), album: str = "", image: str = "") -> dict:
    """What a Cast screen shows beside the stream: Cast's music track metadata (metadataType 3).

    Without it, a Nest Hub showed « Default Media Receiver » and « Live »."""
    metadata = {"metadataType": 3, "title": title}
    if artists:
        metadata["artist"] = _artists(artists)
    if album:
        metadata["albumName"] = album
    if image:
        metadata["images"] = [{"url": image}]
    return metadata


class Music(NamedTuple):
    """What play_music found: the context to play, what the LLM is told it plays, what a screen shows; the track
    the context starts at, and what the LLM is told besides (AD-090)."""

    uri: str
    description: str
    metadata: dict
    start_at: str = ""
    note: str = ""


def _music(kind: str, item: dict) -> Music:
    """A search result as Music."""
    images = (item.get("album") or item).get("images") or []  # largest first
    image = images[0]["url"] if images else ""
    if kind in ("artist", "playlist"):
        metadata = screen_metadata(item["name"], image=image)
    else:
        album = item["album"]["name"] if kind == "track" else ""
        metadata = screen_metadata(item["name"], _artist_names(item), album, image)
    return Music(item["uri"], _describe(kind, item), metadata)


# A key, as said in French or written in English or German, and how its words are written in English
_KEYS = (
    # do mineur, fa dièse mineur, si bémol majeur
    (
        re.compile(r"\b(do|re|mi|fa|sol|la|si)(?: (diese|bemol))? (majeur|mineur)\b"),
        dict(zip(("do", "re", "mi", "fa", "sol", "la", "si"), "cdefgab"))
        | {"diese": "#", "bemol": "b", "majeur": "major", "mineur": "minor"},
    ),
    # C minor, F-sharp minor, E flat major, C# minor
    (
        re.compile(r"\b([a-g])(?:[ -]?(sharp|flat|#|♯|♭))?[ -](major|minor)\b"),
        {"sharp": "#", "flat": "b", "♯": "#", "♭": "b"},
    ),
    # c-Moll, fis-Moll, Es-Dur, B-Dur (B flat), h-Moll (B)
    (
        re.compile(r"\b([a-h])(is|es|s)?-(dur|moll)\b"),
        {"b": "bb", "h": "b", "is": "#", "es": "b", "s": "b", "dur": "major", "moll": "minor"},
    ),
)
# The words before a piece's own number: « numéro deux », « n°2 », « No. 2 »
_NUMBER_MARKERS = ("numero", "number", "num", "nr", "no", "n")
# Words that tell no piece from another. « (version classique) » found « Prelude No. 1 in C
# major (version n.2) » by The Piano Girls before Bach's second prelude
_FILLER = STOPWORDS | set(_NUMBER_MARKERS) | {
    "en", "et", "a", "au", "aux", "par", "pour", "the", "of", "in", "and", "by", "an", "for",
    "version", "musique", "music", "classique", "classical",
}  # fmt: skip
_TOKEN = re.compile(r"[a-z]+|\d+")
# A number Parakeet writes in words after « numéro » or « n° »: « le prélude numéro deux de Bach »
_NUMBER_WORDS = {
    word: str(number) for number, word in enumerate((
        "zero", "un", "deux", "trois", "quatre", "cinq", "six", "sept", "huit", "neuf", "dix",
        "onze", "douze", "treize", "quatorze", "quinze", "seize", "dix-sept", "dix-huit", "dix-neuf", "vingt",
    ))
} | {"une": "1"}  # fmt: skip


_MARKED = "|".join(_NUMBER_MARKERS)
_TOLD_NUMBER = re.compile(rf"\b(?:{_MARKED})[^a-z\d]*(\d+)\b")
# A number never a piece's own: its book's, volume's, disc's, opus's or catalogue's (« Book 2 », « BWV 892 »)
_OTHER_NUMBER = re.compile(r"\b(?:book|livre|vol|volume|cd|disc|disque|op|opus|bwv|kv|k|hob|rv|hwv|d)[^a-z\d]*\d+")
# Who wrote or plays it, last in a French request: « … de Bach », « … joué par Glenn Gould »
_BY = re.compile(r".*\b(?:de|d|par)\b(.+)")


def _numbered(text: str) -> str:
    """A folded text with a number said in words after « numéro » written in digits."""
    return re.sub(
        rf"\b({_MARKED})([^a-z\d]+)(dix-sept|dix-huit|dix-neuf|[a-z]+)\b",
        lambda m: m[1] + m[2] + _NUMBER_WORDS.get(m[3], m[3]),
        text,
    )


def _told_numbers(text: str) -> set[str]:
    """The numbers a folded text tells as a piece's own, after « numéro », « n° » or « No. »."""
    return set(_TOLD_NUMBER.findall(_numbered(text)))


def _own_numbers(title: str) -> set[str]:
    """A folded title's numbers that may be the piece's own: those it tells so, else all but its book's,
    volume's, opus's or catalogue's (« Gymnopédie 1 », « 5ème symphonie »; not « Book 2: Prelude in C Major »)."""
    return _told_numbers(title) or set(re.findall(r"\d+", _OTHER_NUMBER.sub(" ", _numbered(title))))


def _keys(text: str) -> list[str]:
    """The keys named in a folded text, in order (« en do mineur (C minor) » → c minor, c minor)."""
    found = []
    for pattern, english in _KEYS:
        for match in pattern.finditer(text):
            note, accidental, mode = (english.get(part, part) if part else "" for part in match.groups())
            found.append((match.start(), f"{note}{accidental} {mode}"))
    return [key for _, key in sorted(found)]


def _words(text: str) -> list[str]:
    """A folded text's words and numbers, keys and filler left out: « BWV847 » → bwv, 847; « numéro deux » → 2."""
    for pattern, _ in _KEYS:
        text = pattern.sub(" ", text)
    return [w for w in _TOKEN.findall(_numbered(text)) if w not in _FILLER]


def _same_word(asked: str, word: str) -> bool:
    # « symphonie » is « Symphony », « sonate » « Sonata », « préludes » « prélude »
    return asked == word or (len(asked) >= 5 and len(word) >= 5 and asked[:5] == word[:5])


def _score(asked: list[str], keys: set[str], numbers: set[str], item: dict) -> int:
    """How well a search result fits the request: its words found, its key, and its number."""
    name = fold(item["name"])
    album = fold(item.get("album", {}).get("name", ""))
    words = _words(" ".join([name, album, *map(fold, _artist_names(item))]))
    score = sum(any(_same_word(w, word) for word in words) for w in asked if w not in numbers)
    # The number asked after « numéro » is the piece's own, never a book's or a catalogue's: « Prélude numéro 2
    # de Bach » found « Well-Tempered Clavier, Book 2, Prelude and Fugue No. 23 » (2026-10-30)
    score += len(numbers & _own_numbers(name))
    found = _keys(name) or _keys(album)
    if keys and found:
        score += 1 if found[0] in keys else -1  # « en do mineur » is not « en do majeur »
    return score


def best_match(query: str, items: list[dict], liked: frozenset[str] = frozenset()) -> dict:
    """The search result that fits the request best; among equals, a track the account liked, then one by the artist
    the most results credit, then Spotify's order.

    Not an exact name first: « Get Lucky » was L.E.J's cover, « Sonate au clair de lune
    Beethoven » a nursery version titled so. The most credited artist: « Clair de lune de Debussy » found an upload
    by « Johann Debussy », whom no other result credits, where 4 of the 10 credit Claude Debussy (2026-10-30)."""
    plain = fold(query)
    asked, keys, numbers = _words(plain), set(_keys(plain)), _told_numbers(plain)
    credits = Counter(name for item in items for name in set(_artist_names(item)))

    def rank(item: dict) -> tuple:
        credited = max((credits[name] for name in _artist_names(item)), default=0)
        return _score(asked, keys, numbers, item), item["uri"] in liked, credited

    return max(items, key=rank)  # max keeps the first of equals


def search_variant(query: str) -> Optional[str]:
    """The request as Spotify's catalog writes it, for a piece told by its number or key:
    « Prélude numéro 2 de Bach en Do mineur » → « prelude 2 bach c minor »."""
    plain = fold(query)
    words, keys = _words(plain), list(dict.fromkeys(_keys(plain)))
    if not keys and not any(w.isdigit() for w in words):
        return None  # « Get Lucky de Daft Punk » needs one search
    variant = " ".join(words + keys)
    return variant if variant != " ".join(_TOKEN.findall(plain)) else None


def composer_first(query: str, variant: Optional[str]) -> Optional[str]:
    """The request's search_variant with who wrote or plays it first: Spotify gave Glenn Gould's « Prelude & Fugue
    No. 2 » for « bach prelude 2 », and no second prelude for « prelude 2 bach » (2026-10-30)."""
    if not variant or not (by := _BY.match(fold(query))):
        return None
    names = _words(by[1])
    words = [w for w in variant.split() if w not in names]
    return " ".join(names + words) if names and words else None


_PLAYLIST_FILLER = {"playlist", "ma", "mon", "mes", "notre", "nos"}
# How the liked tracks are named: « mes titres likés », « dans mes favoris », « mes coups de cœur »
_LIKED_WORDS = re.compile(
    r"\b(?:dans |parmi |de |d')?(?:mes |nos |vos |les )?(?:titres? |morceaux |chansons |musique )?"
    r"(?:likee?s?|liked|favoris|favorites?|preferes?|preferees?|coups? de c(?:oe|œ)ur|aimes?)\b"
)
# Words left around them that name no track: « tous mes titres likés en aléatoire », « les titres que j'ai likés »
_NAMES_NOTHING = _PLAYLIST_FILLER | {
    "vos", "tous", "toutes", "tout", "titres", "titre", "morceaux", "chansons", "celui", "celle", "ceux", "qui",
    "que", "est", "sont", "dans", "j", "ai", "spotify", "aleatoire", "hasard", "melange", "lecture",
}  # fmt: skip
# What play_music tells the model of a track named among the liked ones (AD-090)
LIKED_THEN_COLLECTION = "dans vos titres likés, qui suivent"
NOT_LIKED = "qui n'est pas dans vos titres likés"


def liked_piece(query: str) -> Optional[str]:
    """The track a request for the liked tracks names among them, folded, or None for the whole collection:
    « prélude numéro deux de Bach Glengould », not « mes favoris »."""
    rest = " ".join(_LIKED_WORDS.sub(" ", fold(query)).split())
    return rest if set(_words(rest)) - _NAMES_NOTHING else None


def match_playlist(playlists: list[dict], query: str) -> Optional[dict]:
    """Find one of the user's playlists by spoken name ("ma playlist du matin" → "Matin")."""
    key = compact(" ".join(w for w in normalize(query).split() if w not in _PLAYLIST_FILLER))
    if not key:
        return None
    names = [(compact(p["name"]), p) for p in playlists]
    exact = next((p for name, p in names if name == key), None)
    return exact or next((p for name, p in names if name and (key in name or name in key)), None)


def _collection(status: dict) -> str:
    """The liked tracks' context, from go-librespot's status."""
    return f"spotify:user:{status['username']}:collection"


class SpotifyError(Exception):
    """Spotify or go-librespot failed. The message never quotes a request: its headers carry tokens."""


class SpotifyNotConnected(SpotifyError):
    """go-librespot runs but has no Spotify session (not paired with an account yet)."""


class SpotifySearchError(SpotifyError):
    """Spotify's Web API, on the internet, failed: nothing was played yet."""


# music_control actions → what the LLM is told
CONTROLS = {
    "pause": "Musique en pause",
    "resume": "Lecture reprise",
    "next": "Titre suivant",
    "previous": "Titre précédent",
}
# go-librespot's change of music shows by its event within this; past it, the stream starts again anyway
_CHANGE_TIMEOUT = 3.0
# « Précédent » this far into a track goes to the one before; further, starts it again, as Spotify's apps do
_RESTART_PAST = 3.0


def _seek(seconds: float) -> tuple[str, dict, str]:
    return "/player/seek", {"position": int(seconds * 1000)}, "seek"


# Steps of a change: go-librespot's API path, its JSON body, and the event that shows it done
_SEEK_START = _seek(0)
_NEXT = ("/player/next", None, "metadata")
_PREVIOUS = ("/player/prev", None, "metadata")


def _steps(tracks: int, offset: float = 0.0) -> list[tuple]:
    """go-librespot's steps to the track `tracks` away from its own (back if negative), `offset` seconds in."""
    if tracks > 0:
        return [_NEXT] * tracks
    # Its previous starts its track again past 3 s: from the start, it goes back
    back = [_SEEK_START, *[_PREVIOUS] * -tracks] if tracks or not offset else []
    return back + ([_seek(offset)] if offset else [])


class _Mark(NamedTuple):
    """A track as it plays from a point of the stream (AD-091)."""

    at: float  # SpotifyStream.position it plays from
    track: dict  # go-librespot's track: uri, name, artist_names, album_name, album_cover_url…
    offset: float  # seconds into it there


class Spotify:
    """go-librespot's API (playback, library) plus Web API search."""

    def __init__(self):
        self.stream = SpotifyStream(on_idle=functools.partial(self.control, "pause"))
        self._session: Optional[aiohttp.ClientSession] = None
        self._app_token = ""
        self._app_token_expiry = 0.0
        # Where each track plays from on the stream, told by go-librespot's events (watch)
        self._marks: deque[_Mark] = deque(maxlen=8)
        self._events = False  # they are followed
        self._restart_on: Optional[str] = None  # the event that starts the stream again (_change)
        self._waiting: dict[str, asyncio.Future] = {}  # by event

    def _http(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10))
        return self._session

    async def close(self) -> None:
        if self._session:
            await self._session.close()

    async def _request(self, method: str, url: str, **kwargs) -> tuple[int, Any]:
        """One HTTP call: (status, JSON body or None). Transport failures become SpotifyError."""
        try:
            async with self._http().request(method, url, **kwargs) as resp:
                body = await resp.json() if resp.content_type == "application/json" else None
                return resp.status, body
        except (aiohttp.ClientError, TimeoutError) as e:
            raise SpotifyError(f"{method} {url}: {type(e).__name__}") from None

    async def _web(self, method: str, url: str, **kwargs) -> tuple[int, Any]:
        """A Web API call, tried again once when the network failed: a search that timed out on 2026-10-20 took
        0.74 s when asked again. An HTTP error is Spotify's answer, kept."""
        for attempt, timeout in enumerate(_WEB_TIMEOUTS):
            try:
                return await self._request(method, url, timeout=timeout, **kwargs)
            except SpotifyError as e:
                if attempt:
                    raise SpotifySearchError(str(e)) from None
                logger.warning(f"Spotify Web API, trying again: {e}")

    async def _api(self, method: str, path: str, **kwargs) -> Any:
        """Call go-librespot; its 204 means no Spotify session."""
        status, body = await self._request(method, LIBRESPOT_URL + path, **kwargs)
        if status == 204:
            raise SpotifyNotConnected("go-librespot has no Spotify session")
        if status >= 400:
            raise SpotifyError(f"go-librespot {path}: HTTP {status}")
        return body

    async def reachable(self) -> bool:
        try:
            await self.status()
        except SpotifyNotConnected:
            pass
        except SpotifyError:
            return False
        return True

    async def status(self) -> dict:
        return await self._api("GET", "/status")

    async def play(self, uri: str, shuffle: bool = False, start_at: str = "") -> None:
        # Shuffle has to be set before play to start on a random track
        await self._api("POST", "/player/shuffle_context", json={"shuffle_context": shuffle})
        body = {"uri": uri} | ({"skip_to_uri": start_at} if start_at else {})
        await self._change(("/player/play", body, "metadata"))

    async def control(self, action: str) -> str:
        """Run a CONTROLS action; returns what the LLM is told."""
        await {"pause": self.pause, "resume": self.resume, "next": self._next, "previous": self._previous}[action]()
        return CONTROLS[action]

    async def pause(self) -> None:
        """go-librespot paused, and taken back to where the speakers were, a head start behind (AD-091): resumed
        from here or from the phone, it starts there."""
        heard = self._heard()
        status = await self.status()
        if status.get("paused") or not _current(status):
            return
        steps = [("/player/pause", None, "paused")]
        if heard:
            mark, ahead = heard
            steps += _steps(-ahead, mark.offset)
        await self._change(*steps)

    async def resume(self) -> bool:
        """False if go-librespot was not paused."""
        if not (await self.status()).get("paused"):
            return False
        await self._change(("/player/resume", None, "playing"))
        return True

    async def _next(self) -> None:
        """The track after the one heard: at its end, go-librespot's own, from its start."""
        heard = self._heard()
        await self._change(*(_steps(1 - heard[1]) if heard else [_NEXT]))

    async def _previous(self) -> None:
        """The track before the one heard, or that one from its start once past its first seconds."""
        if heard := self._heard():
            mark, ahead = heard
            await self._change(*_steps(-ahead - (mark.offset <= _RESTART_PAST)))
        else:
            await self._change(_PREVIOUS)

    async def _change(self, *steps: tuple[str, Optional[dict], str]) -> None:
        """Have go-librespot change what plays, step by step, each once the last showed: the stream drops what
        it read of the old music, and starts again at the last step's event, as the new music comes (AD-091)."""
        self.stream.switch()
        try:
            for i, (path, body, event) in enumerate(steps):
                if i == len(steps) - 1:
                    self._restart_on = event
                waiter = None
                if self._events:
                    waiter = self._waiting[event] = asyncio.get_running_loop().create_future()
                await self._api("POST", path, json=body)
                if waiter:
                    try:
                        await asyncio.wait_for(waiter, _CHANGE_TIMEOUT)
                    except TimeoutError:
                        logger.warning(f"Spotify: no {event} event from go-librespot after {path}")
        finally:
            self._restart_on = None
            if self.stream.switching:
                self.stream.restart()

    async def watch(self) -> None:
        """Follow go-librespot's events, which tell where each track starts on the stream (AD-091)."""
        failing = False
        while True:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(LIBRESPOT_URL + "/events", heartbeat=30) as ws:
                        self._events, failing = True, False
                        await self._mark_status()
                        async for message in ws:
                            if message.type == aiohttp.WSMsgType.TEXT:
                                self._on_event(message.json())
            except (aiohttp.ClientError, OSError, ValueError) as e:
                if not failing:
                    logger.warning(f"Spotify: go-librespot's events: {type(e).__name__} {e}")
                failing = True
            finally:
                self._events = False
            await asyncio.sleep(5)

    async def _mark_status(self) -> None:
        """What plays as the events start to be followed, none having told it."""
        try:
            status = await self.status()
        except SpotifyError:
            return
        if track := _current(status):
            self._mark(track, track["position"] / 1000)

    def _on_event(self, event: dict) -> None:
        kind, data = event.get("type"), event.get("data") or {}
        if kind == self._restart_on:
            self._restart_on = None
            self.stream.restart()
        if kind == "metadata":  # a track loaded: what go-librespot sends from now on
            self._mark(data, data.get("position", 0) / 1000)
        elif kind == "seek":
            if track := next((m.track for m in reversed(self._marks) if m.track["uri"] == data.get("uri")), None):
                self._mark(track, data["position"] / 1000)
        if (waiter := self._waiting.pop(kind, None)) and not waiter.done():
            waiter.set_result(None)

    def _mark(self, track: dict, offset: float) -> None:
        self._marks.append(_Mark(self.stream.position, track, offset))

    def _heard(self) -> Optional[tuple[_Mark, int]]:
        """The track the speakers play, as a mark where they are, and how many tracks go-librespot went on since;
        None unless the events are followed."""
        if not self._events:
            return None
        heard, marks = self.stream.heard, list(self._marks)
        for i in range(len(marks) - 1, -1, -1):
            if (mark := marks[i]).at <= heard:
                offset = mark.offset + heard - mark.at - self.stream.silence(mark.at, heard)
                ahead = sum(a.track["uri"] != b.track["uri"] for a, b in zip(marks[i:], marks[i + 1 :]))
                return mark._replace(at=heard, offset=offset), ahead
        return None

    def _playing(self, status: dict) -> Optional[dict]:
        """The track the speakers play, which go-librespot may have left a head start ago (AD-091); None when
        there is none."""
        if not (track := _current(status)):
            return None
        heard = self._heard()
        return heard[0].track if heard else track

    async def now_playing(self) -> str:
        status = await self.status()
        if not (track := self._playing(status)):
            return "Rien en cours sur Spotify"
        paused = " (en pause)" if status.get("paused") else ""
        return f"En cours sur Spotify : {_said(track['name'], track['artist_names'])}{paused}"

    async def playing(self) -> Optional[str]:
        """The track playing right now, None when nothing plays (or it is paused)."""
        status = await self.status()
        track = None if status.get("paused") else self._playing(status)
        return track and _said(track["name"], track["artist_names"])

    async def _search_token(self) -> str:
        if not CLIENT_ID or not CLIENT_SECRET:
            raise SpotifySearchError("search needs SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET")
        if time.monotonic() < self._app_token_expiry:
            return self._app_token
        auth = aiohttp.BasicAuth(CLIENT_ID, CLIENT_SECRET)
        data = {"grant_type": "client_credentials"}
        status, body = await self._web("POST", "https://accounts.spotify.com/api/token", data=data, auth=auth)
        if status != 200:
            raise SpotifySearchError(f"app token: HTTP {status}")
        self._app_token = body["access_token"]
        self._app_token_expiry = time.monotonic() + body["expires_in"] - 60
        return self._app_token

    async def _search(self, query: str, kind: str) -> list[dict]:
        # An app token has no user to take the market from. 10 is the most Spotify gives (400 above)
        params = {"q": query, "type": kind, "limit": 10, "market": "FR"}
        headers = {"Authorization": f"Bearer {await self._search_token()}"}
        status, body = await self._web("GET", f"{WEB_API}/search", params=params, headers=headers)
        if status != 200:
            raise SpotifySearchError(f"search: HTTP {status}")
        return [item for item in body[f"{kind}s"]["items"] if item]  # playlist results hold nulls

    async def find(self, query: str, kind: Optional[str]) -> Optional[Music]:
        """Resolve a spoken request to what to play, or None if nothing matches.

        kind: artist, track, album, playlist (the user's own first, then search) or liked."""
        if kind == "liked":
            if piece := liked_piece(query):
                return await self._among_liked(piece)
            return Music(_collection(await self.status()), "vos titres likés", screen_metadata("Titres likés"))
        if kind == "playlist":
            library = await self._api("GET", "/library/playlists", params={"limit": 500})
            own = match_playlist(library["items"], query)
            if own:
                metadata = screen_metadata(own["name"], image=own.get("image_url") or "")
                return Music(own["uri"], _describe("playlist", own), metadata)
        if kind not in ("artist", "track", "album", "playlist"):
            kind = "track"
        if not (items := await self._results(query, kind)):
            return None
        liked = await self._liked([item["uri"] for item in items]) if kind == "track" else None
        return _music(kind, best_match(query, items, liked or frozenset()))

    async def _results(self, query: str, kind: str) -> list[dict]:
        """Spotify's search, ten results at most, is fuzzy: the request as said, as its catalog writes it, and with
        its composer first; their results in order, once each."""
        variant = search_variant(query)
        searches = dict.fromkeys(filter(None, (query, variant, composer_first(query, variant))))
        await self._search_token()  # once, not by each search
        results = await asyncio.gather(*(self._search(q, kind) for q in searches), return_exceptions=True)
        if isinstance(results[0], Exception):
            raise results[0]  # the others only help: one failing leaves the request's own results
        found = [items for items in results if not isinstance(items, Exception)]
        return list({item["uri"]: item for item in sum(found, [])}.values())

    async def _among_liked(self, piece: str) -> Optional[Music]:
        """The track a request names among the liked ones, started in their collection, which goes on after it.
        « Celui de Glengould, dans mes titres préférés » played the whole collection, the request left aside, and
        the model said Gould's (2026-10-29). go-librespot tells whether tracks are liked, not which: a search
        finds them, its liked results first."""
        status, items = await asyncio.gather(self.status(), self._results(piece, "track"))
        if not items:
            return None
        if (among := await self._liked([item["uri"] for item in items])) is None:
            return _music("track", best_match(piece, items))  # whether it is among them, nobody can tell
        if not (mine := [item for item in items if item["uri"] in among]):
            return _music("track", best_match(piece, items))._replace(note=NOT_LIKED)
        best = best_match(piece, mine)
        return _music("track", best)._replace(uri=_collection(status), start_at=best["uri"], note=LIKED_THEN_COLLECTION)

    async def _liked(self, uris: list[str]) -> Optional[frozenset[str]]:
        """Those of these tracks the account liked: « j'ai mis ce morceau précis dans mes titres likés »
        (2026-10-30). go-librespot can tell, not list them; None when it cannot, not worth failing the music for."""
        try:
            body = await self._api("GET", "/library/liked", params={"uris": ",".join(uris)}, timeout=_LIKED_TIMEOUT)
        except SpotifyError:
            return None
        return frozenset(item["uri"] for item in body.get("items", []) if item.get("liked"))

    async def now_metadata(self) -> Optional[dict]:
        """What a screen shows for the track the speakers play, None when there is none."""
        if not (track := self._playing(await self.status())):
            return None
        cover = track["album_cover_url"] or ""
        return screen_metadata(track["name"], track["artist_names"], track["album_name"], cover)

    async def now_for_app(self) -> Optional[dict]:
        """What the web app shows of the track the speakers play, None when there is none."""
        status = await self.status()
        if not (track := self._playing(status)):
            return None
        return {
            "title": track["name"],
            "artist": _artists(track["artist_names"]),
            "cover": track["album_cover_url"] or None,
            "paused": bool(status.get("paused")),
        }
