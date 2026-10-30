"""Spotify music: go-librespot plays into a FIFO, served to the speakers as an MP3 stream.

go-librespot is a Spotify Connect player running as its own service (home-infra
charras/go-librespot). It writes raw PCM into a FIFO as fast as the FIFO is read,
so SpotifyStream reads it at real-time pace, fills the gaps (pause, track loading)
with silence so that speakers never starve, encodes MP3 with ffmpeg and fans it out
to every listener. A speaker plays the stream URL like a radio station.

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
# go-librespot answered in 22 ms (2026-10-30), asking Spotify: a liked track only breaks a tie, not worth a wait
_LIKED_TIMEOUT = aiohttp.ClientTimeout(total=0.5)

# go-librespot's PCM: 44.1 kHz, stereo, s16le
_FRAME = 4
_BYTES_PER_SECOND = 44100 * _FRAME
_TICK = 0.05
_CHUNK = int(_BYTES_PER_SECOND * _TICK) // _FRAME * _FRAME
# The encoder outlives its last listener this long: a speaker reconnecting keeps it
_ENCODER_LINGER = 3.0
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
    """Real-time MP3 stream of go-librespot's FIFO, shared by every listener."""

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

    async def run(self) -> None:
        """Pump PCM from the FIFO into the encoder at real-time pace, forever."""
        loop = asyncio.get_running_loop()
        deadline = loop.time()
        grace_until = loop.time() + _STARTUP_GRACE
        alone = unheard = silent = 0.0
        while True:
            deadline += _TICK
            delay = deadline - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)
            elif delay < -1:
                deadline = loop.time()  # the event loop stalled: resync rather than burst
            # An error here killed the stream silently for the life of the process
            try:
                pcm = self._read()
                if self.listeners:
                    alone = unheard = 0.0
                    silent = 0.0 if pcm else silent + _TICK
                    if silent >= _SILENCE_STOP and self.on_silence:
                        logger.info(f"Spotify: silence for {_SILENCE_STOP:.0f} s, stopping its speaker")
                        silent = 0.0
                        self._spawn(self.on_silence())
                    await self._encode(pcm.ljust(_CHUNK, b"\0"))
                    continue
                alone += _TICK
                if self._encoder and alone >= _ENCODER_LINGER:
                    self._stop_encoder()
                unheard = unheard + _TICK if pcm else 0.0
                if unheard >= _IDLE_PAUSE and self.on_idle and loop.time() >= grace_until:
                    logger.info("Spotify: playing to nobody, pausing")
                    unheard = 0.0
                    self._spawn(self.on_idle())
            except Exception:
                logger.exception("Spotify stream error")
                self._stop_encoder()
                await asyncio.sleep(1)

    def _spawn(self, coroutine) -> None:
        """Run a callback in the background, kept referenced and its failure logged."""
        task = asyncio.ensure_future(coroutine)
        self._tasks.add(task)
        task.add_done_callback(self._task_done)

    def _task_done(self, task: asyncio.Task) -> None:
        self._tasks.discard(task)
        if not task.cancelled() and task.exception():
            logger.error(f"Spotify: {task.exception()}")

    def _read(self) -> bytes:
        """Read up to one tick of whole frames from the FIFO, without blocking."""
        if self._fd is None and not self._open():
            return b""
        try:
            data = os.read(self._fd, _CHUNK - len(self._pending))
        except BlockingIOError:  # go-librespot attached but behind
            data = b""
        if not data and not self._pending:
            self._eof_ticks += 1
            # No writer: make sure the FIFO is still the one at the path (recreated at boot)
            if self._eof_ticks % 40 == 0 and not self._same_fifo():
                os.close(self._fd)
                self._fd = None
            return b""
        self._eof_ticks = 0
        data = self._pending + data
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
        try:
            self._encoder.stdin.write(pcm)
            await self._encoder.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            logger.warning("Spotify: encoder exited, restarting")
            self._encoder = None

    async def _fan_out(self, encoder: asyncio.subprocess.Process) -> None:
        while chunk := await encoder.stdout.read(4096):
            for queue in list(self.listeners):
                try:
                    queue.put_nowait(chunk)
                except asyncio.QueueFull:
                    self.listeners.discard(queue)  # too slow: its handler times out and closes

    def _stop_encoder(self) -> None:
        if self._encoder is None:
            return
        self._encoder.stdin.close()  # ffmpeg flushes and exits; _fan_out ends with it
        self._encoder = None
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
        """GET /spotify/stream.mp3: the endless stream."""
        response = web.StreamResponse(headers={"Content-Type": "audio/mpeg", "Cache-Control": "no-store"})
        await response.prepare(request)
        if request.method == "HEAD":
            return response
        queue: asyncio.Queue = asyncio.Queue(maxsize=256)
        self.listeners.add(queue)
        async with self._joined:
            self.joins += 1
            self._joined.notify_all()
        logger.info(f"Spotify: listener {request.remote} joined ({len(self.listeners)} listening)")
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


def _track(status: dict) -> Optional[str]:
    """go-librespot's current track as told to the LLM, None when there is none."""
    track = _current(status)
    return track and f"« {track['name']} » de {_artists(track['artist_names'])}"


def _describe(kind: str, item: dict) -> str:
    """How the LLM is told what is playing."""
    if kind == "artist":
        return item["name"]
    artists = _artists(_artist_names(item))
    if kind == "track":
        return f"« {item['name']} » de {artists}"
    if kind == "album":
        return f"l'album « {item['name']} » de {artists}"
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
    """What play_music found: the context to play, what the LLM is told, what a screen shows."""

    uri: str
    description: str
    metadata: dict


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
    """The search result that fits the request best; among equals, a track the account liked, then Spotify's order.

    Not an exact name first: « Get Lucky » was L.E.J's cover, « Sonate au clair de lune
    Beethoven » a nursery version titled so."""
    plain = fold(query)
    asked, keys, numbers = _words(plain), set(_keys(plain)), _told_numbers(plain)
    # max keeps the first of equals
    return max(items, key=lambda item: (_score(asked, keys, numbers, item), item["uri"] in liked))


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


def match_playlist(playlists: list[dict], query: str) -> Optional[dict]:
    """Find one of the user's playlists by spoken name ("ma playlist du matin" → "Matin")."""
    key = compact(" ".join(w for w in normalize(query).split() if w not in _PLAYLIST_FILLER))
    if not key:
        return None
    names = [(compact(p["name"]), p) for p in playlists]
    exact = next((p for name, p in names if name == key), None)
    return exact or next((p for name, p in names if name and (key in name or name in key)), None)


class SpotifyError(Exception):
    """Spotify or go-librespot failed. The message never quotes a request: its headers carry tokens."""


class SpotifyNotConnected(SpotifyError):
    """go-librespot runs but has no Spotify session (not paired with an account yet)."""


class SpotifySearchError(SpotifyError):
    """Spotify's Web API, on the internet, failed: nothing was played yet."""


# music_control actions → (go-librespot endpoint, what the LLM is told)
CONTROLS = {
    "pause": ("pause", "Musique en pause"),
    "resume": ("resume", "Lecture reprise"),
    "next": ("next", "Titre suivant"),
    "previous": ("prev", "Titre précédent"),
}


class Spotify:
    """go-librespot's API (playback, library) plus Web API search."""

    def __init__(self):
        self.stream = SpotifyStream(on_idle=functools.partial(self.control, "pause"))
        self._session: Optional[aiohttp.ClientSession] = None
        self._app_token = ""
        self._app_token_expiry = 0.0

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

    async def play(self, uri: str, shuffle: bool = False) -> None:
        # Shuffle has to be set before play to start on a random track
        await self._api("POST", "/player/shuffle_context", json={"shuffle_context": shuffle})
        await self._api("POST", "/player/play", json={"uri": uri})

    async def control(self, action: str) -> str:
        """Run a CONTROLS action; returns what the LLM is told."""
        endpoint, label = CONTROLS[action]
        await self._api("POST", f"/player/{endpoint}")
        return label

    async def now_playing(self) -> str:
        status = await self.status()
        track = _track(status)
        if not track:
            return "Rien en cours sur Spotify"
        paused = " (en pause)" if status.get("paused") else ""
        return f"En cours sur Spotify : {track}{paused}"

    async def playing(self) -> Optional[str]:
        """The track playing right now, None when nothing plays (or it is paused)."""
        status = await self.status()
        return None if status.get("paused") else _track(status)

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
            status = await self.status()
            liked = f"spotify:user:{status['username']}:collection"
            return Music(liked, "vos titres likés", screen_metadata("Titres likés"))
        if kind == "playlist":
            library = await self._api("GET", "/library/playlists", params={"limit": 500})
            own = match_playlist(library["items"], query)
            if own:
                metadata = screen_metadata(own["name"], image=own.get("image_url") or "")
                return Music(own["uri"], _describe("playlist", own), metadata)
        if kind not in ("artist", "track", "album", "playlist"):
            kind = "track"
        # Spotify's search, ten results at most, is fuzzy: the request as said, as its catalog writes it,
        # and with its composer first
        variant = search_variant(query)
        searches = dict.fromkeys(filter(None, (query, variant, composer_first(query, variant))))
        await self._search_token()  # once, not by each search
        results = await asyncio.gather(*(self._search(q, kind) for q in searches), return_exceptions=True)
        if isinstance(results[0], Exception):
            raise results[0]  # the others only help: one failing leaves the request's own results
        found = [items for items in results if not isinstance(items, Exception)]
        items = list({item["uri"]: item for item in sum(found, [])}.values())  # in order, once each
        if not items:
            return None
        liked = await self._liked([item["uri"] for item in items]) if kind == "track" else frozenset()
        return _music(kind, best_match(query, items, liked))

    async def _liked(self, uris: list[str]) -> frozenset[str]:
        """Those of these tracks the account liked: « j'ai mis ce morceau précis dans mes titres likés »
        (2026-10-30). go-librespot can tell, not list them."""
        try:
            body = await self._api("GET", "/library/liked", params={"uris": ",".join(uris)}, timeout=_LIKED_TIMEOUT)
        except SpotifyError:
            return frozenset()  # it only breaks a tie: not worth failing the music for
        return frozenset(item["uri"] for item in body.get("items", []) if item.get("liked"))

    async def now_metadata(self) -> Optional[dict]:
        """What a screen shows for the track go-librespot plays, None when there is none."""
        if not (track := _current(await self.status())):
            return None
        cover = track["album_cover_url"] or ""
        return screen_metadata(track["name"], track["artist_names"], track["album_name"], cover)

    async def now_for_app(self) -> Optional[dict]:
        """What the web app shows of the track go-librespot plays, None when there is none."""
        status = await self.status()
        if not (track := _current(status)):
            return None
        return {
            "title": track["name"],
            "artist": _artists(track["artist_names"]),
            "cover": track["album_cover_url"] or None,
            "paused": bool(status.get("paused")),
        }
