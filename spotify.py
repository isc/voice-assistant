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
"""

import asyncio
import logging
import os
import time
from typing import Callable, Optional

import aiohttp
from aiohttp import web

from ha_client import normalize

logger = logging.getLogger(__name__)

LIBRESPOT_URL = os.environ.get("SPOTIFY_LIBRESPOT_URL", "http://127.0.0.1:3678")
PCM_FIFO = os.environ.get("SPOTIFY_PCM_FIFO", "/run/go-librespot/pcm")
WEB_API = "https://api.spotify.com/v1"
# A Spotify developer app (developer.spotify.com), used for search only
CLIENT_ID = os.environ.get("SPOTIFY_CLIENT_ID", "")
CLIENT_SECRET = os.environ.get("SPOTIFY_CLIENT_SECRET", "")

# go-librespot's PCM: 44.1 kHz, stereo, s16le
_FRAME = 4
_BYTES_PER_SECOND = 44100 * _FRAME
_TICK = 0.05
_CHUNK = int(_BYTES_PER_SECOND * _TICK) // _FRAME * _FRAME
# Music read while nobody listens this long gets paused: the speaker was stopped
# some other way, and the account would otherwise stay "playing" on this device
_IDLE_PAUSE = 15.0
# A listener that has received nothing for this long was dropped as too slow
_LISTENER_TIMEOUT = 5.0


class SpotifyStream:
    """Real-time MP3 stream of go-librespot's FIFO, shared by every listener."""

    def __init__(self, fifo_path: str = PCM_FIFO, on_idle: Optional[Callable] = None):
        self.fifo_path = fifo_path
        self.on_idle = on_idle
        self.listeners: set[asyncio.Queue] = set()
        self.joins = 0  # listeners that ever joined: tells a new connection from an old one
        self._fd: Optional[int] = None
        self._pending = b""  # bytes of an incomplete frame, kept for the next read
        self._eof_ticks = 0
        self._encoder: Optional[asyncio.subprocess.Process] = None

    async def run(self) -> None:
        """Pump PCM from the FIFO into the encoder at real-time pace, forever."""
        loop = asyncio.get_running_loop()
        deadline = loop.time()
        unheard = 0.0
        while True:
            deadline += _TICK
            delay = deadline - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)
            elif delay < -1:
                deadline = loop.time()  # the event loop stalled: resync rather than burst
            pcm = self._read()
            if self.listeners:
                unheard = 0.0
                await self._encode(pcm.ljust(_CHUNK, b"\0"))
                continue
            await self._stop_encoder()
            unheard = unheard + _TICK if pcm else 0.0
            if unheard >= _IDLE_PAUSE and self.on_idle:
                logger.info("Spotify: playing to nobody, pausing")
                unheard = 0.0
                asyncio.ensure_future(self.on_idle())

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
                "-f", "s16le", "-ar", "44100", "-ac", "2", "-i", "pipe:0",
                "-c:a", "libmp3lame", "-b:a", "192k", "-f", "mp3", "-flush_packets", "1", "pipe:1",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
            )  # fmt: skip
            asyncio.ensure_future(self._fan_out(self._encoder))
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

    async def _stop_encoder(self) -> None:
        if self._encoder is None:
            return
        encoder, self._encoder = self._encoder, None
        encoder.stdin.close()
        try:
            await asyncio.wait_for(encoder.wait(), 2)
        except TimeoutError:
            encoder.kill()
        logger.info("Spotify: encoder stopped (no listener)")

    async def wait_for_join(self, joins_before: int, timeout: float) -> bool:
        """Wait until a listener connects after `joins_before` was read from self.joins."""
        loop = asyncio.get_running_loop()
        end = loop.time() + timeout
        while self.joins == joins_before and loop.time() < end:
            await asyncio.sleep(0.1)
        return self.joins != joins_before

    async def handle(self, request: web.Request) -> web.StreamResponse:
        """GET /spotify/stream.mp3: the endless stream."""
        response = web.StreamResponse(headers={"Content-Type": "audio/mpeg", "Cache-Control": "no-store"})
        await response.prepare(request)
        if request.method == "HEAD":
            return response
        queue: asyncio.Queue = asyncio.Queue(maxsize=256)
        self.listeners.add(queue)
        self.joins += 1
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


def _artists(item: dict) -> str:
    return ", ".join(a["name"] for a in item.get("artists", [])[:2])


def _describe(kind: str, item: dict) -> str:
    """How the LLM is told what is playing."""
    if kind == "artist":
        return item["name"]
    if kind == "track":
        return f"« {item['name']} » de {_artists(item)}"
    if kind == "album":
        return f"l'album « {item['name']} » de {_artists(item)}"
    return f"la playlist « {item['name']} »"


_PLAYLIST_FILLER = {"playlist", "ma", "mon", "mes", "notre", "nos"}


def match_playlist(playlists: list[dict], query: str) -> Optional[dict]:
    """Find one of the user's playlists by spoken name ("ma playlist du matin" → "Matin")."""
    key = " ".join(w for w in normalize(query).split() if w not in _PLAYLIST_FILLER)
    if not key:
        return None
    names = [(normalize(p["name"]), p) for p in playlists]
    for name, playlist in names:
        if name == key:
            return playlist
    matches = [p for name, p in names if name and (key in name or name in key)]
    return matches[0] if matches else None


class SpotifyNotConnected(Exception):
    """go-librespot runs but has no Spotify session (not paired with an account yet)."""


class SpotifyError(Exception):
    """A Web API failure, described without the request (its headers carry the token)."""


class Spotify:
    """go-librespot's API (playback, library) plus Web API search."""

    def __init__(self, base_url: str = LIBRESPOT_URL):
        self.base_url = base_url
        self.stream = SpotifyStream(on_idle=self.pause)
        self._session: Optional[aiohttp.ClientSession] = None
        self._app_token = ""
        self._app_token_expiry = 0.0

    def _http(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10))
        return self._session

    async def _api(self, method: str, path: str, **kwargs):
        """Call go-librespot. Returns the JSON body, None when there is none."""
        async with self._http().request(method, self.base_url + path, **kwargs) as resp:
            resp.raise_for_status()
            if resp.status == 204:
                raise SpotifyNotConnected
            return await resp.json() if resp.content_type == "application/json" else None

    async def reachable(self) -> bool:
        try:
            await self._api("GET", "/status")
        except SpotifyNotConnected:
            pass
        except (aiohttp.ClientError, TimeoutError):
            return False
        return True

    async def status(self) -> dict:
        return await self._api("GET", "/status")

    async def play(self, uri: str, shuffle: bool = False) -> None:
        # Shuffle has to be set before play to start on a random track
        await self._api("POST", "/player/shuffle_context", json={"shuffle_context": shuffle})
        await self._api("POST", "/player/play", json={"uri": uri})

    async def pause(self) -> None:
        await self._api("POST", "/player/pause")

    async def resume(self) -> None:
        await self._api("POST", "/player/resume")

    async def next(self) -> None:
        await self._api("POST", "/player/next")

    async def previous(self) -> None:
        await self._api("POST", "/player/prev")

    async def _search_token(self) -> str:
        if not CLIENT_ID or not CLIENT_SECRET:
            raise SpotifyError("search needs SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET")
        if time.monotonic() < self._app_token_expiry:
            return self._app_token
        auth = aiohttp.BasicAuth(CLIENT_ID, CLIENT_SECRET)
        data = {"grant_type": "client_credentials"}
        async with self._http().post("https://accounts.spotify.com/api/token", data=data, auth=auth) as resp:
            if resp.status != 200:
                raise SpotifyError(f"app token: HTTP {resp.status}")
            body = await resp.json()
        self._app_token = body["access_token"]
        self._app_token_expiry = time.monotonic() + body["expires_in"] - 60
        return self._app_token

    async def _search(self, query: str, kind: str) -> list[dict]:
        # An app token has no user to take the market from
        params = {"q": query, "type": kind, "limit": 5, "market": "FR"}
        headers = {"Authorization": f"Bearer {await self._search_token()}"}
        async with self._http().get(f"{WEB_API}/search", params=params, headers=headers) as resp:
            if resp.status != 200:
                raise SpotifyError(f"search: HTTP {resp.status}")
            body = await resp.json()
        return [item for item in body[f"{kind}s"]["items"] if item]  # playlist results hold nulls

    async def find(self, query: str, kind: str) -> Optional[tuple[str, str]]:
        """Resolve a spoken request to (context URI, description), or None if nothing matches.

        kind: artist, track, album, playlist (the user's own first, then search) or liked."""
        if kind == "liked":
            status = await self.status()
            return (f"spotify:user:{status['username']}:collection", "vos titres likés")
        if kind == "playlist":
            library = await self._api("GET", "/library/playlists", params={"limit": 500})
            own = match_playlist(library["items"], query)
            if own:
                return own["uri"], _describe("playlist", own)
        if kind not in ("artist", "track", "album", "playlist"):
            kind = "track"
        items = await self._search(query, kind)
        if not items:
            return None
        # Spotify ranks by popularity: prefer an exact name match among the top results
        key = normalize(query)
        item = next((i for i in items if normalize(i["name"]) == key), items[0])
        return item["uri"], _describe(kind, item)
