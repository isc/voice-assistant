#!/usr/bin/env python3
"""
Custom Voice Assistant Server for ESPHome
Replaces Home Assistant for the voice pipeline
"""

import asyncio
import datetime
import locale
import logging
import os
import signal
import socket
import sys
import tempfile
import time
import wave
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urljoin

import aiohttp
from aioesphomeapi import (
    APIClient,
    APIConnectionError,
    MediaPlayerCommand,
    MediaPlayerEntityState,
    MediaPlayerInfo,
    MediaPlayerState,
    VoiceAssistantEventType,
)
from aioesphomeapi.core import TimeoutAPIError
from aioesphomeapi.reconnect_logic import ReconnectLogic
from aiohttp import web

from ha_client import LOCAL_CONFIG, compact, is_whole_house, normalize
from images import find_image
from places import find_place
from recordings import Recordings
from sites import Site, load_sites
from spotify import CONTROLS, Spotify, SpotifyError, SpotifyNotConnected

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

try:
    locale.setlocale(locale.LC_TIME, "fr_FR.UTF-8")
except locale.Error:
    pass  # Fall back to system locale

# Configuration via environment variables; satellite and Home Assistant are per site (sites.py)
LLM_URL = os.environ.get("LLM_URL", "http://localhost:8080/v1/chat/completions")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "")  # empty = llama-server's only model
HTTP_PORT = int(os.environ.get("HTTP_PORT", "8888"))
# Devices re-read from Home Assistant this often (and a Home Assistant that was down, retried)
_HA_REFRESH_SECONDS = 300
# A tool result kept in history this long at most (a weather or calendar answer runs to 1 kB)
_HISTORY_RESULT_CHARS = 300


# French web radio streams. All MP3: the Voice PE does not decode AAC.
# URLs checked 2026-10-10 (audio/mpeg, MP3 frame sync); NRJ group and RMC move theirs.
_STATIONS = [
    # (display name, stream URL, other names it answers to)
    ("France Inter", "https://icecast.radiofrance.fr/franceinter-midfi.mp3", ()),
    ("France Info", "https://icecast.radiofrance.fr/franceinfo-midfi.mp3", ("info",)),
    ("France Culture", "https://icecast.radiofrance.fr/franceculture-midfi.mp3", ("culture",)),
    ("France Musique", "https://icecast.radiofrance.fr/francemusique-midfi.mp3", ("musique",)),
    # France Bleu became "ici" in 2025; fb1071 is ici Paris (ex-France Bleu Paris 107.1)
    ("ici Paris", "https://icecast.radiofrance.fr/fb1071-midfi.mp3", ("ici", "France Bleu", "bleu")),
    ("FIP", "https://icecast.radiofrance.fr/fip-midfi.mp3", ()),
    ("Mouv'", "https://icecast.radiofrance.fr/mouv-midfi.mp3", ()),
    ("RTL", "https://streamer.radio.rtl.fr/rtl-1-48-192", ()),
    ("Europe 1", "https://europe1.lmn.fm/europe1.mp3", ()),
    ("RMC", "https://audio.bfmtv.com/rmcradio_128.mp3", ()),
    ("NRJ", "https://scdn.nrjaudio.fm/audio1/fr/30001/mp3_128.mp3", ()),
    ("Skyrock", "http://icecast.skyrock.net/s/natio_mp3_128k", ()),
    ("Nostalgie", "https://streaming.nrjaudio.fm/oug7girb92oc?origine=fluxradios", ()),
    ("Chérie FM", "https://streaming.nrjaudio.fm/oubqggj25zuw?origine=fluxradios", ("Chérie",)),
    ("Rire et Chansons", "https://streaming.nrjaudio.fm/ou8o8xgk7oiu?origine=fluxradios", ("Rire & Chansons",)),
    ("TSF Jazz", "https://tsfjazz.ice.infomaniak.ch/tsfjazz-high.mp3", ("jazz",)),
    ("Radio Classique", "https://radioclassique.ice.infomaniak.ch/radioclassique-high.mp3", ("classique",)),
    ("RFI", "https://live02.rfi.fr/rfimonde-64.mp3", ()),
]


RADIO_STATIONS: Dict[str, tuple[str, str]] = {
    compact(alias): (display, url) for display, url, aliases in _STATIONS for alias in (display, *aliases)
}


def _resolve_station(name: str) -> Optional[tuple[str, str]]:
    """Fuzzy-resolve a station name to (display_name, url). Returns None if unknown or ambiguous."""
    key = compact(name or "")
    if key in RADIO_STATIONS:
        return RADIO_STATIONS[key]
    # "radio RTL", "la radio France Inter": the word "radio" alone names no station
    key = key.removeprefix("radio")
    if not key:
        return None
    if key in RADIO_STATIONS:
        return RADIO_STATIONS[key]
    # Substring match both ways ("france inter sur FM" → franceinter), only when it points
    # at a single station: "musique classique" matches two and must not pick one at random
    matches = {info for station_key, info in RADIO_STATIONS.items() if station_key in key or key in station_key}
    return matches.pop() if len(matches) == 1 else None


def _speaker_left_on(entities: dict, urls: set[str]) -> Optional[tuple[str, str]]:
    """The HA media player left playing one of urls, with that URL; None if none is."""
    return next(
        (
            (entity_id, info["attributes"]["media_content_id"])
            for entity_id, info in entities.items()
            if info["state"] in ("playing", "paused") and info["attributes"].get("media_content_id") in urls
        ),
        None,
    )


def _no_speaker(room: Optional[str]) -> str:
    return f"Aucune enceinte trouvée pour {room}" if room else "Aucune enceinte disponible"


def _format_duration(total_seconds: int) -> str:
    """Format seconds into a French human-readable duration."""
    if total_seconds >= 3600:
        h = total_seconds // 3600
        m = (total_seconds % 3600) // 60
        if m:
            return f"{h}h{m:02d}"
        return f"{h} heure{'s' if h > 1 else ''}"
    if total_seconds >= 60:
        m = total_seconds // 60
        s = total_seconds % 60
        if s:
            return f"{m} minute{'s' if m > 1 else ''} {s}s"
        return f"{m} minute{'s' if m > 1 else ''}"
    return f"{total_seconds} seconde{'s' if total_seconds > 1 else ''}"


def _describe_timer(timer) -> str:
    if timer.is_alarm and timer.target_time:
        return f"l'alarme de {timer.target_time}"
    if timer.name:
        return f'le timer "{timer.name}"'
    return f"le timer de {_format_duration(timer.total_seconds)}"


def _format_family_for_prompt(family: list) -> str:
    """Format family member list with ages for injection into system prompt."""
    today = datetime.date.today()
    parts = []
    role_fr = {"parent": "parent", "child": "enfant"}
    for m in family:
        role = role_fr.get(m["role"], m["role"])
        birth_date = m.get("birth_date")
        if birth_date:
            bday = datetime.date.fromisoformat(birth_date)
            age = today.year - bday.year - ((today.month, today.day) < (bday.month, bday.day))
            parts.append(f"{m['name']} ({role}, {age} ans)")
        else:
            parts.append(f"{m['name']} ({role})")
    return "Membres de la famille: " + ", ".join(parts)


# The site the running task serves: set where a satellite's request or a web request comes in.
# asyncio copies it into every task created from there (service calls, TTS generation, timers).
_current_site: ContextVar[Site] = ContextVar("current_site")


@dataclass
class _Request:
    """What one request gathers while it runs. Shared by reference with the tasks it starts (tool
    calls run under asyncio.gather), unlike a ContextVar set in them; a concurrent request has its own."""

    tool_calls: list = field(default_factory=list)
    end_conversation: bool = False  # the end_conversation tool: no follow-up, history cleared
    no_followup: bool = False  # a media command: no follow-up, history kept (AD-035)


_current_request: ContextVar[_Request] = ContextVar("current_request")


def _request_attribute(name: str) -> property:
    def request() -> _Request:
        if (current := _current_request.get(None)) is None:
            current = _Request()
            _current_request.set(current)
        return current

    return property(lambda self: getattr(request(), name), lambda self, value: setattr(request(), name, value))


def _site_attribute(name: str) -> property:
    """An attribute of the site this task serves."""
    return property(lambda self: getattr(self.site, name), lambda self, value: setattr(self.site, name, value))


class VoiceAssistantServer:
    """
    Custom voice assistant server for ESPHome
    Handles the STT -> LLM -> TTS pipeline
    """

    # Conversation history (per site) is forgotten after this much silence. Long enough that a typed
    # follow-up ("et dans la salle de jeux ?") still resolves, short enough that a wake word much
    # later starts clean. Voice conversations usually end sooner anyway, via the end_conversation tool.
    CONVERSATION_TIMEOUT = 600  # seconds

    # Per home, read from the active site (sites.py)
    ha_client = _site_attribute("ha_client")
    timer_manager = _site_attribute("timer_manager")
    conversation = _site_attribute("conversation")
    _last_media_target = _site_attribute("last_media_target")
    # Per request (_Request)
    _last_tool_calls = _request_attribute("tool_calls")
    _end_conversation = _request_attribute("end_conversation")
    _no_followup = _request_attribute("no_followup")

    def __init__(self):
        self.llm_config = (LLM_API_KEY, LLM_MODEL, LLM_URL)  # for /health
        self.sites: list[Site] = load_sites()
        # Whose satellite holds the recording, VAD and pipeline state below: one conversation at a
        # time. Also the site served by a task that names none, such as an untagged web request.
        self._voice_site = self.sites[0]
        self._pipeline_running = False
        # Site name -> its satellite, once connected
        self.devices: Dict[str, APIClient] = {}
        self._reconnect_logics: list[ReconnectLogic] = []

        # Voice pipeline state
        self.audio_buffer = bytearray()
        self.is_recording = False
        self.recording_task = None
        self.last_audio_time = 0

        # Sub-engines (initialized in start())
        self.tts = None  # tts.KokoroTTS
        self.stt = None  # stt.ParakeetSTT

        # Google Calendar client
        self.calendar_client = None

        # Continuous conversation
        self.is_followup = False  # True when pipeline was started without wake word

        # Media playback state. Media targets are ("esp", site name) or ("ha", entity_id),
        # the last one per site for follow-ups (_last_media_target).
        # Site name -> its satellite's media_player entity key (discovered at connect time)
        self.device_media_keys: Dict[str, int] = {}
        # Site name -> what its satellite's own player does (announcements left out)
        self.device_media_states: Dict[str, MediaPlayerState] = {}
        # Spotify through go-librespot (None when it is not running here)
        self.spotify: Optional[Spotify] = None
        self._spotify_task: Optional[asyncio.Task] = None
        # The speaker playing the Spotify stream, with its site: one stream, one speaker at a time
        self._spotify_at: Optional[tuple[Site, tuple[str, str]]] = None

        # Exchange log for web UI (persisted to disk)
        from web_ui import ExchangeLog

        self.exchange_log = ExchangeLog(Path(__file__).parent / "exchanges.json")
        self.recordings = Recordings()
        # Cap on concurrent tool calls: fast enough for a whole-house command while
        # staying gentle on the single HomeKit bridge relaying every Velux.
        self._tool_semaphore = asyncio.Semaphore(4)
        # Actuation service calls are fired here and drain in the background — held
        # so the tasks aren't garbage-collected mid-flight, discarded on completion.
        self._pending_service_calls: set = set()
        self._background_tasks: set = set()  # held until done, not awaited at shutdown

        # HTTP server for serving TTS audio files
        self.http_server = None
        self.tts_dir = Path(tempfile.gettempdir()) / "voice_assistant_tts"
        self.tts_dir.mkdir(exist_ok=True)
        logger.info(f"TTS directory: {self.tts_dir}")

    @property
    def site(self) -> Site:
        """The site this task serves; for a task that names none, where someone last spoke."""
        return _current_site.get(self._voice_site)

    @site.setter
    def site(self, site: Site) -> None:
        _current_site.set(site)

    @property
    def device(self) -> Optional[APIClient]:
        """This site's satellite, when connected."""
        return self.devices.get(self.site.name)

    def claim_voice(self, site: Site) -> bool:
        """A site's satellite starts a request; refused while another site's conversation holds the voice state."""
        if site is not self._voice_site and (self.is_recording or self._pipeline_running):
            logger.warning(f"Site {site.label} refused: a conversation is running at {self._voice_site.label}")
            return False
        self._voice_site = self.site = site
        return True

    @staticmethod
    def new_request() -> None:
        """A request starts in this task: its own tool calls and follow-up flags."""
        _current_request.set(_Request())

    def site_named(self, name: str) -> Optional[Site]:
        return next((site for site in self.sites if site.name == name.strip().lower()), None)

    # === INIT ===

    async def start(self):
        """Start the server and connect to devices"""
        logger.info("Voice Assistant Server started")
        for site in self.sites:
            logger.info(
                f"Site {site.label}: ESP {site.esp_host or '(none)'}:{site.esp_port} | HA {site.ha_url or '(none)'}"
            )
        logger.info(f"LLM: {LLM_MODEL or '(server default)'} at {LLM_URL} | HTTP: {HTTP_PORT}")

        from stt import ParakeetSTT
        from tts import KokoroTTS

        self.stt = ParakeetSTT()
        await self.stt.init()

        self.tts = KokoroTTS()
        await self.tts.init()

        await self.init_ha_clients()
        self._ha_refresh_task = asyncio.create_task(self._keep_ha_fresh())
        await self.init_calendar_client()
        await self.init_spotify()
        await self.start_http_server()
        self._recover_media_target()
        for site in self.sites:
            if site.esp_host:
                await self.connect_to_device(site)
            else:
                logger.info(f"Site {site.label}: no satellite configured")

    async def init_ha_clients(self):
        """Connect to every site's Home Assistant, all at once, and discover entities."""
        await asyncio.gather(*(self._connect_ha(site) for site in self.sites))
        # Local time is the home's: the server may run elsewhere, or in UTC (charras). Homes in
        # several time zones would need it per site; both are in Europe/Paris.
        time_zone = next((site.ha_client.home.get("time_zone") for site in self.sites if site.ha_client), None)
        if time_zone and not os.environ.get("TZ"):
            os.environ["TZ"] = time_zone
            time.tzset()
            logger.info(f"Time zone: {time_zone} (from Home Assistant)")

    async def _connect_ha(self, site: Site):
        from ha_client import HAClient

        if not site.ha_token:
            logger.info(f"Site {site.label}: no HA token configured, device control disabled")
            return
        client = HAClient(site.ha_url, site.ha_token)
        if not await client.connect():
            await client.close()
            # A reload during an HA blip keeps the client that worked
            state = "keeping the previous connection" if site.ha_client else "device control disabled"
            logger.warning(f"Site {site.label}: Home Assistant not reachable, {state}")
            return
        logger.info(f"Site {site.label}: Home Assistant, {len(client.entities)} entities discovered")
        # Swapped in once ready, so that a reload (SIGHUP) leaves no gap
        previous, site.ha_client = site.ha_client, client
        if previous:
            await previous.close()

    async def _keep_ha_fresh(self):
        """Every few minutes, re-read each site's devices, or connect a Home Assistant that was down.

        Without it, devices added in HA would stay unknown until a restart, and a Home Assistant
        down at startup (booting slower after a power cut) would never be used.
        """
        while True:
            await asyncio.sleep(_HA_REFRESH_SECONDS)
            for site in self.sites:
                try:
                    if site.ha_client:
                        await site.ha_client.refresh_entities()
                    elif site.ha_token:
                        await self._connect_ha(site)
                except Exception:
                    logger.exception(f"Site {site.label}: Home Assistant refresh failed")

    async def init_calendar_client(self):
        """Connect to Google Calendar API."""
        from calendar_client import CalendarClient

        project_dir = Path(__file__).parent
        credentials_path = project_dir / os.environ.get("CALENDAR_CREDENTIALS", "client_secret.json")
        token_path = project_dir / os.environ.get("CALENDAR_TOKEN", "token.json")

        if not token_path.exists():
            logger.info("No calendar token found (run setup_calendar.py), calendar disabled")
            return

        self.calendar_client = CalendarClient(credentials_path, token_path)
        if await self.calendar_client.connect():
            logger.info("Google Calendar connected")
        else:
            logger.warning("Google Calendar not available, calendar disabled")
            self.calendar_client = None

    async def init_spotify(self):
        """Enable music when a go-librespot player answers on this host."""
        spotify = Spotify()
        if not await spotify.reachable():
            logger.info("No go-librespot player reachable, Spotify disabled")
            await spotify.close()
            return
        self.spotify = spotify
        spotify.stream.on_silence = self._stop_silent_spotify
        self._spotify_task = asyncio.create_task(spotify.stream.run())
        logger.info("Spotify enabled (go-librespot)")

    @staticmethod
    def _spotify_url(site: Site) -> str:
        # Beside /tts/, so that a relay serving the site's TTS_BASE_URL serves it too
        return urljoin(site.http_base_url, "../spotify/stream.mp3")

    @property
    def _spotify_here(self) -> Optional[tuple[str, str]]:
        """The speaker playing the Spotify stream, if it is at this site."""
        return self._spotify_at[1] if self._spotify_at and self._spotify_at[0] is self.site else None

    def _recover_media_target(self) -> None:
        """Pick up, per site, the HA speaker a previous run left on a radio or on the Spotify stream.

        A restart forgot it, so media commands without a room went to the voice device
        while the music went on, on the Sonos.
        """
        radios = {url for _, url in RADIO_STATIONS.values()}
        for site in self.sites:
            if not site.ha_client:
                continue
            stream = self._spotify_url(site) if self.spotify else None
            found = _speaker_left_on(site.ha_client.entities, radios | {stream} - {None})
            if not found:
                continue
            speaker, url = found
            site.last_media_target = ("ha", speaker)
            if url == stream:
                self._spotify_at = (site, ("ha", speaker))
            logger.info(f"Site {site.label}: {speaker} left on {url}, media commands without a room go there")

    async def start_http_server(self):
        """Start HTTP server to host TTS audio files and web UI"""
        from web_ui import setup_routes

        app = web.Application()
        setup_routes(app, self)
        # No listing: the LAN may fetch a reply by its URL, not browse every one
        app.router.add_static("/tts/", self.tts_dir)
        if self.spotify:
            app.router.add_get("/spotify/stream.mp3", self.spotify.stream.handle)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", HTTP_PORT)
        await site.start()

        # Get local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
        finally:
            s.close()

        logger.info(f"HTTP TTS server started on http://{local_ip}:{HTTP_PORT}/tts/")
        logger.info(
            f'Test endpoint: curl -X POST http://{local_ip}:{HTTP_PORT}/test -d \'{{"text": "allume la lumière"}}\''
        )
        for home in self.sites:
            # A site whose satellite cannot reach this host's LAN address names a relay
            home.http_base_url = home.tts_base_url or f"http://{local_ip}:{HTTP_PORT}/tts/"
            logger.info(f"Site {home.label}: TTS URL announced to the ESP: {home.http_base_url}")

    # === ESP CONNECTION ===

    async def connect_to_device(self, site: Site):
        """Connect to a site's satellite with automatic reconnection."""
        host = site.esp_host
        logger.info(f"Site {site.label}: connecting to {host}...")

        if site.esp_noise_psk:
            logger.info("Auth: Noise encryption")
            api = APIClient(
                host,
                site.esp_port,
                None,
                noise_psk=site.esp_noise_psk,
                client_info="voice-assistant",
            )
        else:
            api = APIClient(
                host,
                site.esp_port,
                site.esp_password,
                client_info="voice-assistant",
            )

        async def on_connect() -> None:
            device_info = await api.device_info()
            logger.info(f"Site {site.label}: connected to {host}: {device_info.name} (v{device_info.esphome_version})")

            if device_info.voice_assistant_feature_flags:
                logger.info(f"Voice assistant supported (flags: {device_info.voice_assistant_feature_flags})")
            else:
                logger.warning("Voice assistant may not be supported")

            self.devices[site.name] = api
            await self._discover_device_entities(api, site)
            api.subscribe_states(partial(self._on_satellite_state, site))
            await self.setup_voice_assistant(api, site)

        async def on_disconnect(expected_disconnect: bool) -> None:
            reason = "expected" if expected_disconnect else "unexpected"
            logger.warning(f"Site {site.label}: ESP {host} disconnected ({reason}), will reconnect automatically")
            self.devices.pop(site.name, None)
            # A recording cut short would hold the voice state, refusing the other site for 30 s
            if site is self._voice_site and self.is_recording:
                self.is_recording = False

        # No name: ReconnectLogic derives it from the host, and matches mDNS announcements with it
        # (an x.local satellite reconnects as soon as it appears; an IP gets no listener)
        reconnect_logic = ReconnectLogic(client=api, on_connect=on_connect, on_disconnect=on_disconnect)
        self._reconnect_logics.append(reconnect_logic)
        await reconnect_logic.start()

    def _on_satellite_state(self, site: Site, state) -> None:
        if isinstance(state, MediaPlayerEntityState) and state.key == self.device_media_keys.get(site.name):
            # An announcement plays over the media, which resumes after it
            if state.state != MediaPlayerState.ANNOUNCING:
                self.device_media_states[site.name] = state.state

    def _satellite_plays(self, site: Optional[Site] = None) -> bool:
        """Whether this site's (or `site`'s) satellite plays music or a radio itself."""
        return self.device_media_states.get((site or self.site).name) == MediaPlayerState.PLAYING

    async def _discover_device_entities(self, api: APIClient, site: Site):
        """List the satellite's entities and remember its media_player key (for direct radio playback)."""
        try:
            entities, _ = await api.list_entities_services()
        except Exception as e:
            logger.warning(f"Site {site.label}: could not list ESP entities: {e}")
            return
        media_players = [e for e in entities if isinstance(e, MediaPlayerInfo)]
        if not media_players:
            logger.info(f"Site {site.label}: ESP has no media_player entity — radio playback via HA only")
            return
        mp = media_players[0]
        self.device_media_keys[site.name] = mp.key
        logger.info(f"Site {site.label}: ESP media_player key={mp.key} name={mp.name!r}")

    async def setup_voice_assistant(self, api: APIClient, site: Site):
        """Set up the voice assistant subscription; the handlers know which site they serve."""
        logger.info(f"Setting up voice assistant for {site.label}")

        try:
            api.subscribe_voice_assistant(
                handle_start=partial(self.handle_voice_assistant_start, site),
                handle_stop=partial(self.handle_voice_assistant_stop, site),
                handle_audio=partial(self.handle_voice_assistant_audio, site),
            )
            logger.info("Voice assistant subscription successful")

        except Exception as e:
            logger.error(f"Voice assistant setup error: {e}")

    # === VOICE ASSISTANT EVENT HANDLERS ===

    async def handle_voice_assistant_start(
        self,
        site: Site,
        conversation_id: str,
        flags: int,
        audio_settings: Any,
        wake_word_phrase: str | None,
    ) -> int | None:
        """Called when a site's voice assistant starts"""
        logger.info(f"Voice Assistant START ({site.label})")
        logger.info(f"   Conversation ID: {conversation_id}")
        logger.info(f"   Flags: {flags}")
        logger.info(f"   Wake word: {wake_word_phrase}")
        logger.info(f"   Audio settings: {audio_settings}")

        if not self.claim_voice(site):
            return None  # the ESP takes it as an error

        self.is_followup = wake_word_phrase is None
        # Only overwrite conversation_id for new conversations (wake word detected).
        # Follow-ups keep the server-assigned ID so exchanges group correctly.
        if not self.is_followup:
            self.conversation.id = None

        # Cancel any previous recording timeout
        if self.recording_task and not self.recording_task.done():
            self.recording_task.cancel()

        self.audio_buffer = bytearray()
        self.is_recording = True
        self.vad_speech_frames = 0
        self.vad_silence_frames = 0
        self.vad_has_speech = False
        # In follow-up mode, skip initial audio to avoid picking up TTS echo
        self.skip_audio_until = time.time() + 0.5 if self.is_followup else 0

        if api := self.device:
            try:
                api.send_voice_assistant_event(VoiceAssistantEventType.VOICE_ASSISTANT_STT_START, {})
                logger.info(f"STT_START sent - {'follow-up' if self.is_followup else 'wake word'} mode")
                self.recording_task = asyncio.create_task(self.monitor_recording_timeout(api))
            except Exception as e:
                logger.error(f"Error sending STT_START: {e}")

        return 0

    async def monitor_recording_timeout(self, api: APIClient):
        """Safety timeout. In follow-up mode, wait 5s for speech to start,
        then extend to 30s once speech is detected."""
        # End early if nobody speaks: 5 s in a follow-up, 8 s after a wake word (a false one kept
        # the mic open 30 s, recording the room)
        wait = 5.0 if self.is_followup else 8.0
        await asyncio.sleep(wait)
        if not self.is_recording:
            return
        if not self.vad_has_speech:
            logger.info("No speech - stopping")
            await self.stop_recording(api, "no speech")
            return
        # Speech detected: the normal limit for the rest, 30 s in all
        await asyncio.sleep(30.0 - wait)

        if self.is_recording:
            logger.warning("SAFETY TIMEOUT reached (30s)")
            await self.stop_recording(api, "timeout")

    async def stop_recording(self, api: APIClient, reason: str):
        """Stop recording and process the audio"""
        if not self.is_recording:
            return

        self.is_recording = False
        audio_data = bytes(self.audio_buffer)
        logger.info(f"Recording stopped ({reason}): {len(audio_data)} bytes")

        try:
            # No "thinking" if nobody spoke: the ESP would ignore the RUN_END that follows (AD-036)
            if self.vad_has_speech:
                api.send_voice_assistant_event(VoiceAssistantEventType.VOICE_ASSISTANT_STT_VAD_END, {})
                logger.info("STT_VAD_END sent - LEDs orange (thinking)")

            if audio_data:
                await self.process_voice_pipeline(api, audio_data)
            else:
                # Nothing to process: without RUN_END the ESP would stay listening
                api.send_voice_assistant_event(VoiceAssistantEventType.VOICE_ASSISTANT_RUN_END, {})

        except Exception:
            logger.exception("Error stopping recording")

    async def handle_voice_assistant_stop(self, site: Site, abort: bool):
        """Called when a site's voice assistant stops"""
        logger.info(f"Voice Assistant STOP ({site.label}, abort={abort})")
        if site is not self._voice_site:
            return
        self.site = site

        # The device stopped the run itself (start=False): the user closed the mic, so no reply
        if abort:
            if self.is_recording:
                self.is_recording = False
                if self.recording_task and self.recording_task is not asyncio.current_task():
                    self.recording_task.cancel()
                logger.info("Run stopped by the device - request dropped")
            return

        if self.is_recording and len(self.audio_buffer) > 0:
            logger.info(f"Complete audio received: {len(self.audio_buffer)} bytes")
            self.is_recording = False

            if api := self.device:
                await self.process_voice_pipeline(api, bytes(self.audio_buffer))
        else:
            logger.info("No audio to process")

    async def _announce(self, api: APIClient, audio_url: str, follow_up: bool) -> None:
        """Play a reply on the satellite; with follow_up, it then listens again without wake word.

        AnnounceFinished comes at the end of playback, the follow-up START 30-50 ms later
        (journal, 2026-10-10), so the wait is the audio's length plus a margin.
        """
        timeout = self._wav_seconds(audio_url) + 5
        try:
            logger.info(f"Playing TTS (follow-up: {follow_up}): {audio_url}")
            await api.send_voice_assistant_announcement_await_response(
                media_id=audio_url, timeout=timeout, text="", start_conversation=follow_up
            )
        except TimeoutAPIError:  # aioesphomeapi's own: not a TimeoutError
            logger.warning(f"Announcement not confirmed within {timeout:.0f} s: {audio_url}")
        except APIConnectionError as e:
            logger.warning(f"Announcement failed: {e}")

    def _wav_seconds(self, audio_url: str) -> float:
        try:
            with wave.open(str(self.tts_dir / audio_url.rsplit("/", 1)[-1])) as wav:
                return wav.getnframes() / wav.getframerate()
        except (OSError, wave.Error):
            return 30.0

    async def handle_voice_assistant_audio(self, site: Site, audio_bytes: bytes, _data2: bytes | None = None):
        """Handle audio received from a site's ESP with VAD detection.

        aioesphomeapi 45+ passes a second buffer (data2); a single mic stream is enough here."""
        if self.is_recording and site is self._voice_site:
            self.site = site  # the pipeline may run from this task, once the VAD hears the end
            # Skip early audio in follow-up mode to avoid TTS echo
            if self.skip_audio_until and time.time() < self.skip_audio_until:
                return

            self.audio_buffer.extend(audio_bytes)
            self.last_audio_time = time.time()

            await self.analyze_audio_vad(audio_bytes)

            if len(self.audio_buffer) % 10240 == 0 or len(self.audio_buffer) < 10240:
                logger.debug(f"Audio: {len(self.audio_buffer)} bytes")
        else:
            logger.debug("Audio received after recording stopped (normal)")

    async def analyze_audio_vad(self, audio_bytes: bytes):
        """Analyze audio with Silero VAD to detect end of speech"""
        import array

        from silero_vad_lite import SileroVAD

        if not hasattr(self, "vad"):
            self.vad = SileroVAD(16000)

        frame_samples = 512  # 32ms at 16kHz
        frame_size = frame_samples * 2  # 1024 bytes as int16

        for i in range(0, len(audio_bytes), frame_size):
            frame = audio_bytes[i : i + frame_size]
            if len(frame) != frame_size:
                continue

            try:
                int16_samples = array.array("h")
                int16_samples.frombytes(frame)
                float_samples = array.array("f", [s / 32768.0 for s in int16_samples])

                speech_prob = self.vad.process(float_samples)
                is_speech = speech_prob > 0.5

                if is_speech:
                    self.vad_speech_frames += 1
                    self.vad_silence_frames = 0
                    if not self.vad_has_speech:
                        self.vad_has_speech = True
                        logger.info("Speech detected")
                else:
                    if self.vad_has_speech:
                        self.vad_silence_frames += 1

                if self.vad_has_speech and self.vad_silence_frames >= 31:
                    logger.info(f"Silence detected after speech ({self.vad_silence_frames} frames)")
                    if self.is_recording and (api := self.device):
                        await self.stop_recording(api, "VAD silence")
                    return

            except Exception as e:
                logger.debug(f"VAD error: {e}")
                continue

    # === MAIN PIPELINE ===

    async def process_voice_pipeline(self, api: APIClient, audio_bytes: bytes):
        """Main pipeline: Audio -> STT -> LLM -> TTS"""
        logger.info(f"Starting full voice pipeline ({self.site.label})")
        self.new_request()
        self._pipeline_running = True
        timings = {}
        # What became of the request, for the exchange log and beside its recording
        entry = {
            "site": self.site.name,
            "transcript": "",
            "response": None,
            "timings": timings,
            "audio": self.recordings.save(audio_bytes),
            "wake_word": not self.is_followup,
            "vad_speech": self.vad_has_speech,
        }

        try:
            # 1. STT, skipped if the VAD heard no speech (AD-036)
            t0 = time.time()
            if self.vad_has_speech:
                transcript = await self.stt.transcribe(audio_bytes)
            else:
                logger.info("STT skipped: the VAD heard no speech")
                transcript = ""
            timings["stt"] = round(time.time() - t0, 2)
            entry["transcript"] = transcript
            if not transcript:
                entry["response"] = "[Rien compris]"
                if self.is_followup:
                    # Normal: no speech during follow-up, end conversation quietly
                    logger.info("No speech during follow-up - ending conversation")
                    api.send_voice_assistant_event(VoiceAssistantEventType.VOICE_ASSISTANT_RUN_END, {})
                else:
                    await self.send_error_to_device(api, "STT failed")
                return

            api.send_voice_assistant_event(
                VoiceAssistantEventType.VOICE_ASSISTANT_STT_END,
                {"text": transcript},
            )
            logger.info(f'STT_END sent: "{transcript}"')

            # 2. LLM
            t0 = time.time()
            response_text = await self.process_with_llm(api, transcript)
            timings["llm"] = round(time.time() - t0, 2)
            # Capture conversation_id now — TTS triggers ESP follow-up which overwrites it
            entry.update(
                response=response_text or "[Erreur LLM: pas de réponse]",
                tool_calls=self._last_tool_calls,
                conversation_id=self.conversation.id,
            )
            if not response_text:
                await self.send_error_to_device(api, "LLM processing failed")
                return

            # 3. TTS
            t0 = time.time()
            try:
                tts_timings = await self.text_to_speech(api, response_text)
                timings["tts_gen"] = tts_timings["gen"]
                timings["tts_play"] = tts_timings["play"]
                timings["tts"] = round(time.time() - t0, 2)
            except Exception as e:
                logger.error(f"TTS failed, but exchange will still be logged: {e}")
                await self.send_error_to_device(api, f"TTS error: {e}")
            timings["total"] = round(timings["stt"] + timings["llm"] + timings.get("tts_gen", 0), 2)
            logger.info(
                f"Pipeline: STT={timings['stt']}s LLM={timings['llm']}s "
                f"TTS={timings.get('tts_gen', '?')}s+{timings.get('tts_play', '?')}s "
                f"total={timings['total']}s"
            )

        except Exception as e:
            logger.exception("Pipeline error")
            entry["error"] = str(e)
            await self.send_error_to_device(api, f"Pipeline error: {e}")
        finally:
            self._pipeline_running = False
            self.recordings.describe(entry["audio"], entry)
            # Failures too, with their audio: they are the ones worth hearing. Not a
            # follow-up nobody spoke in, which is how every conversation ends.
            if entry["wake_word"] or entry["vad_speech"] or "error" in entry:
                self.exchange_log.add(
                    "voice",
                    entry["transcript"],
                    entry["transcript"],
                    entry["response"] or f"[Erreur: {entry.get('error', 'inconnue')}]",
                    timings=timings,
                    tool_calls=entry.get("tool_calls"),
                    conversation_id=entry.get("conversation_id"),
                    audio=entry["audio"],
                    site=entry["site"],
                )

    # === FUNCTION CALLING ===

    async def execute_function(self, function_name: str, arguments: dict) -> str:
        """Execute a function call from the LLM."""
        if function_name == "end_conversation":
            self._end_conversation = True
            return "Conversation terminée"

        # Timers — standalone, doesn't need HA
        if function_name == "set_timer":
            # A string from the text fallback ("5" * 60 is a 60-digit number), or nothing at all
            try:
                total_seconds = int(float(arguments.get("duration_minutes") or 0) * 60)
            except (TypeError, ValueError):
                total_seconds = 0
            if total_seconds <= 0:
                return "Durée du timer manquante ou invalide"
            label = arguments.get("label")
            timer = self.timer_manager.start_timer(total_seconds, label, partial(self._handle_timer_event, self.site))
            if label:
                return f"Timer {label} de {_format_duration(total_seconds)} lancé"
            return f"Timer de {_format_duration(total_seconds)} lancé"

        if function_name == "set_alarm":
            time_str = arguments.get("time", "")
            label = arguments.get("label")
            timer = self.timer_manager.start_alarm(time_str, label, partial(self._handle_timer_event, self.site))
            return f"Alarme programmée pour {time_str}"

        if function_name == "cancel_timer":
            label = arguments.get("label")
            timer = self.timer_manager.cancel_timer(name=label)
            if timer:
                from aioesphomeapi import VoiceAssistantTimerEventType

                cancelled = VoiceAssistantTimerEventType.VOICE_ASSISTANT_TIMER_CANCELLED
                await self._handle_timer_event(self.site, cancelled, timer)
                return f"Annulé: {_describe_timer(timer)}"
            if running := self.timer_manager.get_timers():
                names = ", ".join(map(_describe_timer, running))
                return f"Plusieurs timers en cours ({names}) : lequel annuler ?"
            return "Aucun timer actif"

        # Weather — standalone, doesn't need HA
        if function_name == "get_weather":
            from weather import get_weather

            return await get_weather(arguments.get("location", ""), self.ha_client and self.ha_client.home)

        if function_name == "find_place":
            return await find_place(arguments.get("query", ""), self.ha_client and self.ha_client.home)

        # Calendar — standalone, no HA dependency
        if function_name == "query_calendar":
            if not self.calendar_client:
                return "Agenda non disponible"
            return await self.calendar_client.query_events(
                start_date=arguments.get("start_date", ""),
                end_date=arguments.get("end_date"),
                search=arguments.get("search"),
            )

        if function_name == "create_event":
            if not self.calendar_client:
                return "Agenda non disponible"
            return await self.calendar_client.create_event(
                title=arguments.get("title", ""),
                start_datetime=arguments.get("start_datetime", ""),
                duration_minutes=arguments.get("duration_minutes", 60),
            )

        # Media player tools (radio + volume) — default to the current ESP speaker,
        # fall back to HA media_player entities when a room is named.
        if function_name in ("play_music", "music_control"):
            return await self._execute_music_function(function_name, arguments)
        if function_name in ("play_radio", "stop_media", "set_volume", "change_volume"):
            return await self._execute_media_function(function_name, arguments)

        if not self.ha_client:
            return "Home Assistant non disponible"

        if function_name == "show_image":
            return await self._show_image(arguments.get("query", ""))

        # Sensors are searched by keyword, not resolved like actionable entities
        if function_name == "get_sensor":
            return await self.ha_client.search_sensors(arguments.get("query", ""))

        entity_name = arguments.get("entity", "")
        room = arguments.get("room")

        # Every action tool is named after its HA service; this is also the list of them
        domain_map = {
            "turn_on": ["light", "switch", "media_player"],
            "turn_off": ["light", "switch", "media_player"],
            "open_cover": ["cover"],
            "close_cover": ["cover"],
            "set_cover_position": ["cover"],
            "set_temperature": ["climate"],
            "lock": ["lock"],
            "get_state": None,
        }
        if function_name not in domain_map:
            logger.warning(f"Unknown function: {function_name}")
            return f"Fonction {function_name} inconnue"
        domains = domain_map[function_name]

        entity_ids = self.ha_client.resolve_all_entities(entity_name, room=room, domain_hints=domains)
        if not entity_ids:
            return f"Appareil {entity_name} non trouvé"

        if function_name == "get_state":
            # Every device resolved: « les volets sont fermés ? » names several, and only the first was read
            states = await asyncio.gather(*(self.ha_client.get_entity_state(eid) for eid in entity_ids))
            answers = [self.ha_client.format_state_for_speech(eid, st) for eid, st in zip(entity_ids, states) if st]
            return ". ".join(answers) if answers else f"Impossible de lire l'état de {entity_name}"

        if not (reachable := await self.ha_client.reachable(entity_ids)):
            return self.ha_client.offline_reply(entity_ids)
        entity_ids = reachable

        extra = {}
        if function_name == "set_temperature" and "temperature" in arguments:
            extra["temperature"] = arguments["temperature"]
        if function_name == "turn_on" and "brightness" in arguments:
            extra["brightness"] = round(arguments["brightness"] * 255 / 100)
        if function_name == "set_cover_position" and "position" in arguments:
            extra["position"] = arguments["position"]

        calls = [self.ha_client.call_service(eid.split(".")[0], function_name, eid, **extra) for eid in entity_ids]
        if function_name == "lock":
            # Awaited, unlike the rest: whoever leaves on « porte verrouillée » must not
            # be told so when HA could not reach the lock
            return ". ".join(await asyncio.gather(*calls))

        # Fire and forget. An HA service call blocks until the HomeKit bridge
        # acknowledges (~0.5-1.8s/Velux, serialised by the bridge → ~10s for eight),
        # but a 200 only means HA dispatched the command, not that the shutter
        # finished moving — the spoken reply ("en cours d'ouverture") is optimistic
        # either way. So don't make the user wait for acks: schedule the calls and
        # let them drain while the pipeline moves on to the reply. call_service
        # swallows and logs its own errors, so a failed background call never raises.
        for call in calls:
            task = asyncio.ensure_future(call)
            self._pending_service_calls.add(task)
            task.add_done_callback(self._pending_service_calls.discard)

        if len(entity_ids) > 1:
            if is_whole_house(room):
                action_map = {
                    "close_cover": "Tous les volets en cours de fermeture",
                    "open_cover": "Tous les volets en cours d'ouverture",
                    "set_cover_position": f"Tous les volets réglés à {arguments.get('position', '?')}%",
                    "turn_on": "Toutes les lumières allumées",
                    "turn_off": "Toutes les lumières éteintes",
                }
            else:
                room_label = room or "la maison"
                action_map = {
                    "close_cover": f"Volets {room_label} en cours de fermeture",
                    "open_cover": f"Volets {room_label} en cours d'ouverture",
                    "set_cover_position": f"Volets {room_label} réglés à {arguments.get('position', '?')}%",
                    "turn_on": f"Lumières {room_label} allumées",
                    "turn_off": f"Lumières {room_label} éteintes",
                }
            return action_map.get(function_name, "C'est fait")

        return self.ha_client._build_response(
            entity_ids[0].split(".")[0],
            function_name,
            self.ha_client.entities.get(entity_ids[0], {}).get("friendly_name", entity_ids[0]),
            extra,
        )

    # === MEDIA PLAYER ===

    def _resolve_media_target(self, room: Optional[str]) -> Optional[tuple[str, str]]:
        """Resolve a media target to ("esp", host) or ("ha", entity_id).

        Strategy:
        - If room is provided and matches an HA media_player → use HA.
        - Otherwise default to the current ESP (direct playback, no HA needed).
        - With no room, follow-ups fall back to the last-used target.
        """
        # Explicit room → prefer HA (covers naming another room / a non-ESP speaker)
        if room and self.ha_client:
            ids = self.ha_client.resolve_all_entities(room, room=room, domain_hints=["media_player"])
            if not ids:
                # Maybe user named the speaker itself (e.g. "sonos salon")
                ids = self.ha_client.resolve_all_entities(room, room=None, domain_hints=["media_player"])
            if ids:
                return ("ha", ids[0])
            # A room named, no speaker there: not the satellite either, which played anyway while
            # the reply said where. « ici » (and « là », a stopword) mean the satellite.
            if normalize(room) not in ("", "ici"):
                return None

        # Follow-up without room: reuse the last target
        if not room and self._last_media_target:
            kind, ident = self._last_media_target
            if kind == "esp" and ident in self.devices:
                return self._last_media_target
            if kind == "ha" and self.ha_client and ident in self.ha_client.entities:
                return self._last_media_target

        # Default: the current ESP, if it has a media_player
        if self.device and self.site.name in self.device_media_keys:
            return ("esp", self.site.name)

        return None

    async def _offline_reply(self, target: tuple[str, str]) -> Optional[str]:
        """The result when the target is an HA speaker HA cannot reach (the satellite answers, being the one listening)."""
        kind, ident = target
        if kind == "ha" and not await self.ha_client.reachable([ident]):
            return self.ha_client.offline_reply([ident])
        return None

    def _after_media_change(self, target: Optional[tuple[str, str]]) -> None:
        """Remember the speaker for volume follow-ups; no follow-up listening (an open mic would
        transcribe the radio)."""
        if target:
            self._last_media_target = target
        self._no_followup = True

    async def _execute_media_function(self, function_name: str, arguments: dict) -> str:
        """Handle play_radio / stop_media / set_volume / change_volume tool calls.

        Dispatches to the ESP's media_player directly when possible (no HA round-trip),
        or to a HA media_player entity when the user names a specific room.
        """
        room = arguments.get("room")
        target = self._resolve_media_target(room)
        if not target:
            return _no_speaker(room)
        if refusal := await self._offline_reply(target):
            return refusal

        if function_name == "play_radio":
            station_name = arguments.get("station", "")
            station = _resolve_station(station_name)
            if not station:
                return f"Station {station_name} inconnue"
            display_name, url = station
            await self._media_play_url(target, url)
            self._after_media_change(target)
            return f"{display_name} en cours de lecture"

        if function_name == "stop_media":
            await self._media_command(target, MediaPlayerCommand.STOP, ha_service="media_stop")
            return "Lecture arrêtée"

        if function_name == "set_volume":
            level = arguments.get("level")
            if level is None:
                return "Niveau de volume manquant"
            level = max(0, min(100, int(level)))
            await self._media_set_volume(target, level)
            self._after_media_change(target)
            return f"Volume à {level} pour cent"

        if function_name == "change_volume":
            direction = arguments.get("direction", "up")
            if direction == "up":
                await self._media_command(target, MediaPlayerCommand.VOLUME_UP, ha_service="volume_up")
                label = "Plus fort"
            else:
                await self._media_command(target, MediaPlayerCommand.VOLUME_DOWN, ha_service="volume_down")
                label = "Moins fort"
            self._after_media_change(target)
            return label

        return "C'est fait"

    async def _execute_music_function(self, function_name: str, arguments: dict) -> str:
        """Handle play_music / music_control: Spotify through go-librespot.

        Controls act on go-librespot's single queue, whichever speaker plays it; only
        play_music and resume need a speaker to point at the stream."""
        if not self.spotify:
            return "Spotify non disponible"
        action = arguments.get("action")
        try:
            if function_name == "music_control" and action != "resume":
                if action == "current":
                    result = await self.spotify.now_playing()
                elif action in CONTROLS:
                    result = await self.spotify.control(action)
                else:
                    return f"Action {action} inconnue"
                self._after_media_change(self._spotify_here)
                return result

            room = arguments.get("room")
            # "reprends la musique": back where it played, if that is at this site
            target = self._spotify_here if action == "resume" and not room else None
            target = target or self._resolve_media_target(room)
            if not target:
                return _no_speaker(room)
            if refusal := await self._offline_reply(target):
                return refusal
            found = None
            if function_name == "play_music":
                query = arguments.get("query", "")
                if not (found := await self.spotify.find(query, arguments.get("kind"))):
                    return f"Rien trouvé sur Spotify pour {query}"
            if not await self._point_at_spotify(target):
                return f"{self._speaker_name(target)} ne répond pas"
            if found:
                uri, description = found
                await self.spotify.play(uri, shuffle=bool(arguments.get("shuffle")))
                result = f"Lecture sur Spotify : {description}"
            else:
                result = await self.spotify.control("resume")
            self._after_media_change(target)
            return result
        except SpotifyNotConnected:
            return "Spotify n'est pas connecté au compte"
        except SpotifyError as e:
            logger.error(f"Spotify error: {e}")
            return "Spotify ne répond pas"

    async def _music_for_prompt(self) -> Optional[str]:
        """What Spotify plays and where, so that "mets la musique sur la Sonos" moves it rather than replace it."""
        if not self.spotify or not self.spotify.stream.listeners or not self._spotify_at:
            return None
        try:
            track = await self.spotify.playing()
        except SpotifyError:
            return None
        if not track:
            return None
        site, target = self._spotify_at
        where = self._speaker_name(target, site)
        if site is not self.site and site.name:
            where += f", à {site.name.capitalize()}"
        return (
            f"Musique Spotify en cours sur {where} : {track}. Pour la mettre sur une autre enceinte, "
            "music_control resume avec room : elle continue sans changer."
        )

    def _speaker_name(self, target: tuple[str, str], site: Optional[Site] = None) -> str:
        kind, ident = target
        ha_client = (site or self.site).ha_client
        if kind == "ha" and ha_client:
            return ha_client.entities.get(ident, {}).get("friendly_name", ident)
        return "l'enceinte de l'appareil vocal"

    async def _stop_silent_spotify(self) -> None:
        """Paused music kept sending silence to its speaker: stop the speaker after a while."""
        if self._spotify_at:
            site, target = self._spotify_at
            await self._media_command(target, MediaPlayerCommand.STOP, ha_service="media_stop", site=site)

    async def _point_at_spotify(self, target: tuple[str, str]) -> bool:
        """Make the target speaker play the Spotify stream, and move it off the previous one.
        False if it never connected: nothing changed, the music must not start nor be said playing.

        A speaker taken off the stream (radio, stop) needs nothing here: once the music
        plays to nobody, the stream pauses go-librespot."""
        stream = self.spotify.stream
        previous, new = self._spotify_at, (self.site, target)
        if previous == new and stream.listeners:
            return True
        url = self._spotify_url(self.site)
        joins = stream.joins
        await self._media_play_url(target, url)
        # Start the music once the speaker listens, so that the first seconds play
        if not await stream.wait_for_join(joins, timeout=8):
            logger.warning(f"Spotify: {target} did not connect to {url}")
            return False
        # Stopped after the new one joined, so the encoder never runs dry; and only if
        # still on the stream, as it may have moved to a radio since
        if previous and previous != new and len(stream.listeners) > 1:
            previous_site, previous_target = previous
            await self._media_command(
                previous_target, MediaPlayerCommand.STOP, ha_service="media_stop", site=previous_site
            )
        self._spotify_at = new
        return True

    async def _media_play_url(self, target: tuple[str, str], url: str, content_type: str = "music") -> None:
        """Start playback of an arbitrary media URL on the target speaker (or show it, for a picture)."""
        kind, ident = target
        if kind == "esp":
            api = self.devices.get(ident)
            key = self.device_media_keys.get(ident)
            if api and key is not None:
                logger.info(f"ESP media_player_command: play {url} on {ident} (key={key})")
                api.media_player_command(key, media_url=url, announcement=False)
            return
        await self.ha_client.call_service(
            "media_player",
            "play_media",
            ident,
            media_content_id=url,
            media_content_type=content_type,
        )

    async def _media_command(
        self, target: tuple[str, str], command: MediaPlayerCommand, ha_service: str, site: Optional[Site] = None
    ) -> None:
        """Send a media control command (STOP, VOLUME_UP...) to a target of the active site, or of `site`."""
        kind, ident = target
        if kind == "esp":
            api = self.devices.get(ident)
            key = self.device_media_keys.get(ident)
            if api and key is not None:
                logger.info(f"ESP media_player_command: {command.name} on {ident} (key={key})")
                api.media_player_command(key, command=command)
            return
        ha_client = (site or self.site).ha_client
        if ha_client:
            await ha_client.call_service("media_player", ha_service, ident)

    async def _media_set_volume(self, target: tuple[str, str], level: int) -> None:
        """Set absolute volume (0-100) on the target speaker."""
        kind, ident = target
        if kind == "esp":
            api = self.devices.get(ident)
            key = self.device_media_keys.get(ident)
            if api and key is not None:
                logger.info(f"ESP media_player_command: volume={level}% on {ident} (key={key})")
                api.media_player_command(key, volume=level / 100.0)
            return
        await self.ha_client.call_service(
            "media_player",
            "volume_set",
            ident,
            volume_level=level / 100.0,
        )

    # === SCREEN ===

    async def _show_image(self, query: str) -> str:
        """Find a picture on the web and cast it to the screen."""
        if not self.ha_client.screens:
            return "Aucun écran disponible"
        if not (screens := await self.ha_client.reachable(self.ha_client.screens)):
            return self.ha_client.offline_reply(self.ha_client.screens)
        picture = await find_image(query, exclude=self.conversation.shown_images)
        if not picture:
            return f"Aucune image trouvée pour {query}"
        self.conversation.shown_images.add(picture.url)
        screen = screens[0]
        await self._media_play_url(("ha", screen), picture.url, picture.content_type)
        if picture.from_article:
            shown = f"l'image de l'article Wikipédia « {picture.caption} »"
        else:
            # Matched on words, maybe not what was asked: the reply said « le plus grand arbre du
            # monde » over a photo titled « Nepal (11) - 26Sep10, Bardia (Nepal) », a field
            shown = f"une photo trouvée par mots-clés, titrée « {picture.caption} »"
        return f"Image affichée sur {self.ha_client.entities[screen]['friendly_name']} : {shown}"

    # === TIMER EVENTS ===

    async def _handle_timer_event(self, site: Site, event_type, timer):
        """Forward timer events to the satellite of the site the timer was set at. On FINISHED, also announce it."""
        api = self.devices.get(site.name)
        if api:
            try:
                api.send_voice_assistant_timer_event(
                    event_type,
                    timer.id,
                    timer.name,
                    timer.total_seconds,
                    timer.seconds_left,
                    timer.is_active,
                )
            except Exception as e:
                logger.warning(f"Failed to send timer event to ESP: {e}")

        from aioesphomeapi import VoiceAssistantTimerEventType

        if event_type == VoiceAssistantTimerEventType.VOICE_ASSISTANT_TIMER_FINISHED and api:
            await self._announce_timer_finished(api, timer, site)

    async def _announce_timer_finished(self, api, timer, site: Site):
        """Generate TTS and play announcement when a timer finishes."""
        if timer.is_alarm:
            text = f"Il est {timer.target_time}, c'est l'heure{f' : {timer.name}' if timer.name else ''} !"
        elif timer.name:
            text = f"Le timer {timer.name} est terminé !"
        else:
            text = f"Le timer de {_format_duration(timer.total_seconds)} est terminé !"

        try:
            audio_url = await self.text_to_speech_file(text, site)
            logger.info(f"Timer finished announcement: {text}")
            await self._announce(api, audio_url, follow_up=not self._satellite_plays(site))
        except Exception:
            logger.exception("Timer announcement error")

    # === LLM ===

    async def process_with_llm(self, api: Optional[APIClient], text: str, dry_run: bool = False) -> Optional[str]:
        """LLM via OpenAI-compatible API (local llama.cpp or cloud provider)"""
        from llm import (
            chat_completion,
            claims_completed_action,
            get_tool_definitions,
            parse_text_tool_call,
        )

        logger.info(f'LLM input: "{text}"')
        self._keep_llm_loaded()

        is_local = not LLM_API_KEY

        # The system prompt stays the same from one request to the next: Ollama reuses the prompt
        # it has already read, tools included, up to the first change. With the time in it, every
        # new minute re-read all ~3.6K tokens (~1.9 s, measured). What changes goes just before
        # the request instead.
        system_prompt = (
            "Tu es un assistant vocal pour la maison connectée. "
            "Réponds en français, en 1-2 phrases courtes maximum. "
            "Pas de markdown. Parle naturellement comme à l'oral. "
            "Quand on te demande de contrôler un appareil "
            "(même avec un pronom comme 'la' ou 'les'), "
            "utilise TOUJOURS les fonctions disponibles. "
            "Passe le paramètre room quand la pièce est mentionnée. "
            "Utilise les noms de groupes tels quels "
            "(ex: room='enfants'), ne les décompose pas. "
            "Si la demande couvre plusieurs actions ou sujets "
            "(ex: contrôler un appareil ET demander la météo), "
            "appelle TOUTES les fonctions nécessaires "
            "en parallèle dans ta réponse. "
            # Without it, « merci bonne nuit » got a goodbye and no call 5 times out of 5 (dry runs,
            # 2026-10-11), once with « (end_conversation) » written in the text
            "Quand l'utilisateur prend congé (merci, au revoir, bonne nuit, c'est tout), "
            "appelle end_conversation en plus de ta réponse."
        )
        family = LOCAL_CONFIG.get("family", [])
        if family:
            system_prompt += "\n" + _format_family_for_prompt(family)
        if is_local:
            # Not only a Qwen 3 habit: without it, gemma4 sometimes spent a round's 500 tokens
            # thinking, reasoning_effort none notwithstanding (content empty, finish_reason length)
            system_prompt = "/no_think " + system_prompt
        if self.ha_client:
            entity_list = self.ha_client.get_entity_list_for_prompt()
            if entity_list:
                system_prompt += f"\nAppareils disponibles: {entity_list}"

        from timer import format_timers_for_prompt

        context = [f"Date et heure actuelles: {time.strftime('%A %d %B %Y, %H:%M')}."]
        if timers_info := format_timers_for_prompt(self.timer_manager.get_timers()):
            context.append(timers_info)
        if music_info := await self._music_for_prompt():
            context.append(music_info)

        tools = get_tool_definitions(self.ha_client, self.calendar_client, music=self.spotify is not None)
        # Only these run: a name the model made up (« unlock ») or a tool this house lacks does not
        offered = {tool["name"] for tool in tools}

        # A dry run reads and writes only its own conversation (AD-047)
        conversation = self.site.dry_run if dry_run else self.conversation

        # Expire old conversation history
        now = time.time()
        if now - conversation.last_time > self.CONVERSATION_TIMEOUT:
            if conversation.history:
                logger.info("Conversation history expired, starting fresh")
            conversation.history = []
            conversation.id = None
        conversation.last_time = now

        # Assign conversation ID for first exchange in a conversation
        if not conversation.id:
            conversation.id = int(now)
            conversation.shown_images.clear()

        # Build messages
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(conversation.history)
        messages.append({"role": "system", "content": "\n".join(context)})
        messages.append({"role": "user", "content": text})

        message = await chat_completion(
            LLM_URL,
            LLM_API_KEY,
            LLM_MODEL,
            messages,
            tools,
        )
        if not message:
            return None

        # The turn as history keeps it: the calls and their results too. With only « ferme les volets
        # → C'est fait », the model answered the same request again from history, calling nothing
        # (7 repeats out of 10, dry runs).
        turn = [{"role": "user", "content": text}]

        def save_to_history(response_text):
            history = conversation.history + turn + [{"role": "assistant", "content": response_text}]
            # The last 5 exchanges, cut at a request
            starts = [i for i, m in enumerate(history) if m["role"] == "user"]
            conversation.history = history[starts[-5] :] if len(starts) > 5 else history

        # Tool call loop: keep calling LLM until it returns text (max 3 rounds)
        import json

        self._last_tool_calls = []
        done: dict[tuple, str] = {}  # (function, arguments) -> result, this turn
        max_rounds = 3
        challenged_empty_claim = False
        for round_num in range(max_rounds):
            if "tool_calls" not in message or not message["tool_calls"]:
                # The model sometimes answers "c'est fait, j'ai ouvert les volets"
                # having called nothing at all. With no tool call behind it, that
                # claim is false by construction. Give it one chance to actually
                # act before the answer reaches the user.
                claim = (message.get("content") or "").strip()
                if not self._last_tool_calls and not challenged_empty_claim and claims_completed_action(claim):
                    challenged_empty_claim = True
                    logger.warning(f"Action claimed with no function call: {claim!r} — challenging")
                    messages.append(message)
                    messages.append(
                        {
                            "role": "system",
                            "content": (
                                "Tu viens d'affirmer avoir effectué une action alors que tu n'as "
                                "appelé AUCUNE fonction : rien n'a été fait. Appelle maintenant les "
                                "fonctions nécessaires. Si la demande n'exige aucune action, réponds "
                                "sans prétendre avoir agi."
                            ),
                        }
                    )
                    message = await chat_completion(LLM_URL, LLM_API_KEY, LLM_MODEL, messages, tools=tools)
                    if not message:
                        break
                    continue
                break

            # Run this round's calls concurrently. An HA service call blocks until
            # the device acknowledges — ~0.5-1.8s for a Velux over HomeKit, vs 14ms
            # for a state read — so "ferme tous les volets" spent 11 of its 19
            # seconds waiting on eight of them one after another. The LLM emits the
            # batch in one go, so the calls are independent by construction.
            async def run_tool(tool_call):
                function_name = tool_call["function"]["name"]
                try:
                    function_args = json.loads(tool_call["function"]["arguments"])
                except (TypeError, ValueError):
                    # One malformed call must not fail the whole request and the calls beside it
                    logger.warning(f"Malformed arguments for {function_name}: {tool_call['function']['arguments']!r}")
                    return {"function": function_name, "args": {}, "error": "arguments"}, "Erreur: arguments illisibles"
                logger.info(f"Function call: {function_name}({function_args})")
                tc_entry = {"function": function_name, "args": function_args}
                # The same call again in a later round (show_image ran 3 times, 2026-10-11): its result
                key = (function_name, json.dumps(function_args, sort_keys=True))
                if key in done:
                    logger.info(f"Already run this turn: {function_name}")
                    tc_entry["result"] = done[key]
                    return tc_entry, done[key]
                if function_name not in offered:
                    logger.warning(f"Function not offered: {function_name}")
                    tc_entry["result"] = f"Fonction {function_name} non disponible"
                    return tc_entry, tc_entry["result"]
                if dry_run:
                    logger.info(f"Dry run: {function_name}({function_args})")
                    tc_entry["result"] = "OK"
                    return tc_entry, "OK"
                async with self._tool_semaphore:
                    try:
                        result = await self.execute_function(function_name, function_args)
                        logger.info(f"Function result: {result}")
                        tc_entry["result"] = done[key] = result
                    except Exception as e:
                        logger.error(f"Function execution error: {e}")
                        result = f"Erreur: {e}"
                        tc_entry["error"] = str(e)
                return tc_entry, result

            outcomes = await asyncio.gather(*(run_tool(tc) for tc in message["tool_calls"]))

            tool_messages = []
            for (tc_entry, result), tool_call in zip(outcomes, message["tool_calls"]):
                self._last_tool_calls.append(tc_entry)
                tool_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": result,
                    }
                )

            # Follow-up LLM call with tools still available for additional calls
            messages.append(message)  # assistant message with tool_calls
            messages.extend(tool_messages)
            turn.append(message)
            turn.extend({**m, "content": m["content"][:_HISTORY_RESULT_CHARS]} for m in tool_messages)
            # A goodbye needs no second round: with nothing left to do, gemma answered it in English
            # (« No problem, have a good night! ») or spent 500 tokens thinking, content empty
            if all(tc["function"]["name"] == "end_conversation" for tc in message["tool_calls"]):
                response = (message.get("content") or "").strip() or "Avec plaisir, à bientôt !"
                save_to_history(response)
                return response
            # No « Fonctions déjà appelées… » reminder any more: written for GPT-5.4 Nano, it made no
            # extra call in ~45 gemma4 dry runs, and gemma answered it in English (« No other actions
            # needed. ») instead of speaking to the user
            message = await chat_completion(
                LLM_URL,
                LLM_API_KEY,
                LLM_MODEL,
                messages,
                tools=tools,
            )
            if not message:
                # Fallback: join raw results if follow-up call fails
                response = ". ".join(tc.get("result", "") for tc in self._last_tool_calls if tc.get("result"))
                save_to_history(response)
                return response

        if message.get("tool_calls"):
            names = [tc["function"]["name"] for tc in message["tool_calls"]]
            logger.warning(f"Round limit reached, calls not run: {names}")

        if self._last_tool_calls:
            # LLM returned text after tool execution
            response = (message.get("content") or "").strip()
            if not response:
                response = ". ".join(tc.get("result", "") for tc in self._last_tool_calls if tc.get("result"))
            save_to_history(response)
            return response

        # Direct response (possibly with text-encoded tool call)
        content = message.get("content", "")
        if is_local and "</think>" in content:
            content = content.split("</think>")[-1]
        content = content.strip()

        fn_name, fn_args = parse_text_tool_call(content)
        if fn_name in offered and fn_args and not dry_run:
            try:
                logger.info(f"Function call (text fallback): {fn_name}({fn_args})")
                result = await self.execute_function(fn_name, fn_args)
                logger.info(f"Function result: {result}")
                save_to_history(result)
                return result
            except Exception as e:
                logger.warning(f"Failed to execute text tool call: {e}")

        # Last resort: never hand back a claim of an action that never ran. Safe
        # because we only get here with zero tool calls executed this turn.
        if not self._last_tool_calls and claims_completed_action(content):
            logger.error(f"Unbacked action claim survived the challenge: {content!r}")
            response = "Désolé, je n'ai pas réussi à faire ça."
            save_to_history(response)
            return response

        logger.info(f'LLM response: "{content}"')
        if content:
            save_to_history(content)
        return content if content else None

    def _keep_llm_loaded(self) -> None:
        """Ask Ollama to keep the model loaded: it unloads after 5 idle minutes, and the next request
        then waited ~7 s (median of 10 cold requests, against 2.3 s warm). Sent with each request
        rather than on a timer, so that a job that unloads it to free the GPU (beast) can."""
        if LLM_API_KEY or not LLM_MODEL or not LLM_URL.endswith("/v1/chat/completions"):
            return
        url = LLM_URL.removesuffix("/v1/chat/completions") + "/api/generate"

        async def ping():
            try:
                async with aiohttp.ClientSession() as session:
                    await session.post(
                        url, json={"model": LLM_MODEL, "keep_alive": -1}, timeout=aiohttp.ClientTimeout(total=30)
                    )
            except Exception as e:
                logger.debug(f"Ollama keep-alive failed: {e}")

        # Not with the service calls drained at shutdown: a ping must not hold a restart 30 s
        task = asyncio.ensure_future(ping())
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    # === TTS ===

    async def text_to_speech_file(self, text: str, site: Optional[Site] = None) -> str:
        """Generate a TTS WAV file and return its URL, as the active site's (or `site`'s) speakers reach it."""
        return await self.tts.synthesize_to_file(text, self.tts_dir, (site or self.site).http_base_url)

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentences for chunked TTS generation."""
        import re

        # Split on sentence-ending punctuation followed by space
        parts = re.split(r"(?<=[.!?;])\s+", text)
        sentences = [s.strip() for s in parts if s.strip()]
        # Don't split very short texts
        if len(sentences) <= 1:
            return [text]
        return sentences

    async def text_to_speech(self, api: APIClient, text: str) -> dict:
        """Generate TTS sentence-by-sentence with pipelined playback.
        Each sentence is played as soon as generated while the next one
        generates in parallel, reducing perceived latency.
        Returns timing dict with 'gen' (generation) and 'play' (playback) durations.
        """
        logger.info(f'TTS: generating audio for "{text}"')

        api.send_voice_assistant_event(VoiceAssistantEventType.VOICE_ASSISTANT_TTS_START, {"text": text})

        gen_time = 0.0
        play_time = 0.0

        try:
            # No follow-up after a goodbye or a media command, nor while the satellite plays music
            # itself: a reply with follow-up over its own player went unconfirmed and the Voice PE
            # rebooted (2026-10-10, 23:13). Without follow-up, one announcement: several with
            # start_conversation=False made the ESP miss AnnounceFinished.
            follow_up = not (self._end_conversation or self._no_followup or self._satellite_plays())
            sentences = self._split_sentences(text) if follow_up else [text]
            logger.info(f"TTS: {len(sentences)} sentence(s) to generate")

            # Generate first sentence
            t0 = time.time()
            next_url = await self.text_to_speech_file(sentences[0])
            gen_time += time.time() - t0

            # End pipeline before playing announcements
            api.send_voice_assistant_event(VoiceAssistantEventType.VOICE_ASSISTANT_RUN_END, {})

            for i in range(len(sentences)):
                current_url = next_url
                is_last = i == len(sentences) - 1

                # Pipeline: start generating next sentence while playing current
                gen_task = None
                if not is_last:
                    gen_task = asyncio.create_task(self.text_to_speech_file(sentences[i + 1]))

                # Play current sentence
                logger.info(f"TTS: playing sentence {i + 1}/{len(sentences)}")
                t_play = time.time()

                await self._announce(api, current_url, follow_up=is_last and follow_up)
                if is_last and self._end_conversation:
                    self.conversation.history = []
                    self.conversation.id = None
                    logger.info("Conversation ended, history cleared")

                play_time += time.time() - t_play

                # Await next sentence generation (should already be done during playback)
                if gen_task:
                    t0 = time.time()
                    next_url = await gen_task
                    extra_gen = time.time() - t0
                    gen_time += extra_gen
                    if extra_gen > 0.05:
                        logger.info(f"TTS: waited {extra_gen:.2f}s for next sentence generation")

            timings = {"gen": round(gen_time, 2), "play": round(play_time, 2)}
            logger.info(f"TTS done: gen={timings['gen']}s play={timings['play']}s")
            return timings

        except Exception:
            logger.exception("TTS error")
            raise

    async def send_error_to_device(self, api: APIClient, error_message: str):
        """Send an error event and end the pipeline."""
        logger.error(f"Sending error to ESP: {error_message}")
        try:
            api.send_voice_assistant_event(
                VoiceAssistantEventType.VOICE_ASSISTANT_ERROR,
                {"code": "server_error", "message": error_message},
            )
            api.send_voice_assistant_event(VoiceAssistantEventType.VOICE_ASSISTANT_RUN_END, {})
            logger.info("ERROR + RUN_END sent to ESP")
        except Exception as e:
            logger.error(f"Error sending error: {e}")


async def main():
    """Main entry point"""
    server = VoiceAssistantServer()
    reload_event = asyncio.Event()
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def handle_sighup():
        logger.info("SIGHUP received — scheduling reload")
        reload_event.set()

    def handle_sigterm():
        logger.info("SIGTERM received — shutting down")
        stop_event.set()

    loop.add_signal_handler(signal.SIGHUP, handle_sighup)
    loop.add_signal_handler(signal.SIGTERM, handle_sigterm)

    try:
        await server.start()
        logger.info("Server running — send SIGHUP to reload, SIGTERM to stop")

        while not stop_event.is_set():
            reload_task = asyncio.create_task(reload_event.wait())
            stop_task = asyncio.create_task(stop_event.wait())
            done, _ = await asyncio.wait({reload_task, stop_task}, return_when=asyncio.FIRST_COMPLETED)
            for t in {reload_task, stop_task} - done:
                t.cancel()

            if reload_event.is_set():
                reload_event.clear()
                logger.info("Reloading...")
                await server.init_ha_clients()
                logger.info("Reload complete")

    except KeyboardInterrupt:
        logger.info("\nShutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
    finally:
        # Let any fire-and-forget service calls reach HA before we tear down the
        # session — otherwise a restart right after "ferme tout" drops commands.
        if server._pending_service_calls:
            logger.info(f"Draining {len(server._pending_service_calls)} pending service calls...")
            await asyncio.gather(*server._pending_service_calls, return_exceptions=True)
        for reconnect_logic in server._reconnect_logics:
            await reconnect_logic.stop()
        if server.calendar_client:
            await server.calendar_client.close()

    logger.info("Server stopped")


if __name__ == "__main__":
    asyncio.run(main())
