"""Web UI and HTTP routes for the voice assistant."""

import asyncio
import hmac
import html
import ipaddress
import json
import logging
import os
import re
import socket
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import quote, urlencode

from aiohttp import WSMsgType, web
from yarl import URL

import onnx_gpu
from ha_client import quiet
from sites import Site, site_at_address
from spotify import CONTROLS
from vad import MAX_SECONDS, WAIT_FOR_FOLLOW_UP, WAIT_FOR_SPEECH, SpeechEnd

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
# Tailscale's address range: the web UI is reached through it, while :8888 also answers on the LAN
_TAILNET = ipaddress.ip_network("100.64.0.0/10")
# What speakers fetch, from the LAN or through the Paris relay: TTS replies and the Spotify stream
_SPEAKER_PATHS = ("/tts/", "/spotify/")
# What the LAN may fetch: the speakers' files, and /health for checks
_LAN_PATHS = (*_SPEAKER_PATHS, "/health")
# Tailscale Funnel's proxy runs on this machine: the internet arrives from loopback, told apart by these
_PROXY_HEADERS = ("X-Forwarded-For", "Tailscale-Funnel-Request")
# Fetched without the cookie when Chrome installs the app, the icons by Google's WebAPK server. static/
# holds only the shell and its icons: data stays behind api/, tts/, recordings/ and health
_INSTALL_PATHS = ("/manifest.webmanifest", "/static/")
KEY_COOKIE = "assistant_key"
# The names browsers reach this server by, besides IP addresses and MagicDNS's (*.ts.net, Funnel's too)
_OUR_NAMES = {"localhost", socket.gethostname().split(".")[0].lower()}
# The nightly review's briefs (AD-074), and where the Claude Code Desktop session that reads one opens:
# Ivan's SSH connection to this machine, the project's folder
_REVIEW_DIR = "/home/ivan/.local/state/voice-assistant-review"
_REVIEW_BRIEF = re.compile(r"(\d{4}-\d{2}-\d{2})/proposals/[\w.-]+\.md")
_REVIEW_SESSION = {"ssh_host": "charras", "ssh_folder": "/home/ivan/Code/voice-assistant"}
# What the app's microphone sends: 16 kHz 16-bit mono, as long as the satellite's at most
_MAX_RECORDING = int(MAX_SECONDS) * 16000 * 2


def is_trusted_client(remote: str | None) -> bool:
    """This machine and the tailnet. The web UI shows transcripts, plays recordings and runs commands."""
    try:
        address = ipaddress.ip_address(remote or "")
    except ValueError:
        return False
    return address.is_loopback or address in _TAILNET


def _is_key(given: str | None, key: str) -> bool:
    return bool(key and given) and hmac.compare_digest(given.encode(), key.encode())


def access_control(key: str):
    """The LAN sees what speakers fetch; this machine and the tailnet see everything; the internet,
    through Funnel at /assistant/, sees everything with the key: `?k=<key>` once, then a cookie."""

    @web.middleware
    async def middleware(request, handler):
        if not any(header in request.headers for header in _PROXY_HEADERS):
            if request.path.startswith(_LAN_PATHS) or is_trusted_client(request.remote):
                return await handler(request)
            raise web.HTTPForbidden(text="Only speaker files are served on the LAN")
        if request.path.startswith(_INSTALL_PATHS) or _is_key(request.cookies.get(KEY_COOKIE), key):
            return await handler(request)
        if _is_key(request.query.get("k"), key):
            # The same page without the key in its address. Relative, as is the cookie (no Path: the
            # browser scopes it to /assistant), since Funnel strips that prefix before we see the request
            response = web.HTTPFound(request.rel_url.name or "./")
            response.set_cookie(
                KEY_COOKIE, key, path="", max_age=365 * 86400, secure=True, httponly=True, samesite="Lax"
            )
            raise response
        raise web.HTTPUnauthorized(text="Ouvrez le lien avec la clé")

    return middleware


def _funnel_address(request) -> Optional[str]:
    """The internet address a request came from through Funnel; None from the tailnet or this machine."""
    if forwarded := request.headers.get("X-Forwarded-For"):
        # The last address is the one Funnel's proxy added; any before it, the client wrote
        return forwarded.rsplit(",", 1)[-1].strip()
    return None


def write_private(path: Path, text: str) -> None:
    """Written whole to a temporary file then renamed: a crash mid-write left invalid JSON, read as empty
    and overwritten. Mode 600: kept from other users."""
    tmp = path.with_name(path.name + ".tmp")
    with open(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as f:
        f.write(text)
    os.replace(tmp, path)


def _is_our_host(request) -> bool:
    """The Host header names this server: an IP address, or one of its names. No header: no browser."""
    if "Host" not in request.headers:
        return True
    try:
        name = (request.url.host or "").rstrip(".")
    except ValueError:  # malformed
        return False
    try:
        ipaddress.ip_address(name)
        return True
    except ValueError:
        return name in _OUR_NAMES or name.endswith(".ts.net")


@web.middleware
async def cross_site_guard(request, handler):
    """Pages of other sites, open in a browser on this machine or the tailnet, cannot use it (AD-056): a
    POST that is not JSON crosses sites without a CORS preflight, and DNS rebinding leaves the page's
    own name in Host. Speakers and the Paris relay are no browsers: their paths pass."""
    if request.path.startswith(_SPEAKER_PATHS):
        return await handler(request)
    if not _is_our_host(request):
        raise web.HTTPMisdirectedRequest(text="Unknown host")
    if request.method not in ("GET", "HEAD") and request.content_type != "application/json":
        raise _json_error(web.HTTPUnsupportedMediaType, "JSON body only (Content-Type: application/json)")
    if request.headers.get("Upgrade", "").lower() == "websocket" and not _same_origin(request):
        raise web.HTTPForbidden(text="Not this server's page")
    return await handler(request)


def _same_origin(request) -> bool:
    """A WebSocket opened by this server's own page. Browsers open WebSockets across sites without a
    preflight, a GET the JSON rule does not cover, and say which page asks in Origin: a page of any site,
    in a browser on the tailnet, could otherwise speak to the house (AD-059). No Origin: no browser."""
    if (origin := request.headers.get("Origin")) is None:
        return True
    try:
        return URL(origin).host == request.url.host
    except ValueError:
        return False


async def _send(ws: web.WebSocketResponse, message: dict) -> None:
    """To the page, if it is still there: a request it left (the phone locked) still runs."""
    if not ws.closed:
        try:
            await ws.send_json(message)
        except ConnectionError:
            pass


def _page_path(audio_url: str) -> str:
    """Where the page fetches a reply's audio: relative, the app lives at / and at /assistant/."""
    return "tts/" + audio_url.rsplit("/", 1)[-1]


async def _listen(ws: web.WebSocketResponse, follow_up: bool) -> Optional[tuple[bytes, bool]]:
    """Take the app's audio until speech ends, the page says the user is done (« end »), or time is up, with
    the satellite's limits. Once the page says « hold » (the mic button held down), only « end » (released)
    or the 30 s end it: no silence, no wait for speech (AD-062). Returns the recording and whether it holds
    speech; None if the page went away."""
    recording, speech, held = bytearray(), SpeechEnd(), False
    loop = asyncio.get_running_loop()
    start = loop.time()
    while len(recording) < _MAX_RECORDING:
        limit = MAX_SECONDS if speech.heard or held else WAIT_FOR_FOLLOW_UP if follow_up else WAIT_FOR_SPEECH
        if (wait := limit - (loop.time() - start)) <= 0:
            break
        try:
            message = await ws.receive(timeout=wait)
        except TimeoutError:
            break
        if message.type == WSMsgType.BINARY:
            heard = speech.heard
            recording.extend(message.data)
            if speech.read(recording) and not held:
                break
            if speech.heard and not heard:
                await _send(ws, {"type": "speech"})
        elif message.type == WSMsgType.TEXT:
            if message.data != "hold":  # « end »: done speaking
                break
            held = True
        else:
            return None
    return bytes(recording), speech.heard


def _client(params) -> Optional[str]:
    """The id a browser gives itself (`client`): its conversation (AD-061), the one its page shows (AD-060)."""
    client = params.get("client")
    return client if isinstance(client, str) and 0 < len(client) <= 40 and client.isalnum() else None


async def handle_review_open(request):
    """GET /review/open?brief=<day>/proposals/<file>.md: opens a Claude Code Desktop session on a proposal of
    the nightly review. Its email can't link to claude:// itself: HEY shows such a link but doesn't follow
    it. The brief is a path the page checks, not a prompt: a link from elsewhere can't put words in it."""
    match = _REVIEW_BRIEF.fullmatch(request.query.get("brief", ""))
    if not match:
        raise web.HTTPNotFound(text="Pas de proposition à cette adresse")
    prompt = (f"Lis {_REVIEW_DIR}/{match[0]} : une proposition de la revue de nuit de l'assistant vocal "
              f"({match[1]}). Discutons-en avant de coder.")
    link = html.escape(f"claude://code/new?{urlencode({**_REVIEW_SESSION, 'q': prompt}, quote_via=quote)}")
    return web.Response(content_type="text/html", text=f"""<!doctype html>
<meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Revue de nuit</title>
<body style="font:16px -apple-system,Helvetica,sans-serif;margin:2em">
<p><a href="{link}" style="font-size:1.2em">Ouvrir dans Claude Code</a></p>
<p style="color:#666">{html.escape(prompt)}</p>
<script>location.href = document.querySelector("a").href</script>""")


def _json_error(error: type[web.HTTPException], message: str) -> web.HTTPException:
    return error(text=json.dumps({"error": message}), content_type="application/json")


async def _read_body(request) -> dict:
    """The JSON object a POST carries (cross_site_guard has checked its type)."""
    try:
        body = await request.json()
    except ValueError:
        raise _json_error(web.HTTPBadRequest, "invalid JSON") from None
    return body if isinstance(body, dict) else {}


class ExchangeLog:
    """Persisted exchange log for the web UI."""

    def __init__(self, path, max_entries=50):
        self.path = path
        self.max_entries = max_entries
        self.entries = self._load()
        # Changes with each entry, and from one run of the server to the next
        self._started, self._added = time.time_ns(), 0

    @property
    def etag(self) -> str:
        return f'"{self._started}-{self._added}"'

    def _load(self):
        try:
            if self.path.exists():
                data = json.loads(self.path.read_text())
                return data[-self.max_entries :]
        except Exception as e:
            logger.warning(f"Exchange log unreadable, starting empty: {e}")
        return []

    def _save(self):
        """Private (write_private): transcripts and calendar answers."""
        try:
            write_private(self.path, json.dumps(self.entries, ensure_ascii=False, indent=1))
        except Exception as e:
            logger.warning(f"Failed to save exchange log: {e}")

    def add(
        self,
        source,
        input_text,
        transcript,
        response,
        tts_url=None,
        timings=None,
        tool_calls=None,
        conversation_id=None,
        audio=None,
        site=None,
        client=None,
    ):
        entry = {
            # With its offset: the page compares it with the phone's clock, wherever the phone is
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "source": source,
            "input": input_text,
            "transcript": transcript,
            "response": response,
            "tts_url": tts_url,
            "timings": timings or {},
            "tool_calls": tool_calls or [],
            "conversation_id": conversation_id,
            "audio": audio,
            "site": site,
            "client": client,  # the browser that sent it, for its own conversation (AD-060)
        }
        self.entries.append(entry)
        self._added += 1
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries :]
        self._save()


def setup_routes(app, server):
    """Register web UI routes on an aiohttp app."""
    # Empty: nothing answers through Funnel
    app.middlewares.extend([cross_site_guard, access_control(os.environ.get("WEB_UI_KEY", ""))])

    async def read_text(request) -> str:
        """The prompt of a text request; the request then serves the site it names, if any.

        {"text": …, "site": "charras", "client": the browser's id}"""
        body = await _read_body(request)
        text = body.get("text", "")
        if not isinstance(text, str) or not text.strip():
            raise _json_error(web.HTTPBadRequest, "missing 'text' field")
        await serve(request, body)
        return text

    async def serve(request, params: dict) -> None:
        """The request's site (use_site), and a request of its own, its browser's (AD-061)."""
        await use_site(request, params)
        server.new_request(_client(params), _funnel_address(request))

    async def use_site(request, body: dict) -> None:
        """Serve the site the request names, else the one it comes from, for this request only: a
        conversation at another site goes on."""
        if name := body.get("site"):
            if not (site := server.site_named(name)):
                raise _json_error(web.HTTPNotFound, f"unknown site {name}")
            server.site = site
        elif site := await here(request):
            server.site = site

    async def here(request) -> Site | None:
        """The home a request came from through Funnel, by its internet address: a phone on its Wi-Fi."""
        address = _funnel_address(request)
        return await site_at_address(server.sites, address) if address else None

    async def handle_sites(request):
        """The homes for the UI's selector, and the one the request comes from, which needs none."""
        site = await here(request)
        return web.json_response(
            {"sites": [s.name for s in server.sites if s.name], "here": site.name if site else None}
        )

    async def handle_web_ui(request):
        return web.Response(text=(STATIC_DIR / "index.html").read_text(), content_type="text/html")

    async def handle_manifest(request):
        return web.Response(
            text=(STATIC_DIR / "manifest.webmanifest").read_text(),
            content_type="application/manifest+json",
        )

    async def handle_service_worker(request):
        # Beside the page, so its scope is the app's folder (a /static/sw.js could only control
        # /static/*). No-cache so updates propagate.
        return web.Response(
            text=(STATIC_DIR / "sw.js").read_text(),
            content_type="application/javascript",
            headers={"Cache-Control": "no-cache"},
        )

    async def handle_get_exchanges(request):
        # Polled every few seconds by open pages, often on mobile data: unchanged, a 304 and no body
        log = server.exchange_log
        headers = {"ETag": log.etag, "Cache-Control": "no-cache"}
        if request.headers.get("If-None-Match") == log.etag:
            return web.Response(status=304, headers=headers)
        return web.json_response(log.entries, headers=headers)

    async def handle_web_send(request):
        """A request typed in the web app: its reply is read, not spoken."""
        reply = await server.answer(await read_text(request), "web", {}, speak=False)
        if not reply["response"]:
            return web.json_response({"error": "LLM returned no response"}, status=500)
        return web.json_response(reply)

    async def handle_redo(request):
        """A shortcut of the web app: what a request did, run again as it was, without the model (AD-065).

        {"calls": [{"function": …, "args": {…}}], "label": what its button says, "site": …, "client": …}"""
        body = await _read_body(request)
        label = body.get("label")
        if not isinstance(label, str) or not label.strip():
            raise _json_error(web.HTTPBadRequest, "missing 'label' field")
        await serve(request, body)
        if not (reply := await server.redo(body.get("calls"), label.strip()[:200])):
            raise _json_error(web.HTTPBadRequest, "not calls a shortcut runs")
        return web.json_response(reply)

    async def handle_unlock(request):
        """The app's « Déverrouiller », the button a request proposing to unlock a door shows (AD-068).

        {"offer": the button's id, "client": the browser that asked, "site": its page's home}"""
        body = await _read_body(request)
        await serve(request, body)
        if not isinstance(offer := body.get("offer"), str) or not (reply := await server.confirm_unlock(offer)):
            raise _json_error(web.HTTPGone, "no such offer: expired, used, or another browser's")
        return web.json_response(reply)

    async def handle_in_effect(request):
        """Which shortcuts of the web app do what is already the case, by the devices' state (AD-066).

        {"shortcuts": [[{"function": …, "args": {…}}, …], …], "site": …} → {"in_effect": [true, …]}"""
        body = await _read_body(request)
        if not isinstance(shortcuts := body.get("shortcuts"), list) or len(shortcuts) > 20:
            raise _json_error(web.HTTPBadRequest, "'shortcuts': up to 20 lists of calls")
        await serve(request, body)
        token = quiet.set(True)  # asked at each poll of the page: its resolutions stay out of the log
        try:
            return web.json_response({"in_effect": await server.in_effect(shortcuts)})
        finally:
            quiet.reset(token)

    async def handle_test_prompt(request):
        text = await read_text(request)
        logger.info(f'[TEST] Input: "{text}"')
        reply = await server.answer(text, "web", {}, speak=True)
        if not reply["response"]:
            return web.json_response({"error": "LLM returned no response"}, status=500)
        result = {key: reply[key] for key in ("input", "response", "tts_url")}
        logger.info(f"[TEST] Result: {json.dumps(result, ensure_ascii=False)}")
        return web.json_response(result)

    async def handle_listen(request):
        """The app as a satellite (AD-059): its microphone, 16 kHz 16-bit mono in binary messages,
        heard by the satellite's VAD; the reply is the phone's to play. To the page, in order:
        speech (someone speaks), heard (stop sending), transcript, wait (« Un petit instant. », if the reply takes
        long, AD-079), then reply or error. From the page,
        « end » when the user is done speaking, « hold » while the button is held; closing drops the request.

        ?site=… as for a typed request; ?client=… the browser's id (AD-060); ?follow_up=1 for the listening
        that follows a reply."""
        await serve(request, dict(request.query))
        follow_up = request.query.get("follow_up") == "1"
        if not follow_up:
            server.warm_llm()  # while the request is spoken
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        if not (heard := await _listen(ws, follow_up)):
            return ws
        await _send(ws, {"type": "heard"})
        try:
            reply = await server.answer_recording(
                *heard,
                follow_up,
                lambda transcript: _send(ws, {"type": "transcript", "text": transcript}),
                lambda audio_url: _send(ws, {"type": "wait", "audio": _page_path(audio_url)}),
            )
        except Exception:
            logger.exception("App request failed")
            reply = {"response": None}
        if reply and reply["response"]:
            await _send(ws, {"type": "reply", **reply, "audio": _page_path(reply["tts_url"])})
        elif reply:
            await _send(ws, {"type": "error"})
        await ws.close()
        return ws

    async def handle_state(request):
        """What runs now at the page's home: the timers this browser sees, saying which it set; and Spotify,
        wherever it plays (one for every home)."""
        await use_site(request, dict(request.query))
        client = _client(request.query)
        timers = [
            {
                "id": timer.id,
                "name": timer.name,
                "seconds_left": timer.seconds_left,
                "alarm": timer.target_time if timer.is_alarm else None,
                "mine": client is not None and timer.client == client,  # it rings on this phone (AD-070)
            }
            for timer in server.timer_manager.get_timers(client)
        ]
        return web.json_response({"timers": timers, "music": await server.music_now()})

    async def handle_cancel_timer(request):
        """The app's ✕ on a timer: {"id": …, "site": …}."""
        body = await _read_body(request)
        await use_site(request, body)
        if not isinstance(timer_id := body.get("id"), str) or not await server.cancel_timer(timer_id):
            raise _json_error(web.HTTPNotFound, "no such timer")
        return web.json_response({"status": "ok"})

    async def handle_push_key(request):
        """The key a browser subscribes to this server's pushes with (AD-070)."""
        return web.json_response({"key": server.push.public_key})

    async def handle_push_subscribe(request):
        """A browser's push subscription, for its timers: {"client": …, "subscription": PushSubscription.toJSON()};
        without "subscription", the browser's is forgotten."""
        body = await _read_body(request)
        if not (client := _client(body)):
            raise _json_error(web.HTTPBadRequest, "missing 'client' field")
        if body.get("subscription") is None:
            server.push.unsubscribe(client)
        elif not server.push.subscribe(client, body["subscription"], request.headers.get("Origin")):
            raise _json_error(web.HTTPBadRequest, "not a push subscription")
        return web.json_response({"status": "ok"})

    async def handle_music(request):
        """The app's music buttons, as music_control: {"action": "pause" | "resume" | "next" | "previous"}."""
        body = await _read_body(request)
        if not isinstance(action := body.get("action"), str) or action not in CONTROLS:
            raise _json_error(web.HTTPBadRequest, f"action: one of {', '.join(CONTROLS)}")
        await serve(request, body)
        result = await server.execute_function("music_control", {"action": action})
        return web.json_response({"result": result, "music": await server.music_now()})

    async def handle_dry_run(request):
        """LLM-only endpoint: produces tool calls without executing them."""
        text = await read_text(request)

        response_text = await server.process_with_llm(None, text, dry_run=True)
        tool_calls = server._last_tool_calls
        return web.json_response(
            {
                "input": text,
                "response": response_text or "",
                "tool_calls": tool_calls,
            }
        )

    async def handle_reset_conversation(request):
        """Clear a site's dry-run conversation ({"site": …}, else the default), for tests (AD-047)."""
        await use_site(request, await _read_body(request))
        server.site.dry_run.history = []
        return web.json_response({"status": "ok"})

    async def handle_recording(request):
        path = server.recordings.path(request.match_info["recording"])
        if not path.is_file():
            raise web.HTTPNotFound()
        return web.FileResponse(path)

    async def handle_health(request):
        """Health check: status of STT, TTS, LLM, HA, and ESP.

        Returns 200 when all required components are ready, 503 otherwise.
        ESP and calendar are reported but optional (don't affect the verdict).
        LLM is checked by configuration only — not pinged, to avoid token cost.
        STT or TTS on the CPU though the server has CUDA is `degraded` but still 200: slower,
        working (AD-055).
        """
        # From the server, not `from voice_server import`: the service runs voice_server as __main__,
        # so that import loaded a second copy of the module
        LLM_API_KEY, LLM_MODEL, LLM_URL = server.llm_config

        stt_ready = server.stt is not None and server.stt.model is not None
        tts_ready = server.tts is not None and server.tts.engine is not None
        sites = {
            site.label: {
                "ha": site.ha_client is not None,
                "entities": len(site.ha_client.entities) if site.ha_client else 0,
                "esp": site.name in server.devices,
                "timers": len(site.timer_manager.timers),  # a phone's own too (AD-070)
            }
            for site in server.sites
        }
        # One home reachable is a working server: the other may be closed for the season
        ha_ready = any(home["ha"] for home in sites.values())

        is_cloud = bool(LLM_API_KEY)
        llm_ready = bool(LLM_MODEL) if is_cloud else bool(LLM_URL)

        components = {
            "stt": {
                "ready": stt_ready,
                "model": "parakeet-tdt-0.6b-v3",
                "device": server.stt.device if stt_ready else None,
            },
            "tts": {"ready": tts_ready, "engine": "kokoro-82m", "device": server.tts.device if tts_ready else None},
            "llm": {
                "ready": llm_ready,
                "mode": "cloud" if is_cloud else "local",
                "model": LLM_MODEL or None,
                "url": LLM_URL,
            },
            "ha": {
                "ready": ha_ready,
                "entities": sum(home["entities"] for home in sites.values()),
            },
            "esp": {
                "connected": bool(server.devices),
                "device": ", ".join(site.esp_host for site in server.sites if site.name in server.devices) or None,
                "count": len(server.devices),
            },
            "calendar": {"ready": server.calendar_client is not None},
        }

        required_ok = stt_ready and tts_ready and llm_ready and ha_ready
        lost_cuda = (stt_ready and onnx_gpu.lost_cuda(server.stt)) or (tts_ready and onnx_gpu.lost_cuda(server.tts))
        status = "ok" if required_ok and not lost_cuda else "degraded"
        body = {
            "status": status,
            "components": components,
            "timers": sum(home["timers"] for home in sites.values()),
            "sites": sites,
        }
        return web.json_response(body, status=200 if required_ok else 503)

    app.router.add_get("/", handle_web_ui)
    app.router.add_get("/manifest.webmanifest", handle_manifest)
    app.router.add_get("/sw.js", handle_service_worker)
    app.router.add_get("/health", handle_health)
    app.router.add_get("/review/open", handle_review_open)
    app.router.add_get("/api/exchanges", handle_get_exchanges)
    app.router.add_get("/api/sites", handle_sites)
    app.router.add_get("/api/state", handle_state)
    app.router.add_get("/api/listen", handle_listen)
    # The patterns keep the path inside the recordings folder
    app.router.add_get(r"/recordings/{recording:[\d-]+/[\d_]+\.wav}", handle_recording)
    app.router.add_post("/api/send", handle_web_send)
    app.router.add_post("/api/redo", handle_redo)
    app.router.add_post("/api/unlock", handle_unlock)
    app.router.add_post("/api/in-effect", handle_in_effect)
    app.router.add_post("/api/dry-run", handle_dry_run)
    app.router.add_post("/api/timers/cancel", handle_cancel_timer)
    app.router.add_post("/api/music", handle_music)
    app.router.add_get("/api/push", handle_push_key)
    app.router.add_post("/api/push", handle_push_subscribe)
    app.router.add_post("/api/reset-conversation", handle_reset_conversation)
    app.router.add_post("/test", handle_test_prompt)
    app.router.add_static("/static", STATIC_DIR)
