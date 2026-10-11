"""Web UI and HTTP routes for the voice assistant."""

import hmac
import ipaddress
import json
import logging
import os
import time
from pathlib import Path

from aiohttp import web

from sites import Site, site_at_address

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
# Tailscale's address range: the web UI is reached through it, while :8888 also answers on the LAN
_TAILNET = ipaddress.ip_network("100.64.0.0/10")
# What speakers fetch from the LAN: TTS replies and the Spotify stream (and /health, for checks)
_LAN_PATHS = ("/tts/", "/spotify/", "/health")
# Tailscale Funnel's proxy runs on this machine: the internet arrives from loopback, told apart by these
_PROXY_HEADERS = ("X-Forwarded-For", "Tailscale-Funnel-Request")
# Fetched without the cookie when Chrome installs the app, the icons by Google's WebAPK server. static/
# holds only the shell and its icons: data stays behind api/, tts/, recordings/ and health
_INSTALL_PATHS = ("/manifest.webmanifest", "/static/")
KEY_COOKIE = "assistant_key"


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


def _json_error(error: type[web.HTTPException], message: str) -> web.HTTPException:
    return error(text=json.dumps({"error": message}), content_type="application/json")


class ExchangeLog:
    """Persisted exchange log for the web UI."""

    def __init__(self, path, max_entries=50):
        self.path = path
        self.max_entries = max_entries
        self.entries = self._load()

    def _load(self):
        try:
            if self.path.exists():
                data = json.loads(self.path.read_text())
                return data[-self.max_entries :]
        except Exception as e:
            logger.warning(f"Exchange log unreadable, starting empty: {e}")
        return []

    def _save(self):
        """Written whole to a temporary file then renamed: a crash mid-write left invalid JSON, read as
        empty and overwritten. Mode 600: transcripts and calendar answers, kept from other users."""
        tmp = self.path.with_name(self.path.name + ".tmp")
        try:
            with open(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as f:
                f.write(json.dumps(self.entries, ensure_ascii=False, indent=1))
            os.replace(tmp, self.path)
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
    ):
        entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
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
        }
        self.entries.append(entry)
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries :]
        self._save()


def setup_routes(app, server):
    """Register web UI routes on an aiohttp app."""
    # Empty: nothing answers through Funnel
    app.middlewares.append(access_control(os.environ.get("WEB_UI_KEY", "")))

    async def read_text(request) -> str:
        """The prompt of a text request; the request then serves the site it names, if any.

        {"text": …, "site": "charras"}"""
        try:
            body = await request.json()
        except Exception:
            body = {"text": (await request.text()).strip()}
        if not isinstance(body, dict):
            body = {}
        text = body.get("text", "")
        if not isinstance(text, str) or not text.strip():
            raise _json_error(web.HTTPBadRequest, "missing 'text' field")
        await use_site(request, body)
        server.new_request()
        return text

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
        if not (forwarded := request.headers.get("X-Forwarded-For")):
            return None
        # The last address is the one Funnel's proxy added; any before it, the client wrote
        return await site_at_address(server.sites, forwarded.rsplit(",", 1)[-1].strip())

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
        return web.json_response(server.exchange_log.entries)

    async def handle_web_send(request):
        text = await read_text(request)

        timings = {}
        t0 = time.time()
        response_text = await server.process_with_llm(None, text)
        timings["llm"] = round(time.time() - t0, 2)
        if not response_text:
            server.exchange_log.add(
                "web",
                text,
                text,
                "[Erreur LLM: pas de réponse]",
                timings=timings,
                conversation_id=server.conversation.id,
                site=server.site.name,
            )
            return web.json_response({"error": "LLM returned no response"}, status=500)

        t0 = time.time()
        tts_url = await server.text_to_speech_file(text=response_text)
        timings["tts"] = round(time.time() - t0, 2)
        timings["total"] = round(sum(timings.values()), 2)

        tool_calls = server._last_tool_calls
        server.exchange_log.add(
            "web",
            text,
            text,
            response_text,
            tts_url,
            timings=timings,
            tool_calls=tool_calls,
            conversation_id=server.conversation.id,
            site=server.site.name,
        )
        return web.json_response(
            {
                "input": text,
                "response": response_text,
                "tts_url": tts_url,
                "timings": timings,
                "tool_calls": tool_calls,
            }
        )

    async def handle_test_prompt(request):
        text = await read_text(request)

        logger.info(f'[TEST] Input: "{text}"')
        response_text = await server.process_with_llm(None, text)
        if not response_text:
            server.exchange_log.add(
                "web",
                text,
                text,
                "[Erreur LLM: pas de réponse]",
                conversation_id=server.conversation.id,
                site=server.site.name,
            )
            return web.json_response({"error": "LLM returned no response"}, status=500)

        tts_url = await server.text_to_speech_file(text=response_text)
        result = {"input": text, "response": response_text, "tts_url": tts_url}
        logger.info(f"[TEST] Result: {json.dumps(result, ensure_ascii=False)}")
        server.exchange_log.add(
            "web", text, text, response_text, tts_url, conversation_id=server.conversation.id, site=server.site.name
        )
        return web.json_response(result)

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
        try:
            body = await request.json()
        except Exception:
            body = {}
        await use_site(request, body if isinstance(body, dict) else {})
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
                "timers": len(site.timer_manager.get_timers()),
            }
            for site in server.sites
        }
        # One home reachable is a working server: the other may be closed for the season
        ha_ready = any(home["ha"] for home in sites.values())

        is_cloud = bool(LLM_API_KEY)
        llm_ready = bool(LLM_MODEL) if is_cloud else bool(LLM_URL)

        components = {
            "stt": {"ready": stt_ready, "model": "parakeet-tdt-0.6b-v3"},
            "tts": {"ready": tts_ready, "engine": "kokoro-82m"},
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
        status = "ok" if required_ok else "degraded"
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
    app.router.add_get("/api/exchanges", handle_get_exchanges)
    app.router.add_get("/api/sites", handle_sites)
    # The patterns keep the path inside the recordings folder
    app.router.add_get(r"/recordings/{recording:[\d-]+/[\d_]+\.wav}", handle_recording)
    app.router.add_post("/api/send", handle_web_send)
    app.router.add_post("/api/dry-run", handle_dry_run)
    app.router.add_post("/api/reset-conversation", handle_reset_conversation)
    app.router.add_post("/test", handle_test_prompt)
    app.router.add_static("/static", STATIC_DIR)
