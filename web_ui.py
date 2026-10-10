"""Web UI and HTTP routes for the voice assistant."""

import ipaddress
import json
import logging
import time
from pathlib import Path

from aiohttp import web

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
# Tailscale's address range: the web UI is reached through it, while :8888 also answers on the LAN
_TAILNET = ipaddress.ip_network("100.64.0.0/10")
# What speakers fetch from the LAN: TTS replies and the Spotify stream (and /health, for checks)
_LAN_PATHS = ("/tts/", "/spotify/", "/health")


def is_trusted_client(remote: str | None) -> bool:
    """This machine and the tailnet. The web UI shows transcripts, plays recordings and runs commands."""
    try:
        address = ipaddress.ip_address(remote or "")
    except ValueError:
        return False
    return address.is_loopback or address in _TAILNET


def _json_error(error: type[web.HTTPException], message: str) -> web.HTTPException:
    return error(text=json.dumps({"error": message}), content_type="application/json")


@web.middleware
async def lan_sees_only_speaker_paths(request, handler):
    if request.path.startswith(_LAN_PATHS) or is_trusted_client(request.remote):
        return await handler(request)
    raise web.HTTPForbidden(text="The web UI is reachable through Tailscale only")


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
        except Exception:
            pass
        return []

    def _save(self):
        try:
            self.path.write_text(json.dumps(self.entries, ensure_ascii=False, indent=1))
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
    app.middlewares.append(lan_sees_only_speaker_paths)

    async def read_text(request) -> str:
        """The prompt of a text request; the request then serves the site it names, if any: {"text": …, "site": "charras"}."""
        try:
            body = await request.json()
        except Exception:
            body = {"text": (await request.text()).strip()}
        text = body.get("text", "")
        if not text:
            raise _json_error(web.HTTPBadRequest, "missing 'text' field")
        if body.get("site"):
            site = server.site_named(body["site"])
            if not site:
                raise _json_error(web.HTTPNotFound, f"unknown site {body['site']}")
            server.site = site  # for this request only: a conversation at another site goes on undisturbed
        return text

    async def handle_web_ui(request):
        return web.Response(text=(STATIC_DIR / "index.html").read_text(), content_type="text/html")

    async def handle_manifest(request):
        return web.Response(
            text=(STATIC_DIR / "manifest.webmanifest").read_text(),
            content_type="application/manifest+json",
        )

    async def handle_service_worker(request):
        # Served from root so its scope covers "/" (a /static/sw.js could only
        # control /static/*). Sent no-cache so SW updates propagate promptly.
        return web.Response(
            text=(STATIC_DIR / "sw.js").read_text(),
            content_type="application/javascript",
            headers={"Cache-Control": "no-cache", "Service-Worker-Allowed": "/"},
        )

    async def handle_get_exchanges(request):
        return web.json_response(server.exchange_log.entries)

    async def handle_web_send(request):
        text = await read_text(request)

        timings = {}
        t0 = time.time()
        server._last_tool_calls = []
        response_text = await server.process_with_llm(None, text)
        timings["llm"] = round(time.time() - t0, 2)
        if not response_text:
            server.exchange_log.add(
                "web",
                text,
                text,
                "[Erreur LLM: pas de réponse]",
                timings=timings,
                conversation_id=server.conversation_id,
                site=server.site.name,
            )
            return web.json_response({"error": "LLM returned no response"}, status=500)

        t0 = time.time()
        tts_url = await server.text_to_speech_file(text=response_text)
        timings["tts"] = round(time.time() - t0, 2)
        timings["total"] = round(sum(timings.values()), 2)

        tool_calls = server._last_tool_calls
        server._last_tool_calls = []
        server.exchange_log.add(
            "web",
            text,
            text,
            response_text,
            tts_url,
            timings=timings,
            tool_calls=tool_calls,
            conversation_id=server.conversation_id,
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
                conversation_id=server.conversation_id,
                site=server.site.name,
            )
            return web.json_response({"error": "LLM returned no response"}, status=500)

        tts_url = await server.text_to_speech_file(text=response_text)
        result = {"input": text, "response": response_text, "tts_url": tts_url}
        logger.info(f"[TEST] Result: {json.dumps(result, ensure_ascii=False)}")
        server.exchange_log.add(
            "web", text, text, response_text, tts_url, conversation_id=server.conversation_id, site=server.site.name
        )
        return web.json_response(result)

    async def handle_dry_run(request):
        """LLM-only endpoint: produces tool calls without executing them."""
        text = await read_text(request)

        server._last_tool_calls = []
        response_text = await server.process_with_llm(None, text, dry_run=True)
        tool_calls = server._last_tool_calls
        server._last_tool_calls = []
        return web.json_response(
            {
                "input": text,
                "response": response_text or "",
                "tool_calls": tool_calls,
            }
        )

    async def handle_reset_conversation(request):
        """Clear conversation history. Used by tests for isolation."""
        server.conversation_history = []
        server.conversation_id = None
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
        from voice_server import LLM_API_KEY, LLM_MODEL, LLM_URL

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
    # The patterns keep the path inside the recordings folder
    app.router.add_get(r"/recordings/{recording:[\d-]+/[\d_]+\.wav}", handle_recording)
    app.router.add_post("/api/send", handle_web_send)
    app.router.add_post("/api/dry-run", handle_dry_run)
    app.router.add_post("/api/reset-conversation", handle_reset_conversation)
    app.router.add_post("/test", handle_test_prompt)
    app.router.add_static("/static", STATIC_DIR)
