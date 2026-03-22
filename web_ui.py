"""Web UI and HTTP routes for the voice assistant."""

import json
import logging
import time
from pathlib import Path

from aiohttp import web

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


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
        }
        self.entries.append(entry)
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries :]
        self._save()


def setup_routes(app, server):
    """Register web UI routes on an aiohttp app."""

    async def handle_web_ui(request):
        return web.Response(text=(STATIC_DIR / "index.html").read_text(), content_type="text/html")

    async def handle_get_exchanges(request):
        return web.json_response(server.exchange_log.entries)

    async def handle_web_send(request):
        try:
            body = await request.json()
            text = body.get("text", "")
        except Exception:
            text = (await request.text()).strip()

        if not text:
            return web.json_response({"error": "missing 'text' field"}, status=400)

        timings = {}
        t0 = time.time()
        server._last_tool_calls = []
        response_text = await server.process_with_llm(None, text)
        timings["llm"] = round(time.time() - t0, 2)
        if not response_text:
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
        try:
            body = await request.json()
            text = body.get("text", "")
        except Exception:
            text = (await request.text()).strip()

        if not text:
            return web.json_response({"error": "missing 'text' field"}, status=400)

        logger.info(f'[TEST] Input: "{text}"')
        response_text = await server.process_with_llm(None, text)
        if not response_text:
            return web.json_response({"error": "LLM returned no response"}, status=500)

        tts_url = await server.text_to_speech_file(text=response_text)
        result = {"input": text, "response": response_text, "tts_url": tts_url}
        logger.info(f"[TEST] Result: {json.dumps(result, ensure_ascii=False)}")
        server.exchange_log.add("web", text, text, response_text, tts_url, conversation_id=server.conversation_id)
        return web.json_response(result)

    app.router.add_get("/", handle_web_ui)
    app.router.add_get("/api/exchanges", handle_get_exchanges)
    app.router.add_post("/api/send", handle_web_send)
    app.router.add_post("/test", handle_test_prompt)
    app.router.add_static("/static", STATIC_DIR)
