"""Web UI and HTTP routes for the voice assistant."""

import json
import logging
import time

from aiohttp import web

logger = logging.getLogger(__name__)

WEB_UI_HTML = """<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Voice Assistant</title>
<style>
  :root {
    --bg: #fff; --bg2: #f5f5f7; --bg3: #e8e8ed;
    --fg: #1d1d1f; --fg2: #6e6e73; --fg3: #aeaeb2;
    --accent: #007aff; --accent2: #34c759; --border: #d2d2d7;
    --card: #fff; --card-border: #e5e5ea;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #1c1c1e; --bg2: #2c2c2e; --bg3: #3a3a3c;
      --fg: #f5f5f7; --fg2: #98989d; --fg3: #636366;
      --accent: #0a84ff; --accent2: #30d158; --border: #38383a;
      --card: #2c2c2e; --card-border: #38383a;
    }
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, system-ui, sans-serif; background: var(--bg); color: var(--fg); height: 100vh; display: flex; flex-direction: column; }
  .container { width: 100%; max-width: 640px; margin: 0 auto; display: flex; flex-direction: column; height: 100%; }
  header { padding: 14px 20px; background: var(--bg2); border-bottom: 1px solid var(--border); display: flex; align-items: center; gap: 10px; }
  header h1 { font-size: 17px; font-weight: 600; }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--accent2); }
  #exchanges { flex: 1; overflow-y: auto; padding: 12px 20px; display: flex; flex-direction: column; gap: 10px; }
  .exchange { background: var(--card); border: 1px solid var(--card-border); border-radius: 10px; padding: 12px 14px; }
  .meta { font-size: 11px; color: var(--fg2); margin-bottom: 6px; display: flex; align-items: center; gap: 8px; }
  .badge { padding: 1px 6px; border-radius: 4px; font-size: 10px; font-weight: 600; text-transform: uppercase; }
  .badge.voice { background: var(--accent); color: #fff; }
  .badge.web { background: var(--accent2); color: #fff; }
  .timings { margin-left: auto; font-family: SF Mono, Menlo, monospace; font-size: 10px; color: var(--fg3); }
  .input-line { color: var(--fg2); margin-bottom: 4px; font-size: 14px; }
  .response-line { color: var(--fg); font-weight: 500; font-size: 15px; }
  .actions { margin-top: 8px; display: flex; gap: 6px; }
  .actions button { background: var(--bg2); border: 1px solid var(--border); color: var(--fg2); padding: 3px 10px; border-radius: 6px; cursor: pointer; font-size: 11px; }
  .actions button:hover { border-color: var(--accent); color: var(--accent); }
  .debug { display: none; margin-top: 8px; padding: 8px 10px; background: var(--bg2); border-radius: 6px; font-family: SF Mono, Menlo, monospace; font-size: 11px; color: var(--fg2); line-height: 1.5; white-space: pre-wrap; word-break: break-all; }
  .debug.open { display: block; }
  .debug .label { color: var(--fg3); }
  .debug .val { color: var(--fg); }
  .debug .tc { margin-top: 4px; padding: 4px 6px; background: var(--bg3); border-radius: 4px; }
  .empty { color: var(--fg3); text-align: center; margin-top: 40px; font-size: 14px; }
  #input-bar { padding: 10px 20px 12px; background: var(--bg2); border-top: 1px solid var(--border); display: flex; gap: 8px; }
  #input-bar input { flex: 1; background: var(--bg); border: 1px solid var(--border); color: var(--fg); padding: 9px 12px; border-radius: 8px; font-size: 15px; outline: none; }
  #input-bar input:focus { border-color: var(--accent); box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent) 20%, transparent); }
  #input-bar button { background: var(--accent); color: #fff; border: none; padding: 9px 18px; border-radius: 8px; font-weight: 600; cursor: pointer; font-size: 15px; }
  #input-bar button:disabled { opacity: 0.4; cursor: not-allowed; }
</style>
</head>
<body>
<div class="container">
<header>
  <div class="dot"></div>
  <h1>Voice Assistant</h1>
</header>
<div id="exchanges"><div class="empty">Aucun echange pour le moment</div></div>
<div id="input-bar">
  <input type="text" id="cmd" placeholder="Commande..." autocomplete="off">
  <button id="send-btn">Envoyer</button>
</div>
</div>
<script>
const ex = document.getElementById('exchanges');
const cmd = document.getElementById('cmd');
const btn = document.getElementById('send-btn');
let lastHash = '';

function esc(s) { const d = document.createElement('div'); d.textContent = s || ''; return d.innerHTML; }

function fmtTimings(t) {
  if (!t || !Object.keys(t).length) return '';
  const parts = [];
  if (t.stt != null) parts.push('STT ' + t.stt + 's');
  if (t.llm != null) parts.push('LLM ' + t.llm + 's');
  if (t.tts != null) parts.push('TTS ' + t.tts + 's');
  if (t.total != null) parts.push('= ' + t.total + 's');
  return parts.join(' | ');
}

function fmtDebug(e) {
  let s = '';
  if (e.timings && Object.keys(e.timings).length) {
    s += '<span class="label">Latence:</span> <span class="val">' + fmtTimings(e.timings) + '</span>\\n';
  }
  if (e.tool_calls && e.tool_calls.length) {
    s += '<span class="label">Tool calls:</span>\\n';
    e.tool_calls.forEach(tc => {
      s += '<div class="tc">' + esc(tc.function) + '(' + esc(JSON.stringify(tc.args)) + ')';
      if (tc.result) s += '\\n  -> ' + esc(tc.result);
      if (tc.error) s += '\\n  !! ' + esc(tc.error);
      s += '</div>';
    });
  }
  return s;
}

function render(data) {
  if (!data.length) { ex.innerHTML = '<div class="empty">Aucun echange pour le moment</div>'; return; }
  ex.innerHTML = data.map((e, i) => {
    const hasDebug = (e.timings && Object.keys(e.timings).length) || (e.tool_calls && e.tool_calls.length);
    return `<div class="exchange">
      <div class="meta">
        <span class="badge ${e.source}">${e.source}</span>
        <span>${e.timestamp}</span>
        ${e.timings && e.timings.total ? '<span class="timings">' + e.timings.total + 's</span>' : ''}
      </div>
      <div class="input-line">${esc(e.input)}</div>
      <div class="response-line">${esc(e.response)}</div>
      <div class="actions">
        ${e.tts_url ? '<button onclick="playAudio(\\'' + e.tts_url + '\\')">Ecouter</button>' : ''}
        ${hasDebug ? '<button onclick="toggleDebug(' + i + ')">Debug</button>' : ''}
      </div>
      ${hasDebug ? '<div class="debug" id="dbg-' + i + '">' + fmtDebug(e) + '</div>' : ''}
    </div>`;
  }).join('');
  ex.scrollTop = ex.scrollHeight;
}

function toggleDebug(i) {
  const el = document.getElementById('dbg-' + i);
  if (el) el.classList.toggle('open');
}

let currentAudio = null;
function playAudio(url) {
  if (currentAudio) currentAudio.pause();
  currentAudio = new Audio(url);
  currentAudio.play();
}

async function refresh() {
  try {
    const r = await fetch('/api/exchanges');
    const data = await r.json();
    const hash = JSON.stringify(data.map(e => e.timestamp));
    if (hash !== lastHash) { lastHash = hash; render(data); }
  } catch(e) {}
}

async function send() {
  const text = cmd.value.trim();
  if (!text) return;
  cmd.value = '';
  btn.disabled = true;
  try {
    await fetch('/api/send', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({text}) });
    await refresh();
  } catch(e) { alert('Erreur: ' + e.message); }
  btn.disabled = false;
  cmd.focus();
}

btn.addEventListener('click', send);
cmd.addEventListener('keydown', e => { if (e.key === 'Enter') send(); });
refresh();
setInterval(refresh, 3000);
cmd.focus();
</script>
</body>
</html>
"""


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
        }
        self.entries.append(entry)
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries :]
        self._save()


def setup_routes(app, server):
    """Register web UI routes on an aiohttp app."""

    async def handle_web_ui(request):
        return web.Response(text=WEB_UI_HTML, content_type="text/html")

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
        server.exchange_log.add("web", text, text, response_text, tts_url)
        return web.json_response(result)

    app.router.add_get("/", handle_web_ui)
    app.router.add_get("/api/exchanges", handle_get_exchanges)
    app.router.add_post("/api/send", handle_web_send)
    app.router.add_post("/test", handle_test_prompt)
