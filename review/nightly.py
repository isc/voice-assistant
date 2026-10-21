#!/usr/bin/env python3
"""The nightly review: Claude reads yesterday's interactions, fixes obvious bugs, emails Ivan the rest.

What it does, step by step: review/README.md (AD-074).

  review/nightly.py [YYYY-MM-DD]     review that day (default: yesterday, Paris time)
  FORCE_SEND=1 NO_DEPLOY=1 ...       email even when there's nothing; push and deploy nothing
"""
import datetime
import html
import json
import os
import re
import shutil
import subprocess
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from zoneinfo import ZoneInfo

from markdown_it import MarkdownIt  # python3-markdown-it

HERE = Path(__file__).resolve().parent
LIVE = Path("/opt/voice-assistant")
CLAUDE = Path.home() / ".local/bin/claude"
TRANSCRIBE = Path.home() / ".local/bin/transcrire"
MODEL = "claude-opus-5-5"
TIMEOUT = 40 * 60
CONFIG = Path.home() / ".config/voice-assistant-review"
CLONE = Path.home() / ".local/share/voice-assistant-review/repo"
STATE = Path.home() / ".local/state/voice-assistant-review"
REPO_URL = "https://github.com/isc/voice-assistant"
FROM = "Assistant vocal <assistant-vocal@arabesque.app>"
TO = "ivan.schneider@hey.com"
PARIS = ZoneInfo("Europe/Paris")
# The voice server's page that opens a Desktop session on a brief (web_ui.handle_review_open), on the
# tailnet: HEY doesn't follow a claude:// link
OPEN_BRIEF = "http://charras-server.tailcef78d.ts.net:8888/review/open"
# What a nightly fix may touch: the server's top-level code and docs, its web app; never this loop, the
# dependencies, the scripts or the firmware. The topics it must leave alone (locks, access control…) live
# in files it may touch: those limits are the prompt's alone.
ALLOWED = re.compile(r"[\w-]+\.(py|md)|static/[\w./-]+")
SCHEMA = json.dumps({
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "fixes": {"type": "array", "items": {"type": "string"}},
        "proposals": {"type": "array", "items": {
            "type": "object",
            "properties": {k: {"type": "string"} for k in ("title", "problem", "suggestion", "brief")},
            "required": ["title", "problem", "suggestion", "brief"],
        }},
    },
    "required": ["summary", "fixes", "proposals"],
})


def run(*cmd, cwd=None, check=True, timeout=600):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=check, timeout=timeout)


def git(*args, check=True):
    return run("git", "-C", str(CLONE), *args, check=check)


def commits_since(ref):
    return git("log", "--reverse", "--format=%H %s", f"{ref}..HEAD").stdout.splitlines()


def collect(day, out):
    """The day's data into out; returns the interactions, each with Whisper's transcript of its audio."""
    run("sudo", "-n", "/usr/bin/python3", str(HERE / "collect.py"), day, str(out), os.environ["USER"])
    interactions = json.loads((out / "interactions.json").read_text())
    wavs = sorted((out / "audio").glob("*.wav"))
    if wavs:
        # On the CPU: the GPU is Ollama's and the voice server's (AD-053)
        lines = run(str(TRANSCRIBE), "--cpu", *map(str, wavs), timeout=1800).stdout.splitlines()
        # One line per file, in order, each after a « == file » line when there are several
        texts = lines if len(wavs) == 1 else lines[1::2]
        heard = {f"{day}/{wav.name}": text.strip() for wav, text in zip(wavs, texts)}
        for entry in interactions:
            if entry.get("audio") in heard:
                entry["whisper"] = heard[entry["audio"]]
        (out / "interactions.json").write_text(json.dumps(interactions, ensure_ascii=False, indent=1))
    shutil.rmtree(out / "audio")
    return interactions


def previous(day):
    """What earlier reviews fixed or proposed, the last 30 days, for Claude not to say it again."""
    lines = []
    for past in sorted(STATE.glob("20*-*-*"))[-30:]:
        try:
            answer = json.loads((past / "review.json").read_text())
        except (OSError, ValueError):
            continue
        if past.name == day:
            continue
        lines += [f"- {past.name}, corrigé : {fix}" for fix in answer.get("fixes", [])]
        lines += [f"- {past.name}, proposé : {p['title']}" for p in answer.get("proposals", [])]
    return "\n".join(lines) or "(aucune revue précédente)"


def reset_clone():
    if not (CLONE / ".git").exists():
        CLONE.parent.mkdir(parents=True, exist_ok=True)
        run("git", "clone", "--quiet", REPO_URL, str(CLONE))
    git("fetch", "--quiet", "origin")
    git("checkout", "--quiet", "--force", "-B", "main", "origin/main")
    git("clean", "-ffdxq")
    return git("rev-parse", "HEAD").stdout.strip()


def claude(day, out):
    """Claude's review: its answer (structured_output) and session id, or an error string."""
    prompt = (HERE / "prompt.md").read_text().format(day=day, data=out)
    cmd = [str(CLAUDE), "-p", "--model", MODEL, "--restricted", "--tools", "Bash,Read,Grep,Glob,Edit,Write",
           "--settings", str(HERE / "settings.json"), "--permission-mode", "dontAsk", "--strict-mcp-config",
           "--add-dir", str(out), "--output-format", "json", "--json-schema", SCHEMA,
           "--append-system-prompt", "You run unattended from a timer: nobody can approve a refused command, "
           "so don't retry it, find another way. One plain command per Bash call, run in the working "
           "directory, without cd, git -C, shell variables, pipes or redirections."]
    env = {**os.environ, "ENABLE_CLAUDEAI_MCP_SERVERS": "false"}
    env.pop("CLAUDECODE", None)
    try:
        result = subprocess.run(cmd, input=prompt, capture_output=True, text=True, cwd=CLONE, env=env,
                                timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        return None, f"Claude n'a pas fini en {TIMEOUT // 60} min."
    (out / "claude.json").write_text(result.stdout)
    try:
        answer = json.loads(result.stdout)
    except ValueError:
        answer = {}
    review = answer.get("structured_output")
    if result.returncode or answer.get("is_error") or not review:
        return None, (answer.get("result") or result.stderr or result.stdout).strip()[-1500:]
    review["session_id"] = answer.get("session_id")
    review["cost"] = answer.get("total_cost_usd")
    for denial in answer.get("permission_denials") or []:
        print(f"refusé : {denial.get('tool_input', {}).get('command') or denial}", file=sys.stderr)
    return review, None


def get_json(url):
    try:
        with urllib.request.urlopen(url, timeout=5) as answer:
            return json.load(answer)
    except (OSError, ValueError):
        return None


def health():
    return get_json("http://localhost:8888/health")


def busy():
    """Why the server can't restart now, or None: a restart cuts the Spotify stream it serves (radios
    play from their own servers) and loses its timers."""
    music = get_json("http://127.0.0.1:3678/status") or {}
    if music and not music.get("stopped") and not music.get("paused"):
        return "de la musique passe"
    state = health()
    if state is None:
        return "le serveur ne répond pas"
    if state.get("timers"):
        return "un minuteur tourne"


@dataclass
class Shipped:
    commits: list = field(default_factory=list)
    pushed: bool = False
    warning: bool = False
    notes: list = field(default_factory=list)


def ship(base):
    """Check, push and deploy Claude's commits."""
    commits = commits_since(base)
    if not commits:
        return Shipped()
    outside = [f for f in git("diff", "--name-only", base, "HEAD").stdout.splitlines() if not ALLOWED.fullmatch(f)]
    if outside:
        return Shipped(commits, warning=True, notes=[
            f"⚠️ Pas poussé : ces commits touchent {', '.join(outside)}, hors de ce qu'une nuit peut changer."])
    # main may have moved during the review
    git("fetch", "--quiet", "origin")
    if git("rebase", "--quiet", "origin/main", check=False).returncode:
        git("rebase", "--abort", check=False)
        return Shipped(commits, warning=True, notes=["⚠️ Pas poussé : conflit avec `main`."])
    commits = commits_since("origin/main")
    tests = run(str(LIVE / "venv/bin/python"), "-m", "unittest", "test_unit", cwd=CLONE, check=False)
    if tests.returncode:
        return Shipped(commits, warning=True, notes=[
            f"⚠️ Pas poussé : les tests échouent.\n\n```\n{tests.stderr[-1500:]}\n```"])
    if os.environ.get("NO_DEPLOY") == "1":
        return Shipped(commits, notes=["Pas poussé (NO_DEPLOY)."])
    git("push", "--quiet", "origin", "HEAD:main")
    shipped = Shipped(commits, pushed=True, notes=["Poussé sur `main`."])
    reason = busy()
    if reason:
        shipped.notes.append(f"Pas déployé : {reason}. À faire : `sudo git -C /opt/voice-assistant pull` "
                             "puis `sudo systemctl restart voice-assistant`.")
        return shipped
    deployed = run("sudo", "-n", "git", "-C", str(LIVE), "rev-parse", "HEAD").stdout.strip()  # root-owned
    run("sudo", "-n", "git", "-C", str(LIVE), "pull", "--quiet", "--ff-only")
    if restart():
        shipped.notes.append("Déployé, le serveur a redémarré sans erreur.")
        return shipped
    run("sudo", "-n", "git", "-C", str(LIVE), "reset", "--quiet", "--hard", deployed)
    back = "il tourne à nouveau" if restart() else "il ne répond toujours pas, à voir d'urgence"
    shipped.warning = True
    shipped.notes.append(f"⚠️ Déployé, mais le serveur n'a pas redémarré : /opt revenu à {deployed[:7]}, {back}. "
                         "Les commits restent sur `main` : à annuler ou à corriger.")
    return shipped


def restart():
    """Restart the server; whether it is healthy within 5 minutes."""
    run("sudo", "-n", "systemctl", "restart", "voice-assistant")
    return wait_healthy()


def wait_healthy(seconds=300):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        state = health()
        if state and state.get("status") in ("ok", "degraded"):
            return True
        time.sleep(5)
    return False


def slug(title):
    ascii_title = unicodedata.normalize("NFKD", title.lower()).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", ascii_title)[:40].strip("-")


def report(day, out, interactions, review, error, shipped):
    """The email's Markdown, each proposal's brief written to proposals/ for its Desktop session to read."""
    parts = []
    if error:
        parts.append(f"⚠️ La revue a échoué :\n\n```\n{error}\n```")
    if review:
        parts.append(review["summary"])
    if shipped.commits:
        fixes = "\n".join(f"- {fix}" for fix in review.get("fixes", []))
        # An unpushed commit's link would be a 404
        links = " · ".join((f"[`{sha[:7]}`]({REPO_URL}/commit/{sha})" if shipped.pushed else f"`{sha[:7]}`")
                           + f" {subject}" for sha, subject in (c.split(" ", 1) for c in shipped.commits))
        parts.append("\n\n".join(filter(None, ["## Corrigé", fixes, f"Commits : {links}", "\n".join(shipped.notes)])))
    proposals = review.get("proposals", [])
    if proposals:
        (out / "proposals").mkdir(exist_ok=True)
    for n, proposal in enumerate(proposals, 1):
        brief = out / "proposals" / f"{n}-{slug(proposal['title'])}.md"
        brief.write_text(f"# {proposal['title']}\n\n{proposal['brief']}\n")
        link = f"{OPEN_BRIEF}?{urllib.parse.urlencode({'brief': str(brief.relative_to(STATE))})}"
        parts.append(f"## {n}. {proposal['title']}\n\n{proposal['problem']}\n\n**Pistes.** {proposal['suggestion']}"
                     f"\n\n[Ouvrir dans Claude Code]({link}) · ou dans une session : `Lis {brief}`")
    if review.get("session_id"):
        parts.append(f"La revue elle-même se reprend sur charras : `cd {CLONE} && claude --resume "
                     f"{review['session_id']}`.")
    stats = f"{len(interactions)} requêtes"
    if review.get("cost") is not None:
        stats += f", revue {review['cost']:.2f} $"
    footer = (f"voice-assistant-review sur charras, journée du {day} ({stats}) · données : {out} · "
              "arrêter : systemctl --user disable --now voice-assistant-review.timer")
    return "\n\n".join(parts), footer


def to_html(markdown):
    return MarkdownIt("commonmark", {"breaks": True, "html": False}).render(markdown)


def send(subject, text, body):
    key = (CONFIG / "resend-api-key").read_text().strip()
    request = urllib.request.Request("https://api.resend.com/emails", json.dumps(
        {"from": FROM, "to": [TO], "subject": subject, "text": text, "html": body}).encode(),
        {"Authorization": f"Bearer {key}", "Content-Type": "application/json", "User-Agent": "voice-assistant-review"})
    try:
        with urllib.request.urlopen(request, timeout=30) as answer:
            return json.load(answer)["id"]
    except urllib.error.HTTPError as ex:
        sys.exit(f"Resend a refusé l'email : {ex.code} {ex.read().decode(errors='replace')}")


def main():
    os.umask(0o077)  # the family's words
    yesterday = datetime.datetime.now(PARIS).date() - datetime.timedelta(days=1)
    day = sys.argv[1] if len(sys.argv) > 1 else yesterday.isoformat()
    out = STATE / day
    out.mkdir(parents=True, exist_ok=True)
    out.chmod(0o700)
    interactions = collect(day, out)
    if not interactions:
        print(f"Aucune requête le {day} : pas de revue.")
        return
    (out / "previous.md").write_text(previous(day) + "\n")
    base = reset_clone()
    review, error = claude(day, out)
    if review:
        (out / "review.json").write_text(json.dumps(review, ensure_ascii=False, indent=1))
    review = review or {}
    shipped = ship(base)
    markdown, footer = report(day, out, interactions, review, error, shipped)
    fixes, proposals = len(shipped.commits), len(review.get("proposals", []))
    if not fixes and not proposals and not error and os.environ.get("FORCE_SEND") != "1":
        print(f"Rien à signaler le {day} : pas d'email.\n{markdown}")
        return
    counts = [f"{fixes} correction{'s' * (fixes > 1)}"] if fixes else []
    counts += [f"{proposals} proposition{'s' * (proposals > 1)}"] if proposals else []
    flag = "⚠️ " if error or shipped.warning else ""
    subject = f"{flag}Assistant vocal, {day[8:]}/{day[5:7]} : {', '.join(counts) or 'rien à signaler'}"
    body = ('<div style="font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;font-size:15px;'
            'line-height:1.45;color:#222">' + to_html(markdown)
            + f'<p style="margin-top:24px;color:#888;font-size:12px">{html.escape(footer)}</p></div>')
    (out / "email.html").write_text(body)
    text = f"{markdown}\n\n--\n{footer}\n"
    print(f"Email envoyé (Resend {send(subject, text, body)}) : {subject}")
    sys.exit(1 if error else 0)


if __name__ == "__main__":
    main()
