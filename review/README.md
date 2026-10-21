# Nightly review

Every night at 3:30 (Paris), Claude Code reads the previous day's interactions, fixes the bugs it is
sure of, and emails Ivan the rest as proposals, each with a link that opens a Claude Code Desktop
session on it (AD-074).

| File | Role |
|---|---|
| `nightly.py` | The run: collect, transcribe again, review, push and deploy, email |
| `collect.py` | As root: the day's exchanges and recordings merged, the journal, the WAVs |
| `prompt.md` | What Claude is asked: what to look for, what it may fix, how to answer |
| `settings.json` | Claude's permissions: file tools, git without push, the unit tests |
| `voice-assistant-review.{service,timer}` | systemd user units, run as `ivan` |

## What it does

1. `collect.py` (`sudo`, from `/opt/voice-assistant`: never the clone Claude edits) writes into
   `~/.local/state/voice-assistant-review/<day>/` the day's requests (`exchanges.json`'s entries
   merged with `recordings/<day>/*.json`), the journal without the HTTP and streaming noise, and the WAVs.
2. Whisper large-v3 (`~/.local/bin/transcrire --cpu`: the GPU is Ollama's and the server's) transcribes
   each WAV again, beside Parakeet's transcript; the WAVs are then deleted.
3. Claude (Opus, `claude -p --restricted`, `dontAsk`) works in a clone of `main`
   (`~/.local/share/voice-assistant-review/repo`, reset each night): one commit per obvious fix, with
   its test, three at most; never locks, access control, secrets, recordings, dependencies, firmware,
   or the assistant's behaviour where it can be discussed. Its answer: a summary, its fixes, its proposals.
4. The script checks the commits: only `*.py`, `*.md` and `static/` outside `review/`, rebased on
   `main`, unit tests passing. Then pushes to `main` and deploys (`git pull` in `/opt`, restart), unless
   go-librespot plays or a timer runs. The server not healthy within 5 minutes: `/opt` goes back to the
   commit it ran, restarted, and the email says so with ⚠️.
5. One email from `assistant-vocal@arabesque.app` (Resend key `voice-assistant-review`, send only,
   `~/.config/voice-assistant-review/resend-api-key`). Each proposal's brief is written to
   `<day>/proposals/<n>-<slug>.md`; its link goes to the voice server's `/review/open?brief=…`, on the
   tailnet, which sends the browser to `claude://code/new?ssh_host=charras&ssh_folder=…&q=…`: a Desktop
   session on charras whose prompt asks to read it. HEY shows a `claude://` link but doesn't follow it.
   The brief's path is in the email too, to paste. No fix and no proposal: no email.

Each day's folder keeps `interactions.json`, `journal.log`, `claude.json` (Claude's full answer),
`review.json`, `proposals/` and `email.html`. The next nights read `review.json` of the last 30 to
not propose the same thing again.

## Install

```bash
ln -sf /opt/voice-assistant/review/voice-assistant-review.{service,timer} ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now voice-assistant-review.timer
```

`ivan` needs `sudo` without a password (collect, deploy), `gh` logged in (push), and Claude Code's login.

## Run by hand

```bash
/opt/voice-assistant/review/nightly.py 2026-10-20               # that day, pushes and deploys
NO_DEPLOY=1 FORCE_SEND=1 /opt/voice-assistant/review/nightly.py # yesterday, nothing pushed, always an email
journalctl --user -u voice-assistant-review                    # last runs
```
