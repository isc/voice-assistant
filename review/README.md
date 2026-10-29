# Nightly review

Every night at 3:30 (Paris), Claude Code reads the previous day's interactions, fixes the bugs it is
sure of, and emails Ivan the rest as proposals, each with a link that opens a Claude Code Desktop
session on it (AD-074). Then the model-backed tests run, and a turn that regressed is in the email too
(AD-085).

| File | Role |
|---|---|
| `nightly.py` | The run: collect, transcribe again, review, push and deploy, the model's tests, email |
| `collect.py` | As root: the day's exchanges and recordings merged, the journal, the WAVs |
| `prompt.md` | What Claude is asked: what to look for, what it may fix, how to answer |
| `settings.json` | Claude's permissions: file tools, git without push, the unit tests |
| `voice-assistant-review.{service,timer}` | systemd user units, run as `ivan` |

## What it does

1. `collect.py` (`sudo`, from `/opt/voice-assistant`: never the clone Claude edits) writes into
   `~/.local/state/voice-assistant-review/<day>/` the day's requests (`exchanges.json`'s entries
   merged with `recordings/<day>/*.json`), the journal without the HTTP and streaming noise nor the dry
   runs (the server starts their lines with `[dry run]`: the tests', a development session's), and the WAVs.
2. Whisper large-v3 (`~/.local/bin/transcrire`) transcribes
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
   The brief's path is in the email too, to paste. No fix, no proposal and no regression: no email.
6. Before the email, once Claude is done and the deploy over, so that they have the GPU to themselves:
   `test_hard_cases.py` (dry runs on the live server) and `test_scenarios.py` (fake homes, the live
   model), 3 runs each, from `/opt` with its venv: what runs, a fix of the night included. Skipped, and
   said so, unless `/health` says `ok`. Each turn's passes out of 3 go into `<day>/tests.json`, its runs
   into `test_*.jsonl` (the scripts' `-o`), compared with the last nights' (AD-085): a turn
   is flagged when it loses 2 passes, or its last one; one of `UNSTABLE` (`nightly.py`), which moves
   from run to run, only when its fall holds two nights; a turn new tonight is listed as new. A
   regression, or a test that did not run to its end, makes an email; the totals ride along in one sent
   anyway. Also on a night with no request, which gets no review.

Each day's folder keeps `interactions.json`, `journal.log`, `claude.json` (Claude's full answer),
`review.json`, `proposals/`, `tests.json` (each turn's passes, the commit, the model),
`test_hard_cases.jsonl`, `test_scenarios.jsonl` (each run of each turn) and `email.html`. The next nights read `review.json` of the
last 30 to not propose the same thing again.

## Install

```bash
ln -sf /opt/voice-assistant/review/voice-assistant-review.{service,timer} ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now voice-assistant-review.timer
```

Whisper runs on the GPU only when the unit names a card in `CUDA_VISIBLE_DEVICES`, else on the CPU:
left to itself, CUDA would take the first card, Ollama's and the voice server's. On charras, a
drop-in gives it the bottom card (home-infra `charras/SETUP.md` §23).

`ivan` needs `sudo` without a password (collect, deploy), `gh` logged in (push), and Claude Code's login.

## Run by hand

```bash
/opt/voice-assistant/review/nightly.py 2026-10-20               # that day, pushes and deploys
NO_DEPLOY=1 FORCE_SEND=1 /opt/voice-assistant/review/nightly.py # yesterday, nothing pushed, always an email
/opt/voice-assistant/review/nightly.py --tests                  # the model's tests alone, into yesterday's folder
journalctl --user -u voice-assistant-review                    # last runs
```
