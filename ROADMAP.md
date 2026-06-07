# Roadmap

Features and improvements planned for the voice assistant.

## Conversation flow

### ~~Continuous conversation (no wake word between turns)~~ DONE
Implemented via announcement API with `start_conversation=True`. 5s follow-up timeout, `end_conversation` tool for clean closure.

### ~~TTS sentence-level streaming~~ DONE
Split response into sentences, pipeline generation with playback. First sentence plays in ~2s while subsequent ones generate in parallel.

### More concise, less repetitive responses
The LLM tends to repeat context unnecessarily (e.g., "à Paris, avec un risque de pluie à 0%" on every weather response). Tune the system prompt to encourage shorter, more natural follow-ups — especially when answering a continuation like "et demain?", the response should be minimal ("Non plus, 0% de pluie").

## Robustness

### ~~LLM retry on finish_reason=length~~ DONE
`chat_completion()` in `llm.py` detects `finish_reason == "length"` and retries with double the token limit (up to 2 retries, ceiling 2000) before returning. Handles reasoning models that spend the whole budget on internal thinking. If still truncated at the ceiling, returns the partial response rather than failing.

### Fallback TTS
If Kokoro crashes or fails, fall back to raw espeak for audio output. The response quality degrades but the assistant remains functional instead of going silent.

## UX

### ~~Timers and alarms~~ DONE
LLM tools `set_timer`, `set_alarm`, `cancel_timer` with server-side `TimerManager` (asyncio scheduling). Timer events forwarded to ESP via native `send_voice_assistant_timer_event()` API (STARTED/UPDATED/CANCELLED/FINISHED). ESP handles LED animations and sounds. TTS announcement on timer finish with `start_conversation=True` for follow-up.

### Notifications (proactive announcements)
The assistant announces events without wake word: washing machine done, doorbell, reminders. Requires event listeners on HA entities and a push mechanism to the ESP via announcement API.

### Multi-language support
Automatic French/English detection for guests. Would require language-aware STT (Whisper handles this natively, Parakeet is French-only), and language parameter in the system prompt for the LLM response.

### Automatic room detection
Use UniFi Wi-Fi client tracking to detect which room the user is in (phone → nearest AP → room). When room is known, the `room` parameter becomes implicit: "éteins la lumière" targets the current room without needing to specify it.

## Services

### ~~Google Calendar integration~~ DONE
Connected to Google Calendar via OAuth2 (`calendar.events` scope). LLM tools: `query_calendar` (date range, search), `create_event` (title, datetime, duration). V1: single primary calendar. Multi-calendar with voice identification planned for later.

## Memory and context

### ~~Family profile (static knowledge)~~ DONE
Household members (first names, roles, optional birth dates) stored under a `family` key in `config.local.json` (gitignored, documented in `config.local.example.json`). Loaded via `LOCAL_CONFIG` and injected into the system prompt by `_format_family_for_prompt()` in `voice_server.py`, which computes each member's age from `birth_date`. `birth_date` is optional — members without one are listed by name and role only.

### Conversation memory (long-term)
Beyond the current 5-exchange sliding window (2 min expiry), the assistant should retain interesting facts from past conversations. Architecture options to explore:
- **Summary-based**: periodically summarize conversation highlights into a persistent file, inject as context
- **Embedding + retrieval (RAG)**: store conversation snippets with embeddings, retrieve relevant ones before each LLM call
- **Structured extraction**: extract key facts (preferences, recurring topics) into a knowledge base

Trade-offs: prompt size vs relevance, local embedding model (RAM budget), staleness of old memories.

## Mobile client

### ~~Step 1: Mobile-optimized web UI~~ DONE
Safe area insets for notch/home indicator, 44px touch targets, 16px input font (prevents iOS zoom), `100dvh` viewport, add-to-home-screen meta tags. iOS keyboard dictation works natively via the standard text input.

### Step 2: PWA with voice input
Progressive Web App installable on home screen. Use Web Speech API or stream audio to server-side Parakeet for STT. Tap-to-talk button that activates the mic immediately. No App Store needed, reuses existing backend. Limitation: Safari restricts mic access in background.

### Step 3: Native iOS app (optional)
Mic active on app open, background mode, possible local wake word detection, Siri Shortcut integration. Best UX but significant development effort — only worth it if PWA limitations become a real pain point.

## Identity

### Voice identification (speaker recognition)
Identify who is speaking based on voice characteristics.

Two distinct problems, often conflated under "native audio":
- **Coarse classification / diarization** (child vs adult, speaker A vs B): inferable from prosody. An audio-native LLM does this well, possibly near zero-shot.
- **Nominative identification** (map a voice → "Ivan"): a biometric verification problem. Requires per-person enrollment (reference samples) + embedding comparison. An audio LLM does NOT know household voices without references — this stays a dedicated component regardless of the LLM.

**Phase 1 — age adaptation**: simpler language for children, detailed answers for adults. Falls under coarse classification, so it becomes near-free with an audio-native model (see "STT replacement via unified audio model" below).

**Phase 2 — personal accounts**: use the identified speaker's accounts for services like Spotify (their playlists, recommendations). Needs nominative identification → enrollment + verification component.

- **Per-speaker permissions**: restrict which tools/actions are available depending on who is speaking. Example: children cannot create calendar events or modify home automation settings. This is a **security boundary** — must NOT rely on the LLM's probabilistic "sounds like a child"; requires a reliable speaker-verification model.

**Architectural dependency**: with the current Parakeet → text pipeline, all voice information is discarded at the STT stage, so voice ID would require a fully separate parallel audio pipeline. Moving to an audio-native LLM (Gemma 4 12B, see Infrastructure section) preserves the voice signal end-to-end, unlocking Phase 1 nearly for free. Phase 2 + permissions still need a dedicated enrollment/verification component, but it becomes an add-on rather than a full second pipeline. Target architecture: run a small speaker-verification model in parallel on the raw audio, inject the result ("Ivan / child / unknown") as structured context alongside the audio.

## Media

### Spotify voice control
Control Spotify playback on connected speakers (e.g., "mets du jazz dans le salon", "joue la playlist du matin"). Requires Spotify integration via HA media_player entities or direct Spotify Connect API. LLM tools: play_music (artist/song/playlist/genre query + target speaker), pause/resume, skip, volume.

## Infrastructure

### ~~Health check endpoint~~ DONE
`GET /health` (in `web_ui.py`) returns the status of STT, TTS, LLM, HA, ESP, and calendar. HTTP 200 when all required components (STT, TTS, LLM, HA) are ready, 503 otherwise. ESP and calendar are reported but optional (don't affect the verdict). LLM is checked by configuration only (mode/model/url) — not pinged, to avoid token cost; ops can curl the LLM URL directly. Also reports active timer count.

### Isolated test server instance
Currently e2e tests (`test_e2e.py`) run against the live server via `/api/dry-run` — tool calls are not executed but conversation state is shared. If someone uses the assistant during tests, the reset clears their conversation. Refactor to launch a lightweight test instance (LLM + entity list only, no STT/TTS/ESP) on a separate port for full isolation.

### Dedicated hardware migration (Zotac)
Move from M1 Mac + cloud LLM to a Zotac Magnus EN275060TC (RTX 5060 Ti 16GB VRAM). Primary candidate: Gemma 4 26B A4B (MoE, 3.8B active params, ~50-80 tok/s, native function calling). Fallback: Qwen 3 14B Q6_K (~30 tok/s). Would enable removing Python workarounds (room groups, text tool call parser, generic names) as both models handle tool calling natively. See `HARDWARE.md` for detailed specs and benchmarks. Quality vs GPT-5.4 Nano is uncertain — needs benchmarking once hardware is available.

### STT replacement via unified audio model
Gemma 4 12B (released 2026-06-03, Apache 2.0) is an encoder-free multimodal model with **native audio input** that runs on 16GB. If chosen as the LLM, it can absorb the STT stage: instead of `audio → Parakeet → transcript → LLM`, do `audio → Gemma 4 12B → tool calls` directly. This removes `stt.py` + the Parakeet model and eliminates the transcription bottleneck (the LLM hears full acoustic context instead of reasoning on a possibly-misheard transcript). Also unlocks voice ID Phase 1 (see Identity section).

Audio specs: 16kHz mono, float32 normalized [-1,1] (current pipeline sends 16-bit PCM → trivial conversion), 25 tokens/sec, 30s max segment. Prompt: `Transcribe the following speech segment in French into French text.`

Unknowns to de-risk before committing (a short POC on real French command WAVs settles all three):
1. **French ASR quality** — docs say "multilingual" but don't list French explicitly; no published WER vs Parakeet/Whisper.
2. **Audio + tool calling in the same call** — verify the chat template handles both simultaneously.
3. **Runtime on target hardware** — MLX audio path is undocumented; llama.cpp is the cited edge option. Depends on what hardware is bought.

Trade-off: choosing 12B-unified over 26B A4B means one fewer model to load and a simpler architecture, but loses the explicit transcript used by the web UI debug log (mitigation: prompt the model to emit the transcript in its output).
