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

### LLM retry on finish_reason=length
When the LLM runs out of completion tokens (all budget consumed by internal reasoning), automatically retry with a higher token limit instead of failing silently. Currently we bumped max_tokens to 500, but edge cases may still hit the limit.

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

### Google Calendar integration
Connect to the household's Google Calendars to answer scheduling questions ("qu'est-ce que j'ai demain?", "à quelle heure est le rendez-vous?") and create events by voice ("ajoute un rendez-vous dentiste mardi à 14h"). Requires Google Calendar API with OAuth2 or service account. LLM tools: query_calendar (date range, search), create_event (title, datetime, duration, attendees). Combined with voice identification, the assistant can query and write to the correct person's calendar automatically.

## Memory and context

### Family profile (static knowledge)
The assistant should know the household members: first names, dates of birth, relationships. This is static data that rarely changes — could be a simple JSON/YAML file loaded into the system prompt.

### Conversation memory (long-term)
Beyond the current 5-exchange sliding window (2 min expiry), the assistant should retain interesting facts from past conversations. Architecture options to explore:
- **Summary-based**: periodically summarize conversation highlights into a persistent file, inject as context
- **Embedding + retrieval (RAG)**: store conversation snippets with embeddings, retrieve relevant ones before each LLM call
- **Structured extraction**: extract key facts (preferences, recurring topics) into a knowledge base

Trade-offs: prompt size vs relevance, local embedding model (RAM budget), staleness of old memories.

## Mobile client

### Step 1: Mobile-optimized web UI
Adapt the current web UI for mobile use. Full-screen chat interface, large touch targets, iOS keyboard dictation for text input. Quick win, no app to install.

### Step 2: PWA with voice input
Progressive Web App installable on home screen. Use Web Speech API or stream audio to server-side Parakeet for STT. Tap-to-talk button that activates the mic immediately. No App Store needed, reuses existing backend. Limitation: Safari restricts mic access in background.

### Step 3: Native iOS app (optional)
Mic active on app open, background mode, possible local wake word detection, Siri Shortcut integration. Best UX but significant development effort — only worth it if PWA limitations become a real pain point.

## Identity

### Voice identification (speaker recognition)
Identify who is speaking based on voice characteristics. Phase 1: adapt responses to the speaker's age (simpler language for children, detailed answers for adults). Phase 2: use the identified speaker's personal accounts for services like Spotify (play their playlists, recommendations, etc.).

## Media

### Spotify voice control
Control Spotify playback on connected speakers (e.g., "mets du jazz dans le salon", "joue la playlist du matin"). Requires Spotify integration via HA media_player entities or direct Spotify Connect API. LLM tools: play_music (artist/song/playlist/genre query + target speaker), pause/resume, skip, volume.

## Infrastructure

### Health check endpoint
`/health` endpoint returning status of STT, TTS, LLM, and HA connectivity. Useful for monitoring and alerting when a service goes down.

### Isolated test server instance
Currently e2e tests (`test_e2e.py`) run against the live server via `/api/dry-run` — tool calls are not executed but conversation state is shared. If someone uses the assistant during tests, the reset clears their conversation. Refactor to launch a lightweight test instance (LLM + entity list only, no STT/TTS/ESP) on a separate port for full isolation.

### Dedicated hardware migration (Zotac)
Move from M1 Mac + cloud LLM to a Zotac Magnus EN275060TC (RTX 5060 Ti 16GB VRAM). Target: Qwen 3 14B Q6_K (~30 tok/s, ~1.0s latency). Would enable removing Python workarounds (room groups, text tool call parser, generic names) as a 14B model handles tool calling natively. See `HARDWARE.md` for detailed specs and benchmarks. Quality vs GPT-5.4 Nano is uncertain — needs benchmarking once hardware is available.
