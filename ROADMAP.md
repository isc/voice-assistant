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

### Timers and alarms
"Réveille-moi à 7h", "mets un timer de 5 minutes". The ESP already supports TIMERS (feature flag 8). Needs LLM tools for set_timer/set_alarm and a callback mechanism to trigger the ESP announcement when the timer fires.

### Notifications (proactive announcements)
The assistant announces events without wake word: washing machine done, doorbell, reminders. Requires event listeners on HA entities and a push mechanism to the ESP via announcement API.

### Multi-language support
Automatic French/English detection for guests. Would require language-aware STT (Whisper handles this natively, Parakeet is French-only), and language parameter in the system prompt for the LLM response.

### Automatic room detection
Use UniFi Wi-Fi client tracking to detect which room the user is in (phone → nearest AP → room). When room is known, the `room` parameter becomes implicit: "éteins la lumière" targets the current room without needing to specify it.

## Memory and context

### Family profile (static knowledge)
The assistant should know the household members: first names, dates of birth, relationships. This is static data that rarely changes — could be a simple JSON/YAML file loaded into the system prompt.

### Conversation memory (long-term)
Beyond the current 5-exchange sliding window (2 min expiry), the assistant should retain interesting facts from past conversations. Architecture options to explore:
- **Summary-based**: periodically summarize conversation highlights into a persistent file, inject as context
- **Embedding + retrieval (RAG)**: store conversation snippets with embeddings, retrieve relevant ones before each LLM call
- **Structured extraction**: extract key facts (preferences, recurring topics) into a knowledge base

Trade-offs: prompt size vs relevance, local embedding model (RAM budget), staleness of old memories.

## Media

### Spotify voice control
Control Spotify playback on connected speakers (e.g., "mets du jazz dans le salon", "joue la playlist du matin"). Requires Spotify integration via HA media_player entities or direct Spotify Connect API. LLM tools: play_music (artist/song/playlist/genre query + target speaker), pause/resume, skip, volume.

## Infrastructure

### Health check endpoint
`/health` endpoint returning status of STT, TTS, LLM, and HA connectivity. Useful for monitoring and alerting when a service goes down.

### Dedicated hardware migration (Zotac)
Move from M1 Mac + cloud LLM to a Zotac Magnus EN275060TC (RTX 5060 Ti 16GB VRAM). Target: Qwen 3 14B Q6_K (~30 tok/s, ~1.0s latency). Would enable removing Python workarounds (room groups, text tool call parser, generic names) as a 14B model handles tool calling natively. See `HARDWARE.md` for detailed specs and benchmarks. Quality vs GPT-5.4 Nano is uncertain — needs benchmarking once hardware is available.
