# Roadmap

Features and improvements planned for the voice assistant.

## Conversation flow

### ~~Continuous conversation (no wake word between turns)~~ DONE
Implemented via announcement API with `start_conversation=True`. 5s follow-up timeout, `end_conversation` tool for clean closure.

### More concise, less repetitive responses
The LLM tends to repeat context unnecessarily (e.g., "à Paris, avec un risque de pluie à 0%" on every weather response). Tune the system prompt to encourage shorter, more natural follow-ups — especially when answering a continuation like "et demain?", the response should be minimal ("Non plus, 0% de pluie").

## Latency

### TTS streaming
Currently the full TTS audio is generated before playback starts. Stream chunks to the ESP as they are generated to reduce perceived latency. Kokoro generates audio in a single pass so this may require chunking the text and generating segments independently, or switching to a streaming-capable TTS engine.

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
