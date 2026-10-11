# Roadmap

What is still open. Shipped work is recorded in `DECISIONS.md` and the git history, not here.

## Robustness

### Know why the Voice PE restarts
On 2026-10-10 the Voice PE most likely rebooted after a reply asked for follow-up while its own player was busy; since 2026-10-11 the server asks for no follow-up while the satellite plays (AD-045). The reboot was inferred from the LED (solid warm white is the firmware's init-without-Wi-Fi state). On 2026-10-11 at 12:52 it froze instead: a wake word while it played the Spotify stream itself (left on it by a server bug, fixed since) brought no audio, then no answer to the API at all until unplugged. Adding ESPHome's `debug:` component to the Voice PE firmware would report the reset reason. Its firmware went from 25.6.0 to 26.9.0 on 2026-10-11, whose release notes fix stuck states on long or streamed replies and media stopping on HTTP timeouts, then to that configuration built with ESPHome 2026.9.1 (AD-054): first see whether freezes still happen. `debug:` would need our own build anyway (`firmware/README.md`).

### Back to the official Voice PE firmware
The Voice PE runs our build of 26.9.0 with ESPHome 2026.9.1, for the mixer fix (AD-054). Once a release is built with 2026.9.1 or later (`esphome-version` in its `build.yml`), install it from the update entity. If an announcement hangs again before then (no sound, « Queue full » in the device log), it is a second bug: compare with [#643](https://github.com/esphome/home-assistant-voice-pe/issues/643) and take it upstream.

### Media target from live player state
Media commands without a room go to `_last_media_target`, a value kept in memory: it is lost on restart (picked up from HA at startup since 2026-10-10) and goes stale when music is started or stopped from a phone or the Sonos app. Reading HA player states at command time would send "stop" and "plus fort" to whatever actually plays. It would also let the conversation stay open after a media command when nothing plays near the satellite (AD-035 ends it unconditionally). Cost: one HA read per media command, or a websocket subscription.

Ducking (AD-049) would drop its template read at each wake word too. Moving Spotify would stop the previous HA speaker only if it still plays the stream, not when the Sonos app has since put something else on it. The same subscription would make the device states live: they are polled every 5 minutes, so a device gone offline since is still reported done, and an offline one is read again before each refusal.

## UX

### Proactive announcements
The assistant speaks without wake word: washing machine done, doorbell, reminders. Simplest shape: Home Assistant already knows when these happen, so an HA automation calls a server endpoint (`POST /announce` with the text, through a `rest_command`) and the server plays it like a timer announcement. Needs quiet hours (no washing-machine news at 23:00) and a choice of satellite once there are several.

### Room of the voice satellite
"Éteins la lumière" should target the room the request came from. The room of the satellite that heard the wake word is the reliable signal: give each satellite an HA area and use it as the default `room`. This replaces the earlier idea of tracking phones through UniFi Wi-Fi access points, set aside because an AP covers several rooms and the phone is not always on the person. Only matters once a site has more than one satellite. Ducking (AD-049) would use it too: it lowers every HA speaker playing at the site, where only those near the satellite need it.

### Answer in the speaker's language
Parakeet v3 already transcribes 25 European languages with automatic detection: spoken to in English, it understands (seen 2026-10-10), but the assistant answers in French. Two pieces are missing:
- The LLM answering in the language of the request: the system prompt is French and says nothing about it.
- The TTS speaking that language: `ff_siwis` and the misaki G2P are set up for French. Kokoro has English voices; the reply's language has to pick the voice and the G2P, either detected on the reply text or stated by the LLM.

## Pictures

### Generated pictures
"Dessine-moi un dragon" → generate the picture locally and show it on the Nest Hub, next to `show_image` (AD-038). charras already has `stable-diffusion.cpp` with Z-Image-Turbo q8 in `/opt/sd/` (~22 s per 1024² image; exact command in home-infra `charras/trellis/beast.sh`). Blocked by VRAM today: Z-Image needs ~6.3 GB beside the voice assistant's ~4 GB and `gemma4:12b`'s 8.1 GB on the 16 GB card. A second GPU is ordered for charras. Unlike web pictures, the result is a local file, so it must be served over HTTP, beside `/tts/`, which the Paris relay forwards.

### Family photos from Immich
"Montre-moi des photos de Zoé", "les vacances en Bretagne l'été dernier", "Charlie à la plage" → find them in Immich and show them on the Nest Hub as a slideshow. Immich on charras (v2.7.5, port 2283) already has smart search (CLIP), facial recognition and reverse geocoding enabled, so: people for names, place and date for trips, smart search for descriptions. Cast shows one picture at a time, so a slideshow means casting every few seconds, with "suivante" and "arrête". Immich needs an API key and the Paris Nest cannot reach the charras LAN, so the server fetches each preview and serves it under `/tts/`. Prerequisites on the family's side: migrate the photos, name the face clusters.

## Media

### Spotify started from a phone
"Assistant vocal" is in the account's Spotify Connect device list, but music started from a phone plays to nobody until a speaker listens to the stream. Watch go-librespot's `/events` websocket and point the default speaker at the stream when playback starts from elsewhere.

### The track playing on the screen
The Nest Hub shows the title and cover of what was asked (AD-057), not of the following tracks: the Default Media Receiver shows new metadata only by loading the stream again. Two ways: load it again at each track change from go-librespot's `/events`, holding the FIFO meanwhile so that no music is lost, for a gap of a few seconds between tracks on screens; or a Cast receiver of our own (Google Cast developer account, US$5), which plays the stream and reads the track from the server, with no gap.

### Two rooms, two musics
A single go-librespot is a single queue. Different music in two rooms needs a second player and a second Spotify account, as Spotify allows one stream per account.

## Memory and context

### Long-term memory
Beyond the 5-exchange window (expires after 10 minutes), the assistant could keep facts across conversations. Start with explicit memory: "souviens-toi que la piscine ferme le 15 septembre" calls a tool that appends the fact to a file injected into the system prompt, like the family profile, and "oublie que…" removes it. Predictable, no embedding model, nothing remembered behind the family's back. Automatic extraction or retrieval over past conversations (RAG) only if the facts file outgrows the prompt.

## Identity

### Voice identification (speaker recognition)
Identify who is speaking. Two distinct problems:
- **Coarse classification** (child vs adult, speaker A vs B): inferable from prosody. An audio-native LLM may do this near zero-shot.
- **Nominative identification** (this voice is Ivan): biometric verification. Needs per-person enrollment and embedding comparison; no LLM knows household voices without references.

**Phase 1 — age adaptation**: simpler language for children. Coarse classification, so near-free with an audio-native model (see "STT through the LLM").

**Phase 2 — personal accounts**: the speaker's own Spotify, calendar, reminders. Needs nominative identification.

**Per-speaker permissions**: children cannot create calendar events or change home automation settings. A security boundary: it must rely on a speaker-verification model, never on the LLM's guess that a voice sounds young.

Today Parakeet turns audio into text and every voice cue is lost there, so voice ID means a separate component on the raw audio: a small speaker-verification model whose verdict ("Ivan / child / unknown") is injected into the prompt.

## Infrastructure

### Re-evaluate the small-model workarounds
Written for Qwen 3 4B. The French-verb text fallback went on 2026-10-11 (never fired in 223 requests, and it could take a reply for a command), and so did the « already called » reminder. The structured text fallback (`fn({...})` written as text) remains, harmless; it never fired either. Remove it after a few hundred more requests without a `Function call (text fallback)` line. Generic names stay: they drive group resolution and the shutter/roof-window split.

### Left over from the 2026-10-11 audit
- **Timers that survive a restart**: every deploy drops them, an evening deploy cancels the next morning's alarm without a word. Save alarms to a file and re-arm them at startup; when one fires with no satellite connected, wait for it or announce on an HA speaker.
- **Alarms beyond 24 hours**: « réveille-moi lundi à 7h » sets the next 7:00. A `date` parameter, and the reply naming the day.
- **Duplicate device names**: Paris has three « Plafonnier » and three « Volet »; with no room, the first in HA's order wins and the reply does not say which. Ask, or prefer the satellite's room, and name the room in the reply.
- **Sensor search in French**: « chlore », « sel », « consommation », « batterie » miss the English sensor names (« Free Chlorine », « Salinity »…); « eau » matches « Bureau ». Word prefixes and a small French→English map.
- **VAD minimum speech**: one 32 ms frame above 0.5 counts as speech for the whole recording; Silero's own default asks for 250 ms.

### STT through the LLM
Gemma 4 12B, now the LLM, takes audio input natively. It could absorb the STT stage, so that `audio → Gemma → tool calls` replaces `audio → Parakeet → transcript → LLM`. The LLM would hear the request instead of reasoning on a possibly misheard transcript, which also unlocks voice ID Phase 1. Parakeet is not slow (0.07–0.35 s on the GPU), so the case rests on understanding, not latency; it would also free part of the ~4 GB the STT/TTS models hold. It would not have saved the 2026-10-10 request drowned by the Sonos: that is an acoustic problem (see "Duck the music while listening").

Audio specs: 16 kHz mono float32 in [-1, 1] (the pipeline has 16-bit PCM, a trivial conversion), 25 tokens/s, 30 s max segment.

**Tested on 2026-10-11: not ready.** Ollama 0.30.11, `gemma4:12b-8k` (Q4_K_M), the day's 107 recordings (Paris):
1. **Runtime**: yes. Ollama takes the WAV as recorded, through `/v1/chat/completions`, the endpoint `llm.py` uses (`{"type": "input_audio", "input_audio": {"data": <base64>, "format": "wav"}}`), or `/api/chat` (`images`). The prompt grows by the expected 25 tokens a second.
2. **French recognition**: well behind Parakeet. On 28 requests whose words are known (references written from both transcripts and the conversation around them, not yet checked by ear), 25 % of words wrong against Parakeet's 3.6 %, and 4 requests mostly invented against none (« Tu peux ouvrir le volet dans la chambre invitée » → « Je ne vous ai pas encore parlé de ce sujet »). It does hear a few names better (« Chopin », « Sonos », where Parakeet wrote « goûter du chopin », « sonneuse »). On 10 of the 11 recordings the VAD found silent, it wrote English comments (« I'm sorry, … no audio provided »). With Google's ASR prompt, instruction before the audio; the audio first, it answers the request instead (144 %); a system message with the house's names, worse (35 %). 0.8 s a transcription (median), Parakeet's 0.08 s.
3. **Audio and tools in one call**: no. Behind the real prompt (~4.1K tokens, 24 tools), the audio is not heard: « Ouvre le volet de la chambre invitée » → `get_weather`, « Tu peux fermer la porte ? » → RMC in the living room. Replayed as dry runs, of the 28 requests where the transcript led to calls, 1 got the same calls from the audio; of the 56 that called nothing, 33 called something (bedroom lights off, a pasta timer). Length, not tools: the system prompt alone (~500 tokens, no tools) gets 5 of 6 commands right (« C'est fait, j'ouvre le volet de la chambre invités »); with 4.4K tokens of neutral notes appended, 2 of 6 at most; with the tools written in the prompt instead of `tools`, none.

What would reopen it: audio heard behind a long prompt, from a newer Ollama (the length effect may be its own, a short reproduction is worth reporting) or another runtime, then French recognition near Parakeet's on children and on speech over music. Splitting it in two calls (Gemma transcribes behind a short prompt, then the usual text request) works today, but gets point 2's accuracy for 0.8 s more.

Trade-off: the web UI's debug log loses the transcript, unless the model is asked to write it out.
