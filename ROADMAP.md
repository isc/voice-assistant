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

## Screen

### Our own Cast receiver
The Nest Hub runs Google's Default Media Receiver today: it plays one URL and shows the label it was given at launch, nothing more. A receiver of our own is a web page the Nest Hub shows full screen, holding a connection to the server, which tells it what to show after each request, with no reload in between:
- the music playing, title and cover updated at each track: the Default Media Receiver shows new metadata only by loading the stream again, a cut at each track (AD-057);
- `show_image`'s pictures, generated pictures and Immich slideshows below, run by the page instead of casting one picture at a time;
- later: timers counting down, the week's weather, the day's calendar, a shop's hours and map, the last reply's text.

First version: the music playing and `show_image`, the two uses of the Default Media Receiver today, to try the limits before going further. Needed: a Google Cast developer account (US$5, the family's Google account) for an app ID; the Nest Hub registered by its serial number while the app is unpublished; the page served over HTTPS; HA starts the app by its ID. To find out:
- whether a page served over HTTPS may play the stream and read the server over plain HTTP, as the Paris relay serves them; otherwise both through HTTPS;
- how long a receiver playing nothing stays on screen (HA's own Cast receiver loops a media to keep a dashboard up);
- what touch a third-party receiver gets on a Nest Hub, likely little beyond the media controls: a screen driven by voice;
- one app at a time: a cast from a phone, or "Hey Google", takes the screen back.

### Generated pictures
"Dessine-moi un dragon" → generate the picture locally and show it on the Nest Hub, next to `show_image` (AD-038). charras already has `stable-diffusion.cpp` with Z-Image-Turbo q8 in `/opt/sd/` (~22 s per 1024² image; exact command in home-infra `charras/trellis/beast.sh`). No longer blocked by VRAM: since 2026-10-25 it runs on the second card (home-infra `charras/SETUP.md` §23), with gemma resident on the first; a whole `beast` run took 4 min 36 s, `sd-cli` at 7.2 GB on the bottom card. That card also takes the house's long jobs (ACE-Step ~8–9 GB, ddmr's OCR up to 11 GB), so a picture asked while one runs must wait or say so. Unlike web pictures, the result is a local file, so it must be served over HTTP, beside `/tts/`, which the Paris relay forwards.

### Family photos from Immich
"Montre-moi des photos de Zoé", "les vacances en Bretagne l'été dernier", "Charlie à la plage" → find them in Immich and show them on the Nest Hub as a slideshow. Immich on charras (v2.7.5, port 2283) already has smart search (CLIP), facial recognition and reverse geocoding enabled, so: people for names, place and date for trips, smart search for descriptions. The Default Media Receiver shows one picture at a time, so a slideshow means casting every few seconds, or our own receiver (above), with "suivante" and "arrête". Immich needs an API key and the Paris Nest cannot reach the charras LAN, so the server fetches each preview and serves it under `/tts/`. Prerequisites on the family's side: migrate the photos, name the face clusters.

## Media

### Spotify started from a phone
"Assistant vocal" is in the account's Spotify Connect device list, but music started from a phone plays to nobody until a speaker listens to the stream. Watch go-librespot's `/events` websocket and point the default speaker at the stream when playback starts from elsewhere.

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

## Sites

### A third home: Ivan's parents
They have a patchwork of ecosystems side by side (Alexa, Google Home, a HomePod, Velux roof windows, lights of several brands, a recent air conditioner, a Somfy alarm, a Nuki lock) and find each assistant poor at running the house, LLM or not. Served like Paris: a Home Assistant at their place gathers everything, and charras does STT, LLM and TTS for a third site in `SITES`. It makes Ivan their support: with charras or its line down, their voice control is down too (HA and the vendors' apps still work).

- **At their place**: an HA Green or an N100 mini-PC with the Tailscale add-on, as `homeassistant-paris`; a Voice PE per living room, the web app on their phones; a relay for the satellite's API port and the reply audio, as the Paris web VM does (home-infra `charras/voice-paris/`), since their box is likely on `192.168.1.0/24` too. Echo, Nest and HomePod stay: HA can expose its devices back to them, so Siri keeps working, and the Nest speakers become HA players for the radio or a picture. Matter multi-admin shares a device already in Apple, Google or Alexa with HA without pairing it again.
- **Their devices**: the Nuki through its bridge or Matter (Paris's is on Matter). Velux through HomeKit Controller with a Velux Active (KIX 300, which then leaves Apple Home) or the Velux integration with a KLF 200; a TaHoma also drives io-homecontrol Velux. The Somfy alarm depends on which system: Home Alarm with a TaHoma is Overkiz, local; Somfy Protect has no official integration (Charras's comes into HA through MQTT). The alarm tool written for Charras serves theirs once it is an `alarm_control_panel`. The air conditioner through its brand's integration: `set_temperature` sets its setpoint, its mode (cooling, heating, dry) has no tool yet. Alexa-only lights without Matter are a dead end.
- **The server first** (checked 2026-10-26):
  - Homes kept apart. Any request may reach another home (`home` on `lock`, `set_heating`, `get_state`…), and the web app has a single key (`WEB_UI_KEY`): with its link, they could set Charras's heating or ask to unlock its door. A key per household, naming the homes it reaches, and `home` limited to those.
  - The calendar is one Google account, Ivan's: « qu'est-ce que j'ai demain ? » there would read his. Off for their site, or an account per site.
  - Spotify is one go-librespot on one account, one stream: their music would stop ours. Off for their site to begin with (see "Two rooms, two musics").
  - `family` and `room_groups` in `config.local.json` are shared by all sites, and the family goes into the prompt: one of each per site.
  - One voice conversation at a time across all sites (`claim_voice`): an « Ok Nabu » there while someone speaks at Charras gets an error, likelier with three homes. Voice state per site; STT and TTS stay serialised by `onnx_gpu.run`.
  - Their voices are kept 90 days in `recordings/` and the nightly review reads their exchanges: tell them, or keep their site out of both.
- **Order**: the inventory first (brand and model of each device, or screenshots of their apps: Alexa, Google Home, Apple Home, Somfy, Velux, the air conditioner's), since it decides most of the above, above all which Somfy system and which Velux gateway; HA at their place, used on its own for a few weeks; the server's prerequisites; then a Voice PE and the site.

## Infrastructure

### Models worth trying (review of 2026-10-21)
Open models that run on charras, checked against ours on 2026-10-21. Mistral's Voxtral 3 (2026-10-20) is not among them: API only, self-hosting on request through their sales, no weights published.

- **TTS: Qwen3-TTS or Kyutai TTS instead of Kokoro.** The most to gain: Kokoro is English-first, and its French is one voice (`ff_siwis`) through the misaki G2P. Qwen3-TTS 0.6B / 1.7B (Apache 2.0, French among its languages, voice cloning) and Kyutai TTS 1.6B (CC-BY 4.0, English and French only) are the candidates; Voxtral 4B TTS (2026-03) is CC BY-NC, acceptable for us but 4B. Replay twenty real replies (`recordings/`) through each, compare by ear and on latency against Kokoro's, and on VRAM: both are 20× Kokoro's size.
- **STT: Qwen3-ASR 1.7B beside Parakeet.** Apache 2.0, 2026-06, French: 4.75 % WER on FLEURS French in its report, against 5.15 % for Parakeet TDT 0.6B v3 in NVIDIA's (different normalisations, so indicative). Three times Parakeet's size. Run the 2026-10-11 recordings (the 28 with known words, from "STT through the LLM") through it: children, speech over music, the house's names. Voxtral Mini 4B Realtime (Apache 2.0) is more accurate still on paper, but 4B. NVIDIA has published no newer multilingual Parakeet.
- **LLM: the 12B stays (tried 2026-10-25, AD-081).** Gemma 4 26B-A4B fits a 16 GB card only as its QAT build, alone and without the full SWA cache: 9 points less accurate than the 12B on the hard cases (goodbyes without `end_conversation`, English answered in English), 0.1–0.3 s faster a turn, and no audio. Its Q4_K_M build is about as accurate as the 12B (but a calendar request came back empty, its call malformed) and needs 17 GB; Qwen3.8-27B is dense, 7 s a turn. What would reopen it: a 26B build that keeps the Q4_K_M's accuracy within ~14 GB, or a larger card. Next to try instead: the registry's `gemma4:12b`, changed since production pulled it (other weights, and an MTP draft for speculative decoding). `test_e2e.py` and `test_hard_cases.py` take `VOICE_SERVER`, so a model on trial runs both on a test server (AD-081 says how).

Silero VAD stays: nothing newer replaces it for French.

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

**2026-10-25** (AD-081): the 12B still hears audio on Ollama 0.40.7 (two Kokoro sentences, word for word); neither Gemma 4 26B-A4B nor Qwen3.8-27B takes any, so a larger LLM would close this path.

What would reopen it: audio heard behind a long prompt, from a newer Ollama (the length effect may be its own, a short reproduction is worth reporting) or another runtime, then French recognition near Parakeet's on children and on speech over music. Splitting it in two calls (Gemma transcribes behind a short prompt, then the usual text request) works today, but gets point 2's accuracy for 0.8 s more.

Trade-off: the web UI's debug log loses the transcript, unless the model is asked to write it out.
