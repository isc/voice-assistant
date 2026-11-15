# Echo Dot 5 with clock: teardown notes

Notes from opening an Amazon Echo Dot with clock, 5th generation, on 2026-11-15. The aim is a
replacement main board that keeps the shell, the speaker, the microphones and, if possible, the clock
display, and makes the device a satellite of this assistant instead of Alexa's.

Nothing is designed yet. This file records what was seen, so that the work can resume without the
device open on the table. Each claim says how it is known: **read** on the part, **counted** on a
photo, **inferred**, or **from** a source listed at the end.

No such board exists for this model. [LocalDot](https://github.com/ale1800/LocalDot) does it for the
3rd generation; the Home Assistant forum thread on the 5th generation has no board and no firmware.

## Telling the model without opening it

- A sphere with no 3.5 mm jack at the back, only the power socket: 5th generation (the 4th has the jack).
- The clock is a dot matrix, not seven-segment digits.
- On the network, the device answers Spotify Connect's discovery. It advertises
  `_spotify-connect._tcp` over mDNS, and `http://<ip>:4070/spotifyConnect?action=getInfo` returns
  `modelDisplayName: Echo_Dot_with_clock` and the name given in the Alexa app. It does not give the
  generation.

## Opening it

Nothing was broken or unglued, and the steps reverse.

1. Peel the rubber foot off the base (adhesive and locating pins).
2. Four Torx screws at the bottom of the deep holes (T5, from Dorey's teardown;
   [photo](photos/base-screws.jpg), the serial number covered). The 3×3 grid of pads next to them is
   the debug header `J9`: a forum poster measured about 12 V on one of them, leave it alone.
3. The two halves separate. The lower one holds the light ring's diffuser.
4. The main board is held by four small silver Torx screws. Lift it from the edge opposite the
   `FLEX SIDE` marking, like a lid: two ribbon cables hold it on that side.
5. Both cables sit in flip-lock connectors, whose grey flap lifts on the side opposite the cable. They
   came out without force.
6. Remove the grey metal chassis (Torx screws). Under it: the speaker in its sealed box, the microphone
   board at the top of the shell, the display against the front.

The microphone board is screwed (four Torx) and, from Dorey's teardown, glued. It was left in place:
its visible face is the one with the microphones and the converter.

## The parts

| Part | Marking | Kept? |
|---|---|---|
| Main board | `30-007260 REV 01` | Replaced |
| Microphone and button board | `30-007247 REV 01` | Kept |
| Clock display, a flexible circuit | `30-007230 REV-01`, `MFC 3422` | Kept if it can be driven |
| Speaker, in a sealed plastic box | none electrical | Kept |
| Shell, chassis, light ring diffuser | | Kept |

### Main board (replaced)

A disc with four mounting holes and four larger holes the case screws pass through. Its diameter is
not measured yet.

Face towards the base ([photo](photos/main-board-bottom.jpg)):

- Twelve RGB LEDs around the edge (`DS1`…`DS12`), the light ring. **Counted.**
- `U15`: TI **LP5036**, a 36-channel I2C LED driver; twelve RGB LEDs are 36 channels. **Read.**
- `U7`: MediaTek **MT6398BN**, the processor's power management. **Read.**
- `AR1`: marking unreadable. **Inferred** to be the amplifier, from its designator and from Dorey's
  MAX98396 on this face.
- `C194`: 330 µF, 16 V. Two 10 µH inductors beside it (`L74`, `L75`). **Read.**
- `J9`: the debug header, seen from inside.

Face towards the chassis ([photo](photos/main-board-top.jpg), [in place](photos/main-board-lifted.jpg)):

- A large shield (processor and memory, **from** Dorey: MediaTek MT8519) and a Wi-Fi module under its
  own shield (`U5`), with two antennas printed on the board.
- `J4`: 25-contact ribbon connector, to the microphone board. `J1`: 10-contact ribbon connector, to
  the display. **Counted.**
- `P1`, `P2`: two spring contacts that press on the two pads of the speaker box. No wire to unplug.
  **Read.**
- `J3`: the barrel power socket. The adapter (model `LA23EU`) gives **12.0 V, 1.25 A, 15 W**, the
  plug's **centre positive**. **Read** on its label. `D22` next to the socket is marked `15A`, and
  `L66` is a 3.3 µH inductor.
- The chassis carries a thermal pad that presses on the large shield.

### Microphone and button board (kept)

[Photo](photos/mic-board.jpg), lit from the side so that the laser markings show.

- `U23`: marked `5704` / `2176`, 28 pins (7 a side). **Inferred** to be an AKM **AK5704**, a
  four-channel 32-bit converter with microphone preamplifiers in a 4 × 4 mm 28-pin package, which is
  what marking and package say together. Its datasheet is public.
- Three microphones are fitted: `MK1`, `MK2` and one whose label does not read on the photos
  (presumably `MK4`). A fourth footprint, `MK3`, is empty. **Read.** They are analog, since they go
  through the converter. **Inferred.**
- `U1`, `U3`, `U6`, `VR1`: markings not read. `U25` is an empty footprint.
- About thirty gold test pads labelled `TM1`…`TM39` and `TP1`…`TP4`.
- The other face, not seen, carries the four buttons, two LEDs and a light sensor (**from** Dorey).

### Clock display (kept if it can be driven)

[Photo](photos/display-flex-back.jpg) of the face towards the inside.

- A flexible circuit held on two pegs against the front of the shell, with three foam blocks behind it.
- This face has no chip: three rows of 19 gold points, and four gold strips along one edge. **Counted.**
- The LEDs are on the other face, against the fabric. Not seen.
- **Where its driver is, is not known.** Either on the hidden face, and the 10-contact cable carries a
  bus and power, or on Amazon's main board, and the cable carries the matrix itself. Ten contacts are
  few for a matrix of this size, which favours the first.

### Speaker (kept)

[Photo](photos/speaker.jpg). A driver of about 1.7 inches (**from** Dorey: 1.73) in a sealed plastic
box, reached through two pads. Its impedance is not measured.

## The two ribbon cables

| | To the microphone board | To the display |
|---|---|---|
| Connector on the main board | `J4` ([photo](photos/j4-mic-connector.jpg)) | `J1` ([photo](photos/j1-display-connector.jpg)) |
| Contacts | **25**, fine and staggered ([photo](photos/mic-cable-tip.jpg)) | **10**, wide ([photo](photos/display-cable-tip.jpg)) |
| Printed on the cable | `MLB` at the main board's end, `MIC` at the other | |
| Width of the tip | not measured | not measured |

The 25 were counted as 24 gaps along four lines across the enlarged photo; the 10 on the connector and
again on the cable. Each cable has `P1` and a small triangle printed at its tip: the mark of contact 1.

## What differs from the published teardown

Brian Dorey opened a 5th generation without clock. This unit, with clock and made in 2022, differs:

- the converter is an AKM AK5704, where his is a TI TLV320ADC5140;
- the light ring's driver is a TI LP5036, where he reads `W4970 3246`;
- there is the display, its cable and its connector `J1`.

## What a replacement board would carry

- The same outline and holes, the barrel socket and the two speaker contacts at the same places.
- A 25-contact and a 10-contact ribbon connector, where the cables reach.
- A microcontroller with Wi-Fi running ESPHome, which is what this assistant's server already talks
  to. LocalDot pairs an ESP32-S3 with an ESP32-P4 for the microphones' processing: an ESP32-S3 alone
  used one microphone, with an echo cancellation its author called good but not perfect.
- An amplifier for the speaker.
- Twelve LEDs at the ring's positions. Addressable ones need one pin and no driver.

## Still to find out

Tools were missing on 2026-11-15: no multimeter, no logic analyzer.

1. **The 25-contact cable's pinout**: which contact is power, ground, I2C, the audio clocks and data,
   each button, each LED. With a multimeter, by continuity from the cable to the test pads and to the
   AK5704's pins, whose roles its datasheet gives. With a logic analyzer on the running device, the
   I2C traffic would also give the settings Amazon writes to the converter.
2. **The display's driver**: look at the hidden face, or probe the 10-contact cable while it runs.
3. **The power plug's dimensions.** Its polarity is read on the label: measure it once before
   powering a board of ours.
4. **The speaker's impedance**: across its two pads.
5. **Dimensions**: the board's diameter, its holes, the width and pitch of both cable tips.
6. The unread chips of the microphone board, and the amplifier `AR1`.

## Sources

- Brian Dorey, [Echo Dot 5th gen smart speaker teardown](https://briandorey.com/post/echo-dot-5th-gen-smart-speaker-teardown)
- Home Assistant forum, [De-Amazonizing an Echo Dot 5th Generation](https://community.home-assistant.io/t/de-amazonizing-an-echo-dot-5th-generation-for-use-as-a-wyoming-satellite-and-multiroom-speaker-in-home-assistant/726094)
- [LocalDot](https://github.com/ale1800/LocalDot), a replacement board for the 3rd generation, and its
  [forum thread](https://community.home-assistant.io/t/re-using-an-echo-dot-3rd-gen-with-voice-assistant/1020256)
- AKM, [AK5704EN datasheet](https://www.mouser.com/datasheet/3/5939/1/ak5704en-en-datasheet.pdf)
