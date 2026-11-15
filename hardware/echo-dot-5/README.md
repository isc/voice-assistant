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

A disc with a flat cut where the power socket sits, four mounting holes and four larger holes the
case screws pass through. A caliper across it reads **76.7 mm**, and **74.9 mm** from the flat to the
opposite edge, each within 0.3 mm (see [the caliper](#the-caliper)). Both are taken over the white
ring described below, soft and a little wider than the board itself, which is then about 76 mm.

#### Where things sit

Both faces were photographed square on (the outline fits a circle to 0.1 %, and 0.4 %). The two
photos in `photos/` are cut and scaled from them: **20 pixels a millimetre**, the disc's centre at the
image's centre, taking the outline as 77 mm across (the ring's, 76.7 mm by the caliper since: 0.4 %
too much). A point at pixel (x, y) is at ((x − 800) / 20,
(800 − y) / 20) mm.

Seen from the face with the connectors, the power socket up, origin at the disc's centre, x to the
right and y up, in millimetres:

| What | Where | Size |
|---|---|---|
| Flat cut of the outline | y ≈ +36.6 | |
| Passages of the four case screws | (−17.4, +16.3), (+16.7, +16.3), (−16.8, −17.3), (+16.5, −17.4) | holes of 5.4 to 6 mm (see below) |
| Holes of the board's four screws | (−13.6, +20.9), (+13.7, +21.6), (−18.9, −10.7), (+19.2, −9.6) | 2.9 mm by the caliper (3.1 on the photos) |
| Two more holes, either side of the debug header | (−6.0, +21.8), (+5.4, +21.8) | 3.25 mm by the caliper, one of them (3.1 to 3.5 on the photos) |
| Twelve LEDs, on the other face | a circle of radius 33.0, every 30°, one on each axis | each in a window of the ring |
| Power socket `J3` | centre about (−2, +32.5) | body about 17 × 10 |
| Microphone connector `J4` | centre about (−28.5, +12.7), its length along y | about 9 long |
| Display connector `J1` | centre about (−24, −24.5), turned some 35° | |
| Speaker contacts `P1`, `P2` | about (+32, +13) and (+32, +10) | |

The holes and the LEDs are fitted on the photos (a circle on each hole's edge; the LEDs' dark
outlines), the holes on both faces, which agree to about 1 mm. The connectors, the socket and the
contacts are read by eye on the scaled photo. The outline that sets the scale is the ring's, 1 to
2 mm nearer or farther than the face the holes are seen on, from a camera some 15 cm away: a
percent or so. All of it is good to **about 1 mm**: enough for a
paper outline printed at full size and tried on the chassis, not for ordering a board.

#### Checked with the caliper

The six distances between the board's four screw holes, one inside jaw in each hole, spread apart,
read on photos (see [the caliper](#the-caliper)). A centre distance is that reading less the reading
with both jaws in one such hole (3.1): the scale's offset and the jaws' own width cancel.

| Between (coordinates of the table above) | Reading | Centres, caliper | Centres, photos |
|---|---|---|---|
| (+13.7, +21.6) and (−13.6, +20.9) | 30.0 | 26.9 | 27.3 |
| (−13.6, +20.9) and (−18.9, −10.7) | 34.5 | 31.4 | 32.0 |
| (−18.9, −10.7) and (+19.2, −9.6) | 41.2 | 38.1 | 38.1 |
| (+19.2, −9.6) and (+13.7, +21.6) | 35.1 | 32.0 | 31.7 |
| (+13.7, +21.6) and (−18.9, −10.7), a diagonal | 49.4 | 46.3 | 45.9 |
| (−13.6, +20.9) and (+19.2, −9.6), the other | 48.1 | 45.0 | 44.8 |

The two agree within 0.6 mm, 0.4 on average, with no scale error between them (+0.3 %). The caliper
read this way is no finer than the photos: its first five distances put the sixth 1.2 mm from where
it was read. So the plan is confirmed to about 0.5 mm, not refined. Both give one diagonal about
1 mm longer than the other: the pattern is most likely not symmetrical, as the photos had shown
(each right hole 0.5 to 1 mm higher than its left one, on both faces).

Across the four passages of the case screws, the two diagonals read 54.0 and 53.4: with the photos'
centres, holes of 5.7 to 6 mm, where both jaws in one hole had read 5.4 on a blurred photo.

Face towards the base ([photo](photos/main-board-bottom.jpg)):

- Twelve RGB LEDs around the edge (`DS1`…`DS12`), the light ring. **Counted.**
- A white ring lies on this face along the edge, with a window for each LED. It is soft and looks
  glued to the board (felt by hand, not peeled). Seen edge on, it is 0.3 to 0.5 mm wider than the
  board all round. Most likely a gasket pressed against the diffuser, which also keeps each LED's
  light to its own window (**inferred**). Ring and board together are **1.25 mm** between the
  caliper's jaws at the rim, which squeeze the ring. On a photo of the board's edge held beside the
  caliper's scale, the green layer is about 1.15 mm and the ring about 0.7 mm, unsqueezed. So the
  board is thinner than the usual 1.6 mm: **1.0 or 1.2 mm**, the two standard thicknesses either
  side of what was read.
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
  plug's **centre positive**. **Read** on its label. The plug is **4.0 mm** across by the caliper, its
  hole about 1.7 mm by the photo's proportions: the common 4.0 × 1.7 mm barrel. `D22` next to the
  socket is marked `15A`, and `L66` is a 3.3 µH inductor.
- The chassis carries a thermal pad that presses on the large shield.

### Microphone and button board (kept)

[Photo](photos/mic-board.jpg), lit from the side so that the laser markings show,
[another](photos/mic-board-tracks.jpg) in even light, where the labels and the tracks read better,
and a [close-up](photos/mic-board-closeup.jpg) of the half with `U23` and `U6`.

- `U23`: marked `5704` / `217G`, 28 pins (7 a side). **Inferred** to be an AKM **AK5704**, a
  four-channel 32-bit converter with microphone preamplifiers in a 4 × 4 mm 28-pin package, which is
  what marking and package say together. Its datasheet is public.
- Three microphones are fitted: `MK1`, `MK2` and `MK4`. A fourth footprint, `MK3`, is empty: a
  port hole and bare pads. **Read.** They are analog, since they go through the converter.
  **Inferred.** Each can carries two lines of text and a data matrix, not read.
- `U6`: a black block about 2 mm square and 1 mm high, with pads on its sides, marked with a dot,
  three characters (a `2`, then `3` or `9`, then `5` or `S`) and a data matrix. **Inferred** to be
  the accelerometer behind the tap gestures Amazon gave this generation: the package is that of
  one, and nothing else on this face would be.
- `RT1`, `RT2`: thermistors by their designators, for the temperature this generation reports.
  **Inferred.**
- `U1`, `U3`, `VR1`: markings not read. `U1`, the largest after `U23`, has pads on its sides and a
  marking on two lines. `U5` (six bare pads) and `U25` are empty footprints.
- About thirty gold test pads labelled `TM1`…`TM39` and `TP1`…`TP4`.
- The tracks seen on this face leave the connector `J1` for a millimetre or two, then go through
  vias to inner layers: the cable's pinout does not read on a photo.
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
| Width of the tip | about **7.6 mm**, so a **0.3 mm** pitch | about **5.3 mm**, so a **0.5 mm** pitch |

The widths are read on photos of each tip against a ruler. A 10-contact cable at the common 0.5 mm
pitch is 5.5 mm wide. The 25 were counted as 24 gaps along four lines across the enlarged photo; the 10 on the connector and
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
- A thickness near the original's, without it being critical. The board is screwed from the LED
  face, so its connector face rests on the chassis: the socket, both connectors and the speaker
  contacts keep their place whatever the thickness, and the soft ring takes up the difference on
  the other face (**inferred**).

## Still to find out

Tools were missing on 2026-11-15: no multimeter, no logic analyzer.

### The caliper

A digital caliper is at hand, without its battery. Its beam is graduated in millimetres, and the
opening reads where the slider's edge crosses that scale. On a photo of the jaws closed on the power
plug, the edge reads 4.3 where the jaws, measured on the same photo against the scale, are 4.05 mm
apart: the edge reads **0.25 mm too much**, taken off the readings above. The holes are taken with
its inside jaws. A reading is good to 0.3 mm at best, and less on the hole distances: the slider's
end is a wall some millimetre high, bright as a mirror, and where it meets the scale is judged on
photos taken at an angle. With a battery the caliper reads to 0.01 mm, and none of this matters.

1. **The 25-contact cable's pinout**: which contact is power, ground, I2C, the audio clocks and data,
   each button, each LED. With a multimeter, by continuity from the cable to the test pads and to the
   AK5704's pins, whose roles its datasheet gives. With a logic analyzer on the running device, the
   I2C traffic would also give the settings Amazon writes to the converter.
2. **The display's driver**: look at the hidden face, or probe the 10-contact cable while it runs.
3. **The power socket's polarity**, read on the label: measure it once before powering a board of
   ours.
4. **The speaker's impedance**: across its two pads.
5. **Dimensions to a tenth of a millimetre**, with the caliper and its battery: the six distances
   between the screw holes again, the board's own diameter under the ring, the socket and both
   connectors. Photos and caliper read on its beam give them to about 0.5 mm.
6. **The white ring**: its thickness and what it is made of. Soft and glued, it may not come off
   whole: a new board needs it moved over, or another cut to its shape.
7. The unread chips of the microphone board (`U1` first, then `U3` and `VR1`), `U6`'s three
   characters, and the amplifier `AR1`.

## Sources

- Brian Dorey, [Echo Dot 5th gen smart speaker teardown](https://briandorey.com/post/echo-dot-5th-gen-smart-speaker-teardown)
- Home Assistant forum, [De-Amazonizing an Echo Dot 5th Generation](https://community.home-assistant.io/t/de-amazonizing-an-echo-dot-5th-generation-for-use-as-a-wyoming-satellite-and-multiroom-speaker-in-home-assistant/726094)
- [LocalDot](https://github.com/ale1800/LocalDot), a replacement board for the 3rd generation, and its
  [forum thread](https://community.home-assistant.io/t/re-using-an-echo-dot-3rd-gen-with-voice-assistant/1020256)
- AKM, [AK5704EN datasheet](https://www.mouser.com/datasheet/3/5939/1/ak5704en-en-datasheet.pdf)
