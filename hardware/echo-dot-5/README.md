# Echo Dot 5 with clock: teardown notes

Notes from opening an Amazon Echo Dot with clock, 5th generation, on 2026-11-15. The aim is a
replacement main board that keeps the shell, the speaker, the microphones and, if possible, the clock
display, and makes the device a satellite of this assistant instead of Alexa's.

Nothing is designed yet. This file records what was seen, so that the work can resume without the
device open on the table. Each claim says how it is known: **read** on the part, **counted** on a
photo, **inferred**, or **from** a source listed at the end.

No such board exists for this model. Most of what it must talk to is known since 2026-11-18, from
Amazon's own published source code and from a replacement board begun in 2024 and left at its
connectors: see [What Amazon's source code says](#what-amazons-source-code-says) and
[Prior work](#prior-work-searched-on-2026-11-15).

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

#### Against esphome-dot's outline

esphome-dot's KiCad board (see [Prior work](#prior-work-searched-on-2026-11-15)), drawn from a unit
without clock, has the same eight holes: 3.0 mm for the board's screws, 5.6 mm for the case's.
Taken from its own disc's centre (77.0 mm across, the flat 37.3 mm from it), each hole sits 0.9 to
1.2 mm higher than in the table above, towards the flat, and 0.3 mm to the right, within 0.5 mm of
that shift each. So the two surveys agree on the holes between them, and differ by a millimetre on
where the outline lies around them: its flat is 75.8 mm from the opposite edge, ours 74.9 over the
ring. The caliper, with its battery, settles it.

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
and two close-ups: the half with [`U23` and `U6`](photos/mic-board-closeup.jpg), and the half with
[`U1`, `U3` and `VR1`](photos/mic-board-closeup-u1.jpg).

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
- `U1`: marked `45575` / `M2113` / `BAA`, a package with pads on its sides, the largest after `U23`.
  **Read**, on two photos. A Silego (now Renesas) GreenPAK programmed for Amazon, `SLG4BZ45575`, at
  I2C address 0x08: buttons, the microphone-off state, which it holds in hardware, the mute LED and
  the converter's reset (**from** Amazon's source code, see
  [below](#what-amazons-source-code-says)). Aggarwal's unit carries `45575` / `N211J` / `UAA`:
  `45575` is the part, the rest its lot.
- `U3`: marked `CM5` after a bar, five or six pins. `CM5` is the marking of TI's SN74LVC1G125 in an
  SC70 package, a single buffer whose output can be switched off (**from** TI's and distributors'
  marking tables). What it gates is not known: a line of the converter, cut when the microphones are
  switched off, would be its obvious use.
- `VR1`: a five-pin regulator marked like `3T=225`, its first character and the `=` uncertain. The
  Echo Spot's boards carry regulators marked `3T=653` (**from** Dorey). Not identified.
- `U5` (six bare pads) and `U25` are empty footprints.

That Echo Spot board, of 2024, is a close relative of this one: an AK5704, three microphones, a
25-contact connector, a controller marked in the same way. Dorey found the converter's I2C on two
of its test points and a button interrupt on a third. Its test points are not named as ours, so
none of it maps over as is, but it says what to look for on the 25 contacts.
- About thirty gold test pads labelled `TM1`…`TM39` and `TP1`…`TP4`.
- The tracks seen on this face leave the connector `J1` for a millimetre or two, then go through
  vias to inner layers: the cable's pinout does not read on a photo.
- The other face, not seen, carries the four buttons, two LEDs and a light sensor (**from** Dorey).

### Clock display (kept if it can be driven)

[Photo](photos/display-flex-back.jpg) of the face towards the inside.

- A flexible circuit held on two pegs against the front of the shell, with three foam blocks behind it.
- This face has no chip: three rows of 19 gold points, and four gold strips along one edge. **Counted.**
- The LEDs are on the other face, against the fabric. Not seen.
- **Its drivers are on the flex**, out of sight: three 36-channel LED drivers on the main board's
  I2C bus, at 0x30, 0x32 and 0x33 (**from** Amazon's source code). Nothing near `J1` on Amazon's
  main board is big enough to drive a matrix: a 6-pin part marked `SKB`, two transistors
  (**read**). So the 10-contact cable carries the I2C bus, an enable, a fault line and power
  (**inferred**). The channel maps run in groups of five, a matrix five dots high, which fits the
  « 5 × 21 » a forum poster counted (**inferred**).

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

### The 25 contacts, as far as known

From esphome-dot's `pinout.md` (2024, a unit without clock, traced with a multimeter to the
AK5704's pins), its numbering, unchecked on ours:

| Contact | Signal | Contact | Signal |
|---|---|---|---|
| 1 | ground | 2 | AVDD, 3.3 V |
| 3 | action button, pulled up | 4 | interrupt, low on a button press |
| 6 | I2C SDA | 8 | LRCK, the frame clock |
| 10, 12, 14, 15, 16 | ground | 11 | MCKI, the master clock |
| 13 | BCLK, the bit clock | 17 | SDTO1, the microphones' data, through the buffer `U3` |
| 21 | I2C SCL | 23 | TVDD, 1.8 V |

Contacts 5, 7, 9, 18 to 20, 22, 24 and 25 are not named. Amazon's source code would put there the
accelerometer's own I2C bus, the thermistors' analog lines, perhaps a second supply (**inferred**).
TVDD at 1.8 V is the converter's interface supply: the bus and the clocks are 1.8 V logic, which an
ESP32-S3's 3.3 V pins need translating to (**inferred**).

## What Amazon's source code says

Amazon publishes the GPL part of each Echo's software on
[amazon.com/echonotices](https://www.amazon.com/echonotices). The release
`Echo_Dot_5th_Gen_Clock_src-6.5.7.1-20251027.tar.bz2` holds a Linux 5.4 tree (`kernel/mediatek/5.4`)
whose device tree describes the board: `cannoli` is the Echo Dot 5, `cheesecake` the one with
clock, both built on `arch/arm/boot/dts/mediatek/cannoli-cheesecake.dtsi`. Read there on
2026-11-18:

- **The microphone board's I2C bus** (the SoC's `i2c4`): the controller `U1` at 0x08
  (`amzn,cpld45575`, « SLG4BZ45575 », a 3-button variant; `cpld45165` has 4); the converter, an
  AK5704 at 0x10 or a TLV320ADC5140 at 0x4c, both disabled in the tree, the bootloader enabling the
  one fitted; the light sensor, an ams TSL2540 at 0x39; a TMP108 thermometer at 0x48.
- **`U1`** (`drivers/gpio/gpio-amzncpld.c`, `include/dt-bindings/gpio/amzncpld-gpio.h`): an I2C
  expander with registers 0xF0 to 0xF6 and an interrupt line to the SoC (its GPIO86). Its lines:
  volume down, volume up, microphone-off button, microphone-off state, a software mute pulsed
  50 ms, the mute LED's PWM, the converter's reset. « On button press Silego locks the current
  state »: the microphones are cut in hardware. On a 45575, the action button is a direct line to
  the SoC (GPIO41).
- **The microphones' audio**: TDM, four channels of 32 bits at 96 kHz, `dsp_b` format, the bit
  clock inverted; the converter drives the bit and frame clocks, the SoC gives it a 24.576 MHz
  master clock.
- **The display**: three LED drivers of 36 channels on the main bus (`i2c0`) at 0x30, 0x32 and
  0x33, TI LP5036, ISSI IS31FL3246 or Awinic AW21036 depending on the unit; an enable (GPIO63), a
  power switch (GPIO17), a fault line (GPIO61). The light ring's driver shares the bus at 0x31, in
  the same three makes: Dorey's « W4970 3246 » is the ISSI.
- **The amplifier**: a MAX98396 or a TI TAS2764, I2S from the SoC, 2 × 32 bits at 96 kHz.
- **The accelerometer**: on a bus of its own, read by the DSP, not by Linux; Amazon's build adds
  ST's LIS2DW12 driver, the likely `U6` (**inferred**).

The tree gives no supply voltage for the microphone board or the display, and no contact numbers.

## What differs from the published teardowns

Brian Dorey opened a 5th generation without clock. This unit, with clock and made in 2022, differs:

- the converter is an AKM AK5704, where his is a TI TLV320ADC5140, as is Pallav Aggarwal's, whose
  microphone board is otherwise this one (`MK1`, `MK2`, `MK4` fitted, `MK3` empty, `U1` a `45575`):
  **inferred**, Amazon built this board with either converter, and a replacement board that must
  serve other units than this one has to drive both;
- the light ring's driver is a TI LP5036, where he reads `W4970 3246`;
- there is the display, its cable and its connector `J1`.

## Prior work, searched on 2026-11-15

Searched: the web in English and German, GitHub's repositories and LocalDot's issues, the Home
Assistant forum, the three board numbers and `U1`'s marking. Searched again on 2026-11-18, with
Amazon's source code and the FCC filings. For this model:

- **Teardowns** by Dorey and by Aggarwal: the chips of the main board, photos of both boards. Neither
  traces a cable, counts its contacts or names a test point's signal.
- **A replacement board begun, for the model without clock**:
  [esphome-dot](https://github.com/gcormier/esphome-dot), by Greg Cormier, June to November 2024,
  missed on 2026-11-15. A KiCad board with its outline, holes and connectors placed (the rest of
  its parts not), a schematic with a bare ESP32-S3 and a chip antenna, a TAS5805M, a TPS62932; the
  25-contact cable's contacts half traced (see [above](#the-25-contacts-as-far-as-known)); `U1`'s
  registers and the AK5704's start-up sequence read with a logic analyser, the captures with them.
  Nothing on the display. Its connectors: Molex 502598-2593 for the 25, Adam Tech ADC-055B-9 for
  the socket, SF2212 spring contacts for the speaker.
- **Amazon's GPL source code**, which describes the board: see
  [What Amazon's source code says](#what-amazons-source-code-says).
- **The FCC filings**: `2A4DH-1115` with clock, `2A4DH-0821` without, granted in September 2022.
  Their internal photos are of prototypes (main board `30-007131`, microphone board `30-007043`)
  at a resolution where no marking reads. They show the microphone board's hidden face: four
  buttons in a cross, two LEDs, the light sensor under a button. Not the display's LED face, which
  no published teardown shows either.
- **No other replacement board.** LocalDot (ESP32-S3 and ESP32-P4, on Crowd Supply since November
  2026, boards due in April 2027) is for the 3rd generation, whose microphone board and 40-contact
  cable are others; its repository and its issues name no later generation. echo-419 drew an
  EasyEDA schematic for the 4th generation in April 2023, four commits, nothing since.
- **Amazon's own board made to run other software: not possible yet.** The projects that do it
  (echo-dot-assist, EchoLocal, EchoMuse, TECHO5) cover the 2nd and 3rd generations and the Echo
  Show; the 4th and 5th generations and the Echo Pop have no known exploit. One person has been
  reading the 5th generation's boot chain (MT8519, « cannoli ») since October 2026, asking for a
  signed download agent, with no way in so far.
- **The nine pads under the base** (`J9`): one set of multimeter readings on the forum, its author's
  own numbering, 12 V on one pad, no protocol found. Of use to a root, not to this board.
- **The shell and the speaker reused**: the « Franken Dot » of August 2026 keeps those two, wires
  soldered to the speaker's two pads, a Waveshare ESP32-S3 audio board and its screen glued over
  the buttons. All of the Echo's electronics are taken out.

So the 25-contact cable is half traced and its signals are all named; the display's cable is not
traced, but what it must carry is known. Worth searching again before a board is drawn: LocalDot's
author, or a buyer of his board, is the likeliest to turn to the later generations.

### The Echo Pop, the household's other model

Not opened. **From** Dorey's teardown of it, and a search like the one above: no replacement board
and no pinout published, no exploit (« churro », secure boot on); one repository reuses its speaker
alone. It is built otherwise: one main board carries the microphones (three under the top, one at
an edge), the three buttons, four LEDs behind a light pipe, the converter (an ES7243, from a
reader of his) and the amplifier (a TI TAS5805M). The shell holds besides only the speaker, on
spring tabs, and a small board for the 12 V jack. A board for it would have no cable to trace, and
nothing to reuse but the shell, the speaker and the jack: its own microphones, placed under the
shell's holes, and its own buttons under the caps. It opens by prying off the fabric front, held
by foam tape.

## What a replacement board would carry

- The same outline and holes, the barrel socket and the two speaker contacts at the same places.
- A 25-contact and a 10-contact ribbon connector, where the cables reach. esphome-dot chose a Molex
  502598-2593 for the 25.
- A microcontroller with Wi-Fi running ESPHome, which is what this assistant's server already talks
  to: an ESP32-S3.
- Level translators between its 3.3 V pins and the microphone board's 1.8 V bus and clocks
  (**inferred** from TVDD, to measure).
- On the microphone board's bus: `U1` to release the converter's reset and read the buttons and
  the microphone-off state, then the converter itself, an AK5704 or a TLV320ADC5140 as the unit
  has. On the main bus, the display's three drivers behind `J1`, their enable and power switch.
- An amplifier for the speaker. A TAS5805M runs from the 12 V, ESPHome drives it since 2026.10
  (`tas58xx`), and its `SDOUT` pin gives back what it plays: wired to an I2S input of the ESP32,
  a reference for echo cancellation, locked to the samples (**inferred** from TI's datasheet).
- Twelve LEDs at the ring's positions. Addressable ones need one pin and no driver.
- A thickness near the original's, without it being critical. The board is screwed from the LED
  face, so its connector face rests on the chassis: the socket, both connectors and the speaker
  contacts keep their place whatever the thickness, and the soft ring takes up the difference on
  the other face (**inferred**).

### What the firmware would need

- **The converters**: no ESPHome component for either. The AK5704's register sequence is in
  esphome-dot (`ak_comp.h`, `i2c.md`, `pll.md`) and in Amazon's Linux driver; the TLV320ADC5140's
  in mainline Linux (`tlv320adcx140.c`). A component shaped as ESPHome's `es7243e`, a hundred and
  fifty lines, for each (**inferred**). `U1` and the display's drivers likewise.
- **Echo cancellation**, to hear the wake word over music: not in ESPHome as of 2026.11. Two
  external components try it on an ESP32-S3, neither proven there
  ([esphome-audio-stack](https://github.com/n-IA-hane/esphome-audio-stack), ESP-SR;
  [esphome-aec](https://github.com/matt123p/esphome-aec), SpeexDSP). LocalDot's ESP32-S3 alone did
  it « quite good, not perfect », then its author moved it to an ESP32-P4. The Voice PE does it on
  its XMOS. The open question that most moves the design: an ESP32-S3 alone, or with an XMOS or a
  P4 beside it.

## What it would cost

Worked out on 2026-11-18 for the household's three units: the bare board from JLCPCB's quote page,
its assembly from JLCPCB's published fees and its parts pages, the parts at LCSC's prices. With no
design yet, it prices a sketch of one. It replaces an estimate of 2026-11-15 (700 to 1900 €) that
put the fixed fees at about twice these. PCBWay quoted MiciMike's Home Mini board at about twice
what JLCPCB's grid gives (935 $ for five): a dearer maker, not a reason to double this.

- **The bare board**: 4 layers, 77 × 77 mm, 1.2 mm thick, five of them: 8 $.
- **Parts, a board**: about 13 $. An ESP32-S3-WROOM-1 module (5.4 $), a TAS5805M amplifier fed
  from the 12 V (1.3 $), twelve addressable LEDs (1 to 1.4 $), the two ribbon connectors (0.2 $),
  two spring contacts on the speaker's pads, a 4.0 × 1.7 mm barrel socket (Hroparts DC-098-165A,
  LCSC C129001, 0.25 $), a 12 V to 5 V converter and a 3.3 V regulator, level translators, a USB-C
  socket to flash it, some fifty passives. The display's drivers are on its flex: nothing to add.
  The Voice PE's XMOS XU316 adds about 20 $ (18 $ the chip, ten in stock on 2026-11-18) and an
  XTAG4 to load it.
- **Assembly, fixed per run**: setup, stencil, a fee per distinct part (about 31), fixtures, hand
  soldering. The joints, 0.0016 $ each, come to 0.6 $ a board, and the module's X-ray 1.6 $.
  JLCPCB places ESP32-S3 modules in Standard assembly only (« Standard Only, X-ray Inspection
  Required » on their parts pages); the ribbon connectors and the barrel socket in either.

  | Assembly | When | Fixed |
  |---|---|---|
  | Standard, one face | every part on the chassis face, the LEDs reverse-mounted (SK6812MINI-E, lit through a hole in the board, as in keyboards) | about 100 $ |
  | Standard, both faces | the LEDs on the base face as Amazon's, the rest on the chassis face | about 135 $ |

  Whether everything fits on the chassis face depends on the room under the chassis, not measured:
  the module is 3.1 mm thick. Economic assembly, about 60 $, would take the bare ESP32-S3 chip and
  an antenna of our own, as esphome-dot drew it, and the SK6812MINI-EA, Economic where the -E is
  not: cheaper by 40 to 75 $ a run, against a radio design to get right.
- **A run**: five bare boards, two or five of them assembled, UPS shipping (26 $) and 20 % VAT:
  175 to 220 € on one face, 210 to 260 € on both.
- **Runs**: two, perhaps three; the first ones with two boards assembled. Since Amazon's source
  code and esphome-dot, the signals are known, and the contacts mostly: what remains is to check
  them on this unit and to fit the board. A first board can still fail on a connector with its
  contacts on the wrong side (all of them mirrored), an outline a millimetre off, a 12 V class-D
  amplifier beside the microphone lines, a Wi-Fi antenna beside the metal chassis. On a 0.3 mm
  connector or under a module, a wrong track cannot be mended with a wire.
- **Before any board**: a multimeter, a logic analyser, the caliper's battery, a breakout for the
  25-contact cable, an ESP32-S3 kit and an amplifier on the bench: 60 to 150 €.

In all, without the XMOS: 450 to 600 € in two runs, 630 to 830 € in three. Once a board works,
one more in a run of five costs about 45 to 50 €, its parts and its share of the run. Three Voice
PEs cost about 180 $. What this buys is the speaker, the clock and a design others can have made,
not a saving.

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

1. **The 25-contact cable's pinout, checked on this unit**: esphome-dot's table (above), its
   unnamed contacts, and the voltages: 1.8 V logic or 3.3 V. With a multimeter, by continuity from
   the cable to the AK5704's pins, whose roles its datasheet gives, and by voltage on the running
   device. A logic analyzer on the running device would show `U1` at 0x08 and the converter at 0x10
   on the bus, and the settings Amazon writes to it.
2. **The display's cable**: which of the 10 contacts is the I2C bus (it should answer at 0x30, 0x32
   and 0x33), the enable, the fault line, the supply and its voltage. Probed while it runs.
3. **The power socket's polarity**, read on the label: measure it once before powering a board of
   ours.
4. **The speaker's impedance**: across its two pads.
5. **Dimensions to a tenth of a millimetre**, with the caliper and its battery: the six distances
   between the screw holes again, the board's own diameter under the ring, the socket and both
   connectors. Photos and caliper read on its beam give them to about 0.5 mm.
6. **The white ring**: its thickness and what it is made of. Soft and glued, it may not come off
   whole: a new board needs it moved over, or another cut to its shape.
7. **The room between the board's connector face and the chassis**, at the centre and towards the
   edge, with the caliper: whether a 3.1 mm module and every other part can go on that face.
8. `U6`'s three characters, if the accelerometer is to be used.

## Sources

- Brian Dorey, [Echo Dot 5th gen smart speaker teardown](https://briandorey.com/post/echo-dot-5th-gen-smart-speaker-teardown)
- Home Assistant forum, [De-Amazonizing an Echo Dot 5th Generation](https://community.home-assistant.io/t/de-amazonizing-an-echo-dot-5th-generation-for-use-as-a-wyoming-satellite-and-multiroom-speaker-in-home-assistant/726094)
- [LocalDot](https://github.com/ale1800/LocalDot), a replacement board for the 3rd generation, and its
  [forum thread](https://community.home-assistant.io/t/re-using-an-echo-dot-3rd-gen-with-voice-assistant/1020256)
- AKM, [AK5704EN datasheet](https://www.mouser.com/datasheet/3/5939/1/ak5704en-en-datasheet.pdf)
- Brian Dorey, [Amazon Echo Spot teardown](https://www.briandorey.com/post/echo-spot-smart-speaker-teardown),
  for its microphone board
- Texas Instruments, [SN74LVC1G125](https://www.ti.com/lit/gpn/sn74lvc1g125b)
- Pallav Aggarwal, [Inside the Amazon Echo Dot (5 Gen): a complete teardown](https://pallavaggarwal.in/2023/11/07/amazon-echo-dot-5gen-teardown/)
- [echo-419](https://github.com/renaiku/echo-419), a schematic for the 4th generation
- [echo-dot-assist](https://github.com/Gamer92000/echo-dot-assist), [EchoLocal](https://github.com/ygelfand/echolocal),
  [EchoMuse](https://github.com/wilbowes/EchoMuse) and [TECHO5](https://github.com/HuskerMinion/techo5):
  other software on Amazon's own board, older models
- [kaeru issue 118](https://github.com/R0rt1z2/kaeru/issues/118), the 5th generation's boot chain
- Liam Pettigrew, [The Franken Dot](https://fulloch.com/posts/the-franken-dot/)
- Brian Dorey, [Amazon Echo Pop teardown](https://www.briandorey.com/post/echo-pop-smart-speaker-teardown)
- Greg Cormier, [esphome-dot](https://github.com/gcormier/esphome-dot), a replacement board begun
  for the model without clock
- Amazon, [Echo device software notices and source code](https://www.amazon.com/echonotices):
  `Echo_Dot_5th_Gen_Clock_src-6.5.7.1-20251027.tar.bz2`
- FCC, [2A4DH-1115](https://fcc.report/FCC-ID/2A4DH-1115/) (with clock) and
  [2A4DH-0821](https://fcc.report/FCC-ID/2A4DH-0821/) (without)
- JLCPCB, [assembly fees](https://jlcpcb.com/help/article/pcb-assembly-price) and
  [assembly capabilities](https://jlcpcb.com/capabilities/pcb-assembly-capabilities); its parts
  pages for the assembly class of each part
- Hroparts, [DC-098-165A](https://www.lcsc.com/datasheet/lcsc_datasheet_2108131530_Korean-Hroparts-Elec-DC-098-165A_C129001.pdf),
  a 4.0 × 1.7 mm socket
- ESPHome, [tas58xx](https://esphome.io/components/audio_dac/tas58xx/)
- [Turning an old Echo Dot into a local voice assistant](https://dev.to/adamjrg/turning-an-old-echo-dot-into-a-local-voice-assistant-1nlb),
  for which models have an exploit
