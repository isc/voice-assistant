# Voice PE firmware

Since 2026-10-11 the Voice PE runs the official 26.9.0 configuration built with ESPHome 2026.9.1,
where the release was built with 2026.6.0 (its `.github/workflows/build.yml`). 2026.9.1 carries
[esphome/esphome#19368](https://github.com/esphome/esphome/pull/19368): a mixer start request
that arrives while the mixer task is stopping was cleared, so an announcement right after another
never played and the Voice PE hung on it (AD-053, AD-054).

The device reports it: `esphome_version` 2026.9.1 in the server's log (`connected to … (v2026.9.1)`).
Its project version stays 26.9.0, so its update entity offers nothing until the next official
release, which replaces this build.

## Rebuild

```bash
git clone --depth 1 --branch 26.9.0 https://github.com/esphome/home-assistant-voice-pe.git vpe
cd vpe
sed -i 's/^    version: dev$/    version: 26.9.0/' home-assistant-voice.factory.yaml   # what CI sets
sed -i 's/^      ref: dev$/      ref: 26.9.0/' home-assistant-voice.yaml              # voice_kit pinned
python3 -m venv ../esphome-venv && ../esphome-venv/bin/pip install esphome==2026.9.1
../esphome-venv/bin/esphome compile home-assistant-voice.factory.yaml
```

`home-assistant-voice.factory.yaml` is the variant the device came with (device name
`home-assistant-voice-…`, 16 MB flash); `home-assistant-voice.8mb.yaml` is for 8 MB units. The
first compile, ESP-IDF download included, took about three minutes on charras-server; later ones reuse
`~/.cache/esphome`.

## Flash

Native OTA on port 3232, no password. From charras, through the Paris web VM:

```bash
ssh -4 -f -N -L 127.0.0.1:3232:192.168.1.195:3232 -p 60337 freebox@yge3sf9e.fbxos.fr
../esphome-venv/bin/esphome upload home-assistant-voice.factory.yaml --device 127.0.0.1
```

About 25 s, then the device reboots and the server reconnects. Wi-Fi credentials and the API
encryption key live in flash and survive it (`ESP_NOISE_PSK`, AD-051).

## Back to the official firmware

Once a release is built with ESPHome 2026.9.1 or later (`esphome-version` in its
`.github/workflows/build.yml`), install it from the device's update entity: from Paris's Home
Assistant, or from any API client (`update_command(key, UpdateCommand.INSTALL)`), as on
2026-10-11 for 26.9.0.
