"""Sites: one per home, each with its voice satellite and its Home Assistant, served by one server.

`SITES=paris,charras` names them; each then reads its settings with its name as prefix
(`PARIS_HA_URL`, `CHARRAS_ESP_HOST`…). The satellite's port and keys may be shared, unprefixed.
Without `SITES`, the unprefixed variables describe a single site, as before.
"""

import os
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from timer import TimerManager

# Read as NAME_KEY for a named site; these may also be set once, unprefixed, for all sites
_SHARED_KEYS = {"ESP_PORT", "ESP_PASSWORD", "ESP_NOISE_PSK"}


# Compared by identity: a site is a home with live state, not a value
@dataclass(eq=False)
class Site:
    name: str  # "" for a single, unnamed site
    esp_host: str = ""  # "" when the site has no satellite (still usable from the web UI)
    esp_port: int = 6053
    esp_password: str = ""
    esp_noise_psk: str = ""
    ha_url: str = ""
    ha_token: str = ""
    tts_base_url: str = ""  # "" for this host's LAN address

    # Runtime state, per home: a request at one site must not see the other's
    ha_client: Optional[Any] = None  # ha_client.HAClient
    http_base_url: str = ""  # TTS URL announced to this site's speakers
    timer_manager: TimerManager = field(default_factory=TimerManager)
    conversation_history: list = field(default_factory=list)
    # Dry runs (tests) keep a conversation of their own: a multi-turn test needs one, the family's
    # must not see it
    dry_run_history: list = field(default_factory=list)
    conversation_id: Optional[int] = None
    last_interaction_time: float = 0
    last_media_target: Optional[tuple[str, str]] = None
    shown_images: set = field(default_factory=set)

    @property
    def label(self) -> str:
        return self.name or "default"


def load_sites(env: Mapping[str, str] = os.environ) -> list[Site]:
    names = [name.strip().lower() for name in env.get("SITES", "").split(",") if name.strip()] or [""]
    # A name is an env prefix: « paris-nord » could never be configured, a duplicate would share a satellite
    if bad := [n for n in names if n and not re.fullmatch(r"[a-z0-9_]+", n)]:
        raise ValueError(f"SITES: invalid site names {bad} (letters, digits, _)")
    if len(set(names)) != len(names):
        raise ValueError(f"SITES: duplicate site names in {names}")

    def setting(name: str, key: str, default: str = "") -> str:
        if name and f"{name.upper()}_{key}" in env:
            return env[f"{name.upper()}_{key}"]
        if not name or key in _SHARED_KEYS:
            return env.get(key, default)
        return default

    return [
        Site(
            name=name,
            esp_host=setting(name, "ESP_HOST"),
            esp_port=int(setting(name, "ESP_PORT") or 6053),
            esp_password=setting(name, "ESP_PASSWORD"),
            esp_noise_psk=setting(name, "ESP_NOISE_PSK"),
            ha_url=setting(name, "HA_URL"),
            ha_token=setting(name, "HA_TOKEN"),
            tts_base_url=setting(name, "TTS_BASE_URL"),
        )
        for name in names
    ]
