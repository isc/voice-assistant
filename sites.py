"""Sites: one per home, each with its voice satellite and its Home Assistant, served by one server.

`SITES=paris,charras` names them; each then reads its settings with its name as prefix
(`PARIS_HA_URL`, `CHARRAS_ESP_HOST`…). The satellite's port and keys may be shared, unprefixed.
Without `SITES`, the unprefixed variables describe a single site, as before.
"""

import asyncio
import ipaddress
import os
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from timer import TimerManager

_Address = ipaddress.IPv4Address | ipaddress.IPv6Address
# Read as NAME_KEY for a named site; these may also be set once, unprefixed, for all sites
_SHARED_KEYS = {"ESP_PORT", "ESP_PASSWORD", "ESP_NOISE_PSK"}


@dataclass(eq=False)
class Conversation:
    """What the model hears again of a conversation, forgotten after a silence (CONVERSATION_TIMEOUT)."""

    history: list = field(default_factory=list)
    id: Optional[int] = None
    last_time: float = 0
    shown_images: set = field(default_factory=set)  # pictures not to show twice


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
    public_host: str = ""  # name of the home's internet address (its box): a phone on its Wi-Fi comes from it

    # Runtime state, per home: a request at one site must not see the other's
    ha_client: Optional[Any] = None  # ha_client.HAClient
    http_base_url: str = ""  # TTS URL announced to this site's speakers
    timer_manager: TimerManager = field(default_factory=TimerManager)
    conversation: Conversation = field(default_factory=Conversation)
    # Dry runs (tests) have one of their own: a multi-turn test needs one, and the family's must
    # neither see it nor be kept alive by it (AD-047)
    dry_run: Conversation = field(default_factory=Conversation)
    last_media_target: Optional[tuple[str, str]] = None

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
            public_host=setting(name, "PUBLIC_HOST"),
        )
        for name in names
    ]


def _same_connection(client: _Address, home: _Address) -> bool:
    """IPv4: the box's address, shared by the whole home. IPv6: each device has its own, in the box's /64."""
    if client.version != home.version:
        return False
    if client.version == 4:
        return client == home
    return ipaddress.ip_network(f"{home}/64", strict=False) == ipaddress.ip_network(f"{client}/64", strict=False)


async def site_at_address(sites: list[Site], address: str) -> Optional[Site]:
    """The site whose internet connection a client's public address belongs to, if any."""
    try:
        client = ipaddress.ip_address(address)
    except ValueError:
        return None
    client = getattr(client, "ipv4_mapped", None) or client
    loop = asyncio.get_running_loop()
    for site in sites:
        if not site.public_host:
            continue
        try:
            found = await loop.getaddrinfo(site.public_host, None)
        except OSError:
            continue
        if any(_same_connection(client, ipaddress.ip_address(sockaddr[0])) for *_, sockaddr in found):
            return site
    return None
