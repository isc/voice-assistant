"""
Home Assistant REST API client for voice assistant integration.
Handles entity discovery with area grouping, fuzzy matching, and service calls.
"""

import asyncio
import difflib
import json
import logging
import os
import re
import unicodedata
from contextvars import ContextVar
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)
# Set by a request that only reads, every few seconds (the web app's shortcuts, AD-066): the resolutions it
# runs stay out of the log, but for errors
quiet: ContextVar[bool] = ContextVar("quiet", default=False)
logger.addFilter(lambda record: record.levelno >= logging.ERROR or not quiet.get())

# Domains we expose as LLM tools
SUPPORTED_DOMAINS = {"light", "switch", "cover", "climate", "media_player", "lock"}

# French stopwords to strip for fuzzy matching
STOPWORDS = {"le", "la", "les", "l", "du", "de", "des", "un", "une", "d"}

# Room groups for multi-room commands — loaded from config.local.json (gitignored)
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.local.json")
try:
    with open(_CONFIG_PATH) as _f:
        _local_config = json.load(_f)
    ROOM_GROUPS: dict[str, list[str]] = _local_config.get("room_groups", {})
    LOCAL_CONFIG = _local_config
except FileNotFoundError:
    ROOM_GROUPS = {}
    LOCAL_CONFIG = {}


def fold(text: str) -> str:
    """Lowercase, accents stripped."""
    return "".join(c for c in unicodedata.normalize("NFD", text.lower()) if unicodedata.category(c) != "Mn")


def normalize(text: str) -> str:
    """Normalize text for fuzzy matching: lowercase, strip accents and stopwords."""
    return " ".join(w for w in fold(text).split() if w not in STOPWORDS)


def without_home(name: str, home: str) -> str:
    """A device's name without its home's (« Sonos Paris » at Paris → « Sonos »): a request is already that
    home's, by the satellite it reaches or the phone's Wi-Fi or choice. A name that is only the home's stays."""
    kept = [word for word in name.split() if fold(word) != fold(home)] if home else []
    return " ".join(kept) if kept else name


def compact(text: str) -> str:
    """normalize() without spaces or punctuation, for names said many ways ("Rire & Chansons")."""
    return "".join(c for c in normalize(text) if c.isalnum())


# Rooms that mean the whole house: every room, so no list of rooms to keep in config
_WHOLE_HOUSE = {normalize(room) for room in ("tout", "toute la maison", "partout")}


def is_whole_house(room: Optional[str]) -> bool:
    return bool(room) and normalize(room) in _WHOLE_HOUSE


# A request asks for every room only in so many words, those of _WHOLE_HOUSE and more; « tout de
# suite » or « c'est tout » do not
_WHOLE_HOUSE_WORDS = re.compile(r"\b(?:tout|tous|toute|toutes|partout|maison|appart|appartement)\b")
_NOT_WHOLE_HOUSE = re.compile(r"\btout (?:de suite|a l'heure|a fait|le monde)\b|\bc'est tout\b")


def asks_whole_house(text: str) -> bool:
    """Whether a request says the whole house: tout, tous, partout, la maison, l'appart…"""
    return bool(_WHOLE_HOUSE_WORDS.search(_NOT_WHOLE_HOUSE.sub(" ", fold(text))))


def _group_rooms(room: str) -> Optional[list[str]]:
    """The rooms of the group a room names, also said with more words (« les chambres des enfants »)."""
    room_words = set(normalize(room).split())
    return next((rooms for name, rooms in ROOM_GROUPS.items() if set(normalize(name).split()) <= room_words), None)


# Built in, whatever an older config.local.json still lists for them
for _name in [name for name in ROOM_GROUPS if is_whole_house(name)]:
    del ROOM_GROUPS[_name]


# difflib ratio for a misheard name: at 0.4, « télé » found the light « Entrée » (0.5) and « lumière »
# Charlie's « Plafonnier » (0.47); a typo or a plural stays above 0.8
_FUZZY_CUTOFF = 0.75


def _area_matches(norm_area: str, norm_room: str) -> bool:
    """Loose room match, either name inside the other.

    No area means in no room, not in every one ("" is in any name)."""
    return bool(norm_area) and (norm_room in norm_area or norm_area in norm_room)


# Sub-kinds inside the `cover` domain. A house can have roller shutters and roof
# windows side by side — both are HA covers, but they are not interchangeable
# (opening a skylight when asked for a shutter lets the rain in).
# The name's word comes first (« Volet salon » reported as window by Z-Wave JS); device_class
# for names that say nothing (HomeKit Controller exposes Velux shutters with no class).
# Mosquito screens have no device_class of their own: their name is all that tells them apart
# (the Paris RFPlayer reports none), and « ferme les volets » must leave them up.
COVER_KIND_BY_DEVICE_CLASS = {
    "window": "fenetre",
    "shutter": "volet",
    "blind": "volet",
    "shade": "volet",
    "curtain": "volet",
    "awning": "volet",
}
COVER_KIND_KEYWORDS = {
    "volet": ("volet", "volets", "store", "stores"),
    "fenetre": ("fenetre", "fenetres", "velux"),
    "moustiquaire": ("moustiquaire", "moustiquaires"),
}


def cover_kind(info: dict) -> Optional[str]:
    """Classify a cover as 'volet', 'fenetre' or 'moustiquaire', None if undetermined.

    The name's word first: Z-Wave JS reports the Paris living-room shutters as device_class
    "window", and « Volet salon » says what it is. device_class for names that say nothing.
    """
    words = set(normalize(info.get("friendly_name", "")).split())
    for kind, keywords in COVER_KIND_KEYWORDS.items():
        if words.intersection(keywords):
            return kind
    return COVER_KIND_BY_DEVICE_CLASS.get(info.get("attributes", {}).get("device_class"))


class HAClient:
    """Home Assistant REST API client with area-based entity discovery."""

    def __init__(self, base_url: str, token: str, home: str = ""):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.home_name = home  # the site's name, left out of device names
        # entity_id -> {friendly_name, domain, state, attributes, area_id, area_name}
        self.entities: dict[str, dict] = {}
        self.areas: dict[str, str] = {}  # area_id -> area_name
        # entity_id -> area_id, for every domain — sensors aren't in self.entities
        # but search_sensors still wants to know which room they belong to.
        self.entity_areas: dict[str, str] = {}
        # The home as configured in HA: latitude, longitude, time_zone
        self.home: dict = {}
        # media_player entities with a screen, to show pictures on
        self.screens: list[str] = []
        self._session: Optional[aiohttp.ClientSession] = None

    async def connect(self) -> bool:
        """Create session, verify connectivity, fetch areas and entities."""
        self._session = aiohttp.ClientSession(
            headers={"Authorization": f"Bearer {self.token}"},
            timeout=aiohttp.ClientTimeout(total=10),
        )
        try:
            async with self._session.get(f"{self.base_url}/api/") as resp:
                if resp.status != 200:
                    logger.error(f"HA API returned {resp.status}")
                    return False
                data = await resp.json()
                logger.info(f"Home Assistant connected: {data.get('message', 'OK')}")
            async with self._session.get(f"{self.base_url}/api/config") as resp:
                config = await resp.json()
                self.home = {
                    "latitude": config.get("latitude"),
                    "longitude": config.get("longitude"),
                    "time_zone": config.get("time_zone"),
                }
        except Exception as e:
            logger.error(f"Cannot reach Home Assistant at {self.base_url}: {e}")
            return False

        await self.refresh_entities()
        return True

    async def close(self):
        """Close the aiohttp session."""
        if self._session:
            await self._session.close()

    async def _render_template(self, template: str, timeout: Optional[float] = None) -> Optional[str]:
        """Render a template with HA's template API — the REST way into the registries. None on failure."""
        try:
            # Never timeout=None: aiohttp reads it as no timeout at all, not the session's 10 s. On
            # 2026-10-14 one call caught by a reboot of the Paris HA, through the SSH tunnel, waited
            # 45 min, and every home's refresh with it.
            async with self._session.post(
                f"{self.base_url}/api/template",
                json={"template": template},
                timeout=aiohttp.ClientTimeout(total=timeout) if timeout else self._session.timeout,
            ) as r:
                if r.status != 200:
                    logger.warning(f"HA template API returned {r.status}")
                    return None
                return await r.text()
        except Exception as e:
            logger.warning(f"HA template API error: {e}")
            return None

    async def _fetch_areas(self):
        """Fetch area registry via HA template API."""
        text = await self._render_template("{% for a in areas() %}{{ a }}|{{ area_name(a) }}\n{% endfor %}")
        if text is None:
            logger.warning("Could not fetch areas")
            return
        areas = {}
        for line in text.strip().split("\n"):
            if "|" in line:
                area_id, area_name = line.split("|", 1)
                areas[area_id.strip()] = area_name.strip()
        if areas != self.areas:
            self.areas = areas
            logger.info(f"Areas: {', '.join(areas.values())}")

    async def _fetch_entity_areas(self) -> Optional[dict[str, str]]:
        """Fetch entity -> area_id mapping via HA template API, all areas in one call. None on failure."""
        text = await self._render_template(
            "{% for a in areas() %}{% for e in area_entities(a) %}{{ e }}|{{ a }}\n{% endfor %}{% endfor %}"
        )
        if text is None:
            return None
        return dict(line.split("|", 1) for line in text.split() if "|" in line)

    async def _fetch_hidden(self) -> set[str]:
        """Entities hidden in HA: replaced by another (Paris hides two Fibaro lights for their endpoints)."""
        text = await self._render_template(
            "{% for s in states if is_hidden_entity(s.entity_id) %}{{ s.entity_id }}\n{% endfor %}"
        )
        return set((text or "").split())

    async def _fetch_screens(self) -> list[str]:
        """media_player entities with a screen — Google's Nest Hubs, told apart by device model.

        HA has no "has a display" attribute: a Nest Hub Max reports the same supported
        features as a Google Home speaker.
        """
        text = await self._render_template(
            "{% for s in states.media_player %}"
            "{% if 'Hub' in (device_attr(s.entity_id, 'model') or '') %}{{ s.entity_id }}\n{% endif %}"
            "{% endfor %}"
        )
        return [eid for eid in (text or "").split() if eid in self.entities]

    async def refresh_entities(self):
        """Fetch all states from HA, filter to supported domains, attach area info."""
        try:
            async with self._session.get(f"{self.base_url}/api/states") as resp:
                if resp.status != 200:
                    logger.error(f"Failed to fetch HA states: {resp.status}")
                    return
                states = await resp.json()
        except Exception as e:
            logger.error(f"Error fetching HA states: {e}")
            return

        await self._fetch_areas()
        # A failed call keeps the rooms already known, rather than placing every device nowhere
        if (fetched := await self._fetch_entity_areas()) is not None:
            self.entity_areas = fetched
        entity_to_area = self.entity_areas
        hidden = await self._fetch_hidden()
        known = set(self.entities)

        self.entities.clear()
        for state in states:
            entity_id = state["entity_id"]
            domain = entity_id.split(".")[0]
            if domain not in SUPPORTED_DOMAINS:
                continue

            friendly_name = state.get("attributes", {}).get("friendly_name", entity_id)

            # Skip Fibaro sub-entities
            if "Basic" in friendly_name or "basic" in entity_id:
                continue
            if friendly_name.endswith("(2)"):
                continue
            if "_2_2_" in entity_id or entity_id.endswith("_2_2"):
                continue
            # Skip entities with generic names
            if friendly_name.strip() in (
                "Dimmer 2",
                "Double Smart Module",
                "Single Switch 2",
                "Smart Module",
                "",
            ):
                continue
            # An unavailable one stays known: a sleeping battery speaker (the Paris Sonos Roam), left
            # out, let « la radio sur Sonos » go to the one speaker of the room the LLM made up
            if entity_id in hidden:
                continue

            area_id = entity_to_area.get(entity_id)
            area_name = self.areas.get(area_id, "") if area_id else ""

            self.entities[entity_id] = {
                "friendly_name": without_home(friendly_name.strip(), self.home_name),
                "domain": domain,
                "state": state["state"],
                "attributes": state.get("attributes", {}),
                "area_id": area_id,
                "area_name": area_name,
            }

        # Logged grouped by area when it changed: refreshed every few minutes
        if set(self.entities) != known:
            by_area: dict[str, list[str]] = {}
            for eid, info in self.entities.items():
                area = info["area_name"] or "Sans pièce"
                by_area.setdefault(area, []).append(f"{info['friendly_name']} ({info['domain']})")
            for area, items in sorted(by_area.items()):
                logger.info(f"  {area}: {', '.join(items)}")
            logger.info(f"Total: {len(self.entities)} entities in {len(by_area)} areas")

        screens = await self._fetch_screens()
        if screens != self.screens:
            self.screens = screens
            logger.info(f"Screens: {', '.join(screens) or 'none'}")

    def get_entity_list_for_prompt(self) -> str:
        """Return entity names grouped by room for the system prompt."""
        by_area: dict[str, dict[str, list[str]]] = {}
        for info in self.entities.values():
            # A switch HA puts in no room is a setting more than a plug (Charras's 56 alarm-zone
            # switches, ~600 tokens; the Paris Sonos's loudness): still reachable by its name
            if info["domain"] == "switch" and not info["area_name"]:
                continue
            area = info["area_name"] or "Autre"
            domain_label = {
                "light": "lumières",
                "switch": "prises",
                "cover": "volets",
                "climate": "thermostats",
                "media_player": "médias",
                "lock": "serrures",
            }.get(info["domain"], info["domain"])
            # Roof windows and mosquito screens are covers too — list them apart so the LLM never
            # offers a skylight, or a screen, when the user asked for a roller shutter.
            if info["domain"] == "cover":
                domain_label = {"fenetre": "fenêtres", "moustiquaire": "moustiquaires"}.get(
                    cover_kind(info), domain_label
                )
            by_area.setdefault(area, {}).setdefault(domain_label, []).append(info["friendly_name"])

        parts = []
        for area, devices in sorted(by_area.items()):
            device_parts = []
            for dtype, names in sorted(devices.items()):
                device_parts.append(f"{dtype}: {', '.join(names)}")
            parts.append(f"{area} ({'; '.join(device_parts)})")
        # Append room groups
        if ROOM_GROUPS:
            group_parts = [f"{name} = {' + '.join(rooms)}" for name, rooms in ROOM_GROUPS.items()]
            parts.append(f"Groupes: {', '.join(group_parts)}")
        return ". ".join(parts)

    # Generic names that mean "all entities of this domain in the room".
    # Keys are normalize()d; the second element narrows covers to a sub-kind.
    _GENERIC_NAMES = {
        **{word: ("cover", kind) for kind, words in COVER_KIND_KEYWORDS.items() for word in words},
        "lumiere": ("light", None),
        "lumieres": ("light", None),
        "lampe": ("light", None),
        "lampes": ("light", None),
        "prise": ("switch", None),
        "prises": ("switch", None),
        "porte": ("lock", None),
        "portes": ("lock", None),
        "serrure": ("lock", None),
        "serrures": ("lock", None),
        "verrou": ("lock", None),
    }

    def is_generic(self, name: str) -> bool:
        """Whether a name says a kind of device (« volets », « la lumière ») rather than one device."""
        return normalize(name) in self._GENERIC_NAMES

    def knows_room(self, room: Optional[str]) -> bool:
        """Whether a room is one HA has, a group of them, or the whole house."""
        if not room:
            return False
        return is_whole_house(room) or self._area_exists(normalize(room)) or _group_rooms(room) is not None

    def _area_exists(self, norm_room: str) -> bool:
        """Whether HA knows this area, using the same loose matching as the filter."""
        return any(_area_matches(normalize(area), norm_room) for area in self.areas.values())

    def _filter_by_cover_kind(self, entity_ids: list[str], kind: Optional[str]) -> list[str]:
        """Keep only covers of the requested kind (volet / fenetre / moustiquaire)."""
        if not kind:
            return entity_ids
        covers = [e for e in entity_ids if self.entities[e]["domain"] == "cover"]
        # If no cover here is named for its kind, this house doesn't make the
        # distinction — keep everything rather than resolving to nothing.
        if not any(cover_kind(self.entities[e]) for e in covers):
            return entity_ids
        return [e for e in entity_ids if self.entities[e]["domain"] != "cover" or cover_kind(self.entities[e]) == kind]

    def resolve_all_entities(
        self, name: str, room: str | None = None, domain_hints: list[str] | None = None
    ) -> list[str]:
        """Resolve entity name, expanding room groups and generic names into multiple entity_ids."""
        # « lumière du salon » with no room: a generic word, then where
        words = normalize(name).split()
        if not room and len(words) > 1 and words[0] in self._GENERIC_NAMES:
            name, room = words[0], " ".join(words[1:])
        whole_house = is_whole_house(room)
        is_group = False
        rooms_to_search = [room]
        # Generic name (e.g. "volet" -> all shutters in the room), with its cover sub-kind
        target_domain, generic_kind = self._GENERIC_NAMES.get(normalize(name), (None, None))
        if target_domain:
            # A generic name of another kind names nothing this action can reach: « ferme la
            # porte » sent as a shutter must close no shutter, nor send a cover service to the lock
            if domain_hints and target_domain not in domain_hints:
                logger.info(f"'{name}' is a {target_domain}, not one of {domain_hints}")
                return []
            # It keeps to its own domain, fallbacks included: « la lumière » of a room without
            # one is not its speaker or a plug; « la porte » is the lock, not a switch named Porte
            domain_hints = [target_domain]

        # Expand room groups (e.g. "enfants" -> ["Chambre Zoé", "Chambre Charlie"])
        if room and not whole_house and (group_rooms := _group_rooms(room)) is not None:
            rooms_to_search, is_group = group_rooms, True

        # A group sweeps under a name that says a kind of cover: for « ferme les volets des enfants »,
        # gemma4:12b names the shutters of one room (« Petit volet », « Grand volet », 6 times out of
        # 6) and never Charlie's « Volet ». Any other name is that device in each of the group's
        # rooms: « la lampe de chevet des enfants » is not every light there.
        name_kind = cover_kind({"friendly_name": name})
        named_device = is_group and not (target_domain or name_kind or is_whole_house(name))
        sweep = (is_group or whole_house) and not named_device

        # ALL entities in scope, for a room group, the whole house, or a generic name in a room
        # ("les volets des enfants" -> all covers in all children's rooms)
        if target_domain:
            domains = [target_domain]
        elif sweep and domain_hints:
            # A sweep that names no kind (« éteins tout ») takes no switch: at Charras one cuts
            # the Freebox Wi-Fi, others are speaker settings
            domains = [d for d in domain_hints if d != "switch"]
        else:
            domains = None
        if domains and room:
            areas = [normalize(r) for r in rooms_to_search]

            def in_scope(info: dict) -> bool:
                norm_area = normalize(info.get("area_name", ""))
                if whole_house:
                    # Every light, placed in a room or not; a sweep over several domains keeps to
                    # devices in a room, leaving out what HA has nowhere (the Paris Sonos)
                    return bool(target_domain or norm_area)
                return any(_area_matches(norm_area, a) for a in areas)

            entity_ids = [eid for eid, info in self.entities.items() if info["domain"] in domains and in_scope(info)]
            # The kind the name says (« Fenêtre… », « Volet… »); a sweep naming none leaves the
            # roof windows shut, only the shutters move
            kind = generic_kind or name_kind or ("volet" if sweep else None)
            entity_ids = self._filter_by_cover_kind(entity_ids, kind)
            if entity_ids:
                logger.info(f"Resolve all: '{name}' (room={room}) -> {entity_ids}")
                return entity_ids

        # Specific name: resolve individually per room
        entity_ids = []
        for r in rooms_to_search:
            eid = self.resolve_entity(name, room=r, domain_hints=domain_hints)
            if eid and eid not in entity_ids:
                entity_ids.append(eid)
        return entity_ids

    def _candidates(self, norm_name: str, norm_room: Optional[str], domain_hints: Optional[list[str]]) -> dict:
        """entity_id -> normalized name, for the domains and room given."""
        found = {}
        for eid, info in self.entities.items():
            if domain_hints and info["domain"] not in domain_hints:
                continue
            if norm_room and not _area_matches(normalize(info.get("area_name", "")), norm_room):
                continue
            found[eid] = normalize(info["friendly_name"])
        # "les volets" must never fall back onto a roof window (both are covers)
        kind = self._GENERIC_NAMES.get(norm_name, (None, None))[1]
        if kind:
            found = {eid: found[eid] for eid in self._filter_by_cover_kind(list(found), kind)}
        return found

    def resolve_entity(
        self, name: str, room: str | None = None, domain_hints: list[str] | None = None
    ) -> Optional[str]:
        """Match a spoken name to an entity_id, optionally scoped to a room."""
        norm_input = normalize(name)
        norm_room = normalize(room) if room else None
        generic = self.is_generic(name)

        candidates = self._candidates(norm_input, norm_room, domain_hints)
        in_known_room = bool(candidates) and bool(norm_room)
        if not candidates and norm_room:
            # When the area exists and simply holds no such device, searching the rest of the house
            # is how "ouvre tous les volets sauf celui de Charlie" ended up opening Charlie's: the LLM
            # invented a shutter in Zoé's room, nothing matched there, and the unscoped retry landed
            # on the one room to avoid. Admitting there is no such device beats actuating another room's.
            if self._area_exists(norm_room):
                logger.info(f"No match for '{name}' in room '{room}' (room exists, no such device)")
                return None
            # A room HA doesn't know: Paris names some lights after their room instead of placing
            # them in one (« Couloir », « Bibliothèque »), so the room's words in a name locate it
            room_words = set(norm_room.split())
            candidates = {
                eid: n
                for eid, n in self._candidates(norm_input, None, domain_hints).items()
                if room_words <= set(n.split())
            }
            if not candidates:
                # "salon" for an area called "Living Room": the name alone may still say which
                logger.info(f"Room '{room}' unknown to HA, retrying without room filter")
                return self.resolve_entity(name, room=None, domain_hints=domain_hints)
        if not candidates:
            return None

        # A generic name says which device only within a room: without one, « volets » held the
        # « Volet » of the guest room, first of three so named among nine shutters (AD-058)
        if norm_input and (norm_room or not generic):
            # The closest name containing the request (the exact name first: it is the shortest);
            # else the longest name the request contains (« lumière du salon » holds « salon »)
            containing = sorted(
                (eid for eid, n in candidates.items() if norm_input in n), key=lambda e: len(candidates[e])
            )
            contained = sorted(
                (eid for eid, n in candidates.items() if n and n in norm_input), key=lambda e: -len(candidates[e])
            )
            for how, found in (("containing", containing), ("contained", contained)):
                if found:
                    logger.info(f"Entity match: '{name}' (room='{room}') -> {found[0]} ({how})")
                    return found[0]

        # The one device left is meant when the request names no particular device (« la lumière »,
        # or the room itself for a speaker), when a one-domain action is scoped to a room HA knows
        # (the only cover of the bedroom), or for a lock, whose names are said many ways. Not
        # otherwise: "télé" is not the light Entrée, nor a Sonos left out the Google Home.
        unnamed = not norm_input or generic or norm_input == norm_room
        one_domain = bool(domain_hints) and len(domain_hints) == 1
        any_name = domain_hints == ["lock"] or (in_known_room and one_domain)
        if len(candidates) == 1 and (unnamed or any_name):
            eid = next(iter(candidates))
            logger.info(f"Entity match: '{name}' (room='{room}') -> {eid} (only match in room)")
            return eid

        # « la lumière » of a room with several: the first one
        if norm_room and generic:
            eid = next(iter(candidates))
            logger.info(f"Entity match: '{name}' (room='{room}') -> {eid} (generic name, first in room)")
            return eid

        # A misheard or slightly different name; never for a generic one, which is no name at all
        if norm_input and not generic:
            norm_to_eid = {v: k for k, v in candidates.items()}
            matches = difflib.get_close_matches(norm_input, norm_to_eid.keys(), n=1, cutoff=_FUZZY_CUTOFF)
            if matches:
                eid = norm_to_eid[matches[0]]
                logger.info(f"Entity match: '{name}' -> {eid} (fuzzy: {matches[0]})")
                return eid

        logger.warning(f"No entity match for '{name}' (room={room})")
        return None

    # Never by voice, whatever route a call takes to get here (AD-041); unlocking is unlock()'s alone (AD-068)
    _FORBIDDEN_SERVICES = {("lock", "unlock"), ("lock", "open")}

    async def call_service(self, domain: str, service: str, entity_id: str, **kwargs) -> str:
        """Call a HA service and return a French result string for TTS."""
        if (domain, service) in self._FORBIDDEN_SERVICES:
            logger.warning(f"Refused {domain}.{service} -> {entity_id}")
            return "Le déverrouillage ne se fait pas à la voix"
        return await self._call_service(domain, service, entity_id, **kwargs)

    async def unlock(self, entity_id: str) -> str:
        """Unlock a door: only for a tap on the web app's confirmation button, never for the model (AD-068)."""
        return await self._call_service("lock", "unlock", entity_id)

    async def _call_service(self, domain: str, service: str, entity_id: str, **kwargs) -> str:
        payload = {"entity_id": entity_id}
        payload.update(kwargs)

        display_name = self.display_name(entity_id)

        try:
            url = f"{self.base_url}/api/services/{domain}/{service}"
            logger.info(f"HA service call: {domain}.{service} -> {entity_id} {kwargs}")

            async with self._session.post(url, json=payload) as resp:
                if resp.status == 200:
                    return self._build_response(domain, service, display_name, kwargs)
                else:
                    error = await resp.text()
                    logger.error(f"HA service error {resp.status}: {error}")
                    return f"Erreur lors du contrôle de {display_name}"
        except Exception as e:
            logger.error(f"HA service call failed: {e}")
            return "Impossible de contacter Home Assistant"

    async def search_sensors(self, query: str, limit: int = 15) -> str:
        """Search read-only sensors by keyword and return their current values.

        Sensors are deliberately kept out of `self.entities` and out of the system
        prompt: there are hundreds of them (273 at Charras vs 92 actionable
        entities) and they are only ever read, never controlled. Searching on
        demand keeps the prompt small while leaving every sensor reachable —
        including those HA gives no device_class (pool pH, salinity, chlorine).
        States are read live rather than from the discovery snapshot, since the
        whole point of a sensor is that its value moves.
        """
        if not normalize(query).split():
            return "Précise ce que tu cherches"

        try:
            async with self._session.get(f"{self.base_url}/api/states") as resp:
                if resp.status != 200:
                    return "Capteurs non disponibles"
                states = await resp.json()
        except Exception as e:
            logger.error(f"Error fetching sensor states: {e}")
            return "Capteurs non disponibles"

        return self.match_sensors(states, query, limit)

    def match_sensors(self, states: list[dict], query: str, limit: int = 15) -> str:
        """Rank sensor states against a keyword query. Pure — see search_sensors."""
        words = normalize(query).split()
        matches = []
        for state in states:
            eid = state["entity_id"]
            if not eid.startswith("sensor."):
                continue
            value = state["state"]
            if value in ("unknown", "unavailable", ""):
                continue
            attrs = state.get("attributes", {})
            name = attrs.get("friendly_name") or eid
            area = self.areas.get(self.entity_areas.get(eid, ""), "")
            score = sum(1 for w in words if w in normalize(f"{name} {area}"))
            if not score:
                continue
            unit = attrs.get("unit_of_measurement") or ""
            # A measurement beats a raw status code when results get truncated.
            matches.append((score, bool(unit), f"{name}: {value} {unit}".strip()))

        if not matches:
            return f"Aucun capteur trouvé pour « {query} »"

        matches.sort(key=lambda m: (-m[0], not m[1], m[2]))
        lines = [m[2] for m in matches[:limit]]
        if len(matches) > limit:
            lines.append(f"({len(matches) - limit} autres capteurs non affichés)")
        logger.info(f"Sensor search '{query}' -> {len(matches)} matches")
        return "\n".join(lines)

    async def get_entity_state(self, entity_id: str) -> Optional[dict]:
        """Get the current state of an entity."""
        try:
            async with self._session.get(f"{self.base_url}/api/states/{entity_id}") as resp:
                if resp.status == 200:
                    return await resp.json()
                return None
        except Exception as e:
            logger.error(f"Error getting state for {entity_id}: {e}")
            return None

    def display_name(self, entity_id: str) -> str:
        """The device's name for the reply, with its room for a generic one like "Plafonnier" or "Volet"."""
        info = self.entities.get(entity_id, {})
        friendly = info.get("friendly_name", entity_id)
        area = info.get("area_name", "")
        return f"{friendly} {area}" if area and friendly in ("Plafonnier", "Volet") else friendly

    def offline_reply(self, entity_ids: list[str]) -> str:
        """The result for devices HA cannot reach, which the reply must not report done. A count
        past a few: the closed Charras house has 18 Velux, which the reply would read out."""
        if len(entity_ids) == 1:
            return f"{self.display_name(entity_ids[0])} ne répond pas (hors ligne)"
        names = f"{len(entity_ids)} appareils" if len(entity_ids) > 3 else ", ".join(map(self.display_name, entity_ids))
        return f"{names} ne répondent pas (hors ligne)"

    def _unavailable(self, entity_id: str) -> bool:
        return self.entities.get(entity_id, {}).get("state") == "unavailable"

    async def reachable(self, entity_ids: list[str]) -> list[str]:
        """The devices HA can reach among these. Those the last refresh found unavailable are read
        again, in one short call: a speaker woken up since would otherwise be refused for five minutes."""
        if suspects := [eid for eid in entity_ids if self._unavailable(eid)]:
            template = "{% for e in " + json.dumps(suspects) + " %}{{ e }} {{ states(e) }}\n{% endfor %}"
            text = await self._render_template(template, timeout=2)
            for line in (text or "").splitlines():
                eid, _, state = line.partition(" ")
                if eid in self.entities:
                    self.entities[eid]["state"] = state
        return [eid for eid in entity_ids if not self._unavailable(eid)]

    async def playing_volumes(self) -> dict[str, float]:
        """The speakers playing, with their volume; not a voice satellite, which lowers its own music."""
        text = await self._render_template(
            "{% for s in states.media_player if s.state == 'playing' and s.attributes.volume_level is number"
            " and not s.attributes.is_volume_muted and 'Voice' not in (device_attr(s.entity_id, 'model') or '') %}"
            "{{ s.entity_id }} {{ s.attributes.volume_level }}\n{% endfor %}",
            timeout=2,
        )
        return {eid: float(level) for eid, _, level in (line.partition(" ") for line in (text or "").splitlines())}

    async def set_volumes(self, volumes: dict[str, float]) -> None:
        calls = (self.call_service("media_player", "volume_set", eid, volume_level=v) for eid, v in volumes.items())
        await asyncio.gather(*calls)

    def _build_response(self, domain: str, service: str, display_name: str, kwargs: dict) -> str:
        """Build a natural French TTS response for a service call."""
        # Include brightness in response when specified
        if domain == "light" and service == "turn_on" and "brightness" in kwargs:
            pct = round(kwargs["brightness"] / 255 * 100)
            return f"{display_name} réglé à {pct}%"

        responses = {
            ("light", "turn_on"): f"{display_name} allumé",
            ("light", "turn_off"): f"{display_name} éteint",
            ("switch", "turn_on"): f"{display_name} allumé",
            ("switch", "turn_off"): f"{display_name} éteint",
            ("cover", "open_cover"): f"{display_name} en cours d'ouverture",
            ("cover", "close_cover"): f"{display_name} en cours de fermeture",
            ("cover", "set_cover_position"): f"{display_name} réglé à {kwargs.get('position', '?')}%",
            (
                "climate",
                "set_temperature",
            ): f"Température réglée à {kwargs.get('temperature', '?')} degrés",
            ("media_player", "turn_on"): f"{display_name} allumé",
            ("media_player", "turn_off"): f"{display_name} éteint",
            ("media_player", "play_media"): f"Lecture sur {display_name}",
            ("media_player", "media_stop"): f"{display_name} arrêté",
            ("media_player", "media_pause"): f"{display_name} en pause",
            ("media_player", "media_play"): f"Lecture sur {display_name}",
            (
                "media_player",
                "volume_set",
            ): f"Volume à {round(kwargs.get('volume_level', 0) * 100)} pour cent",
            ("media_player", "volume_up"): "Plus fort",
            ("media_player", "volume_down"): "Moins fort",
            ("lock", "lock"): f"{display_name} en cours de verrouillage",
            ("lock", "unlock"): f"{display_name} en cours de déverrouillage",
        }
        return responses.get((domain, service), "C'est fait")

    def format_state_for_speech(self, entity_id: str, state_data: dict) -> str:
        """Format an entity state as a French sentence for TTS."""
        friendly = state_data.get("attributes", {}).get("friendly_name", entity_id)
        state = state_data.get("state", "inconnu")
        domain = entity_id.split(".")[0]

        # Not "éteint" for a switch nobody can reach
        if state == "unavailable":
            return self.offline_reply([entity_id])
        if state == "unknown":
            return f"L'état de {friendly} est inconnu"

        if domain == "light":
            if state == "on":
                brightness = state_data.get("attributes", {}).get("brightness")
                if brightness:
                    pct = round(brightness / 255 * 100)
                    return f"{friendly} est allumé à {pct} pourcent"
                return f"{friendly} est allumé"
            return f"{friendly} est éteint"

        if domain == "cover":
            state_map = {
                "open": "ouvert",
                "closed": "fermé",
                "opening": "en cours d'ouverture",
                "closing": "en cours de fermeture",
            }
            position = state_data.get("attributes", {}).get("current_position")
            label = state_map.get(state, state)
            if position is not None and 0 < position < 100:
                return f"{friendly} est {label} à {position} pourcent"
            return f"{friendly} est {label}"

        if domain == "climate":
            temp = state_data.get("attributes", {}).get("current_temperature")
            target = state_data.get("attributes", {}).get("temperature")
            if temp:
                msg = f"Il fait {temp} degrés"
                if target:
                    msg += f", consigne à {target} degrés"
                return msg
            return f"{friendly} est {state}"

        if domain == "switch":
            return f"{friendly} est {'allumé' if state == 'on' else 'éteint'}"

        if domain == "lock":
            state_map = {
                "locked": "verrouillée",
                "unlocked": "déverrouillée",
                "locking": "en cours de verrouillage",
                "unlocking": "en cours de déverrouillage",
                "jammed": "bloquée",
                "open": "ouverte",
                "opening": "en cours d'ouverture",
            }
            return f"{friendly} est {state_map.get(state, state)}"

        return f"{friendly} est {state}"
