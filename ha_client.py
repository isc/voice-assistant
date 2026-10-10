"""
Home Assistant REST API client for voice assistant integration.
Handles entity discovery with area grouping, fuzzy matching, and service calls.
"""

import difflib
import json
import logging
import os
import unicodedata
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)

# Domains we expose as LLM tools
SUPPORTED_DOMAINS = {"light", "switch", "cover", "climate", "media_player"}

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


def normalize(text: str) -> str:
    """Normalize text for fuzzy matching: lowercase, strip accents and stopwords."""
    text = text.lower().strip()
    text = "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
    words = [w for w in text.split() if w not in STOPWORDS]
    return " ".join(words)


def compact(text: str) -> str:
    """normalize() without spaces or punctuation, for names said many ways ("Rire & Chansons")."""
    return "".join(c for c in normalize(text) if c.isalnum())


# Sub-kinds inside the `cover` domain. A house can have roller shutters and roof
# windows side by side — both are HA covers, but they are not interchangeable
# (opening a skylight when asked for a shutter lets the rain in).
# HA's device_class is authoritative; the name is only consulted for integrations
# that leave it unset (HomeKit Controller exposes Velux shutters with no class).
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
}


def cover_kind(info: dict) -> Optional[str]:
    """Classify a cover as 'volet' or 'fenetre', None if undetermined."""
    device_class = info.get("attributes", {}).get("device_class")
    if device_class in COVER_KIND_BY_DEVICE_CLASS:
        return COVER_KIND_BY_DEVICE_CLASS[device_class]
    words = set(normalize(info.get("friendly_name", "")).split())
    for kind, keywords in COVER_KIND_KEYWORDS.items():
        if words.intersection(keywords):
            return kind
    return None


class HAClient:
    """Home Assistant REST API client with area-based entity discovery."""

    def __init__(self, base_url: str, token: str):
        self.base_url = base_url.rstrip("/")
        self.token = token
        # entity_id -> {friendly_name, domain, state, attributes, area_id, area_name}
        self.entities: dict[str, dict] = {}
        self.areas: dict[str, str] = {}  # area_id -> area_name
        # entity_id -> area_id, for every domain — sensors aren't in self.entities
        # but search_sensors still wants to know which room they belong to.
        self.entity_areas: dict[str, str] = {}
        # The home as configured in HA: latitude, longitude, time_zone
        self.home: dict = {}
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

        await self._fetch_areas()
        await self.refresh_entities()
        return True

    async def close(self):
        """Close the aiohttp session."""
        if self._session:
            await self._session.close()

    async def _fetch_areas(self):
        """Fetch area registry via HA template API."""
        try:
            r = await self._session.post(
                f"{self.base_url}/api/template",
                json={"template": "{% for a in areas() %}{{ a }}|{{ area_name(a) }}\n{% endfor %}"},
            )
            if r.status != 200:
                logger.warning("Could not fetch areas")
                return
            text = await r.text()
            self.areas.clear()
            for line in text.strip().split("\n"):
                if "|" in line:
                    area_id, area_name = line.split("|", 1)
                    self.areas[area_id.strip()] = area_name.strip()
            logger.info(f"Areas: {', '.join(self.areas.values())}")
        except Exception as e:
            logger.warning(f"Error fetching areas: {e}")

    async def _fetch_entity_areas(self) -> dict[str, str]:
        """Fetch entity -> area_id mapping via HA template API."""
        entity_to_area = {}
        for area_id in self.areas:
            try:
                r = await self._session.post(
                    f"{self.base_url}/api/template",
                    json={"template": f"{{% for e in area_entities('{area_id}') %}}{{{{ e }}}}\n{{% endfor %}}"},
                )
                if r.status == 200:
                    text = await r.text()
                    for line in text.strip().split("\n"):
                        eid = line.strip()
                        if eid:
                            entity_to_area[eid] = area_id
            except Exception as e:
                logger.warning(f"Error fetching entities for area {area_id}: {e}")
        return entity_to_area

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

        entity_to_area = await self._fetch_entity_areas()
        self.entity_areas = entity_to_area

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
            if state["state"] == "unavailable":
                continue

            area_id = entity_to_area.get(entity_id)
            area_name = self.areas.get(area_id, "") if area_id else ""

            self.entities[entity_id] = {
                "friendly_name": friendly_name.strip(),
                "domain": domain,
                "state": state["state"],
                "attributes": state.get("attributes", {}),
                "area_id": area_id,
                "area_name": area_name,
            }

        # Log grouped by area
        by_area: dict[str, list[str]] = {}
        for eid, info in self.entities.items():
            area = info["area_name"] or "Sans pièce"
            by_area.setdefault(area, []).append(f"{info['friendly_name']} ({info['domain']})")
        for area, items in sorted(by_area.items()):
            logger.info(f"  {area}: {', '.join(items)}")
        logger.info(f"Total: {len(self.entities)} entities in {len(by_area)} areas")

    def get_entity_list_for_prompt(self) -> str:
        """Return entity names grouped by room for the system prompt."""
        by_area: dict[str, dict[str, list[str]]] = {}
        for info in self.entities.values():
            area = info["area_name"] or "Autre"
            domain_label = {
                "light": "lumières",
                "switch": "prises",
                "cover": "volets",
                "climate": "thermostats",
                "media_player": "médias",
            }.get(info["domain"], info["domain"])
            # Roof windows are covers too — list them apart so the LLM never
            # offers a skylight when the user asked for a roller shutter.
            if info["domain"] == "cover" and cover_kind(info) == "fenetre":
                domain_label = "fenêtres"
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
        "volet": ("cover", "volet"),
        "volets": ("cover", "volet"),
        "store": ("cover", "volet"),
        "stores": ("cover", "volet"),
        "fenetre": ("cover", "fenetre"),
        "fenetres": ("cover", "fenetre"),
        "velux": ("cover", "fenetre"),
        "lumiere": ("light", None),
        "lumieres": ("light", None),
    }

    def _area_exists(self, norm_room: str) -> bool:
        """Whether HA knows this area, using the same loose matching as the filter."""
        return any(norm_room in normalize(area) or normalize(area) in norm_room for area in self.areas.values() if area)

    def _filter_by_cover_kind(self, entity_ids: list[str], kind: Optional[str]) -> list[str]:
        """Keep only covers of the requested kind (volet / fenetre)."""
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
        is_group = False
        rooms_to_search = [room]
        # Generic name (e.g. "volet" -> all shutters in the room), with its cover sub-kind
        target_domain, generic_kind = self._GENERIC_NAMES.get(normalize(name), (None, None))

        # Expand room groups (e.g. "enfants" -> ["Chambre Zoé", "Chambre Charlie"])
        if room:
            norm_room = normalize(room)
            for group_name, group_rooms in ROOM_GROUPS.items():
                if norm_room == normalize(group_name):
                    rooms_to_search = group_rooms
                    is_group = True
                    break

        # For room groups: return ALL entities of the matching domain(s)
        # (user says "les volets des enfants" -> all covers in all children's rooms)
        if is_group and domain_hints:
            entity_ids = []
            for r in rooms_to_search:
                norm_r = normalize(r)
                for eid, info in self.entities.items():
                    if info["domain"] not in domain_hints:
                        continue
                    norm_area = normalize(info.get("area_name", ""))
                    if not norm_area:
                        continue
                    if norm_r not in norm_area and norm_area not in norm_r:
                        continue
                    if eid not in entity_ids:
                        entity_ids.append(eid)
            entity_ids = self._filter_by_cover_kind(entity_ids, generic_kind)
            if entity_ids:
                logger.info(f"Group resolve: '{name}' (group={room}) -> {entity_ids}")
                return entity_ids

        if target_domain and any(r is not None for r in rooms_to_search):
            entity_ids = []
            for r in rooms_to_search:
                norm_r = normalize(r) if r else None
                for eid, info in self.entities.items():
                    if info["domain"] != target_domain:
                        continue
                    if norm_r:
                        norm_area = normalize(info.get("area_name", ""))
                        if not norm_area:
                            continue
                        if norm_r not in norm_area and norm_area not in norm_r:
                            continue
                    if eid not in entity_ids:
                        entity_ids.append(eid)
            entity_ids = self._filter_by_cover_kind(entity_ids, generic_kind)
            if entity_ids:
                logger.info(f"Generic resolve: '{name}' (room={room}) -> {entity_ids}")
                return entity_ids

        # Specific name: resolve individually per room
        entity_ids = []
        for r in rooms_to_search:
            eid = self.resolve_entity(name, room=r, domain_hints=domain_hints)
            if eid and eid not in entity_ids:
                entity_ids.append(eid)
        return entity_ids

    def resolve_entity(
        self, name: str, room: str | None = None, domain_hints: list[str] | None = None
    ) -> Optional[str]:
        """Fuzzy-match a natural language name to an entity_id, optionally scoped to a room."""
        norm_input = normalize(name)
        norm_room = normalize(room) if room else None

        # Build candidate list filtered by domain and optionally room
        candidates = {}
        for eid, info in self.entities.items():
            if domain_hints and info["domain"] not in domain_hints:
                continue
            if norm_room:
                norm_area = normalize(info.get("area_name", ""))
                # No area means in no room, not in every one ("" is in any name)
                if not norm_area or (norm_room not in norm_area and norm_area not in norm_room):
                    continue
            candidates[eid] = normalize(info["friendly_name"])

        # "les volets" must never fall back onto a roof window (both are covers)
        kind = self._GENERIC_NAMES.get(norm_input, (None, None))[1]
        if kind:
            kept = self._filter_by_cover_kind(list(candidates), kind)
            candidates = {eid: candidates[eid] for eid in kept}

        if not candidates:
            # Dropping the room filter rescues a room name HA doesn't know ("salon"
            # for an area called "Living Room"). But when the area *does* exist and
            # simply holds no such device, searching the rest of the house is how
            # "ouvre tous les volets sauf celui de Charlie" ended up opening
            # Charlie's: the LLM invented a shutter in Zoé's room, nothing matched
            # there, and the unscoped retry landed on the one room to avoid.
            # Admitting there is no such device beats actuating another room's.
            if norm_room and not self._area_exists(norm_room):
                logger.info(f"Room '{room}' unknown to HA, retrying without room filter")
                return self.resolve_entity(name, room=None, domain_hints=domain_hints)
            if norm_room:
                logger.info(f"No match for '{name}' in room '{room}' (room exists, no such device)")
            return None

        # If room narrows to a single entity of the matching domain, use it directly
        if len(candidates) == 1:
            eid = next(iter(candidates))
            logger.info(f"Entity match: '{name}' (room='{room}') -> {eid} (only match in room)")
            return eid

        # Exact substring match
        for eid, norm_name in candidates.items():
            if norm_input and (norm_input in norm_name or norm_name in norm_input):
                logger.info(f"Entity match: '{name}' -> {eid} (substring)")
                return eid

        # If we have a room and multiple candidates, pick the first one
        # (common case: user says "la lumière" and there's only one light in that room)
        if norm_room and norm_input in self._GENERIC_NAMES:
            eid = next(iter(candidates))
            logger.info(f"Entity match: '{name}' (room='{room}') -> {eid} (generic name, first in room)")
            return eid

        # Fuzzy match with difflib
        if norm_input:
            norm_to_eid = {v: k for k, v in candidates.items()}
            matches = difflib.get_close_matches(norm_input, norm_to_eid.keys(), n=1, cutoff=0.4)
            if matches:
                eid = norm_to_eid[matches[0]]
                logger.info(f"Entity match: '{name}' -> {eid} (fuzzy: {matches[0]})")
                return eid

        logger.warning(f"No entity match for '{name}' (room={room})")
        return None

    async def call_service(self, domain: str, service: str, entity_id: str, **kwargs) -> str:
        """Call a HA service and return a French result string for TTS."""
        payload = {"entity_id": entity_id}
        payload.update(kwargs)

        info = self.entities.get(entity_id, {})
        friendly = info.get("friendly_name", entity_id)
        area = info.get("area_name", "")
        # Include room name in response for generic names like "Plafonnier" or "Volet"
        display_name = f"{friendly} {area}" if area and friendly in ("Plafonnier", "Volet") else friendly

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
            ("cover", "stop_cover"): f"{display_name} arrêté",
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
        }
        return responses.get((domain, service), "C'est fait")

    def format_state_for_speech(self, entity_id: str, state_data: dict) -> str:
        """Format an entity state as a French sentence for TTS."""
        friendly = state_data.get("attributes", {}).get("friendly_name", entity_id)
        state = state_data.get("state", "inconnu")
        domain = entity_id.split(".")[0]

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

        return f"{friendly} est {state}"
