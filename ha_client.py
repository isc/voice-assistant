"""
Home Assistant REST API client for voice assistant integration.
Handles entity discovery with area grouping, fuzzy matching, and service calls.
"""

import logging
import unicodedata
import difflib
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)

# Domains we expose as LLM tools
SUPPORTED_DOMAINS = {"light", "switch", "cover", "climate", "media_player"}

# French stopwords to strip for fuzzy matching
STOPWORDS = {"le", "la", "les", "l", "du", "de", "des", "un", "une", "d"}

# Room groups for multi-room commands (e.g. "ferme les volets des enfants")
ROOM_GROUPS = {
    "enfants": ["Chambre Zoé", "Chambre Charlie"],
    "tout": ["Chambre Zoé", "Chambre Charlie", "Chambre parents", "Chambre invités", "Living Room"],
    "toute la maison": ["Chambre Zoé", "Chambre Charlie", "Chambre parents", "Chambre invités", "Living Room"],
    "partout": ["Chambre Zoé", "Chambre Charlie", "Chambre parents", "Chambre invités", "Living Room"],
}


def normalize(text: str) -> str:
    """Normalize text for fuzzy matching: lowercase, strip accents and stopwords."""
    text = text.lower().strip()
    text = "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )
    words = [w for w in text.split() if w not in STOPWORDS]
    return " ".join(words)


class HAClient:
    """Home Assistant REST API client with area-based entity discovery."""

    def __init__(self, base_url: str, token: str):
        self.base_url = base_url.rstrip("/")
        self.token = token
        # entity_id -> {friendly_name, domain, state, attributes, area_id, area_name}
        self.entities: dict[str, dict] = {}
        self.areas: dict[str, str] = {}  # area_id -> area_name
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
            if friendly_name.strip() in ("Dimmer 2", "Double Smart Module", "Single Switch 2", "Smart Module", ""):
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

    # Generic names that mean "all entities of this domain in the room"
    _GENERIC_NAMES = {
        "volet": "cover", "volets": "cover", "les volets": "cover",
        "lumière": "light", "lumières": "light", "les lumières": "light",
        "lumiere": "light", "lumieres": "light",
    }

    def resolve_all_entities(
        self, name: str, room: str | None = None, domain_hints: list[str] | None = None
    ) -> list[str]:
        """Resolve entity name, expanding room groups and generic names into multiple entity_ids."""
        is_group = False
        rooms_to_search = [room]

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
            if entity_ids:
                logger.info(f"Group resolve: '{name}' (group={room}) -> {entity_ids}")
                return entity_ids

        # Check if name is generic (e.g. "volet" -> all covers in room)
        target_domain = self._GENERIC_NAMES.get(normalize(name))

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
                if norm_room not in norm_area and norm_area not in norm_room:
                    continue
            candidates[eid] = normalize(info["friendly_name"])

        if not candidates:
            # Retry without room filter if no match
            if norm_room:
                logger.info(f"No match in room '{room}', retrying without room filter")
                return self.resolve_entity(name, room=None, domain_hints=domain_hints)
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
        if norm_room and norm_input in ("lumiere", "lumières", "lumiere", "volet", "volets"):
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
        responses = {
            ("light", "turn_on"): f"{display_name} allumé",
            ("light", "turn_off"): f"{display_name} éteint",
            ("switch", "turn_on"): f"{display_name} allumé",
            ("switch", "turn_off"): f"{display_name} éteint",
            ("cover", "open_cover"): f"{display_name} en cours d'ouverture",
            ("cover", "close_cover"): f"{display_name} en cours de fermeture",
            ("cover", "stop_cover"): f"{display_name} arrêté",
            ("climate", "set_temperature"): f"Température réglée à {kwargs.get('temperature', '?')} degrés",
            ("media_player", "turn_on"): f"{display_name} allumé",
            ("media_player", "turn_off"): f"{display_name} éteint",
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
            state_map = {"open": "ouvert", "closed": "fermé", "opening": "en cours d'ouverture", "closing": "en cours de fermeture"}
            return f"{friendly} est {state_map.get(state, state)}"

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
