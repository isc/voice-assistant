"""
Home Assistant REST API client for voice assistant integration.
Handles entity discovery, fuzzy matching, and service calls.
"""

import logging
import unicodedata
import difflib
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)

# Domains we expose as LLM tools
SUPPORTED_DOMAINS = {"light", "switch", "cover", "climate", "media_player"}

# Local name overrides for entities with bad/missing friendly_name in HA
# Maps entity_id -> friendly_name
ENTITY_NAME_OVERRIDES = {}

# French stopwords to strip for fuzzy matching
STOPWORDS = {"le", "la", "les", "l", "du", "de", "des", "un", "une", "d"}


def normalize(text: str) -> str:
    """Normalize text for fuzzy matching: lowercase, strip accents and stopwords."""
    text = text.lower().strip()
    # Strip accents
    text = "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )
    # Remove stopwords
    words = [w for w in text.split() if w not in STOPWORDS]
    return " ".join(words)


class HAClient:
    """Home Assistant REST API client with entity discovery and fuzzy matching."""

    def __init__(self, base_url: str, token: str):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.entities: dict[str, dict] = {}  # entity_id -> {friendly_name, domain, state, attributes}
        self._session: Optional[aiohttp.ClientSession] = None

    async def connect(self) -> bool:
        """Create session, verify connectivity, fetch initial states."""
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

        await self.refresh_entities()
        return True

    async def close(self):
        """Close the aiohttp session."""
        if self._session:
            await self._session.close()

    async def refresh_entities(self):
        """Fetch all states from HA and filter to supported domains."""
        try:
            async with self._session.get(f"{self.base_url}/api/states") as resp:
                if resp.status != 200:
                    logger.error(f"Failed to fetch HA states: {resp.status}")
                    return
                states = await resp.json()
        except Exception as e:
            logger.error(f"Error fetching HA states: {e}")
            return

        self.entities.clear()
        for state in states:
            entity_id = state["entity_id"]
            domain = entity_id.split(".")[0]
            if domain not in SUPPORTED_DOMAINS:
                continue

            has_override = entity_id in ENTITY_NAME_OVERRIDES
            friendly_name = ENTITY_NAME_OVERRIDES.get(
                entity_id,
                state.get("attributes", {}).get("friendly_name", entity_id),
            )

            # Skip filters for entities with explicit name overrides
            if not has_override:
                # Skip Fibaro sub-entities (Basic channels and (2) duplicates)
                if "Basic" in friendly_name or "basic" in entity_id:
                    continue
                if friendly_name.endswith("(2)"):
                    continue
                # Skip Fibaro Dimmer 2 secondary channels (entity_id pattern: dimmer_2_2_N)
                if "_2_2_" in entity_id or entity_id.endswith("_2_2"):
                    continue
                # Skip entities with no useful name
                if friendly_name.strip() in ("Dimmer 2", "Double Smart Module", "Single Switch 2", "Smart Module", ""):
                    continue
                # Skip unavailable entities
                if state["state"] == "unavailable":
                    continue

            self.entities[entity_id] = {
                "friendly_name": friendly_name.strip(),
                "domain": domain,
                "state": state["state"],
                "attributes": state.get("attributes", {}),
            }

        by_domain = {}
        for eid, info in self.entities.items():
            by_domain.setdefault(info["domain"], []).append(info["friendly_name"])

        for domain, names in sorted(by_domain.items()):
            logger.info(f"  {domain}: {', '.join(names)}")
        logger.info(f"Total: {len(self.entities)} entities across {len(by_domain)} domains")

    def get_entity_list_for_prompt(self) -> str:
        """Return a compact string of entity names grouped by type for the system prompt."""
        by_domain = {}
        for info in self.entities.values():
            label = {
                "light": "Lumières",
                "switch": "Prises",
                "cover": "Volets",
                "climate": "Thermostats",
                "media_player": "Médias",
            }.get(info["domain"], info["domain"])
            by_domain.setdefault(label, []).append(info["friendly_name"])

        parts = []
        for label, names in sorted(by_domain.items()):
            parts.append(f"{label}: {', '.join(names)}")
        return ". ".join(parts)

    def resolve_entity(self, name: str, domain_hints: list[str] | None = None) -> Optional[str]:
        """Fuzzy-match a natural language name to an entity_id."""
        norm_input = normalize(name)
        if not norm_input:
            return None

        # Filter candidates by domain if hints provided
        candidates = {}
        for eid, info in self.entities.items():
            if domain_hints and info["domain"] not in domain_hints:
                continue
            candidates[eid] = normalize(info["friendly_name"])

        if not candidates:
            return None

        # Exact substring match (both directions)
        for eid, norm_name in candidates.items():
            if norm_input in norm_name or norm_name in norm_input:
                logger.info(f"Entity match: '{name}' -> {eid} (substring)")
                return eid

        # Fuzzy match with difflib
        norm_to_eid = {v: k for k, v in candidates.items()}
        matches = difflib.get_close_matches(norm_input, norm_to_eid.keys(), n=1, cutoff=0.4)
        if matches:
            eid = norm_to_eid[matches[0]]
            logger.info(f"Entity match: '{name}' -> {eid} (fuzzy: {matches[0]})")
            return eid

        logger.warning(f"No entity match for '{name}'")
        return None

    async def call_service(self, domain: str, service: str, entity_id: str, **kwargs) -> str:
        """Call a HA service and return a French result string for TTS."""
        payload = {"entity_id": entity_id}
        payload.update(kwargs)

        friendly = self.entities.get(entity_id, {}).get("friendly_name", entity_id)

        try:
            url = f"{self.base_url}/api/services/{domain}/{service}"
            logger.info(f"HA service call: {domain}.{service} -> {entity_id} {kwargs}")

            async with self._session.post(url, json=payload) as resp:
                if resp.status == 200:
                    # Build natural French response
                    return self._build_response(domain, service, friendly, kwargs)
                else:
                    error = await resp.text()
                    logger.error(f"HA service error {resp.status}: {error}")
                    return f"Erreur lors du contrôle de {friendly}"
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

    def _build_response(self, domain: str, service: str, friendly_name: str, kwargs: dict) -> str:
        """Build a natural French TTS response for a service call."""
        responses = {
            ("light", "turn_on"): f"{friendly_name} allumé",
            ("light", "turn_off"): f"{friendly_name} éteint",
            ("switch", "turn_on"): f"{friendly_name} allumé",
            ("switch", "turn_off"): f"{friendly_name} éteint",
            ("cover", "open_cover"): f"{friendly_name} en cours d'ouverture",
            ("cover", "close_cover"): f"{friendly_name} en cours de fermeture",
            ("cover", "stop_cover"): f"{friendly_name} arrêté",
            ("climate", "set_temperature"): f"Température réglée à {kwargs.get('temperature', '?')} degrés",
            ("media_player", "turn_on"): f"{friendly_name} allumé",
            ("media_player", "turn_off"): f"{friendly_name} éteint",
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
