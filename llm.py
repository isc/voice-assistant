"""LLM integration — chat completion, tool calling, text fallback parser."""

import json
import logging
import re
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)

_TOOL_NAMES = {
    "turn_on",
    "turn_off",
    "open_cover",
    "close_cover",
    "set_cover_position",
    "set_temperature",
    "get_state",
    "set_timer",
    "set_alarm",
    "cancel_timer",
    "query_calendar",
    "create_event",
    "play_radio",
    "stop_media",
    "set_volume",
    "change_volume",
}


def get_tool_definitions(ha_client, calendar_client=None) -> list:
    """Return LLM tool definitions."""
    # Weather is always available (no HA dependency)
    weather_tool = {
        "name": "get_weather",
        "description": "Obtenir la météo actuelle et les prévisions. Par défaut Paris.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "Ville (ex: Paris, Angoulême, Lyon). Par défaut: Paris",
                },
            },
            "required": [],
        },
    }

    # Timer tools are always available (no HA dependency)
    timer_tools = [
        {
            "name": "set_timer",
            "description": "Mettre un minuteur (ex: 5 minutes, 1 heure 30)",
            "parameters": {
                "type": "object",
                "properties": {
                    "duration_minutes": {
                        "type": "number",
                        "description": "Durée en minutes (ex: 5, 1.5, 90)",
                    },
                    "label": {
                        "type": "string",
                        "description": "Nom optionnel du timer (ex: pâtes, lessive)",
                    },
                },
                "required": ["duration_minutes"],
            },
        },
        {
            "name": "set_alarm",
            "description": "Mettre une alarme ou un réveil à une heure précise",
            "parameters": {
                "type": "object",
                "properties": {
                    "time": {
                        "type": "string",
                        "description": "Heure au format HH:MM (ex: 07:00, 14:30)",
                    },
                    "label": {
                        "type": "string",
                        "description": "Nom optionnel (ex: réveil, sieste)",
                    },
                },
                "required": ["time"],
            },
        },
        {
            "name": "cancel_timer",
            "description": "Annuler un minuteur ou une alarme en cours",
            "parameters": {
                "type": "object",
                "properties": {
                    "label": {
                        "type": "string",
                        "description": "Nom du timer à annuler",
                    },
                },
                "required": [],
            },
        },
    ]

    # Calendar tools (standalone, no HA dependency)
    calendar_tools = []
    if calendar_client:
        calendar_tools = [
            {
                "name": "query_calendar",
                "description": "Consulter l'agenda. Chercher des événements par date ou mot-clé.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "start_date": {
                            "type": "string",
                            "description": "Date de début au format AAAA-MM-JJ (ex: 2026-03-24)",
                        },
                        "end_date": {
                            "type": "string",
                            "description": "Date de fin au format AAAA-MM-JJ. Par défaut: même jour que start_date",
                        },
                        "search": {
                            "type": "string",
                            "description": "Texte à chercher dans les titres (ex: dentiste, école)",
                        },
                    },
                    "required": ["start_date"],
                },
            },
            {
                "name": "create_event",
                "description": "Ajouter un événement à l'agenda",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "Titre de l'événement (ex: Dentiste, Réunion école)",
                        },
                        "start_datetime": {
                            "type": "string",
                            "description": "Date et heure au format AAAA-MM-JJ HH:MM (ex: 2026-03-25 14:00)",
                        },
                        "duration_minutes": {
                            "type": "integer",
                            "description": "Durée en minutes. Par défaut: 60",
                        },
                    },
                    "required": ["title", "start_datetime"],
                },
            },
        ]

    # Media tools: work without HA (default to the current ESP speaker).
    # With HA, the `room` parameter can route to other media_players.
    media_room_param = {
        "type": "string",
        "description": "Pièce où jouer le média. Par défaut: l'enceinte de l'appareil vocal.",
    }
    media_tools = [
        {
            "name": "play_radio",
            "description": (
                "Lancer une radio en direct (ex: « mets France Inter », « joue FIP »). "
                "Stations disponibles: France Inter, France Info, France Culture, "
                "France Musique, France Bleu, FIP, Mouv', RTL, Europe 1, RMC, "
                "NRJ, Skyrock, Nostalgie, Chérie FM, Rire et Chansons, "
                "TSF Jazz, Radio Classique, RFI."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "station": {
                        "type": "string",
                        "description": "Nom de la station (ex: france inter, fip, rtl, nrj)",
                    },
                    "room": media_room_param,
                },
                "required": ["station"],
            },
        },
        {
            "name": "stop_media",
            "description": "Arrêter la lecture (radio, musique) en cours",
            "parameters": {
                "type": "object",
                "properties": {"room": media_room_param},
                "required": [],
            },
        },
        {
            "name": "set_volume",
            "description": (
                "Régler le volume à un niveau précis. "
                "Utiliser uniquement quand l'utilisateur donne un pourcentage "
                "ou un niveau (ex: « mets le son à 40 », « volume à 70 »)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "level": {
                        "type": "integer",
                        "description": "Niveau du volume de 0 à 100",
                    },
                    "room": media_room_param,
                },
                "required": ["level"],
            },
        },
        {
            "name": "change_volume",
            "description": (
                "Augmenter ou baisser le volume d'un cran. "
                "À utiliser pour « plus fort », « moins fort », « monte le son », "
                "« baisse le son »."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": ["up", "down"],
                        "description": "« up » pour plus fort, « down » pour moins fort",
                    },
                    "room": media_room_param,
                },
                "required": ["direction"],
            },
        },
    ]

    if not ha_client:
        return [weather_tool] + timer_tools + calendar_tools + media_tools

    room_param = {
        "type": "string",
        "description": "Pièce ou groupe (ex: chambre Charlie, salon, enfants, partout)",
    }

    return [
        {
            "name": "turn_on",
            "description": "Allumer un appareil (lumière, prise, etc.)",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity": {"type": "string", "description": "Nom de l'appareil"},
                    "room": room_param,
                    "brightness": {
                        "type": "integer",
                        "description": "Luminosité 0-100, seulement pour les lumières",
                    },
                },
                "required": ["entity"],
            },
        },
        {
            "name": "turn_off",
            "description": "Éteindre un appareil (lumière, prise, etc.)",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity": {"type": "string", "description": "Nom de l'appareil"},
                    "room": room_param,
                },
                "required": ["entity"],
            },
        },
        {
            "name": "open_cover",
            "description": "Ouvrir un volet ou un store",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity": {"type": "string", "description": "Nom du volet"},
                    "room": room_param,
                },
                "required": ["entity"],
            },
        },
        {
            "name": "close_cover",
            "description": "Fermer un volet ou un store entièrement",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity": {"type": "string", "description": "Nom du volet"},
                    "room": room_param,
                },
                "required": ["entity"],
            },
        },
        {
            "name": "set_cover_position",
            "description": "Régler la position d'un volet (0 = complètement fermé, 100 = complètement ouvert). Exemples : « ferme à moitié » = 50, « ouvre un peu » = 30, « ferme presque » = 10.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity": {"type": "string", "description": "Nom du volet"},
                    "room": room_param,
                    "position": {
                        "type": "integer",
                        "description": "Position du volet en pourcentage (0 = fermé, 100 = ouvert)",
                    },
                },
                "required": ["entity", "position"],
            },
        },
        {
            "name": "set_temperature",
            "description": "Régler la température d'un thermostat",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity": {"type": "string", "description": "Nom du thermostat"},
                    "room": room_param,
                    "temperature": {
                        "type": "number",
                        "description": "Température en degrés Celsius",
                    },
                },
                "required": ["entity", "temperature"],
            },
        },
        {
            "name": "get_state",
            "description": "Obtenir l'état actuel d'un appareil (allumé/éteint, température, ouvert/fermé)",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity": {"type": "string", "description": "Nom de l'appareil"},
                    "room": room_param,
                },
                "required": ["entity"],
            },
        },
        weather_tool,
        *timer_tools,
        *calendar_tools,
        *media_tools,
        {
            "name": "end_conversation",
            "description": (
                "Terminer la conversation. Appeler quand l'utilisateur dit "
                "au revoir, merci, ok merci, bonne nuit, c'est tout, etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    ]


def parse_text_tool_call(content: str, has_history: bool) -> tuple[Optional[str], Optional[dict]]:
    """Parse tool calls that local LLMs output as plain text instead of structured tool_calls.
    Handles: fn({"entity":"X","room":"Y"}), fn("X"), fn(entity="X", room="Y"),
    and French natural language actions as fallback.
    """
    # Pattern 1: fn({"key": "val", ...})
    m = re.match(r"(\w+)\s*\(\s*(\{.*\})\s*\)", content, re.DOTALL)
    if m and m.group(1) in _TOOL_NAMES:
        try:
            return m.group(1), json.loads(m.group(2))
        except json.JSONDecodeError:
            pass

    # Pattern 2: fn("value")
    m = re.match(r'(\w+)\s*\(\s*"([^"]+)"\s*\)', content)
    if m and m.group(1) in _TOOL_NAMES:
        return m.group(1), {"entity": m.group(2)}

    # Pattern 3: fn(entity="X", room="Y") — Python-style kwargs
    m = re.match(r"(\w+)\s*\((.+)\)", content, re.DOTALL)
    if m and m.group(1) in _TOOL_NAMES:
        args = {}
        for kv in re.findall(r'(\w+)\s*=\s*"([^"]*)"', m.group(2)):
            args[kv[0]] = kv[1]
        if args:
            return m.group(1), args

    # Pattern 4: French natural language action (e.g. "Éteins les appliques salon.")
    # Only triggered when conversation history exists (follow-up command)
    if has_history:
        lower = content.lower().rstrip(".!?")

        # Volume follow-up shortcuts ("plus fort", "moins fort", "monte le son"...)
        volume_phrases = {
            "up": ("plus fort", "monte le son", "augmente le son", "monte le volume", "augmente le volume"),
            "down": ("moins fort", "baisse le son", "baisse le volume"),
        }
        for direction, phrases in volume_phrases.items():
            for phrase in phrases:
                if phrase in lower:
                    logger.info(f"French action fallback: '{content}' -> change_volume({direction})")
                    return "change_volume", {"direction": direction}

        action_map = {
            "allume": "turn_on",
            "rallume": "turn_on",
            "éteins": "turn_off",
            "eteins": "turn_off",
            "ferme": "close_cover",
            "ouvre": "open_cover",
        }
        for verb, func in action_map.items():
            if lower.startswith(verb):
                entity_part = lower[len(verb) :].strip()
                for article in ("les ", "le ", "la ", "l'", "l\u2019"):
                    if entity_part.startswith(article):
                        entity_part = entity_part[len(article) :]
                        break
                if entity_part:
                    logger.info(f"French action fallback: '{content}' -> {func}('{entity_part}')")
                    return func, {"entity": entity_part}

    return None, None


async def chat_completion(
    url: str,
    api_key: str,
    model: str,
    messages: list,
    tools: list | None = None,
    temperature: float = 0.3,
    max_tokens: int = 500,
) -> dict | None:
    """Send a chat completion request to an OpenAI-compatible API.
    Returns the message dict from the first choice, or None on error.
    """
    max_tokens_key = "max_completion_tokens" if api_key else "max_tokens"
    payload = {
        "messages": messages,
        max_tokens_key: max_tokens,
        "temperature": temperature,
    }
    if model:
        payload["model"] = model
    if tools:
        payload["tools"] = [{"type": "function", "function": func} for func in tools]

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    tool_names = [t["function"]["name"] for t in payload.get("tools", [])]
    logger.info(f'LLM prompt: [system] ...voice assistant... [user] "{messages[-1]["content"]}"')
    if tool_names:
        logger.info(f"Tools: {tool_names}")

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"LLM raw response: {json.dumps(result, ensure_ascii=False, indent=2)}")
                    return result["choices"][0]["message"]
                else:
                    error_text = await response.text()
                    logger.error(f"LLM error {response.status}: {error_text}")
                    return None
    except Exception as e:
        logger.error(f"LLM error: {e}")
        import traceback

        traceback.print_exc()
        return None
