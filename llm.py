"""LLM integration — chat completion, tool calling, text fallback parser."""

import json
import logging
import os
import re
import time
from typing import Optional

import aiohttp

from ha_client import fold

logger = logging.getLogger(__name__)

# Reasoning budget for thinking-capable models. Empty = leave the field out.
# Local models default to "none" (see chat_completion); cloud APIs validate this
# field strictly, so there it is only sent when explicitly configured.
REASONING_EFFORT = os.environ.get("LLM_REASONING_EFFORT", "")

# What the text fallback may run; its call has a log entry, for unlock's button (AD-068) as for a picture (AD-069)
_TOOL_NAMES = {
    "turn_on",
    "turn_off",
    "open_cover",
    "close_cover",
    "set_cover_position",
    "set_temperature",
    "set_heating",
    "get_state",
    "get_sensor",
    "set_timer",
    "set_alarm",
    "cancel_timer",
    "query_calendar",
    "create_event",
    "update_event",
    "play_radio",
    "stop_media",
    "set_volume",
    "change_volume",
    "play_music",
    "music_control",
    "show_image",
    "lock",
    "unlock",
}


# Claims of having carried something out, or of being about to, matched on
# accent-stripped lowercase. Deliberately about *actions*: a state report ("le
# volet est fermé") must not match. Every verb also takes its re- form (« je
# rouvre », « j'ai refermé », « je relance »), and a promise counts (« je vais
# essayer de le rouvrir »): a retry after « ça n'a pas marché » is where the
# model answered with no call at all (AD-078).
_PAST = "ouvert|ferm|allum|eteint|regl|lanc|programm|activ|desactiv|mis|baiss|mont|verrouill"
_PRESENT = "ferme|ouvre|allume|eteins|regle|lance|programme|mets|baisse|monte|verrouille"
_INFINITIVE = "ouvrir|fermer|allumer|eteindre|regler|lancer|programmer|mettre|baisser|monter|verrouiller|activer|desactiver"
_ACTION_CLAIM = re.compile(
    r"c'est fait|c'est entendu|voila|je m'en occupe|"
    rf"j'ai (?:bien )?(?:re|r)?(?:{_PAST})|"
    rf"\bje (?:re|r)?(?:{_PRESENT})\b|"
    rf"\bje vais (?:(?:essayer|tenter) (?:de |d'))?(?:(?:le|la|les|lui|leur|te|vous) |l')?(?:re|r)?(?:{_INFINITIVE})\b|"
    # Any promised attempt, and an order named as a noun (« je vais essayer de forcer
    # l'ouverture », « je viens de renvoyer l'ordre »): the model rephrases its retries.
    r"\bje vais (?:essayer|tenter|reessayer) d|\bje (?:re)?essaie\b|"
    r"\bje (?:viens de |viens d'|vais )?\w+ (?:l'ouverture|la fermeture|l'ordre|la commande)\b"
)


def claims_completed_action(text: str) -> bool:
    """Whether a reply asserts an action was carried out.

    Only meaningful when the model called no function at all — in that case the
    claim is false by construction, since nothing was executed. Questions are left
    out: « souhaitez-vous que je lance de la musique ? » offers an action, and
    challenging it once pushed the model into playing music nobody asked for.
    """
    flat = fold(text or "").replace("’", "'")
    return bool(_ACTION_CLAIM.search(re.sub(r"[^.!?]*\?", "", flat)))


def get_tool_definitions(ha_client, calendar_client=None, music=False, homes=(), locks=False, heating=()) -> list:
    """Return LLM tool definitions. `homes`: the homes' names, which the door and heating tools, get_state and
    get_sensor may name when there are several (AD-068, AD-071); `locks`: whether some home has a lock;
    `heating`: the modes of the homes' heating, if some home's can be set (AD-071)."""
    # Weather is always available (no HA dependency)
    weather_tool = {
        "name": "get_weather",
        "description": "Obtenir la météo actuelle et les prévisions. Par défaut: à la maison.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "Ville (ex: Paris, Angoulême, Lyon). À omettre pour la maison",
                },
            },
            "required": [],
        },
    }

    # Shops and places from OpenStreetMap, always available too. Without it, « est-ce que l'Altermundi
    # de la rue du Commerce est ouvert le dimanche ? » got « je ne peux pas vérifier les horaires »
    # (2026-10-11)
    place_tool = {
        "name": "find_place",
        "description": (
            "Chercher un commerce ou un lieu (magasin, pharmacie, boulangerie, restaurant, musée, "
            "piscine, poste…), les plus proches de la maison d'abord : adresse, horaires de la semaine, "
            "ouvert ou fermé maintenant, téléphone. Ex : « est-ce que l'Altermundi de la rue du Commerce "
            "est ouvert le dimanche ? », « jusqu'à quelle heure la pharmacie est ouverte ? », "
            "« le Monoprix ouvre à quelle heure demain ? »"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Nom ou type du lieu, avec la rue ou la ville si elles sont dites "
                        "(ex : « Altermundi rue du Commerce », « pharmacie », « Monoprix », « Leclerc Angoulême »)"
                    ),
                },
            },
            "required": ["query"],
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
        # Its address is looked up by the server (AD-072)
        location_param = {
            "type": "string",
            "description": "Lieu, s'il est dit : son nom ou son adresse, tels que dits (ex: Le Récepteur, 3 rue Cler)",
        }
        calendar_tools = [
            {
                "name": "query_calendar",
                "description": (
                    "Consulter l'agenda : rendez-vous, trajets en train ou en avion réservés, vacances. "
                    "Chercher des événements par date ou mot-clé. "
                    "Dès que la demande couvre une période (« cette semaine », « ce week-end », "
                    "« en août », « dans les prochains jours »), renseigne OBLIGATOIREMENT end_date : "
                    "sans elle la recherche ne couvre que la seule journée start_date."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "start_date": {
                            "type": "string",
                            "description": "Date de début au format AAAA-MM-JJ (ex: 2026-03-24)",
                        },
                        "end_date": {
                            "type": "string",
                            "description": (
                                "Date de fin au format AAAA-MM-JJ, incluse. Obligatoire pour toute "
                                "période. Ex. « cette semaine » = aujourd'hui → dimanche qui suit ; "
                                "« ce week-end » = samedi → dimanche. Par défaut : start_date seule."
                            ),
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
                # « rajoute sur ce rendez-vous qu'il a lieu au… » made a second event (2026-10-19, AD-072)
                "description": (
                    "Ajouter un nouvel événement à l'agenda. Jamais pour changer un événement déjà dans "
                    "l'agenda : c'est update_event."
                ),
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
                        "location": location_param,
                    },
                    "required": ["title", "start_datetime"],
                },
            },
            {
                "name": "update_event",
                "description": (
                    "Modifier un événement déjà dans l'agenda : son lieu, son heure, sa durée ou son titre "
                    "(« ajoute l'adresse », « il a lieu au… », « décale-le à 14h », « le dentiste de jeudi "
                    "dure deux heures »)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "event": {
                            "type": "string",
                            "description": "Titre de l'événement à modifier, tel quel (ex: Déjeuner avec Clémence)",
                        },
                        "date": {
                            "type": "string",
                            "description": "Sa date au format AAAA-MM-JJ, avant modification",
                        },
                        "location": location_param,
                        "start_datetime": {
                            "type": "string",
                            "description": "Nouvelles date et heure AAAA-MM-JJ HH:MM, seulement s'il est déplacé",
                        },
                        "duration_minutes": {
                            "type": "integer",
                            "description": "Nouvelle durée en minutes, seulement si elle change",
                        },
                        "title": {"type": "string", "description": "Nouveau titre, seulement s'il est renommé"},
                    },
                    "required": ["event", "date"],
                },
            },
        ]

    # Media tools: work without HA (default to the current ESP speaker).
    # With HA, the `room` parameter can route to other media_players.
    media_room_param = {
        "type": "string",
        "description": (
            "Pièce ou enceinte où jouer le média (ex: « salon », « Google Home »), parmi les "
            "« médias » de la liste des appareils. À omettre pour l'enceinte de l'appareil vocal "
            "(« ici », ou aucune pièce dite)."
        ),
    }
    media_tools = [
        {
            "name": "play_radio",
            "description": (
                "Lancer une radio en direct (ex: « mets France Inter », « joue FIP »). "
                "Stations disponibles: France Inter, France Info, France Culture, "
                "France Musique, ici Paris (ex-France Bleu), FIP, Mouv', RTL, Europe 1, RMC, "
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
            "name": "move_media",
            "description": (
                "Passer le son en cours, radio ou musique, sur une autre enceinte, sans changer ce qui "
                "joue ; l'ancienne s'arrête (ex: « passe le son sur la Sonos », « mets la radio dans le "
                "salon plutôt », « transfère sur la Google Home »)."
            ),
            "parameters": {
                "type": "object",
                "properties": {"room": {**media_room_param, "description": "Enceinte ou pièce où passer le son"}},
                "required": ["room"],
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

    if music:
        media_tools += [
            {
                "name": "play_music",
                "description": (
                    "Mettre de la musique sur Spotify : un artiste, un titre, un album, une "
                    "playlist, un genre ou une ambiance (ex: « mets du Daft Punk », « joue "
                    "Get Lucky », « mets ma playlist du matin », « mets du jazz »). "
                    "Pas pour les radios en direct (play_radio)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "Ce qu'il faut chercher, tel que dit (ex: « Daft Punk », "
                                "« Get Lucky Daft Punk », « du matin », « jazz »)"
                            ),
                        },
                        "kind": {
                            "type": "string",
                            "enum": ["artist", "track", "album", "playlist", "liked"],
                            "description": (
                                "artist pour un artiste ou groupe, track pour un titre, album, "
                                "playlist pour une playlist, un genre ou une ambiance, "
                                "liked pour les titres likés / favoris"
                            ),
                        },
                        "shuffle": {
                            "type": "boolean",
                            "description": "Lecture aléatoire, si demandée",
                        },
                        "room": media_room_param,
                    },
                    "required": ["query", "kind"],
                },
            },
            {
                "name": "music_control",
                "description": (
                    "Piloter la musique Spotify : pause, reprendre (avec room, sur cette enceinte), titre "
                    "suivant ou précédent, ou savoir ce qui joue (« c'est quoi cette chanson ? »). "
                    "Pour passer la musique qui joue sur une autre enceinte : move_media."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["pause", "resume", "next", "previous", "current"],
                        },
                        "room": media_room_param,
                    },
                    "required": ["action"],
                },
            },
        ]

    # Shown in the web app that asks, else cast to a screen (a Nest Hub): offered without one (AD-069)
    image_tool = {
        "name": "show_image",
        "description": (
            "Montrer une image : une photo de quelque chose ou de quelqu'un "
            "(ex: « montre-moi la tour Eiffel », « à quoi ressemble un okapi ? », "
            "« montre-moi un chat roux »). Pour une autre image du même sujet "
            "(« une autre »), rappeler avec la même requête."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    # The name alone finds the Wikipedia article and its picture; with the
                    # request's description around it (« La Joconde au Louvre », « l'Everest,
                    # la plus haute montagne du monde »), the photo search showed a Monet, or nothing
                    "type": "string",
                    "description": (
                        "Le nom de ce qu'il faut montrer, seul, comme le titre de son article "
                        "Wikipédia : « tour Eiffel », « okapi », « Burj Khalifa » (pas « Burj "
                        "Khalifa, le plus haut bâtiment du monde »). Une demande qui désigne une chose "
                        "précise sans la nommer (un superlatif : « le plus haut bâtiment du monde », "
                        "« le plus grand lac ») prend le nom de cette chose (« Burj Khalifa », "
                        "« mer Caspienne ») ; si plusieurs conviennent, en choisir une et la nommer "
                        "dans la réponse. "
                        "Un mot de plus si le nom seul désigne autre chose (« Mercure planète »). "
                        "Sans nom propre, quelques mots (« chat roux »)."
                    ),
                },
                "room": {
                    "type": "string",
                    "description": (
                        "L'écran où la montrer, seulement si la demande en nomme un : « écran » "
                        "(« sur l'écran »), ou sa pièce (« sur l'écran du salon »). Sinon, rien"
                    ),
                },
            },
            "required": ["query"],
        },
    }

    standalone_tools = [weather_tool, place_tool, *timer_tools, *calendar_tools, *media_tools, image_tool]
    if not ha_client:
        return standalone_tools

    room_param = {
        "type": "string",
        "description": "Pièce ou groupe (ex: chambre Charlie, salon, enfants, partout)",
    }
    # « Nom exact » made gemma list every shutter by name: 9 calls (5.3 s) for « ouvre tous les volets »
    cover_entity_param = {
        "type": "string",
        "description": (
            "Nom exact d'un volet, d'une fenêtre ou d'une moustiquaire ; ou « volets » / « fenêtres » / "
            "« moustiquaires » avec room pour tous ceux d'une pièce, d'un groupe ou de partout, en un seul appel"
        ),
    }

    # Another home, named in the request: its door from this one, its heating, its state (AD-068, AD-071)
    elsewhere = {}
    if len(homes) > 1:
        elsewhere["home"] = {
            "type": "string",
            "enum": list(homes),
            "description": (
                "La maison, seulement si la demande en nomme une (« la porte de Charras », « à Paris »). Ses "
                "appareils ne sont pas dans la liste : entity tel que la demande le dit (« porte », « chauffage »)"
            ),
        }

    # No lock, no tool. Unlocking only proposes: a tap in the web app confirms, and the satellite refuses (AD-068)
    lock_tools = []
    if locks:
        door_params = {
            "type": "object",
            "properties": {
                "entity": {"type": "string", "description": "Nom de la serrure, ou « porte »"},
                "room": room_param,
                **elsewhere,
            },
            "required": ["entity"],
        }
        lock_tools = [
            {
                "name": "lock",
                # « à chaque demande » measured on gemma4:12b (AD-041): without it, a door locked earlier
                # in the conversation got no call. The last sentence keeps « ouvre la porte » off lock
                "description": (
                    "Verrouiller une porte, la fermer à clé (« ferme la porte », « tu peux fermer "
                    "la porte ? », « verrouille la porte »). Appelle-la à chaque demande, même si la "
                    "porte a déjà été verrouillée plus tôt : quelqu'un a pu la rouvrir depuis. "
                    "Jamais pour ouvrir ou déverrouiller : c'est unlock."
                ),
                "parameters": door_params,
            },
            {
                "name": "unlock",
                "description": (
                    "Ouvrir ou déverrouiller une porte (« ouvre la porte », « déverrouille la porte »). "
                    "N'ouvre rien tout de suite : le résultat dit si une confirmation est demandée dans "
                    "l'application ou si c'est refusé. Répète-le sans dire que la porte est ouverte."
                ),
                "parameters": door_params,
            },
        ]

    # A home whose heating HA sets by a mode (Charras's Ökofen boiler), from either home (AD-071)
    heating_tools = []
    if heating:
        heating_tools = [
            {
                "name": "set_heating",
                "description": (
                    "Allumer, éteindre ou changer le mode du chauffage (« allume le chauffage », « coupe le "
                    "chauffage », « mets le chauffage en confort »)"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entity": {
                            "type": "string",
                            "description": "« chauffage » pour tout le chauffage, ou le circuit nommé (« radiateurs »)",
                        },
                        "mode": {
                            "type": "string",
                            "enum": list(heating),
                            "description": (
                                # « (horaires inconnus) »: the box does not give them (AD-075)
                                "Auto : selon le programme horaire de la chaudière (horaires inconnus), pour "
                                "« allume » ou « active » sans autre précision. Confort : à la température de "
                                "confort en permanence. Réduit : à la température réduite. Off : arrêté (« éteins », "
                                "« coupe »)"
                            ),
                        },
                        **elsewhere,
                    },
                    "required": ["entity", "mode"],
                },
            }
        ]

    return [
        {
            "name": "turn_on",
            # « et baisse à 30% » after a light went to set_cover_position (dry run, 2026-10-11)
            "description": (
                "Allumer un appareil (lumière, prise, etc.), ou régler la luminosité d'une lumière "
                "(« baisse la lumière à 30% », « monte à 80% »)"
            ),
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
            "description": (
                "Ouvrir un volet, un store ou une fenêtre, ou remonter une moustiquaire. Volets, fenêtres et "
                "moustiquaires sont des équipements distincts : n'agir que sur celui qui est demandé."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entity": cover_entity_param,
                    "room": room_param,
                },
                "required": ["entity"],
            },
        },
        {
            "name": "close_cover",
            "description": (
                "Fermer entièrement un volet, un store ou une fenêtre, ou baisser une moustiquaire. Volets, "
                "fenêtres et moustiquaires sont des équipements distincts : n'agir que sur celui qui est demandé."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entity": cover_entity_param,
                    "room": room_param,
                },
                "required": ["entity"],
            },
        },
        {
            "name": "set_cover_position",
            "description": (
                "Régler la position d'un volet ou d'une fenêtre, jamais d'une lumière "
                "(0 = complètement fermé, 100 = complètement ouvert). "
                "Exemples : « ferme à moitié » = 50, « ouvre un peu » = 30, « ferme presque » = 10."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entity": cover_entity_param,
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
            "description": "Obtenir l'état actuel d'un appareil (allumé/éteint, température, ouvert/fermé, verrouillé)",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity": {"type": "string", "description": "Nom de l'appareil"},
                    "room": room_param,
                    **elsewhere,
                },
                "required": ["entity"],
            },
        },
        {
            "name": "get_sensor",
            "description": (
                "Consulter les capteurs de mesure de la maison par mot-clé : piscine "
                "(température de l'eau, pH, chlore, sel), chauffage et chaudière, "
                "températures intérieures/extérieure, CO2, humidité, consommation "
                "électrique, batteries. Ces capteurs ne figurent PAS dans la liste "
                "des appareils ci-dessus — utilise cet outil dès qu'on te demande une "
                "mesure qui n'y est pas."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Mots-clés du capteur cherché (ex: « piscine », « piscine ph », "
                            "« chaudière », « température extérieure »). Rester large : "
                            "la recherche renvoie tous les capteurs correspondants."
                        ),
                    },
                    **elsewhere,
                },
                "required": ["query"],
            },
        },
        *standalone_tools,
        *lock_tools,
        *heating_tools,
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


def parse_text_tool_call(content: str) -> tuple[Optional[str], Optional[dict]]:
    """Parse tool calls that local LLMs output as plain text instead of structured tool_calls.
    Handles: fn({"entity":"X","room":"Y"}), fn("X"), fn(entity="X", room="Y").
    No French-verb guess: « Fermeture des volets impossible » would have closed a shutter, and it
    never fired in 223 requests with gemma4:12b.
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

    return None, None


def _max_tokens_key(api_key: str) -> str:
    return "max_completion_tokens" if api_key else "max_tokens"


def _payload(api_key: str, model: str, messages: list, tools: list | None, temperature: float, max_tokens: int) -> dict:
    """A chat completion's request, built alike for every call: all but the last two arguments make the prompt
    the model reads, and a warm-up's must be the request's (the reasoning effort too, which a template may
    write into the system turn)."""
    payload = {
        "messages": messages,
        _max_tokens_key(api_key): max_tokens,
        "temperature": temperature,
    }
    # Thinking models burn the whole token budget on internal deliberation before
    # emitting anything: gemma4:12b spent 441 tokens reasoning about "et dans la
    # salle de jeux ?" and 9.8s, vs 13 tokens and 0.7s with reasoning off. Worse,
    # it usually hits the cap first, leaving `content` empty and triggering the
    # length-retry below — which regenerates from scratch with a *bigger* budget
    # to ruminate in. Nothing here needs a scratchpad: the pipeline is a tool call
    # or a one-sentence answer. Beware "low" — it thinks anyway, then truncates.
    effort = REASONING_EFFORT or ("" if api_key else "none")
    if effort:
        payload["reasoning_effort"] = effort
    if model:
        payload["model"] = model
    if tools:
        payload["tools"] = [{"type": "function", "function": func} for func in tools]
    return payload


async def warm_up(url: str, model: str, messages: list, tools: list) -> None:
    """Has the local model read a prompt and answer one token, for nothing: loaded, if Ollama had let it go,
    and the prompt in its cache, where the request that follows finds what it shares."""
    t0 = time.monotonic()
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, json=_payload("", model, messages, tools, 0, 1), timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                await response.read()
    except Exception as e:
        logger.debug(f"LLM warm-up failed: {e}")
        return
    if (seconds := time.monotonic() - t0) > 1:  # ~0.1 s when the cache had it all
        logger.info(f"LLM warmed up in {seconds:.1f} s")


async def chat_completion(
    url: str,
    api_key: str,
    model: str,
    messages: list,
    tools: list | None = None,
    temperature: float = 0.3,
    # A round is 79 tokens at most (900 rounds, a month to 2026-10-25; p99 49): 200 leaves room, and a
    # model looping past its answer stops after ~4.5 s instead of ~11.5 s at 500 (AD-080)
    max_tokens: int = 200,
) -> dict | None:
    """Send a chat completion request to an OpenAI-compatible API.
    Returns the message dict from the first choice, or None on error.
    """
    max_tokens_key = _max_tokens_key(api_key)
    payload = _payload(api_key, model, messages, tools, temperature, max_tokens)
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    logger.debug(f"LLM request: {len(messages)} messages, last {messages[-1]['role']}: {messages[-1]['content']!r}")

    # finish_reason=length with the budget spent. A long answer cut short gets a bigger budget. But gemma4
    # mostly hits it by looping past its answer, with no hidden reasoning (AD-080): it said a few words or made
    # its call, and those are kept; or it said nothing, and a bigger budget only looped longer (500, 1000,
    # 2000 tokens: 67 s and an empty reply, 2026-10-24), so it samples again, warmer, at the same budget.
    MAX_TOKEN_CEILING = 2000
    MAX_LENGTH_RETRIES = 2
    RETRY_TEMPERATURE = 0.7

    try:
        async with aiohttp.ClientSession() as session:
            for attempt in range(MAX_LENGTH_RETRIES + 1):
                async with session.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"LLM error {response.status}: {error_text}")
                        return None

                    result = await response.json()
                    choice = result["choices"][0]
                    finish_reason = choice.get("finish_reason")
                    usage = result.get("usage") or {}
                    called = [tc["function"]["name"] for tc in choice["message"].get("tool_calls") or []]
                    # Where a reply's tokens went when its text is empty: Ollama returns hidden thinking as
                    # `reasoning`, other servers as `reasoning_content` (2026-10-24: 3 × empty, 67 s, AD-080)
                    reasoning = choice["message"].get("reasoning") or choice["message"].get("reasoning_content") or ""
                    # One line per round: the raw response, indented, took ~30 journal lines each
                    logger.info(
                        f"LLM: {finish_reason}, prompt {usage.get('prompt_tokens')} + completion "
                        f"{usage.get('completion_tokens')} tokens, calls {called}, "
                        f"text {(choice['message'].get('content') or '')[:120]!r}"
                        + (f", reasoning {len(reasoning)} chars {reasoning[:200]!r}" if reasoning else "")
                    )

                    # A completion stopped well short of its budget hit the context, not the budget:
                    # more tokens cannot help (Ollama's 4096-token context, 2026-10-11: 3 retries, 7 tokens each)
                    hit_budget = (usage.get("completion_tokens") or 0) >= payload[max_tokens_key] * 0.9
                    budget = payload[max_tokens_key]
                    content = (choice["message"].get("content") or "").strip()
                    if finish_reason == "length" and not hit_budget:
                        logger.error(f"LLM context full: prompt {usage.get('prompt_tokens')} tokens")
                    elif finish_reason == "length" and (called or content and len(content) < budget):
                        # Fewer characters than tokens: the answer is there, the rest was a loop
                        logger.warning(f"LLM looped past its reply to {budget} tokens: keeping {content!r}, calls {called}")
                        choice["message"]["content"] = content
                    elif finish_reason == "length" and attempt < MAX_LENGTH_RETRIES:
                        if not content:
                            logger.warning(
                                f"LLM looped to {budget} tokens with nothing said, "
                                f"sampling again at temperature {RETRY_TEMPERATURE}"
                            )
                            payload["temperature"] = RETRY_TEMPERATURE
                            continue
                        if budget < MAX_TOKEN_CEILING:
                            new_limit = min(budget * 2, MAX_TOKEN_CEILING)
                            logger.warning(f"LLM truncated (finish_reason=length) at {budget} tokens, retrying with {new_limit}")
                            payload[max_tokens_key] = new_limit
                            continue
                    elif finish_reason == "length":
                        logger.warning(
                            f"LLM still truncated after {MAX_LENGTH_RETRIES} retries "
                            f"(max {budget} tokens), returning partial response"
                        )

                    return choice["message"]
    except Exception:
        logger.exception("LLM error")
        return None
