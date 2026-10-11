"""Shops and places from OpenStreetMap: address, opening hours, phone (free, no API key).

Photon finds them, nearest to the home first; Nominatim then gives each its OSM tags, the opening
hours among them, which opening-hours-py turns into a week the LLM reads without parsing OSM's syntax.
"""

import asyncio
import json
import logging
import math
from datetime import date, datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

import aiohttp
from opening_hours import OpeningHours, ParserError, State

logger = logging.getLogger(__name__)

# Nominatim's usage policy asks for an identifying User-Agent
_HEADERS = {"User-Agent": "voice-assistant/1.0 (https://github.com/isc/voice-assistant)"}
_TIMEOUT = aiohttp.ClientTimeout(total=5)
# The voice device waits on it
_DEADLINE = 8
_RESULTS = 3
# Measured on 2026-10-11 around Paris 15e: with Photon's defaults (zoom 14, scale 0.2) « pharmacie »
# found pharmacies 3–6 km away; zoom 16 and scale 0.1 found the six within 400 m, and still found
# « musée d'Orsay » 2.6 km away and « Leclerc Angoulême » 387 km away
_BIAS = {"zoom": 16, "location_bias_scale": 0.1}
# Named like the place but not one: « musée d'Orsay » is also an RER station and a bus stop,
# « Leclerc Angoulême » first found two streets
_NOT_PLACES = set("highway railway public_transport place boundary landuse natural waterway building".split())
# Named after the landmark beside them: Vélib' stations above all
_NOT_PLACE_TYPES = set("bicycle_rental ferry_terminal parking parking_entrance taxi charging_station".split())

_DAYS = "lundi mardi mercredi jeudi vendredi samedi dimanche".split()
_MONTHS = "janvier février mars avril mai juin juillet août septembre octobre novembre décembre".split()


def _error(message: str) -> str:
    return json.dumps({"error": message}, ensure_ascii=False)


def _distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    x = math.radians(lon2 - lon1) * math.cos(math.radians((lat1 + lat2) / 2))
    y = math.radians(lat2 - lat1)
    return 6371 * math.hypot(x, y)


def _day_label(day: date, today: date) -> str:
    named = f"{_DAYS[day.weekday()]} {day.day} {_MONTHS[day.month - 1]}"
    if day == today:
        return f"aujourd'hui ({named})"
    if day == today + timedelta(days=1):
        return f"demain ({named})"
    return named


def _span(start: datetime, end: datetime, state: State, comment: str) -> str:
    # Spans are cut at midnight: one that ends on the next day ends at midnight
    text = f"{start:%H:%M}-{'24:00' if end.date() > start.date() else f'{end:%H:%M}'}"
    if state == State.UNKNOWN:
        text += " (incertain)"
    return f"{text} ({comment})" if comment else text


def week(hours: str, coords: tuple[float, float], now: Optional[datetime] = None) -> dict:
    """Seven days of opening hours from today, and whether it is open now, in French.

    The place's own time zone and public holidays (« PH off ») come from its coordinates.
    Raises ParserError for hours OSM's syntax can't read (opening-hours-py documents SyntaxError, wrongly).
    """
    oh = OpeningHours(hours, coords=coords)
    # A ZoneInfo: opening-hours-py reads its key
    now = now or datetime.now(ZoneInfo("UTC"))
    # The first interval starts at now, in the place's time zone
    local_now = next(oh.intervals(now))[0]
    today = local_now.date()
    midnight = local_now.replace(hour=0, minute=0, second=0, microsecond=0)

    days = {}
    for offset in range(7):
        day = today + timedelta(days=offset)
        start = midnight.replace(year=day.year, month=day.month, day=day.day)
        end = start + timedelta(days=1)
        spans = [
            (max(s, start), min(e, end), state, comment)
            for s, e, state, comment in oh.intervals(start, end)
            if state != State.CLOSED
        ]
        if len(spans) == 1 and spans[0][:2] == (start, end) and spans[0][2] == State.OPEN and not spans[0][3]:
            text = "ouvert toute la journée"
        else:
            text = ", ".join(_span(*span) for span in spans) or "fermé"
        days[_day_label(day, today)] = text

    state, comment = oh.state(now)
    change = oh.next_change(now)
    # "" today, else "demain " or "mardi 13 octobre "
    day = "" if not change or change.date() == today else _day_label(change.date(), today).split(" (")[0] + " "
    if state == State.UNKNOWN:
        status = "incertain"
    elif state == State.OPEN:
        status = f"ouvert jusqu'à {day}{change:%H:%M}" if change else "ouvert"
    else:
        status = f"fermé, ouvre {day}à {change:%H:%M}" if change and change - local_now < timedelta(days=7) else "fermé"
    if comment:
        status += f" ({comment})"
    return {"maintenant": status, "horaires": days}


async def _search(session: aiohttp.ClientSession, query: str, home: dict) -> list[dict]:
    """Photon: the places named query, the nearest to home first."""
    params = {"q": query, "lang": "fr", "limit": 10}
    if home.get("latitude") is not None:
        params |= {"lat": home["latitude"], "lon": home["longitude"], **_BIAS}
    async with session.get("https://photon.komoot.io/api/", params=params) as r:
        r.raise_for_status()
        data = await r.json()
    return [
        feature
        for feature in data.get("features", [])
        if feature["properties"].get("name")
        and feature["properties"].get("osm_key") not in _NOT_PLACES
        and feature["properties"].get("osm_value") not in _NOT_PLACE_TYPES
    ][:_RESULTS]


async def _tags(session: aiohttp.ClientSession, features: list[dict]) -> dict[str, dict]:
    """Nominatim: each place's OSM tags (opening_hours, phone…), by "N123"-style id."""
    ids = [f"{f['properties']['osm_type']}{f['properties']['osm_id']}" for f in features]
    params = {"osm_ids": ",".join(ids), "format": "jsonv2", "extratags": 1}
    async with session.get("https://nominatim.openstreetmap.org/lookup", params=params) as r:
        r.raise_for_status()
        data = await r.json()
    return {f"{p['osm_type'][0].upper()}{p['osm_id']}": p.get("extratags") or {} for p in data}


def _describe(feature: dict, tags: dict, home: dict, now: Optional[datetime]) -> dict:
    props = feature["properties"]
    lon, lat = feature["geometry"]["coordinates"]
    street = " ".join(filter(None, (props.get("housenumber"), props.get("street"))))
    place = {
        "nom": props["name"],
        "type": props.get("osm_value"),
        "adresse": ", ".join(filter(None, (street, props.get("city")))),
    }
    if home.get("latitude") is not None:
        place["distance_km"] = round(_distance_km(home["latitude"], home["longitude"], lat, lon), 1)
    if hours := tags.get("opening_hours"):
        try:
            place |= week(hours, (lat, lon), now)
        except ParserError:
            logger.info(f"Unreadable opening hours for {props['name']}: {hours!r}")
            place["horaires"] = hours
    else:
        place["horaires"] = "inconnus"
    if phone := tags.get("phone") or tags.get("contact:phone"):
        place["telephone"] = phone
    return place


async def find_place(query: str, home: Optional[dict] = None, now: Optional[datetime] = None) -> str:
    """Places matching query, nearest to home first, with their opening hours, as JSON for the LLM."""
    query = (query or "").strip()
    if not query:
        return _error("Lieu à chercher manquant")
    home = home or {}
    try:
        async with asyncio.timeout(_DEADLINE), aiohttp.ClientSession(headers=_HEADERS, timeout=_TIMEOUT) as session:
            features = await _search(session, query, home)
            if not features:
                return _error(f"Aucun lieu trouvé pour « {query} »")
            try:
                tags = await _tags(session, features)
            except (aiohttp.ClientError, TimeoutError, ValueError) as e:
                # Still the address and the distance
                logger.warning(f"Nominatim lookup failed: {e}")
                tags = {}
    except (aiohttp.ClientError, TimeoutError, ValueError) as e:
        logger.warning(f"Place search for {query!r} failed: {e!r}")
        return _error("Recherche de lieux indisponible")
    places = [
        _describe(f, tags.get(f"{f['properties']['osm_type']}{f['properties']['osm_id']}", {}), home, now)
        for f in features
    ]
    return json.dumps({"lieux": places}, ensure_ascii=False)
