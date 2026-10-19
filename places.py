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
from opening_hours import OpeningHours, ParserError, State, UnknownCountryError

from ha_client import fold, words

# Nominatim's usage policy, like Wikimedia's, asks for an identifying User-Agent
from images import _HEADERS
from weather import _error

logger = logging.getLogger(__name__)

_TIMEOUT = aiohttp.ClientTimeout(total=5)
# The voice device waits on it
_DEADLINE = 8
_NETWORK_ERRORS = (aiohttp.ClientError, TimeoutError, ValueError)
_RESULTS = 3
# Measured on 2026-10-11 around Paris 15e: with Photon's defaults (zoom 14, scale 0.2) « pharmacie »
# found pharmacies 3–6 km away; zoom 16 and scale 0.1 found the six within 400 m, and still found
# « musée d'Orsay » 2.6 km away and « Leclerc Angoulême » 387 km away
_BIAS = {"zoom": 16, "location_bias_scale": 0.1}
# Named like the place but not one: « musée d'Orsay » is also an RER station and a bus stop,
# « Leclerc Angoulême » first found two streets. Filtered here: Photon's own filter (osm_tag=!highway)
# reranks, and put a stadium and a social centre before the Leclerc supermarket
_NOT_PLACES = set("highway railway public_transport place boundary landuse natural waterway building".split())
# Named after the landmark beside them: Vélib' stations above all
_NOT_PLACE_TYPES = set("bicycle_rental ferry_terminal parking parking_entrance taxi charging_station".split())

# An event's place, from the home's: a place of the name said farther off is another one
_NEARBY_KM = 50
_STREETS = set("rue avenue av boulevard bd place allee chemin impasse quai route cours square passage".split())

_DAYS = "lundi mardi mercredi jeudi vendredi samedi dimanche".split()
_MONTHS = "janvier février mars avril mai juin juillet août septembre octobre novembre décembre".split()


def _distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    x = math.radians(lon2 - lon1) * math.cos(math.radians((lat1 + lat2) / 2))
    y = math.radians(lat2 - lat1)
    return 6371 * math.hypot(x, y)


def _date(day: date) -> str:
    return f"{_DAYS[day.weekday()]} {day.day} {_MONTHS[day.month - 1]}"


def _relative(day: date, today: date) -> Optional[str]:
    return {0: "aujourd'hui", 1: "demain"}.get((day - today).days)


def _span(start: datetime, end: datetime, state: State, comment: str) -> str:
    # Spans are cut at midnight: one that ends on the next day ends at midnight
    text = f"{start:%H:%M}-{'24:00' if end.date() > start.date() else f'{end:%H:%M}'}"
    if state == State.UNKNOWN:
        text += " (incertain)"
    return f"{text} ({comment})" if comment else text


def week(
    hours: str,
    coords: tuple[float, float],
    tz: Optional[ZoneInfo] = None,
    country: Optional[str] = None,
    now: Optional[datetime] = None,
) -> dict:
    """Seven days of opening hours from today, and whether it is open now, in French.

    The country gives the public holidays (« PH off »). Time zone and country are best given:
    guessed from the coordinates, they load a 110 MB map, kept for good (measured).
    Raises ParserError for hours OSM's syntax can't read (opening-hours-py documents SyntaxError, wrongly).
    """
    try:
        oh = OpeningHours(hours, timezone=tz, country=country, coords=coords, auto_country=False)
    except UnknownCountryError:  # no public holidays known there
        oh = OpeningHours(hours, timezone=tz, coords=coords, auto_country=False)
    # A ZoneInfo: opening-hours-py reads its key
    now = now or datetime.now(ZoneInfo("UTC"))
    # From now, in the place's time zone, to the next change
    local_now, change, state, comment = next(oh.intervals(now))
    today = local_now.date()
    midnight = local_now.replace(hour=0, minute=0, second=0, microsecond=0)

    days = {}
    for offset in range(7):
        start = midnight + timedelta(days=offset)
        end = start + timedelta(days=1)
        spans = [span for span in oh.intervals(start, end) if span[2] != State.CLOSED]
        if spans == [(start, end, State.OPEN, "")]:
            text = "ouvert toute la journée"
        else:
            text = ", ".join(_span(*span) for span in spans) or "fermé"
        relative = _relative(start.date(), today)
        days[f"{relative} ({_date(start.date())})" if relative else _date(start.date())] = text

    if change:
        relative = _relative(change.date(), today)
        # "" today, else "demain " or "mardi 13 octobre "
        day = "" if relative == "aujourd'hui" else f"{relative or _date(change.date())} "
    if state == State.UNKNOWN:
        status = "incertain"
    elif state == State.OPEN:
        status = f"ouvert jusqu'à {day}{change:%H:%M}" if change else "ouvert"
    else:
        status = f"fermé, ouvre {day}à {change:%H:%M}" if change and change - local_now < timedelta(days=7) else "fermé"
    if comment:
        status += f" ({comment})"
    return {"maintenant": status, "horaires": days}


def _osm_id(props: dict) -> str:
    """ "N123": Photon's osm_type is "N", Nominatim's "node"."""
    return f"{props['osm_type'][0].upper()}{props['osm_id']}"


async def _search(session: aiohttp.ClientSession, query: str, origin: Optional[tuple[float, float]]) -> list[dict]:
    """Photon: the places named query, the nearest to origin first."""
    params = {"q": query, "lang": "fr", "limit": 10}
    if origin:
        params |= {"lat": origin[0], "lon": origin[1], **_BIAS}
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


async def _tags(session: aiohttp.ClientSession, features: list[dict]) -> list[dict]:
    """Nominatim: each place's OSM tags (opening_hours, phone…), in the order of features."""
    params = {"osm_ids": ",".join(_osm_id(f["properties"]) for f in features), "format": "jsonv2", "extratags": 1}
    async with session.get("https://nominatim.openstreetmap.org/lookup", params=params) as r:
        r.raise_for_status()
        data = await r.json()
    by_id = {_osm_id(p): p.get("extratags") or {} for p in data}
    return [by_id.get(_osm_id(f["properties"]), {}) for f in features]


def _origin(home: dict) -> Optional[tuple[float, float]]:
    return (home["latitude"], home["longitude"]) if home.get("latitude") is not None else None


def _street(props: dict) -> str:
    return " ".join(filter(None, (props.get("housenumber"), props.get("street"))))


def _describe(feature: dict, tags: dict, origin: Optional[tuple[float, float]], tz: Optional[ZoneInfo]) -> dict:
    props = feature["properties"]
    lon, lat = feature["geometry"]["coordinates"]
    place = {
        "nom": props["name"],
        "type": props.get("osm_value"),
        "adresse": ", ".join(filter(None, (_street(props), props.get("city")))),
    }
    if origin:
        place["distance_km"] = round(_distance_km(*origin, lat, lon), 1)
    if hours := tags.get("opening_hours"):
        try:
            place |= week(hours, (lat, lon), tz, props.get("countrycode"))
        except ParserError:
            logger.info(f"Unreadable opening hours for {props['name']}: {hours!r}")
            place["horaires"] = hours
    else:
        place["horaires"] = "inconnus"
    if phone := tags.get("phone") or tags.get("contact:phone"):
        place["telephone"] = phone
    return place


async def find_place(query: str, home: Optional[dict] = None) -> str:
    """Places matching query, nearest to home first, with their opening hours, as JSON for the LLM.

    home: latitude, longitude, time_zone, as Home Assistant has them.
    """
    query = (query or "").strip()
    if not query:
        return _error("Lieu à chercher manquant")
    home = home or {}
    origin = _origin(home)
    # The home's, for every place found: one far enough to be in another is a rare request
    tz = ZoneInfo(home["time_zone"]) if home.get("time_zone") else None
    try:
        async with asyncio.timeout(_DEADLINE), aiohttp.ClientSession(headers=_HEADERS, timeout=_TIMEOUT) as session:
            features = await _search(session, query, origin)
            if not features:
                return _error(f"Aucun lieu trouvé pour « {query} »")
            try:
                tags = await _tags(session, features)
            except _NETWORK_ERRORS as e:
                # Still the address and the distance
                logger.warning(f"Nominatim lookup failed: {e!r}")
                tags = [{}] * len(features)
    except _NETWORK_ERRORS as e:
        logger.warning(f"Place search for {query!r} failed: {e!r}")
        return _error("Recherche de lieux indisponible")
    places = [_describe(feature, t, origin, tz) for feature, t in zip(features, tags)]
    return json.dumps({"lieux": places}, ensure_ascii=False)


async def address_of(query: str, home: Optional[dict] = None) -> Optional[str]:
    """An event's place as a calendar keeps it, its name and address (« Le Récepteur, 3 Avenue Théophile Gautier,
    75016 Paris »), when a place nearest the home bears the name said (« au bistrot Le Récepteur »); else None,
    and the words said are kept (AD-072)."""
    first = next(iter(query.split()), "")
    # An address already (« 12 rue Cler »): Photon names no house, and took 0.6 to 1 s to find nothing. And
    # « chez Clémence » is her home, not the restaurant « Chez Clémence », found in Kinshasa
    if not first or first[0].isdigit() or fold(first) in _STREETS | {"chez"}:
        return None
    origin = _origin(home or {})
    try:
        # A nicety: the event is made with the words said rather than wait for a slow search
        async with asyncio.timeout(2), aiohttp.ClientSession(headers=_HEADERS, timeout=_TIMEOUT) as session:
            features = await _search(session, query, origin)
    except _NETWORK_ERRORS as e:
        logger.warning(f"Address search for {query!r} failed: {e!r}")
        return None
    for feature in features:
        props = feature["properties"]
        far = origin and _distance_km(*origin, *reversed(feature["geometry"]["coordinates"])) > _NEARBY_KM
        if words(props["name"]) <= words(query) and not far:
            town = " ".join(filter(None, (props.get("postcode"), props.get("city"))))
            return ", ".join(filter(None, (props["name"], _street(props), town)))
    return None
