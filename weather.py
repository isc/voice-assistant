"""Weather via Open-Meteo API (free, no API key)."""

import json
import logging
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)

# Paris 15e as default
DEFAULT_LAT = 48.8416
DEFAULT_LON = 2.3001
DEFAULT_CITY = "Paris"

# WMO weather codes → French descriptions
_WMO_CODES = {
    0: "ciel dégagé",
    1: "plutôt dégagé",
    2: "partiellement nuageux",
    3: "couvert",
    45: "brouillard",
    48: "brouillard givrant",
    51: "bruine légère",
    53: "bruine",
    55: "bruine forte",
    61: "pluie légère",
    63: "pluie",
    65: "forte pluie",
    66: "pluie verglaçante",
    67: "forte pluie verglaçante",
    71: "neige légère",
    73: "neige",
    75: "forte neige",
    80: "averses légères",
    81: "averses",
    82: "fortes averses",
    85: "averses de neige",
    86: "fortes averses de neige",
    95: "orage",
    96: "orage avec grêle",
    99: "orage violent avec grêle",
}


async def _geocode(city: str) -> Optional[tuple[float, float, str]]:
    """Resolve city name to (lat, lon, display_name) via Open-Meteo geocoding."""
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=fr"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as r:
                if r.status != 200:
                    return None
                data = await r.json()
                results = data.get("results", [])
                if not results:
                    return None
                loc = results[0]
                return loc["latitude"], loc["longitude"], loc["name"]
    except Exception as e:
        logger.error(f"Geocoding error: {e}")
        return None


async def get_weather(location: str = "") -> str:
    """Fetch weather and return structured JSON string for the LLM to interpret."""
    city = location.strip() if location else DEFAULT_CITY
    lat, lon = DEFAULT_LAT, DEFAULT_LON
    display_name = DEFAULT_CITY

    if city.lower() not in ("paris", ""):
        geo = await _geocode(city)
        if geo:
            lat, lon, display_name = geo
        else:
            return json.dumps({"error": f"Ville {city} introuvable"})

    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&current=temperature_2m,apparent_temperature,weather_code"
        f"&daily=temperature_2m_max,temperature_2m_min,precipitation_probability_max,weather_code"
        f"&timezone=Europe/Paris&forecast_days=5"
    )

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as r:
                if r.status != 200:
                    return json.dumps({"error": "API météo indisponible"})
                data = await r.json()
    except Exception as e:
        logger.error(f"Weather API error: {e}")
        return json.dumps({"error": "API météo indisponible"})

    current = data.get("current", {})
    daily = data.get("daily", {})

    # Build structured result for the LLM
    result = {
        "ville": display_name,
        "maintenant": {
            "temperature": current.get("temperature_2m"),
            "ressenti": current.get("apparent_temperature"),
            "conditions": _WMO_CODES.get(current.get("weather_code", -1), "inconnu"),
        },
        "previsions": [],
    }

    days = daily.get("time", [])
    maxs = daily.get("temperature_2m_max", [])
    mins = daily.get("temperature_2m_min", [])
    rain = daily.get("precipitation_probability_max", [])
    codes = daily.get("weather_code", [])

    for i, day in enumerate(days):
        result["previsions"].append(
            {
                "date": day,
                "min": mins[i] if i < len(mins) else None,
                "max": maxs[i] if i < len(maxs) else None,
                "pluie_pct": rain[i] if i < len(rain) else None,
                "conditions": _WMO_CODES.get(codes[i], "inconnu")
                if i < len(codes)
                else "inconnu",
            }
        )

    return json.dumps(result, ensure_ascii=False)
