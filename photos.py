"""The family's photos, from Immich: found by what they show, who is in them, where and when they were taken, shown
in the web app or on a screen as a slideshow (AD-112).

The server asks Immich with a read-only key and hands each photo out itself: a screen fetches it under /tts/photos/
by a token that lasts an hour (the LAN sees /tts/, the Paris relay forwards nothing else), the web app under
api/photos/, behind its own access control."""

import asyncio
import calendar
import itertools
import logging
import random
import re
import secrets
import time
from collections import Counter
from datetime import date, datetime, timedelta
from typing import Awaitable, Callable, Collection, Iterable, NamedTuple, Optional

import aiohttp
from aiohttp import web

from ha_client import fold, words
from images import _HEADERS
from places import _NETWORK_ERRORS, MONTHS, osm_place
from spotify import _NUMBER_WORDS
from vacuum_schedules import date_words

logger = logging.getLogger(__name__)

COUNT = 20  # photos a request shows
SECONDS = 8  # each one on a screen
TOKEN_SECONDS = 3600  # how long the LAN may fetch a photo given to a screen
SIZES = ("preview", "thumbnail")  # Immich's: 1440 px JPEG, 250 px WebP
_TIMEOUT = aiohttp.ClientTimeout(total=10)
# Immich's places one asked may stand for, a search each (20 smart ones take 0.35 s): « Paris » is Paris and its 20
# arrondissements, « Paris 11 Popincourt »…
_PLACES = 20
# Degrees: a place this large or more is searched only by the name Immich gives it, as its bounds are too loose
# (France's, with its overseas territories, span the globe)
_SPAN = 2.0


class Photo(NamedTuple):
    id: str
    taken: datetime  # local time
    city: str

    @classmethod
    def of(cls, asset: dict) -> "Photo":
        # Immich writes the local time with a Z it does not mean
        taken = datetime.fromisoformat(asset["localDateTime"]).replace(tzinfo=None)
        return cls(asset["id"], taken, (asset.get("exifInfo") or {}).get("city") or "")

    @property
    def caption(self) -> str:
        return f"{_day(self.taken)} · {self.city}" if self.city else _day(self.taken)


# Their first and last days each year (month, day, month, day). Noël is its holidays, la Toussaint its own about
_YEARLY = {
    "printemps": (3, 20, 6, 20), "ete": (6, 21, 9, 22), "automne": (9, 23, 12, 20), "hiver": (12, 21, 3, 19),
    "grandes vacances": (7, 1, 8, 31), "toussaint": (10, 18, 11, 3), "noel": (12, 20, 1, 3),
    "nouvel an": (12, 31, 1, 1), "jour de l'an": (12, 31, 1, 1),
}  # fmt: skip
_BY_END = {"hiver", "nouvel an", "jour de l'an"}  # named by the year they end in: « l'hiver 2024 » skied in February
_MONTHS = [fold(month) for month in MONTHS]


def _month(year: int, month: int) -> tuple[date, date]:
    return date(year, month, 1), date(year, month, calendar.monthrange(year, month)[1])


def _easter(year: int) -> date:
    """Easter Sunday, by the anonymous Gregorian algorithm."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = divmod(b, 4)
    g = (b - (b + 8) // 25 + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return date(year, month, day + 1)


def period(text: str, today: date) -> Optional[tuple[date, date]]:
    """The first and last days of a period said in French: a year, years, a decade, a month, a day, a season, the
    holidays, Noël, Pâques, the week-end, the week, yesterday… With no year, the latest one begun, or over for
    « dernier » (« l'été dernier », on 20 November 2026, is 2026's: the model said 2025's, whatever its tool's
    description said). None if it is none of these."""
    t = re.sub(r"\s+", " ", fold(text).replace("’", "'"))
    last = bool(re.search(r"\b(derni|passe)", t))
    years = [int(year) for year in re.findall(r"\b(?:19|20)\d\d\b", t)]
    if ago := re.search(rf"il y a (\d+|{'|'.join(_NUMBER_WORDS)}) ans?\b", t):
        years.append(today.year - int(_NUMBER_WORDS.get(ago[1], ago[1])))
    if re.search(r"\bl'(annee|an) (derni|passe)", t):
        years.append(today.year - 1)
        last = False  # the year's, not the season's
    elif re.search(r"\bcette annee\b", t):
        years.append(today.year)
    if decade := re.search(r"\bannees (19|20)?(\d)0\b", t):  # « les années 90 »
        first = int(decade[1] or (19 if decade[2] > "2" else 20)) * 100 + int(decade[2]) * 10
        return date(first, 1, 1), date(first + 9, 12, 31)
    if re.search(r"\baujourd'hui\b", t):
        return today, today
    if re.search(r"\bhier\b", t):
        return today - timedelta(days=1), today - timedelta(days=1)
    if re.search(r"\bweek-?end\b", t):
        saturday = today - timedelta(days=(today.weekday() - 5) % 7 + (7 if last and today.weekday() >= 5 else 0))
        return saturday, saturday + timedelta(days=1)
    if re.search(r"\bsemaine\b", t):
        monday = today - timedelta(days=today.weekday() + (7 if last else 0))
        return monday, monday + timedelta(days=6)
    month = next((i + 1 for i, name in enumerate(_MONTHS) if re.search(rf"\b{name}\b", t)), None)
    if re.search(r"\bmois\b", t) and not month:
        first = today.replace(day=1)
        return _month(*((first - timedelta(days=1)) if last else first).timetuple()[:2])
    if len(years) > 1:
        return date(min(years), 1, 1), date(max(years), 12, 31)

    # A year's occurrence of the period: its first and last days
    if named := next(((name, days) for name, days in _YEARLY.items() if re.search(rf"\b{name}\b", t)), None):
        name, (m1, d1, m2, d2) = named
        later = m2 < m1  # it ends the year after it begins
        shift = later and name in _BY_END

        def span(year):
            return date(year - shift, m1, d1), date(year - shift + later, m2, d2)

    elif re.search(r"\bpaques\b", t):

        def span(year):
            return _easter(year) - timedelta(days=2), _easter(year) + timedelta(days=1)

    elif month:
        day = re.search(rf"\b(\d{{1,2}})(?:er)? {_MONTHS[month - 1]}\b", t)

        def span(year):
            return (date(year, month, int(day[1])),) * 2 if day else _month(year, month)

    elif years:

        def span(year):
            return date(year, 1, 1), date(year, 12, 31)

    else:
        return None
    try:
        if years:
            return span(years[0])
        for year in range(today.year + 1, today.year - 3, -1):
            first, end = span(year)
            if (end < today) if last else (first <= today):
                return first, end
    except ValueError:  # « le 31 février »
        return None
    return None


def _day(when: datetime) -> str:
    return f"{date_words(when.date())} {when.year}"


def summary(photos: list[Photo]) -> str:
    """When and where, for the model to say: « prises de 2013 à 2020, à Binic, Sarzeau, Mimizan »."""
    first, last = photos[0].taken, photos[-1].taken
    if first.date() == last.date():
        when = f"le {_day(first)}"
    elif (first.year, first.month) == (last.year, last.month):
        when = f"en {MONTHS[first.month - 1]} {first.year}"
    elif first.year == last.year:
        when = f"en {first.year}"
    else:
        when = f"de {first.year} à {last.year}"
    cities = [city for city, _ in Counter(p.city for p in photos if p.city).most_common(3)]
    return f"prises {when}" + (f", à {', '.join(cities)}" if cities else "")


def _named(places: dict[tuple[str, str], set[str]], names: Iterable[str]) -> list[tuple[str, str]]:
    """Immich's places whose name holds one of these, or is it (« Netherlands »: The Netherlands; « Paris »: Paris and
    Paris 11 Popincourt). `places`: each one's words, made once: Nominatim gives Spain 332 names."""
    asked = [w for name in names if (w := words(name))]
    return [key for key, have in places.items() if any(a <= have for a in asked)]


def _within(markers: list[dict], bounds: Optional[list]) -> list[tuple[str, str]]:
    """The cities of the photos taken within OpenStreetMap's bounds of a place, if under _SPAN across."""
    if not bounds:
        return []
    south, north, west, east = map(float, bounds)
    if max(north - south, east - west) >= _SPAN:
        return []
    inside = (m for m in markers if south <= m["lat"] <= north and west <= m["lon"] <= east)
    return [("city", city) for city in {m["city"] for m in inside if m.get("city")}]


class Immich:
    def __init__(self, url: str, key: str):
        self.url = url.rstrip("/")
        self._key = {"x-api-key": key}  # sent to Immich alone, never to OpenStreetMap
        self._tokens: dict[str, tuple[str, float]] = {}

    async def _call(self, session: aiohttp.ClientSession, path: str, body: Optional[dict] = None, **params):
        url = self.url + path
        async with session.post(url, json=body) if body is not None else session.get(url, params=params) as r:
            r.raise_for_status()
            return await r.json()

    async def find(
        self,
        query: str = "",
        people: Collection[str] = (),
        place: str = "",
        when: str = "",
        exclude: Collection[str] = (),
    ) -> list[Photo] | str:
        """Photos for a request, COUNT at most, in the order they were taken: the most like `query` (Immich's smart
        search, multilingual), else drawn at random. Or why there are none, in French for the model."""
        criteria: dict = {"type": "IMAGE", "withExif": True}
        if when:
            if not (days := period(when, date.today())):
                return f"Aucune photo : période non comprise, « {when} » (une année, un mois, une saison, Noël…)"
            criteria["takenAfter"] = f"{days[0]}T00:00:00.000Z"
            criteria["takenBefore"] = f"{days[1] + timedelta(days=1)}T00:00:00.000Z"
        try:
            async with aiohttp.ClientSession(headers={**_HEADERS, **self._key}, timeout=_TIMEOUT) as session:
                if people:
                    ids, unknown = await self._people(session, people)
                    if unknown:
                        unknown = " ni ".join(unknown)
                        return f"Aucune personne nommée {unknown} dans les photos : son visage n'y a pas de nom"
                    criteria["personIds"] = ids
                where = await self._places(session, place) if place else [{}]
                if not where:
                    return f"Aucune photo prise à {place}"
                size = min(COUNT + len(exclude), 1000)
                found = await asyncio.gather(*(self._search(session, query, {**criteria, **w}, size) for w in where))
        except (*_NETWORK_ERRORS, KeyError) as e:
            logger.warning(f"Immich: {e!r}")
            return "Erreur : la photothèque ne répond pas"
        if query:  # each place's best in turn
            assets = [a for a in itertools.chain.from_iterable(itertools.zip_longest(*found)) if a]
        else:
            assets = [a for each in found for a in each]
            random.shuffle(assets)  # each place's draw mixed with the others'
        photos = list({a["id"]: Photo.of(a) for a in assets if a["id"] not in exclude}.values())[:COUNT]
        logger.info(f"Immich: {len(photos)} photos for {query=} {people=} {place=} {when=} in {where}")
        return sorted(photos, key=lambda p: p.taken) or "Aucune photo trouvée"

    async def _search(self, session: aiohttp.ClientSession, query: str, criteria: dict, size: int) -> list[dict]:
        if query:
            body = {**criteria, "query": query, "language": "fr", "size": size}
            return (await self._call(session, "/api/search/smart", body))["assets"]["items"]
        return await self._call(session, "/api/search/random", {**criteria, "size": size})

    async def _people(self, session: aiohttp.ClientSession, names: Collection[str]) -> tuple[list[str], list[str]]:
        """Immich's people by the names asked, the closest name first (« Zoé »: Zoé before Zoé Martin); the unknown."""
        data = await self._call(session, "/api/people", withHidden="false", size="1000")
        named = sorted(((words(p["name"]), p["id"]) for p in data["people"] if p["name"]), key=lambda n: len(n[0]))
        ids, unknown = [], []
        for name in names:
            asked = words(name)
            if found := next((pid for known, pid in named if asked and asked <= known), None):
                ids.append(found)
            else:
                unknown.append(name)
        return ids, unknown

    async def _places(self, session: aiohttp.ClientSession, place: str) -> list[dict]:
        """The places of Immich's photos one asked stands for, as search criteria, the most photos first: its city,
        region or country as Immich names them, in English but for cities (« Bretagne » is Brittany, « Barcelone »
        Barcelona, « Bruges » Brugge), OpenStreetMap telling; else the cities of the photos taken within its bounds
        (« les Arcs », a ski resort: Bourg-Saint-Maurice, Villaroger)."""
        markers = await self._call(session, "/api/map/markers")
        counts = Counter((key, m[key]) for m in markers for key in ("city", "state", "country") if m.get(key))
        places = {key: words(key[1]) for key in counts}
        if not (found := _named(places, [place])):
            osm = await osm_place(place) or {}
            found = _named(places, osm.get("namedetails", {}).values()) or _within(markers, osm.get("boundingbox"))
        return [{key: value} for key, value in sorted(found, key=counts.__getitem__, reverse=True)[:_PLACES]]

    def token(self, asset_id: str) -> str:
        """A name a screen may fetch the photo by, for TOKEN_SECONDS."""
        now = time.monotonic()
        self._tokens = {token: kept for token, kept in self._tokens.items() if kept[1] > now}
        token = secrets.token_urlsafe(16)
        self._tokens[token] = (asset_id, now + TOKEN_SECONDS)
        return token

    def asset_of(self, token: str) -> Optional[str]:
        asset_id, until = self._tokens.get(token, (None, 0))
        return asset_id if until > time.monotonic() else None

    async def response(self, asset_id: str, size: str = "preview") -> web.Response:
        """The photo, as Immich makes it for a screen or a thumbnail."""
        try:
            async with aiohttp.ClientSession(headers=self._key, timeout=_TIMEOUT) as session:
                url = f"{self.url}/api/assets/{asset_id}/thumbnail"
                async with session.get(url, params={"size": size}) as r:
                    if r.status in (400, 403, 404):
                        raise web.HTTPNotFound()
                    r.raise_for_status()
                    body, content_type = await r.read(), r.content_type
        except _NETWORK_ERRORS as e:
            logger.warning(f"Immich, photo {asset_id}: {e!r}")
            raise web.HTTPBadGateway()
        return web.Response(body=body, content_type=content_type, headers={"Cache-Control": "private, max-age=86400"})


class Slideshow:
    """Photos shown one after another on a screen, SECONDS each, the screen let go after the last.

    show(photo, shown) puts a photo on the screen and returns what the screen now shows (its URL), given what it
    showed before; None when the screen has since been given something else (« Hey Google », a phone's cast): the
    slideshow then ends there, leaving the screen to it."""

    def __init__(
        self,
        screen: str,
        photos: list[Photo],
        show: Callable[[Photo, Optional[str]], Awaitable[Optional[str]]],
        end: Callable[[], Awaitable],
    ):
        self.screen, self.photos, self.index, self.paused = screen, photos, 0, False
        self._moved = asyncio.Event()
        self._task = asyncio.create_task(self._run(show, end))

    async def _run(self, show, end) -> None:
        index = shown = None
        try:
            while True:
                self._moved.clear()
                if self.index != index:
                    if not (shown := await show(self.photos[index := self.index], shown)):
                        return
                try:  # « suivante », a pause, sooner
                    await asyncio.wait_for(self._moved.wait(), None if self.paused else SECONDS)
                except TimeoutError:
                    if not self.move(1):
                        break
            await end()
        except Exception:
            logger.exception("Slideshow failed")

    @property
    def running(self) -> bool:
        return not self._task.done()

    def move(self, step: int) -> bool:
        if not 0 <= self.index + step < len(self.photos):
            return False
        self.index += step
        self._moved.set()
        return True

    def pause(self, paused: bool) -> None:
        self.paused = paused
        self._moved.set()  # resumed, the photo shown has its SECONDS anew

    def stop(self) -> None:
        self._task.cancel()
