"""Pictures from the web for a screen: a Wikipedia article's lead image, else an Openverse photo (free, no API key)."""

import asyncio
import logging
import re
from typing import Collection, NamedTuple, Optional

import aiohttp

from ha_client import normalize

logger = logging.getLogger(__name__)

# Wikimedia turns away API clients without a descriptive User-Agent
_HEADERS = {"User-Agent": "voice-assistant/1.0 (https://github.com/isc/voice-assistant)"}
_TIMEOUT = aiohttp.ClientTimeout(total=5)
# For the whole search, however many candidates fail: the voice device waits on it
_DEADLINE = 8
# The Nest Hub Max screen is 1280×800: anything bigger only takes longer to load
_SIZE = 1280
# Openverse hands out originals, up to 13000 px wide for some sources
_MAX_SIDE = 4096
_SCREEN_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}


class Picture(NamedTuple):
    url: str
    content_type: str
    caption: str


def _words(text: str) -> set[str]:
    # Words only: « Paris, Texas » kept « paris, » and missed the article « Paris (Texas) »
    return set(re.findall(r"\w+", normalize(text)))


def is_about(query: str, title: str) -> bool:
    """Whether a Wikipedia article is about what was asked: every word of the query is in its title.

    "Arc de Triomphe" → "Arc de triomphe de l'Étoile" is; "coucher de soleil sur la mer" →
    "Le Coucher du soleil", a painting, is not — a photo search does better there.
    """
    words = _words(query)
    return bool(words) and words <= _words(title)


async def _wikipedia(session: aiohttp.ClientSession, query: str) -> list[tuple[str, str]]:
    """Lead image of the French Wikipedia article that best matches the query, if it is about it."""
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": query,
        "gsrlimit": 1,
        "prop": "pageimages",
        "piprop": "thumbnail",
        "pithumbsize": _SIZE,
    }
    async with session.get("https://fr.wikipedia.org/w/api.php", params=params) as r:
        r.raise_for_status()
        data = await r.json()
    return [
        (page["thumbnail"]["source"], page["title"])
        for page in data.get("query", {}).get("pages", {}).values()
        if "thumbnail" in page and is_about(query, page["title"])
    ]


async def _openverse(session: aiohttp.ClientSession, query: str) -> list[tuple[str, str]]:
    """Free-licence photos, mostly from Flickr, best first."""
    async with session.get("https://api.openverse.org/v1/images/", params={"q": query, "page_size": 5}) as r:
        r.raise_for_status()
        data = await r.json()
    return [
        (result["url"], result.get("title") or query)
        for result in data.get("results", [])
        if max(result.get("width") or 0, result.get("height") or 0) <= _MAX_SIDE
    ]


async def _image_type(session: aiohttp.ClientSession, url: str) -> Optional[str]:
    """Content type of the image at url, None when it answers with no image (a deleted Flickr photo)."""
    try:
        async with session.head(url, allow_redirects=True) as r:
            # What a Cast screen shows: an Openverse original may be an SVG or a TIFF
            if r.status == 200 and r.content_type in _SCREEN_TYPES:
                return r.content_type
            logger.info(f"Image skipped ({r.status}, {r.content_type}): {url}")
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        logger.info(f"Image skipped ({e!r}): {url}")
    return None


async def find_image(query: str, exclude: Collection[str] = ()) -> Optional[Picture]:
    """Find a picture of query on the web, checked to answer: the screen fetches it itself.

    Wikipedia first, for whatever has an article (a monument, an animal, a person); Openverse
    for the rest, and for "another one" once the article's image is in exclude.
    """
    try:
        async with asyncio.timeout(_DEADLINE), aiohttp.ClientSession(headers=_HEADERS, timeout=_TIMEOUT) as session:
            for source, search in (("Wikipédia", _wikipedia), ("Openverse", _openverse)):
                try:
                    candidates = await search(session, query)
                except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as e:
                    logger.warning(f"{source} image search failed: {e}")
                    continue
                for url, caption in candidates:
                    if url not in exclude and (content_type := await _image_type(session, url)):
                        logger.info(f"Image for {query!r} from {source}: {caption}")
                        return Picture(url, content_type, caption)
    except TimeoutError:
        logger.warning(f"Image search for {query!r} gave up after {_DEADLINE} s")
    return None
