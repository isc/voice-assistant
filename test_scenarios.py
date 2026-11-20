#!/usr/bin/env python3
"""Conversations replayed against fake homes, the server's own code running every call (AD-084).

test_hard_cases.py replays requests as dry runs: the server runs no call and answers each « OK », so a speaker
that does not answer, or what a read finds, cannot be played out. Here each scenario sets the devices' states,
and the server's code, this checkout's, runs every call in this process against a fake Home Assistant that
holds them: the homes' devices as the live server knows them (its /api/snapshot), the scenario's states, the
service calls kept and nothing sent. Spotify is fake too, and the calendar empty, Google's API stubbed: its tools,
which the live server offers, change what the model searches (AD-097). There is no satellite. The model is the
live server's, through the same Ollama: run it on the server.

Usage:
    python test_scenarios.py                 # every scenario, 3 times
    python test_scenarios.py -k AD-084 -n 10
    python test_scenarios.py -o runs.jsonl   # each turn kept, its calls and their results
"""

import argparse
import asyncio
import contextlib
import copy
import json
import os
import re
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from yarl import URL

from test_e2e import SERVER


def get(path: str) -> dict:
    try:
        response = urllib.request.urlopen(f"{SERVER}{path}", timeout=10)
    except urllib.error.HTTPError as e:
        response = e  # /health's 503, a part not ready: it still says which model answers
    return json.loads(response.read())


# === FAKE HOME ===


class _Response:
    def __init__(self, status: int, body):
        self.status, self.body = status, body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def json(self):
        return self.body

    async def text(self):
        return self.body if isinstance(self.body, str) else json.dumps(self.body)


# What a service call leaves its device in, as HA would report it
_EFFECTS = {
    "turn_on": "on", "turn_off": "off", "toggle": "on", "open_cover": "open", "close_cover": "closed",
    "lock": "locked", "play_media": "playing", "media_play": "playing", "media_pause": "paused",
    "media_stop": "idle", "alarm_disarm": "disarmed", "start": "cleaning", "pause": "paused",
    "return_to_base": "returning",
}  # fmt: skip
# The template HAClient.reachable reads states with
_STATES_TEMPLATE = re.compile(r"\{% for e in (\[.*?\]) %\}\{\{ e \}\} \{\{ states\(e\) \}\}")


class FakeHA:
    """A home's Home Assistant, as the aiohttp session its HAClient talks through. `states` is HA's side: a
    change there reaches the server only by a read, as a speaker woken up between two refreshes does."""

    timeout = None

    def __init__(self, entities: dict, building_door: Optional[dict] = None):
        self.entities = entities
        self.states = {eid: info["state"] for eid, info in entities.items()}
        if building_door:  # its button and ring sensor, which are no devices of the snapshot (AD-111)
            self.states.update({building_door["button"]: "unknown", building_door["ring"]: "off"})
        # (since when, entity, state): HA's history, which the ring is read from; each state since ever at first
        self.changes: list[tuple[float, str, str]] = [(0.0, eid, state) for eid, state in self.states.items()]
        self.sensors: dict[str, dict] = {}  # a scenario's sensors, HA's whole state: the snapshot has no sensor
        self.calls: list[tuple[str, str, dict]] = []
        from test_unit import FakeRoombaSchedule

        self.robot = FakeRoombaSchedule()  # the Roomba's own schedules, through roomba_schedule (AD-108)

    def _state(self, entity_id: str) -> dict:
        attributes = self.entities.get(entity_id, {}).get("attributes", {})
        return {"entity_id": entity_id, "state": self.states[entity_id], "attributes": attributes}

    def set(self, entity_id: str, state, ago: float = 0):
        """HA's side: a device's state, since `ago` seconds, or a sensor's whole state (sensors)."""
        if isinstance(state, dict):
            self.sensors[entity_id] = state
        else:
            self.states[entity_id] = state
            self.changes.append((time.time() - ago, entity_id, state))

    def history(self, entity_id: str, since: float) -> list[dict]:
        """HA's history of a device from `since`: its state then, and each change after."""
        changes = sorted((when, state) for when, eid, state in self.changes if eid == entity_id)
        then = [state for when, state in changes if when <= since][-1:]
        return [{"state": state} for state in then + [state for when, state in changes if when > since]]

    def get(self, url, **kwargs):
        path = URL(url).path
        if path == "/api/states":
            return _Response(200, [self._state(eid) for eid in self.states] + list(self.sensors.values()))
        if (eid := path.removeprefix("/api/states/")) in self.states:
            return _Response(200, self._state(eid))
        if since := re.fullmatch(r"/api/history/period/(.+)", path):
            eid = kwargs.get("params", {}).get("filter_entity_id")
            when = datetime.fromisoformat(since[1]).timestamp()
            return _Response(200, [self.history(eid, when)] if eid in self.states else [])
        return _Response(404, {"message": "Entity not found."})

    def post(self, url, **kwargs):
        path, payload = URL(url).path, kwargs.get("json") or {}
        if path == "/api/template":
            if found := _STATES_TEMPLATE.search(payload["template"]):
                ids = json.loads(found[1])
                return _Response(200, "".join(f"{eid} {self.states.get(eid, 'unknown')}\n" for eid in ids))
            return _Response(200, "")  # the others (ducking, registries) as if they found nothing
        if found := re.fullmatch(r"/api/services/(\w+)/(\w+)", path):
            domain, service = found.groups()
            self.calls.append((domain, service, payload))
            if domain == "roomba_schedule":
                data = {key: value for key, value in payload.items() if key != "entity_id"}
                return _Response(200, {"changed_states": [], "service_response": self.robot.answer(service, data)})
            eid = payload.get("entity_id")
            if self.states.get(eid) not in (None, "unavailable"):
                if service.startswith("alarm_arm_"):
                    self.states[eid] = "armed_" + service.removeprefix("alarm_arm_")
                elif service == "select_option":
                    self.states[eid] = payload["option"]
                elif service in _EFFECTS:
                    self.states[eid] = _EFFECTS[service]
                elif service == "send_command" and payload.get("command") in _EFFECTS:  # a vacuum's rooms (AD-107)
                    self.states[eid] = _EFFECTS[payload["command"]]
            return _Response(200, [])
        return _Response(404, {})

    async def close(self):
        pass


class FakeStream:
    """The Spotify stream, which a speaker joins at once."""

    def __init__(self):
        self.joins, self.listeners = 0, set()

    async def head_start(self):
        pass

    async def wait_for_join(self, joins_before: int, timeout: float) -> bool:
        self.joins += 1
        self.listeners.add(self.joins)
        return True


# What Spotify's search found for the prelude on 2026-10-26
PRELUDE = "« Prelude & Fugue No. 2 in C Minor, BWV 847: Prelude » de Johann Sebastian Bach, Glenn Gould"


class FakeSpotify:
    """go-librespot and the search: whatever is asked is found, and plays. Of the liked tracks, the prelude."""

    COLLECTION = "spotify:user:replay:collection"

    def __init__(self):
        self.stream, self.found, self.track = FakeStream(), None, None
        self.next = None  # what the next search finds, as Spotify did that day (searched)

    async def find(self, query: str, kind):
        from ha_client import fold
        from spotify import LIKED_THEN_COLLECTION, NOT_LIKED, Music, liked_piece

        self.found, self.next = self.next or (PRELUDE if "prelude" in fold(query) else f"« {query} »"), None
        if kind != "liked":
            return Music(f"spotify:{kind}:replay", self.found, {})
        if not liked_piece(query):
            self.found = "« Symphony No. 3 in C Minor, Op. 78 » de Camille Saint-Saëns"  # one of them
            return Music(self.COLLECTION, "vos titres likés", {})
        if self.found == PRELUDE:
            return Music(self.COLLECTION, self.found, {}, "spotify:track:replay", LIKED_THEN_COLLECTION)
        return Music("spotify:track:replay", self.found, {}, note=NOT_LIKED)

    async def play(self, uri: str, shuffle: bool = False, start_at: str = ""):
        self.track = self.found

    async def playing(self):
        return self.track

    async def now_playing(self):
        return f"En cours sur Spotify : {self.track}" if self.track else "Rien en cours sur Spotify"

    async def control(self, action: str) -> str:
        from spotify import CONTROLS

        return CONTROLS[action]

    async def pause(self):
        pass

    async def resume(self) -> bool:
        return True

    async def now_metadata(self):
        return {}


def empty_calendar():
    """The calendar client, its Google API an empty calendar that refuses writes."""
    from calendar_client import CalendarClient

    client = CalendarClient.__new__(CalendarClient)
    client._service, client._lock = object(), asyncio.Lock()
    client._list_events = lambda *_: {"items": []}
    return client


def ScenarioPhotos():
    """The family's photos (AD-112): every place has some, the children's faces are named, and each search finds new
    ones, so that « d'autres » finds others."""
    from test_unit import FakeImmich, asset

    class Photos(FakeImmich):
        def __init__(self):
            super().__init__()
            self.people = [{"id": f"p-{n}", "name": name} for n, name in enumerate(("Zoé", "Charlie", "Gisèle", "Émilie", "Emily"))]
            self.found = 0

        async def _places(self, session, place):
            return [{"city": place}]

        async def _call(self, session, path, body=None, **params):
            if path.startswith("/api/search/"):
                self.found += 6
                self.smart = self.drawn = [asset(f"a{n}", f"2025-07-{n % 28 + 1:02}", "Sarzeau") for n in range(self.found - 6, self.found)]
            return await super()._call(session, path, body, **params)

    return Photos()


def build_server(snapshot: dict, site: str):
    """A server with the snapshot's homes, each behind its FakeHA, serving `site`."""
    import ha_client
    from ha_client import HAClient
    from test_unit import make_server

    config = snapshot["config"]
    ha_client.LOCAL_CONFIG.clear()
    ha_client.LOCAL_CONFIG.update(config)
    ha_client.ROOM_GROUPS.clear()  # without the whole house's names, as ha_client loads its own
    groups = config.get("room_groups", {}).items()
    ha_client.ROOM_GROUPS.update((name, rooms) for name, rooms in groups if not ha_client.is_whole_house(name))

    homes = {home["name"]: copy.deepcopy(home) for home in snapshot["sites"]}
    server = make_server({"SITES": ",".join(homes)})
    for each in server.sites:
        home = homes[each.name]
        each.ha_client = client = HAClient("http://replay", "", home.pop("home_name"))
        for key, value in home.items():
            if key != "name":
                setattr(client, key, value)
        # getattr: code before AD-111 has none, as « before » runs this file on main's (CLAUDE.md, Tests)
        client._session = FakeHA(client.entities, getattr(client, "building_door", None))
    server.site = server._voice_site = server.site_named(site)
    server.calendar_client = empty_calendar()
    server.spotify = FakeSpotify()
    with contextlib.suppress(ImportError):  # code before AD-112, as « before » runs this file on main's (CLAUDE.md, Tests)
        server.immich = ScenarioPhotos()
    server._keep_llm_loaded = lambda: None
    return server


# === SCENARIOS ===


@dataclass
class Turn:
    """A request's answer: the reply, the calls with their results, the service calls HA got, and the speaker
    Spotify plays on after it."""

    reply: str
    calls: list[dict]
    sent: list[tuple[str, str, dict]]
    music: Optional[tuple[str, str]]
    wall: float
    robot_schedule: list  # the Roomba's schedules after it (AD-108)
    pressed: list[str]  # the homes whose HA got a button pressed, the other home's too (AD-111)

    @property
    def said(self) -> str:
        from ha_client import fold

        return fold(self.reply)

    @property
    def played(self) -> list[str]:
        return [payload["entity_id"] for _, service, payload in self.sent if service == "play_media"]


def plays_on(entity_id: str):
    return lambda t: (t.music == ("ha", entity_id), f"the music on {entity_id}")


def offers_one_that_answers(*names: str, unlike: str = ""):
    """Nothing played elsewhere unasked, a speaker that answers named, and not that one (offline, 2026-11-07)."""
    named = says(*names, unlike=unlike)
    return lambda t: (not t.played and named(t)[0], "nothing played, " + named(t)[1])


def plays_on_or_asks(entity_id: str, *names: str):
    """There, or an offer naming it: « … ? », « … si tu le souhaites »."""

    def check(t):
        offered = re.search(r"\?|si tu (?:le )?(?:veux|souhaites|preferes)", t.said)
        asked = bool(offered) and offers_one_that_answers(*names)(t)[0]
        return t.music == ("ha", entity_id) or asked, f"the music on {entity_id}, or an offer"

    return check


def searches(*words: str, unlike: str = ""):
    """Spotify searched with one of these words, and not that one."""

    def check(t):
        from ha_client import fold

        queries = [fold(c["args"].get("query", "")) for c in t.calls if c["function"] == "play_music"]
        found = any(any(w in q for w in words) and not (unlike and unlike in q) for q in queries)
        expect = f"a Spotify search with « {' » or « '.join(words)} »"
        return found, expect + (f", without « {unlike} »" if unlike else "")

    return check


def told_playing(description: str):
    """play_music's result says it plays this."""
    return lambda t: (
        any(description in (c.get("result") or "") for c in t.calls if c["function"] == "play_music"),
        f"told playing {description}",
    )


def turns_off(*entity_ids: str, allowed: tuple[str, ...] = ()):
    """These devices turned off, and nothing else got a call but those allowed turned off: by id, or by a prefix
    ending with « . » (« light. »: any light)."""

    def check(t):
        sent = {(service, str(payload.get("entity_id"))) for _, service, payload in t.sent}
        ok = (*entity_ids, *allowed)
        others = sorted(f"{s} {e}" for s, e in sent if s != "turn_off" or not any(
            e == a or (a.endswith(".") and e.startswith(a)) for a in ok
        ))  # fmt: skip
        missed = [e for e in entity_ids if ("turn_off", e) not in sent]
        expect = f"{' and '.join(entity_ids)} turned off, nothing else" + (f" but {', '.join(allowed)}" if allowed else "")
        return not others and not missed, expect + (f" (missed {missed}, others {others})" if missed or others else "")

    return check


def opens_all_the_way(word: str):
    """One shutter opened, the one the call names with this word, and no position set (AD-099)."""

    def check(t):
        from ha_client import fold

        sent = [(service, payload.get("entity_id")) for _, service, payload in t.sent]
        named = any(word in fold(c["args"].get("entity", "")) for c in t.calls if c["function"] == "open_cover")
        return len(sent) == 1 and sent[0][0] == "open_cover" and named, f"the shutter « {word} » opened all the way"

    return check


def moves(**services: str):
    """These shutters, by entity id's object part, got these services, and nothing else got a call."""

    def check(t):
        sent = {(service, payload.get("entity_id")) for _, service, payload in t.sent}
        wanted = {(service, f"cover.{name}") for name, service in services.items()}
        return sent == wanted, ", ".join(f"{service} {name}" for name, service in services.items()) + ", nothing else"

    return check


def vacuums(*actions: str, home: str = ""):
    """The robot vacuum got these actions, and no other; at another home, by the calls that reached it."""

    def check(t):
        if home:
            done = [c["args"].get("action") for c in t.calls if c["function"] == "vacuum"
                    and str(c["args"].get("home", "")).lower() == home and not str(c.get("result")).startswith("Rien")]
            return done == list(actions), f"vacuum {', '.join(actions) or 'nothing'} at {home}"  # fmt: skip
        sent = [service for domain, service, _ in t.sent if domain == "vacuum"]
        return sent == list(actions), f"vacuum {', '.join(actions) or 'nothing'}"

    return check


def cleans(*rooms: str, home: str = ""):
    """The robot vacuum started in these rooms, in any order, by one call, at `home` if given (AD-107); a room
    refused before does not count, nothing having moved. The fake HA has run it: its regions are test_unit's."""
    from ha_client import rooms_named, vacuum_map

    def check(t):
        done = [c for c in t.calls if c["function"] == "vacuum" and str(c["args"].get("home", home)).lower() == home
                and not str(c.get("result")).startswith(("Rien", "Pas de"))]  # fmt: skip
        ok = len(done) == 1 and done[0]["args"].get("action") == "start"
        known = list(vacuum_map("vacuum.parispi").get("rooms", ()))
        ok = ok and sorted(rooms_named(str(done[0]["args"].get("room", "")), known)[0]) == sorted(rooms)
        return ok, f"vacuum start in {', '.join(rooms)}" + (f" at {home}" if home else "")

    return check


def schedules(test, wanted: str):
    """The robot's schedules changed by one call, `test` true of them after (AD-108)."""

    def check(t):
        changes = [s for domain, s, _ in t.sent if domain == "roomba_schedule" and s != "get"]
        return len(changes) == 1 and bool(test(t.robot_schedule)), wanted

    return check


def _on(schedule: list, day: int) -> list:
    return [entry for entry in schedule if day in entry["start"]["day"]]


def _regions(entry: dict) -> list:
    from vacuum_schedules import regions

    return regions(entry)


def pauses_music(t):
    paused = any(c["function"] == "stop_media" or c["args"].get("action") == "pause" for c in t.calls if c["function"] != "vacuum")
    robot = [service for domain, service, _ in t.sent if domain == "vacuum"]
    return paused and not robot, "the music paused, not the vacuum"


def says(*patterns: str, unlike: str = ""):
    """The reply, folded, holds one of these patterns, and not that one."""
    return lambda t: (
        any(re.search(p, t.said) for p in patterns) and not (unlike and re.search(unlike, t.said)),
        f"« {' » or « '.join(patterns)} » said" + (f", not « {unlike} »" if unlike else ""),
    )


# The sensor a ring at the Paris flat turns on, at its building door (AD-111)
RING = "binary_sensor.comelit_incoming_call"


def _offered_unlock(t) -> bool:
    return any(c["function"] == "unlock" and c.get("confirm") for c in t.calls)


# A reply saying the door open, when only the order is known to have gone: its board acts within ~6 s, and says
# nothing back (AD-111)
_SAID_OPEN = re.compile(
    r"\b(?:est|a ete) (?:bien )?(?:ouverte?|deverrouillee?)\b|\bj'ai (?:bien )?(?:ouvert|deverrouille)"
)


def opens_building_door(t):
    """Paris's building door pressed once, the flat's door neither locked nor offered for unlocking, and the reply
    not saying the door is open."""
    locks = [service for domain, service, _ in t.sent if domain == "lock"]
    ok = t.pressed == ["paris"] and not locks and not _offered_unlock(t) and not _SAID_OPEN.search(t.said)
    return ok, "the building door opened once, no lock, not said open"


def offers_flat_door(t):
    """The flat's door offered for a tap in the app (AD-068), the building's left shut."""
    return _offered_unlock(t) and not t.pressed, "the flat's door offered for unlocking, the building's not opened"


def refuses_flat_door(t):
    """From the satellite: the flat's door refused (AD-068), the building's left shut, neither said open."""
    refused = any(c["function"] == "unlock" and "enceinte" in str(c.get("result")) for c in t.calls)
    return refused and not t.pressed and not _SAID_OPEN.search(t.said), "the flat's door refused, nothing opened"


def locks_flat_door(t):
    sent = [(domain, service) for domain, service, _ in t.sent]
    return sent == [("lock", "lock")] and not t.pressed, "the flat's door locked, the building's not opened"


def nothing_opened_asks(t):
    """Charras has no building door: Paris's not opened unasked, the model asks."""
    return not t.pressed and "?" in t.reply, "nothing opened, a question"


def ringing(ended: float = 0, satellite: bool = False):
    """Someone rang at Paris's building door: its ring sensor on, or on until `ended` seconds ago, for 30 s as the
    board keeps it; the request from the satellite rather than the app, if `satellite`."""

    def change(server):
        paris = server.site_named("paris").ha_client._session
        paris.set(RING, "on", ago=ended + 30)
        if ended:
            paris.set(RING, "off", ago=ended)
        if satellite:
            from_satellite(server)

    return change


def photos(expect: str, cond=lambda args: True):
    """show_photos called with arguments as `cond` wants, its own photos, no picture from the web (AD-112)."""

    def check(t):
        shown = [c["args"] for c in t.calls if c["function"] == "show_photos"]
        web = any(c["function"] == "show_image" for c in t.calls)
        return bool(shown) and not web and any(cond(args) for args in shown), f"show_photos {expect}"

    return check


def photo_arg(args: dict, key: str) -> str:
    from ha_client import fold

    value = args.get(key) or ""
    return fold(" ".join(value) if isinstance(value, list) else str(value))


def not_said(check, pattern: str):
    """`check`, and the reply, folded, without `pattern`."""

    def both(t):
        ok, what = check(t)
        return ok and not re.search(pattern, t.said), f"{what}, not « {pattern} » said"

    return both


def slideshow_started(t):
    started = [c for c in t.calls if c["function"] == "show_photos" and str(c.get("result", "")).startswith("Diaporama")]
    return bool(started), "a slideshow on the screen"


def web_image(t):
    names = [c["function"] for c in t.calls]
    return "show_image" in names and "show_photos" not in names, "show_image, no show_photos"


def slideshow(action: str):
    return lambda t: (
        any(c["function"] == "slideshow" and c["args"].get("action") == action for c in t.calls),
        f"slideshow {action}",
    )


def from_satellite(server):
    """The request from the home's satellite rather than a browser of the web app: no button to tap there."""
    server._client = None


def number(value: str) -> str:
    """A pattern of a number with decimals as a reply may write it: 19,9, 19.9, 19 virgule 9."""
    whole, decimals = value.split(".")
    return rf"\b{whole}(?:[,.]| virgule ){decimals}\b"


def none_said_off(t):
    # A Google Home playing nothing says « off »: the reply took it for one that cannot play (2026-10-26)
    return not t.played and "eteint" not in t.said, "nothing played, no speaker said off"


# The speakers of Charras that answered on 2026-10-26, as a reply may name them
CHARRAS_ANSWERING = ("salon", "living", "nest", "hub", "display", "dining", "salle a manger", "charlie")

CHARRAS_SPEAKERS = {
    "media_player.sonos_move": "unavailable",  # Terrasse
    "media_player.sonos_move_2": "unavailable",  # Cuisine
    "media_player.living_room_speaker": "off",  # Salon, a Google Home playing nothing
    "media_player.dining_room_display": "idle",  # the Nest Hub Max, in no room
    "media_player.home_mini_zoe": "unavailable",
    "media_player.home_mini_charlie": "off",
}
SONOS_BACK = {"media_player.sonos_move_2": "idle"}
PARIS_SPEAKERS = {"media_player.google_home": "off", "media_player.sonos_paris": "idle"}
SONOS_PARIS_OFF = {"media_player.sonos_paris": "unavailable"}


def sensors(*rows: tuple[str, str, str, str]) -> dict:
    """Sensors' states by entity id, from (entity id, name, value, unit)."""
    from test_unit import sensor_state

    return {row[0]: sensor_state(*row) for row in rows}


# Charras's temperature sensors, read on 2026-10-31 (those that answered « température intérieure » shown that day,
# 2026-11-08's values for the others), and the upstairs corridor's
CHARRAS_TEMPERATURES = sensors(
    ("sensor.couloir_etage_capteur_temperature_sensor", "Couloir étage température", "19.9", "°C"),
    ("sensor.couloir_etage_capteur_humidity_sensor", "Couloir étage humidité", "54.0", "%"),
    ("sensor.couloir_etage_capteur_carbon_dioxide_sensor", "Couloir étage CO2", "361.0", "ppm"),
    ("sensor.chaudiere_temperature", "Chaudière température", "62.5", "°C"),
    ("sensor.eau_chaude_temperature", "Eau chaude température", "37.9", "°C"),
    ("sensor.temperature_ambiante_chaudiere", "Température ambiante chaudière", "13.8", "°C"),
    ("sensor.temperature_exterieure_okofen", "Température extérieure (Ökofen)", "13.0", "°C"),
    ("sensor.radiateurs_temperature_piece", "Radiateurs température pièce", "15.7", "°C"),
    ("sensor.radiateurs_temperature_depart", "Radiateurs température départ", "16.1", "°C"),
    ("sensor.plancher_temperature_piece", "Plancher température pièce", "16.5", "°C"),
    ("sensor.plancher_temperature_depart", "Plancher température départ", "16.6", "°C"),
    ("sensor.plancher_2_temperature_piece", "Plancher 2 température pièce", "21.7", "°C"),
    ("sensor.plancher_2_temperature_depart", "Plancher 2 température départ", "24.4", "°C"),
    ("sensor.jardin_piscine_water_temperature", "Piscine Water Temperature", "15", "°C"),
    ("sensor.jardin_piscine_air_temperature", "Piscine Air Temperature", "7.2", "°C"),
    ("sensor.freebox_v7_r1_temperature_1", "Freebox v7 (r1) Température 1", "34", "°C"),
    ("sensor.freebox_v7_r1_temperature_2", "Freebox v7 (r1) Température 2", "34", "°C"),
    ("sensor.freebox_v7_r1_temperature_3", "Freebox v7 (r1) Température 3", "33", "°C"),
    ("sensor.freebox_v7_r1_temperature_cpu_ap", "Freebox v7 (r1) Température CPU AP", "53", "°C"),
    ("sensor.freebox_v7_r1_temperature_cpu_cp_master", "Freebox v7 (r1) Température CPU CP Master", "65", "°C"),
    ("sensor.freebox_v7_r1_temperature_cpu_cp_slave", "Freebox v7 (r1) Température CPU CP Slave", "65", "°C"),
    ("sensor.sirene_exterieure_temperature", "Sirene exterieure temperature", "23.9", "°C"),
    ("sensor.entree_temperature", "Entrée temperature", "20.8", "°C"),
    ("sensor.escalier_temperature", "Escalier temperature", "19.3", "°C"),
)
TERRACE_ON = {"light.lumiere_terrasse_sejour": "on", "light.lumiere_terrasse_parents": "on"}
# Paris's living room, kitchen and bar, one HA room
PARIS_LIVING_LIGHTS = (
    "light.dimmer_2_12", "light.dimmer_2_13", "light.dimmer_2_7", "light.dimmer_2_4",  # spots, sink, bar, sconces
    "light.dimmer_2_14", "light.dimmer_2_15", "light.dimmer_2_9", "light.dimmer_2_18",  # billiard, table, sofas, bar
)  # fmt: skip
# An evening in Paris: lights on all over, the bedside lamps' power too, which their own switch turns off
PARIS_EVENING_LIGHTS = PARIS_LIVING_LIGHTS + (
    "light.salle_de_bain_parents_lavabo", "light.salle_de_bain_parents_douche", "light.dimmer_2_6", "light.dimmer_2_8",
    "light.dimmer_2_16", "light.lumiere_balcon",
)  # fmt: skip
PARIS_EVENING = {
    **dict.fromkeys(PARIS_EVENING_LIGHTS, "on"),
    "switch.lampe_de_chevet_ivan": "on", "switch.lampe_de_chevet_emilie": "on", "switch.led_de_la_borne_de_zoe": "on",
}  # fmt: skip
CONCERTO = (
    "« Bach, JS: Concerto for Four Pianos in A Minor, BWV 1065: I. Allegro » de Johann Sebastian Bach, Michel Béroff"
)


def searched(found: str):
    """What Spotify's search found that day for the next request: a change not HA's."""

    def change(server):
        server.spotify.next = found

    return change


def went_on(track: str, forgotten: bool = False):
    """Spotify went on by itself to this track, and the conversation was forgotten since: a change not HA's."""

    def change(server):
        server.spotify.track = track
        if forgotten:
            server.conversation.last_time = 0

    return change


# (name, site, states, [(request, check, changes)]): states as the last refresh read them, a sensor's its whole HA
# state (sensors); changes, HA's before that request, the server's own copy left as it was, or a function of the
# server (went_on)
SCENARIOS = [
    ("AD-084 the Sonos back online", "charras", CHARRAS_SPEAKERS, [
        ("Est-ce que tu peux me jouer le prélude numéro deux de Bach sur la sonos ?",
         offers_one_that_answers(*CHARRAS_ANSWERING), {}),
        ("Y'a pas une sonos qui est joignable ?", none_said_off, {}),
        # « Dining room Display » in HA then, where a question was fair; « Nest Hub Max » since 2026-10-29
        ("Bah mets alors sur le Nest Hub Max.", plays_on("media_player.dining_room_display"), {}),
        ("Tu peux basculer le son sur la sonos, normalement elle est connectée maintenant.",
         plays_on_or_asks("media_player.sonos_move_2", "cuisine", "move 2"), SONOS_BACK),
        ("Oui, vas-y.", plays_on("media_player.sonos_move_2"), {}),
    ]),
    # The music on the screen, then the kitchen's Sonos named
    ("AD-084 c'est parfait ça", "charras", {**CHARRAS_SPEAKERS, **SONOS_BACK}, [
        ("Mets le prélude numéro deux de Bach sur le Nest Hub Max.",
         plays_on("media_player.dining_room_display"), {}),
        ("Y a quoi comme enceinte qui est connectée là ?", none_said_off, {}),
        ("Eh ben la sonos de la cuisine, c'est parfait ça.",
         plays_on_or_asks("media_player.sonos_move_2", "cuisine", "move 2"), {}),
    ]),
    # The prelude over, Spotify went on to a concerto it chose, and the turn that asked for it had left the
    # history: « Prélude numéro 2 » found another pianist's (2026-10-26)
    ("AD-087 remets le prélude", "charras", {**CHARRAS_SPEAKERS, **SONOS_BACK}, [
        ("Mets le prélude numéro deux de Bach sur la Sonos de la cuisine.",
         plays_on("media_player.sonos_move_2"), {}),
        # Its pianist, or its title as Spotify writes it, which finds Gould's (2026-10-30)
        ("Et maintenant j'aimerais que tu remettes le prélude numéro deux.", searches("gould", "bwv 847: prelude"),
         went_on(CONCERTO, forgotten=True)),
        # What was asked before is not what is asked now
        ("Mets plutôt du Chopin.", searches("chopin", unlike="gould"), went_on(CONCERTO)),
    ]),
    # Maurane's song first, as Spotify found that day; then the model asked the liked tracks for Gould's, and all of
    # them played, the request left aside, while the reply said Gould's (2026-10-29)
    ("AD-090 celui de Glenn Gould dans mes titres likés", "charras", {**CHARRAS_SPEAKERS, **SONOS_BACK}, [
        ("Est-ce que tu peux nous mettre le prélude numéro deux de Bach dans le salon ?",
         plays_on("media_player.living_room_speaker"), searched("« Sur un prélude de Bach » de Maurane")),
        ("Non c'est pas ça que je voulais écouter, je voulais jouer par Glengould, celui qui est dans mes titres "
         "préférés.", told_playing(PRELUDE), {}),
        # All of them, as asked on 2026-10-03 (« mes favoris »)
        ("Mets plutôt tous mes titres likés en aléatoire.", told_playing("Lecture sur Spotify : vos titres likés"), {}),
    ]),
    # « la Google Home » heard « la boule de l'homme », the music on the satellite and the Sonos offline: the one
    # speaker it could go to, asked about with the Sonos, and no follow-up to answer (2026-11-07). Here, with no
    # satellite, the music on the Sonos
    ("AD-095 la boule de l'homme", "paris", PARIS_SPEAKERS, [
        ("Mets le prélude numéro deux de Bach sur la Sonos.", plays_on("media_player.sonos_paris"), {}),
        ("Mets le son sur la boule de l'homme.", plays_on("media_player.google_home"), {}),
    ]),
    ("AD-095 sous-douce, the Sonos offline", "paris", {**PARIS_SPEAKERS, **SONOS_PARIS_OFF}, [
        ("Tu peux mettre France Inter sur la sous-douce ?", offers_one_that_answers("google", "living", "salon", unlike="sonos"), {}),
    ]),
    # Two terrace lights, « Lumière terrasse séjour » and « … parents », in an area of the outside: the model
    # sent « Lumière terrasse », and the first of the two names holding it was the one turned off (2026-11-08). The
    # garden's « Spots terrasse potager », there since, may go off too: Ivan, 2026-11-08. So may every light of the
    # area, « Extérieur », which the registry's 12B turns off with them (AD-098): Ivan, 2026-11-09
    ("AD-096 les lumières de la terrasse", "paris", TERRACE_ON, [
        ("Éteins les lumières de la terrasse.", turns_off(*TERRACE_ON, allowed=(
            "light.spots_potager", "light.spots_balcon_enfants", "light.lumiere_balcon",
            "light.dimmer_2_basic_11", "light.dimmer_2_basic_2_10", "light.dimmer_2_2_10",
        )), {}),
    ]),
    ("AD-096 the parents' terrace light alone", "paris", TERRACE_ON, [
        ("Éteins la lumière de la terrasse côté chambre parentale.", turns_off("light.lumiere_terrasse_parents"), {}),
    ]),
    # « température intérieure » searched, the upstairs corridor's sensor named « Capteur Temperature sensor »: the
    # stairs' 19.3 °C was said to be upstairs (2026-10-31). Renamed in HA « Couloir étage température » (AD-097)
    ("AD-097 la température à l'étage", "charras", CHARRAS_TEMPERATURES, [
        ("Il fait quelle température à l'étage ?", says(number("19.9"), unlike=number("19.3")), {}),
    ]),
    # « au niveau de » (by) read as a level: the shutter set to 50 %, half closed as it was open (2026-11-11)
    ("AD-099 le volet au niveau de la table à manger", "paris",
     {"cover.roller_shutter_3_3": "closed", "cover.roller_shutter_3_4": "open"}, [
        ("Ouvre le volet au niveau de la table à manger.", opens_all_the_way("manger"), {}),
    ]),
    # Two orders in one request: one carried out, on the shutter the request had just named otherwise, HA's two
    # living-room names being swapped then (2026-11-11). Names right, both run
    ("AD-099 two orders on the shutters", "paris",
     {"cover.roller_shutter_3_3": "closed", "cover.roller_shutter_3_4": "open"}, [
        ("Ferme le volet du salon.", moves(roller_shutter_3_4="close_cover"), {}),
        ("Rouvre-le jusqu'en haut et ouvre aussi celui de la salle à manger.",
         moves(roller_shutter_3_4="open_cover", roller_shutter_3_3="open_cover"), {}),
    ]),
    # The evening of 2026-11-13: a light HA said off was turned off again, « je viens de l'éteindre », while it shone;
    # then the whole flat's lights named one by one, 14 of 30, and « toutes les lumières de la maison » said
    ("AD-102 the evening of 2026-11-13", "paris", PARIS_EVENING, [
        ("Tu peux éteindre toutes les lumières du salon et de la cuisine.", turns_off(*PARIS_LIVING_LIGHTS), {}),
        ("Ah bah non, t'as pas éteint la lumière de la table à manger.",
         says(r"\bdeja\b", r"\?", unlike=r"viens de|c'est fait"), {}),
        ("Toujours pas, éteins toutes les lumières de l'appartement.", turns_off(*PARIS_EVENING_LIGHTS, allowed=("light.",)), {}),
    ]),
    ("AD-102 every light, the bedside lamps' power kept", "paris", PARIS_EVENING, [
        ("Éteins toutes les lumières de l'appartement.", turns_off(*PARIS_EVENING_LIGHTS, allowed=("light.",)), {}),
    ]),
    # Paris's Roomba, in HA since 2026-11-15: started, paused, resumed, and « arrête » sends it to its dock (AD-106)
    ("AD-106 the vacuum, start to dock", "paris", {"vacuum.parispi": "docked"}, [
        ("Lance l'aspirateur.", vacuums("start"), {}),
        ("Mets-le en pause.", vacuums("pause"), {}),
        ("Tu peux le relancer ?", vacuums("start"), {}),
        ("Arrête l'aspirateur.", vacuums("return_to_base"), {}),
    ]),
    ("AD-106 where is the vacuum", "paris", {"vacuum.parispi": "returning"}, [
        ("Il est où l'aspirateur ?", says(r"rentr|base"), {}),
    ]),
    # The rooms of its map, from Ivan's schedule in iRobot's app, said as HA's areas (AD-107)
    ("AD-107 the vacuum in the kitchen", "paris", {"vacuum.parispi": "docked"}, [
        ("Passe l'aspirateur dans la cuisine.", cleans("Cuisine"), {}),
    ]),
    ("AD-107 two rooms", "paris", {"vacuum.parispi": "docked"}, [
        ("Lance l'aspirateur dans la cuisine et la salle à manger.", cleans("Cuisine", "Salle à manger"), {}),
    ]),
    ("AD-107 the living room", "paris", {"vacuum.parispi": "docked"}, [
        ("Tu peux passer l'aspirateur dans le salon ?", cleans("Salon"), {}),
    ]),
    ("AD-107 the parents' bedroom", "paris", {"vacuum.parispi": "docked"}, [
        ("Passe l'aspirateur dans la chambre des parents.", cleans("Chambre parents"), {}),
    ]),
    ("AD-107 the children's bedrooms", "paris", {"vacuum.parispi": "docked"}, [
        ("Passe l'aspirateur dans les chambres des enfants.", cleans("Chambre Charlie", "Chambre Zoé"), {}),
    ]),
    # Not on its map: nothing started, the whole flat least of all
    ("AD-107 a room it has not", "paris", {"vacuum.parispi": "docked"}, [
        ("Passe l'aspirateur dans le garage.", vacuums(), {}),
    ]),
    ("AD-107 a room from Charras", "charras", {}, [
        ("Passe l'aspirateur dans la cuisine à Paris.", cleans("Cuisine", home="paris"), {}),
    ]),
    # The Roomba's own schedules, read and changed by voice (AD-108). Paris's on 2026-11-15: Monday and Thursday
    # 8:30 the whole flat, Wednesday 9:00 off (Charlie, Zoé, the corridor), Friday 9:00 the dining room and kitchen
    ("AD-108 the schedules said", "paris", {"vacuum.parispi": "docked"}, [
        ("Quand est-ce que l'aspirateur passe dans la semaine ?", says(r"lundi.*vendredi|vendredi.*lundi"), {}),
    ]),
    ("AD-108 a bedroom left out on Mondays", "paris", {"vacuum.parispi": "docked"}, [
        ("Le lundi, l'aspirateur ne doit plus faire la chambre de Zoé.", schedules(
            lambda s: [_regions(e) for e in _on(s, 1)] == [["1", "2", "5", "8", "17", "18", "19", "20", "3", "7"]]
            and [_regions(e) for e in _on(s, 4)] == [[]],
            "Monday without Zoé's room, Thursday still the whole flat"), {}),
    ]),
    ("AD-108 a cleaning added", "paris", {"vacuum.parispi": "docked"}, [
        ("Programme l'aspirateur le samedi à 10 h dans la cuisine.", schedules(
            lambda s: [(e["start"]["hour"], e["start"]["min"], _regions(e), e["enabled"]) for e in _on(s, 6)]
            == [(10, 0, ["7"], True)] and len(s) == 4,
            "Saturday 10:00, the kitchen, added"), {}),
    ]),
    ("AD-108 Friday moved", "paris", {"vacuum.parispi": "docked"}, [
        ("Décale la programmation de l'aspirateur du vendredi à 14 h.", schedules(
            lambda s: [(e["start"]["hour"], e["start"]["min"], _regions(e)) for e in _on(s, 5)]
            == [(14, 0, ["2", "7"])],
            "Friday at 14:00, its rooms kept"), {}),
    ]),
    ("AD-108 Wednesday removed", "paris", {"vacuum.parispi": "docked"}, [
        ("Supprime la programmation du mercredi de l'aspirateur.", schedules(
            lambda s: not _on(s, 3) and len(s) == 2, "Wednesday's gone, the others kept"), {}),
    ]),
    ("AD-108 holidays", "paris", {"vacuum.parispi": "docked"}, [
        ("On part en vacances jusqu'au 28 novembre, coupe les programmations de l'aspirateur.", schedules(
            lambda s: not any(e["enabled"] for e in s), "all off"), {}),
        ("On est rentrés, tu peux remettre les programmations de l'aspirateur ?", schedules(
            lambda s: [e["enabled"] for e in s] == [True, False, True], "back on, Wednesday's still off"), {}),
    ]),
    # Now is the vacuum tool's, not a schedule
    ("AD-108 now is not a schedule", "paris", {"vacuum.parispi": "docked"}, [
        ("Passe l'aspirateur dans l'entrée.", cleans("Entrée"), {}),
    ]),
    # « Mets en pause » with music playing is the music's, though the robot cleans
    ("AD-106 pause is the music's", "paris", {**PARIS_SPEAKERS, "vacuum.parispi": "cleaning"}, [
        ("Mets le prélude numéro deux de Bach sur la Sonos.", plays_on("media_player.sonos_paris"), {}),
        ("Mets en pause.", pauses_music, {}),
    ]),
    # Charras has none: Paris's is never started unasked, the model asks; then it is
    ("AD-106 the vacuum from Charras", "charras", {}, [
        ("Lance l'aspirateur.", vacuums(home="paris"), {}),
        ("Oui, celui de Paris.", vacuums("start", home="paris"), {}),
    ]),
    # Paris's building door, opened by voice at any time, from the satellite as from the app (AD-111). Its words
    ("AD-111 la porte de l'immeuble", "paris", {}, [("Ouvre la porte de l'immeuble.", opens_building_door, {})]),
    ("AD-111 la porte d'en bas", "paris", {}, [("Tu peux ouvrir la porte d'en bas ?", opens_building_door, {})]),
    ("AD-111 la porte du bas", "paris", {}, [("Ouvre la porte du bas.", opens_building_door, {})]),
    ("AD-111 ouvre en bas", "paris", {}, [("Ouvre en bas.", opens_building_door, {})]),
    ("AD-111 la porte de la rue", "paris", {}, [("Ouvre la porte de la rue.", opens_building_door, {})]),
    ("AD-111 l'interphone", "paris", {}, [("Ça sonne à l'interphone, tu peux ouvrir ?", opens_building_door, {})]),
    ("AD-111 la personne qui sonne", "paris", {}, [
        ("Ouvre à la personne qui sonne.", opens_building_door, ringing()),
    ]),
    ("AD-111 the satellite", "paris", {}, [("Ouvre la porte de l'immeuble.", opens_building_door, from_satellite)]),
    # Plain « ouvre la porte »: the building's just after a ring, else the flat's, a tap in the app (AD-068)
    ("AD-111 ouvre la porte, ringing", "paris", {}, [("Ouvre la porte.", opens_building_door, ringing())]),
    ("AD-111 ouvre la porte, rang 40 s ago", "paris", {}, [("Ouvre la porte.", opens_building_door, ringing(40))]),
    ("AD-111 ouvre la porte, ringing, the satellite", "paris", {}, [
        ("Ouvre la porte.", opens_building_door, ringing(satellite=True)),
    ]),
    # Then the building's, said: the satellite's refusal of the flat's door (« l'enceinte n'ouvre aucune porte ») must
    # not make it refuse that one
    ("AD-111 ouvre la porte, no ring", "paris", {}, [
        ("Ouvre la porte.", offers_flat_door, {}),
        ("Non, celle d'en bas.", opens_building_door, {}),
    ]),
    ("AD-111 ouvre la porte, no ring, the satellite", "paris", {}, [
        ("Ouvre la porte.", refuses_flat_door, from_satellite),
        ("Non, celle d'en bas.", opens_building_door, from_satellite),
    ]),
    ("AD-111 verrouille la porte, ringing", "paris", {}, [("Verrouille la porte.", locks_flat_door, ringing())]),
    ("AD-111 ferme la porte", "paris", {}, [("Tu peux fermer la porte ?", locks_flat_door, {})]),
    # From Charras, which has none: Paris's when named, else a question, as the vacuum (AD-106)
    ("AD-111 from Charras, à Paris", "charras", {}, [
        ("Ouvre la porte de l'immeuble à Paris.", opens_building_door, {}),
    ]),
    ("AD-111 from Charras, à Paris, the satellite", "charras", {}, [
        ("Ouvre la porte de l'immeuble à Paris.", opens_building_door, from_satellite),
    ]),
    ("AD-111 from Charras, unnamed", "charras", {}, [
        ("Ouvre la porte de l'immeuble.", nothing_opened_asks, {}),
        ("Oui, celle de Paris.", opens_building_door, {}),
    ]),
    # Paris's ring is not Charras's: « ouvre la porte » there is Charras's door
    ("AD-111 from Charras, ouvre la porte while Paris rings", "charras", {}, [
        ("Ouvre la porte.", offers_flat_door, ringing()),
    ]),
    # The family's photos, from Immich (AD-112): who, where, when, what, and the web's pictures still show_image's
    ("AD-112 des photos de Zoé", "paris", {}, [
        ("Montre-moi des photos de Zoé.", photos("of Zoé, no description", lambda a: "zoe" in photo_arg(a, "people")
                                                 and "zoe" not in photo_arg(a, "query")), {}),
        ("Montre-m'en d'autres.", photos("of Zoé again", lambda a: "zoe" in photo_arg(a, "people")), {}),
    ]),
    ("AD-112 les vacances en Bretagne l'été dernier", "paris", {}, [
        ("Montre-moi les photos des vacances en Bretagne l'été dernier.", photos(
            "in Bretagne, « l'été dernier »", lambda a: "bretagne" in photo_arg(a, "place")
            and "ete dernier" in photo_arg(a, "when")), {}),
    ]),
    ("AD-112 Noël 2024", "charras", {}, [
        # Or 2024's photos most like Christmas, as gemma asked 10 times of 10
        ("Tu peux me montrer nos photos de Noël 2024 ?", photos("« Noël 2024 », or « Noël » in 2024", lambda a: "noel 2024" in photo_arg(a, "when")
                                                                or "noel" in photo_arg(a, "query") and "2024" in photo_arg(a, "when")), {}),
    ]),
    ("AD-112 Charlie à la plage", "paris", {}, [
        ("Montre-moi Charlie à la plage.", photos("of Charlie, « plage »", lambda a: "charlie" in photo_arg(a, "people")
                                                  and "plage" in photo_arg(a, "query") and "charlie" not in photo_arg(a, "query")), {}),
    ]),
    ("AD-112 the Eiffel Tower is the web's", "paris", {}, [("Montre-moi la tour Eiffel.", web_image, {})]),
    ("AD-112 a slideshow on the screen", "paris", {}, [
        ("Montre-moi des photos de la plage.", photos("of « plage » on the screen", lambda a: "plage" in photo_arg(a, "query")),
         from_satellite),
        ("Suivante.", slideshow("next"), from_satellite),
        ("Reviens à la précédente.", slideshow("previous"), from_satellite),
        ("Arrête.", slideshow("stop"), from_satellite),
    ]),
    # From the app: said shown « sur l'écran », then « que sur l'écran principal de la maison », and never cast to the
    # Google Home asked, « qui n'a pas d'écran » (AD-114)
    ("AD-114 the app's photos, then on the Google Home", "paris", {}, [
        ("Est-ce que tu saurais me trouver une photo sur laquelle il y a Gisèle, Zoé et Emily ?", not_said(
            photos("of the three, in the app", lambda a: all(n in photo_arg(a, "people") for n in ("gisele", "zoe", "emil"))
                   and not a.get("room")), r"\bles affiche sur l'ecran|ecran (?:de la maison|principal)"), {}),
        ("Et là tu les affiches sur quel écran ?", says(r"telephone|application", unlike=r"ecran principal"), {}),
        ("Affiche-les sur le Google Home de Paris.", slideshow_started, {}),
    ]),
]  # fmt: skip


async def replay(snapshot: dict, site: str, states: dict, turns: list):
    """One run of a scenario, from a new server; yields each turn with its index."""
    server = build_server(snapshot, site)
    fake = server.ha_client._session
    homes = {each.name: each.ha_client._session for each in server.sites}
    for eid, state in states.items():
        fake.set(eid, state)
        if not isinstance(state, dict):
            server.ha_client.entities[eid]["state"] = state
    for i, (text, _, changes) in enumerate(turns):
        sent, t0 = {name: len(home.calls) for name, home in homes.items()}, time.monotonic()
        server.new_request("replay")  # a browser of the web app, with its own conversation (AD-061)
        if callable(changes):
            changes(server)
        else:
            for eid, state in changes.items():
                fake.set(eid, state)
        reply = await server.process_with_llm(None, text) or ""
        await asyncio.gather(*server._pending_service_calls, *server._background_tasks, return_exceptions=True)
        wall = round(time.monotonic() - t0, 2)
        robot = copy.deepcopy(fake.robot.schedule)
        new = {name: home.calls[sent[name] :] for name, home in homes.items()}
        pressed = [name for name, calls in new.items() for domain, *_ in calls if domain == "button"]
        yield i, Turn(reply, list(server._last_tool_calls), new[site], server._spotify_here, wall, robot, pressed)


async def main():
    parser = argparse.ArgumentParser(description="Conversations replayed against fake homes, with the real LLM")
    parser.add_argument("-k", "--filter", help="Run only scenarios matching this string")
    parser.add_argument("-n", "--repeats", type=int, default=3, help="Runs of each scenario (default 3)")
    parser.add_argument("-o", "--output", help="Append each turn, as JSON lines, to this file")
    args = parser.parse_args()

    try:
        llm = get("/health")["components"]["llm"]
        snapshot = get("/api/snapshot")
    except Exception as e:
        sys.exit(f"ERROR: {SERVER} gave no model or snapshot: {e}")
    # Before voice_server reads them: the live server's model, and its homes' time, which the prompt gives
    os.environ.update(LLM_URL=llm["url"], LLM_MODEL=llm["model"], LLM_API_KEY="")
    os.environ.setdefault("TZ", snapshot["sites"][0]["home"].get("time_zone", "Europe/Paris"))
    time.tzset()
    print(f"{llm['model']} at {llm['url']}\n")

    scenarios = [s for s in SCENARIOS if not args.filter or args.filter.lower() in s[0].lower()]
    passed = total = 0
    failures = []
    with open(args.output, "a") if args.output else contextlib.nullcontext() as out:
        for name, site, states, turns in scenarios:
            passes = [0] * len(turns)
            for run in range(args.repeats):
                async for i, turn in replay(snapshot, site, states, turns):
                    text, check, _ = turns[i]
                    ok, expect = check(turn)
                    passes[i] += ok
                    found = [(c["function"], c["args"], c.get("result")) for c in turn.calls]
                    if not ok:
                        got = "; ".join(f"{fn}({a}) → {r}" for fn, a, r in found) or "no call"
                        failures.append(f"{name} · {text} — expected {expect}; got {got}, « {turn.reply[:150]} »")
                    if out:
                        out.write(json.dumps({
                            "model": llm["model"], "scenario": name, "run": run, "turn": i, "text": text, "pass": ok,
                            "expect": expect, "wall": turn.wall, "response": turn.reply, "calls": found,
                        }, ensure_ascii=False) + "\n")  # fmt: skip
                        out.flush()
            passed += sum(passes)
            total += len(turns) * args.repeats
            print(f"  {name:38} " + "  ".join(f"{p}/{args.repeats}" for p in passes), flush=True)

    print(f"\n{'=' * 50}\n{passed} of {total} turns passed")
    if failures:
        print("\nFailures:")
        for failure in failures:
            print(f"  {failure}")


if __name__ == "__main__":
    asyncio.run(main())
