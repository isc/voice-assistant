#!/usr/bin/env python3
"""Conversations replayed against fake homes, the server's own code running every call (AD-084).

test_hard_cases.py replays requests as dry runs: the server runs no call and answers each « OK », so a speaker
that does not answer, or what a read finds, cannot be played out. Here each scenario sets the devices' states,
and the server's code, this checkout's, runs every call in this process against a fake Home Assistant that
holds them: the homes' devices as the live server knows them (its /api/snapshot), the scenario's states, the
service calls kept and nothing sent. Spotify is fake too; there is no satellite and no calendar. The model is
the live server's, through the same Ollama: run it on the server.

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
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
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
    "media_stop": "idle", "alarm_disarm": "disarmed",
}  # fmt: skip
# The template HAClient.reachable reads states with
_STATES_TEMPLATE = re.compile(r"\{% for e in (\[.*?\]) %\}\{\{ e \}\} \{\{ states\(e\) \}\}")


class FakeHA:
    """A home's Home Assistant, as the aiohttp session its HAClient talks through. `states` is HA's side: a
    change there reaches the server only by a read, as a speaker woken up between two refreshes does."""

    timeout = None

    def __init__(self, entities: dict):
        self.entities = entities
        self.states = {eid: info["state"] for eid, info in entities.items()}
        self.calls: list[tuple[str, str, dict]] = []

    def _state(self, entity_id: str) -> dict:
        attributes = self.entities[entity_id]["attributes"]
        return {"entity_id": entity_id, "state": self.states[entity_id], "attributes": attributes}

    def get(self, url, **_):
        path = URL(url).path
        if path == "/api/states":
            return _Response(200, [self._state(eid) for eid in self.states])
        if (eid := path.removeprefix("/api/states/")) in self.states:
            return _Response(200, self._state(eid))
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
            eid = payload.get("entity_id")
            if self.states.get(eid) not in (None, "unavailable"):
                if service.startswith("alarm_arm_"):
                    self.states[eid] = "armed_" + service.removeprefix("alarm_arm_")
                elif service == "select_option":
                    self.states[eid] = payload["option"]
                elif service in _EFFECTS:
                    self.states[eid] = _EFFECTS[service]
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
        client._session = FakeHA(client.entities)
    server.site = server._voice_site = server.site_named(site)
    server.calendar_client = None
    server.spotify = FakeSpotify()
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

    @property
    def said(self) -> str:
        from ha_client import fold

        return fold(self.reply)

    @property
    def played(self) -> list[str]:
        return [payload["entity_id"] for _, service, payload in self.sent if service == "play_media"]


def plays_on(entity_id: str):
    return lambda t: (t.music == ("ha", entity_id), f"the music on {entity_id}")


def offers_one_that_answers(*names: str):
    """Nothing played elsewhere unasked, and a speaker that answers named."""
    return lambda t: (not t.played and any(name in t.said for name in names), "nothing played, one that answers named")


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


# (name, site, states, [(request, check, changes)]): states as the last refresh read them; changes, HA's before
# that request, the server's own copy left as it was, or a function of the server (went_on)
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
]  # fmt: skip


async def replay(snapshot: dict, site: str, states: dict, turns: list):
    """One run of a scenario, from a new server; yields each turn with its index."""
    server = build_server(snapshot, site)
    fake = server.ha_client._session
    for eid, state in states.items():
        server.ha_client.entities[eid]["state"] = fake.states[eid] = state
    for i, (text, _, changes) in enumerate(turns):
        sent, t0 = len(fake.calls), time.monotonic()
        server.new_request("replay")  # a browser of the web app, with its own conversation (AD-061)
        if callable(changes):
            changes(server)
        else:
            fake.states.update(changes)
        reply = await server.process_with_llm(None, text) or ""
        await asyncio.gather(*server._pending_service_calls, *server._background_tasks, return_exceptions=True)
        wall = round(time.monotonic() - t0, 2)
        yield i, Turn(reply, list(server._last_tool_calls), fake.calls[sent:], server._spotify_here, wall)


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
