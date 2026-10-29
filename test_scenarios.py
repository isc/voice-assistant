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

    async def wait_for_join(self, joins_before: int, timeout: float) -> bool:
        self.joins += 1
        self.listeners.add(self.joins)
        return True


class FakeSpotify:
    """go-librespot and the search: whatever is asked is found, and plays."""

    def __init__(self):
        self.stream, self.found, self.track = FakeStream(), None, None

    async def find(self, query: str, kind):
        from spotify import Music

        self.found = f"« {query} »"
        return Music(f"spotify:{kind}:replay", self.found, {})

    async def play(self, uri: str, shuffle: bool = False):
        self.track = self.found

    async def playing(self):
        return self.track

    async def now_playing(self):
        return f"En cours sur Spotify : {self.track}" if self.track else "Rien en cours sur Spotify"

    async def control(self, action: str) -> str:
        from spotify import CONTROLS

        return CONTROLS[action][1]

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


def none_said_off(t):
    # A Google Home playing nothing says « off »: the reply took it for one that cannot play (2026-10-26)
    return not t.played and "eteint" not in t.said, "nothing played, no speaker said off"


# The speakers of Charras that answered on 2026-10-26, as a reply may name them
CHARRAS_ANSWERING = ("salon", "living", "nest", "hub", "display", "dining", "salle a manger")

CHARRAS_SPEAKERS = {
    "media_player.sonos_move": "unavailable",  # Terrasse
    "media_player.sonos_move_2": "unavailable",  # Cuisine
    "media_player.living_room_speaker": "off",  # Salon, a Google Home playing nothing
    "media_player.dining_room_display": "idle",  # the Nest Hub Max, in no room
    "media_player.home_mini_zoe": "unavailable",
    "media_player.home_mini_charlie": "off",
}
SONOS_BACK = {"media_player.sonos_move_2": "idle"}

# (name, site, states, [(request, check, changes)]): states as the last refresh read them; changes, HA's before
# that request, the server's own copy left as it was
SCENARIOS = [
    ("AD-084 the Sonos back online", "charras", CHARRAS_SPEAKERS, [
        ("Est-ce que tu peux me jouer le prélude numéro deux de Bach sur la sonos ?",
         offers_one_that_answers(*CHARRAS_ANSWERING), {}),
        ("Y'a pas une sonos qui est joignable ?", none_said_off, {}),
        # HA names the Nest Hub Max « Dining room Display »: a question is fair, another speaker is not
        ("Bah mets alors sur le Nest Hub Max.",
         plays_on_or_asks("media_player.dining_room_display", "dining", "display"), {}),
        ("Tu peux basculer le son sur la sonos, normalement elle est connectée maintenant.",
         plays_on_or_asks("media_player.sonos_move_2", "cuisine", "move 2"), SONOS_BACK),
        ("Oui, vas-y.", plays_on("media_player.sonos_move_2"), {}),
    ]),
    # The music on the screen, by its name in HA, whatever the model makes of « Nest Hub Max »
    ("AD-084 c'est parfait ça", "charras", {**CHARRAS_SPEAKERS, **SONOS_BACK}, [
        ("Mets le prélude numéro deux de Bach sur le Dining room Display.",
         plays_on("media_player.dining_room_display"), {}),
        ("Y a quoi comme enceinte qui est connectée là ?", none_said_off, {}),
        ("Eh ben la sonos de la cuisine, c'est parfait ça.",
         plays_on_or_asks("media_player.sonos_move_2", "cuisine", "move 2"), {}),
    ]),
]  # fmt: skip


async def replay(snapshot: dict, site: str, states: dict, turns: list):
    """One run of a scenario, from a new server; yields each turn with its index."""
    server = build_server(snapshot, site)
    fake = server.ha_client._session
    for eid, state in states.items():
        server.ha_client.entities[eid]["state"] = fake.states[eid] = state
    for i, (text, _, changes) in enumerate(turns):
        fake.states.update(changes)
        sent, t0 = len(fake.calls), time.monotonic()
        server.new_request("replay")  # a browser of the web app, with its own conversation (AD-061)
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
