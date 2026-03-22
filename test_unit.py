#!/usr/bin/env python3
"""Unit tests for deterministic Python logic (no server, no LLM, no HA).

Tests entity resolution, room groups, normalize, text tool call parser,
and build_response. Uses a fake entity registry loaded in-memory.

Usage:
    python test_unit.py
    python test_unit.py -v
    python test_unit.py -k normalize
"""

import asyncio
import unittest

from ha_client import HAClient, normalize
from llm import parse_text_tool_call

# ---------------------------------------------------------------------------
# Fixture: fake entity registry (mirrors a typical HA setup)
# ---------------------------------------------------------------------------

FAKE_ENTITIES = {
    "light.appliques_salon": {
        "friendly_name": "Appliques salon",
        "domain": "light",
        "state": "off",
        "attributes": {},
        "area_id": "living_room",
        "area_name": "Living Room",
    },
    "light.spots_cuisine": {
        "friendly_name": "Spots cuisine",
        "domain": "light",
        "state": "on",
        "attributes": {"brightness": 255},
        "area_id": "living_room",
        "area_name": "Living Room",
    },
    "light.plafonnier_charlie": {
        "friendly_name": "Plafonnier",
        "domain": "light",
        "state": "off",
        "attributes": {},
        "area_id": "chambre_charlie",
        "area_name": "Chambre Charlie",
    },
    "light.plafonnier_invites": {
        "friendly_name": "Plafonnier",
        "domain": "light",
        "state": "on",
        "attributes": {"brightness": 128},
        "area_id": "chambre_invites",
        "area_name": "Chambre invités",
    },
    "light.plafonnier_zoe": {
        "friendly_name": "Plafonnier",
        "domain": "light",
        "state": "off",
        "attributes": {},
        "area_id": "chambre_zoe",
        "area_name": "Chambre Zoé",
    },
    "cover.volet_charlie": {
        "friendly_name": "Volet",
        "domain": "cover",
        "state": "open",
        "attributes": {},
        "area_id": "chambre_charlie",
        "area_name": "Chambre Charlie",
    },
    "cover.volet_invites": {
        "friendly_name": "Volet",
        "domain": "cover",
        "state": "closed",
        "attributes": {},
        "area_id": "chambre_invites",
        "area_name": "Chambre invités",
    },
    "cover.volet_zoe": {
        "friendly_name": "Volet",
        "domain": "cover",
        "state": "open",
        "attributes": {},
        "area_id": "chambre_zoe",
        "area_name": "Chambre Zoé",
    },
    "cover.volet_parents": {
        "friendly_name": "Volet",
        "domain": "cover",
        "state": "open",
        "attributes": {},
        "area_id": "chambre_parents",
        "area_name": "Chambre parents",
    },
    "climate.thermostat_salon": {
        "friendly_name": "Thermostat salon",
        "domain": "climate",
        "state": "heat",
        "attributes": {"temperature": 20},
        "area_id": "living_room",
        "area_name": "Living Room",
    },
}


def make_client() -> HAClient:
    """Create an HAClient with fake entities (no HA connection)."""
    client = HAClient.__new__(HAClient)
    client.base_url = "http://fake"
    client.token = "fake"
    client.entities = dict(FAKE_ENTITIES)
    client.areas = {
        "living_room": "Living Room",
        "chambre_charlie": "Chambre Charlie",
        "chambre_invites": "Chambre invités",
        "chambre_zoe": "Chambre Zoé",
        "chambre_parents": "Chambre parents",
    }
    client._session = None
    return client


# ---------------------------------------------------------------------------
# Tests: normalize
# ---------------------------------------------------------------------------


class TestNormalize(unittest.TestCase):
    def test_lowercase_and_accents(self):
        self.assertEqual(normalize("Étagère"), "etagere")

    def test_strips_stopwords(self):
        self.assertEqual(normalize("la lumière du salon"), "lumiere salon")

    def test_apostrophe_not_stripped(self):
        # normalize splits on spaces only, so l' is kept attached
        # This is acceptable because fuzzy matching still works
        self.assertEqual(normalize("l'entrée"), "l'entree")

    def test_empty(self):
        self.assertEqual(normalize(""), "")


# ---------------------------------------------------------------------------
# Tests: resolve_entity (single entity fuzzy match)
# ---------------------------------------------------------------------------


class TestResolveEntity(unittest.TestCase):
    def setUp(self):
        self.client = make_client()

    def test_exact_match(self):
        eid = self.client.resolve_entity("Appliques salon")
        self.assertEqual(eid, "light.appliques_salon")

    def test_match_with_room(self):
        eid = self.client.resolve_entity("Plafonnier", room="Chambre invités")
        self.assertEqual(eid, "light.plafonnier_invites")

    def test_same_name_different_rooms(self):
        eid1 = self.client.resolve_entity("Plafonnier", room="Chambre Charlie")
        eid2 = self.client.resolve_entity("Plafonnier", room="Chambre invités")
        self.assertEqual(eid1, "light.plafonnier_charlie")
        self.assertEqual(eid2, "light.plafonnier_invites")
        self.assertNotEqual(eid1, eid2)

    def test_fuzzy_match(self):
        eid = self.client.resolve_entity("Spot cuisine")
        self.assertEqual(eid, "light.spots_cuisine")

    def test_no_match(self):
        eid = self.client.resolve_entity("Grille-pain")
        self.assertIsNone(eid)

    def test_fallback_without_room(self):
        # Non-existent room → falls back to global search
        eid = self.client.resolve_entity("Appliques salon", room="Garage")
        self.assertEqual(eid, "light.appliques_salon")

    def test_domain_hint_filters(self):
        eid = self.client.resolve_entity("Volet", room="Chambre Charlie", domain_hints=["cover"])
        self.assertEqual(eid, "cover.volet_charlie")


# ---------------------------------------------------------------------------
# Tests: resolve_all_entities (groups, generic names, multi-entity)
# ---------------------------------------------------------------------------


class TestResolveAllEntities(unittest.TestCase):
    def setUp(self):
        self.client = make_client()

    def test_room_group_enfants(self):
        eids = self.client.resolve_all_entities("volet", room="enfants", domain_hints=["cover"])
        self.assertIn("cover.volet_charlie", eids)
        self.assertIn("cover.volet_zoe", eids)
        self.assertNotIn("cover.volet_invites", eids)

    def test_room_group_partout(self):
        eids = self.client.resolve_all_entities("volet", room="partout", domain_hints=["cover"])
        self.assertEqual(len(eids), 4)  # all covers

    def test_generic_name_volet_in_room(self):
        eids = self.client.resolve_all_entities("volet", room="Chambre invités")
        self.assertEqual(eids, ["cover.volet_invites"])

    def test_generic_name_lumiere_in_room(self):
        eids = self.client.resolve_all_entities("lumière", room="Living Room")
        self.assertIn("light.appliques_salon", eids)
        self.assertIn("light.spots_cuisine", eids)

    def test_specific_name(self):
        eids = self.client.resolve_all_entities("Appliques salon")
        self.assertEqual(eids, ["light.appliques_salon"])

    def test_no_area_leak(self):
        """Entities without area should not appear in room-scoped queries."""
        # Add an entity without area
        self.client.entities["light.orphan"] = {
            "friendly_name": "Orphan",
            "domain": "light",
            "state": "off",
            "attributes": {},
            "area_id": None,
            "area_name": "",
        }
        eids = self.client.resolve_all_entities("lumière", room="Living Room")
        self.assertNotIn("light.orphan", eids)


# ---------------------------------------------------------------------------
# Tests: _build_response
# ---------------------------------------------------------------------------


class TestBuildResponse(unittest.TestCase):
    def setUp(self):
        self.client = make_client()

    def test_light_on(self):
        r = self.client._build_response("light", "turn_on", "Plafonnier", {})
        self.assertEqual(r, "Plafonnier allumé")

    def test_light_off(self):
        r = self.client._build_response("light", "turn_off", "Spots", {})
        self.assertEqual(r, "Spots éteint")

    def test_light_brightness(self):
        r = self.client._build_response("light", "turn_on", "Plafonnier", {"brightness": 128})
        self.assertIn("50%", r)

    def test_cover_close(self):
        r = self.client._build_response("cover", "close_cover", "Volet", {})
        self.assertIn("fermeture", r)

    def test_temperature(self):
        r = self.client._build_response("climate", "set_temperature", "Thermostat", {"temperature": 21})
        self.assertIn("21", r)

    def test_unknown_service(self):
        r = self.client._build_response("fan", "turn_on", "Ventilo", {})
        self.assertEqual(r, "C'est fait")


# ---------------------------------------------------------------------------
# Tests: parse_text_tool_call (fallback parser for local LLM)
# ---------------------------------------------------------------------------


class TestParseTextToolCall(unittest.TestCase):
    def test_json_args(self):
        fn, args = parse_text_tool_call('turn_on({"entity": "Plafonnier", "room": "Salon"})', False)
        self.assertEqual(fn, "turn_on")
        self.assertEqual(args["entity"], "Plafonnier")
        self.assertEqual(args["room"], "Salon")

    def test_quoted_string(self):
        fn, args = parse_text_tool_call('turn_off("Appliques salon")', False)
        self.assertEqual(fn, "turn_off")
        self.assertEqual(args["entity"], "Appliques salon")

    def test_python_kwargs(self):
        fn, args = parse_text_tool_call('close_cover(entity="Volet", room="Chambre")', False)
        self.assertEqual(fn, "close_cover")
        self.assertEqual(args["entity"], "Volet")
        self.assertEqual(args["room"], "Chambre")

    def test_french_verb_with_history(self):
        fn, args = parse_text_tool_call("Éteins les appliques salon.", True)
        self.assertEqual(fn, "turn_off")
        self.assertEqual(args["entity"], "appliques salon")

    def test_french_verb_without_history(self):
        fn, args = parse_text_tool_call("Éteins les appliques salon.", False)
        self.assertIsNone(fn)

    def test_plain_text_no_match(self):
        fn, args = parse_text_tool_call("Il fait beau aujourd'hui.", False)
        self.assertIsNone(fn)

    def test_unknown_function(self):
        fn, args = parse_text_tool_call('reboot({"target": "server"})', False)
        self.assertIsNone(fn)


# ---------------------------------------------------------------------------
# Tests: format_state_for_speech
# ---------------------------------------------------------------------------


class TestFormatState(unittest.TestCase):
    def setUp(self):
        self.client = make_client()

    def test_light_on_with_brightness(self):
        state = {
            "state": "on",
            "attributes": {"friendly_name": "Plafonnier", "brightness": 128},
        }
        r = self.client.format_state_for_speech("light.plafonnier_invites", state)
        self.assertIn("allumé", r)
        self.assertIn("50", r)

    def test_light_off(self):
        state = {
            "state": "off",
            "attributes": {"friendly_name": "Plafonnier"},
        }
        r = self.client.format_state_for_speech("light.plafonnier_invites", state)
        self.assertIn("éteint", r)

    def test_cover_open(self):
        state = {
            "state": "open",
            "attributes": {"friendly_name": "Volet"},
        }
        r = self.client.format_state_for_speech("cover.volet_charlie", state)
        self.assertIn("ouvert", r)


# ---------------------------------------------------------------------------
# Tests: TimerManager
# ---------------------------------------------------------------------------


class TestTimerManager(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        from timer import TimerManager

        self.manager = TimerManager()
        self.events = []

    async def _on_event(self, event_type, timer):
        self.events.append((event_type, timer.id, timer.seconds_left, timer.is_active))

    async def test_start_timer(self):
        timer = self.manager.start_timer(10, "test", self._on_event)
        self.assertEqual(timer.total_seconds, 10)
        self.assertEqual(timer.name, "test")
        self.assertTrue(timer.is_active)
        # Wait for STARTED event
        await asyncio.sleep(0.05)
        self.assertTrue(any(e[0].value == 0 for e in self.events))  # STARTED = 0
        self.manager.cancel_all()

    async def test_cancel_timer_by_name(self):
        self.manager.start_timer(60, "pâtes", self._on_event)
        await asyncio.sleep(0.05)
        cancelled = self.manager.cancel_timer(name="pâtes")
        self.assertIsNotNone(cancelled)
        self.assertEqual(cancelled.name, "pâtes")
        self.assertEqual(len(self.manager.get_timers()), 0)

    async def test_cancel_only_timer(self):
        """When there's only one timer, cancel it even if the name doesn't match."""
        self.manager.start_timer(60, None, self._on_event)
        await asyncio.sleep(0.05)
        cancelled = self.manager.cancel_timer(name="whatever")
        self.assertIsNotNone(cancelled)
        self.assertEqual(len(self.manager.get_timers()), 0)

    async def test_cancel_nonexistent(self):
        result = self.manager.cancel_timer(name="nope")
        self.assertIsNone(result)

    async def test_timer_finishes(self):
        self.manager.start_timer(1, None, self._on_event)
        # Wait for timer to finish (1s + margin)
        await asyncio.sleep(1.3)
        finished_events = [e for e in self.events if e[0].value == 3]  # FINISHED = 3
        self.assertEqual(len(finished_events), 1)
        self.assertEqual(len(self.manager.get_timers()), 0)

    async def test_get_timers(self):
        self.manager.start_timer(60, "a", self._on_event)
        self.manager.start_timer(30, "b", self._on_event)
        timers = self.manager.get_timers()
        self.assertEqual(len(timers), 2)
        # Sorted by fire_at (b fires first since it's shorter)
        self.assertEqual(timers[0].name, "b")
        self.manager.cancel_all()

    async def test_seconds_left(self):
        timer = self.manager.start_timer(60, None, self._on_event)
        await asyncio.sleep(0.05)
        self.assertGreater(timer.seconds_left, 55)
        self.assertLessEqual(timer.seconds_left, 60)
        self.manager.cancel_all()


# ---------------------------------------------------------------------------
# Tests: timer._parse_time
# ---------------------------------------------------------------------------


class TestParseTime(unittest.TestCase):
    def test_hhmm_colon(self):
        from timer import _parse_time

        self.assertEqual(_parse_time("07:00"), (7, 0))
        self.assertEqual(_parse_time("14:30"), (14, 30))

    def test_french_h_format(self):
        from timer import _parse_time

        self.assertEqual(_parse_time("7h"), (7, 0))
        self.assertEqual(_parse_time("7h30"), (7, 30))
        self.assertEqual(_parse_time("14h00"), (14, 0))

    def test_invalid(self):
        from timer import _parse_time

        with self.assertRaises(ValueError):
            _parse_time("abc")


# ---------------------------------------------------------------------------
# Tests: format_timers_for_prompt
# ---------------------------------------------------------------------------


class TestFormatTimersForPrompt(unittest.TestCase):
    def test_empty(self):
        from timer import format_timers_for_prompt

        self.assertEqual(format_timers_for_prompt([]), "")

    def test_with_timers(self):
        import time

        from timer import Timer, format_timers_for_prompt

        timers = [
            Timer(id="abc", name="pâtes", total_seconds=300, fire_at=time.time() + 192),
        ]
        result = format_timers_for_prompt(timers)
        self.assertIn("Minuteurs", result)
        self.assertIn("pâtes", result)
        self.assertIn("min", result)


# ---------------------------------------------------------------------------
# Tests: _format_duration (voice_server)
# ---------------------------------------------------------------------------


class TestFormatDuration(unittest.TestCase):
    def test_seconds(self):
        from voice_server import _format_duration

        self.assertEqual(_format_duration(30), "30 secondes")
        self.assertEqual(_format_duration(1), "1 seconde")

    def test_minutes(self):
        from voice_server import _format_duration

        self.assertEqual(_format_duration(60), "1 minute")
        self.assertEqual(_format_duration(300), "5 minutes")
        self.assertEqual(_format_duration(90), "1 minute 30s")

    def test_hours(self):
        from voice_server import _format_duration

        self.assertEqual(_format_duration(3600), "1 heure")
        self.assertEqual(_format_duration(7200), "2 heures")
        self.assertEqual(_format_duration(5400), "1h30")


if __name__ == "__main__":
    unittest.main()
