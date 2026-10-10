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
from unittest.mock import patch

from ha_client import HAClient, cover_kind, normalize
from images import is_about
from llm import claims_completed_action, get_tool_definitions, parse_text_tool_call

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
    client.entity_areas = {}
    client.areas = {
        "living_room": "Living Room",
        "chambre_charlie": "Chambre Charlie",
        "chambre_invites": "Chambre invités",
        "chambre_zoe": "Chambre Zoé",
        "chambre_parents": "Chambre parents",
    }
    client.screens = []
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
    # Room groups come from the gitignored config.local.json: the tests bring their own,
    # over the fake areas, so that they pass on any machine. The whole house needs none.
    ROOM_GROUPS = {"enfants": ["Chambre Charlie", "Chambre Zoé"]}

    def setUp(self):
        self.client = make_client()
        groups = patch.dict("ha_client.ROOM_GROUPS", self.ROOM_GROUPS, clear=True)
        groups.start()
        self.addCleanup(groups.stop)

    def test_room_group_enfants(self):
        eids = self.client.resolve_all_entities("volet", room="enfants", domain_hints=["cover"])
        self.assertIn("cover.volet_charlie", eids)
        self.assertIn("cover.volet_zoe", eids)
        self.assertNotIn("cover.volet_invites", eids)

    def test_whole_house_without_config(self):
        for room in ("partout", "tout", "toute la maison"):
            eids = self.client.resolve_all_entities("volet", room=room, domain_hints=["cover"])
            self.assertEqual(len(eids), 4, room)  # all covers

    def test_whole_house_and_devices_without_a_room(self):
        """Every light, placed or not; but "tout" leaves alone a speaker's settings switches (Paris Sonos)."""
        for eid, domain in (("light.couloir", "light"), ("switch.sonos_loudness", "switch")):
            self.client.entities[eid] = {
                "friendly_name": eid,
                "domain": domain,
                "state": "on",
                "attributes": {},
                "area_id": None,
                "area_name": "",
            }
        hints = ["light", "switch", "media_player"]
        eids = self.client.resolve_all_entities("lumières", room="partout", domain_hints=hints)
        self.assertIn("light.couloir", eids)
        self.assertIn("light.spots_cuisine", eids)
        eids = self.client.resolve_all_entities("tout", room="partout", domain_hints=hints)
        self.assertIn("light.spots_cuisine", eids)
        self.assertNotIn("switch.sonos_loudness", eids)
        self.assertNotIn("light.couloir", eids)

    def test_generic_name_keeps_to_its_domain(self):
        """turn_off hints light, switch and media_player; "les lumières" means the lights only."""
        self.client.entities["switch.veilleuse_charlie"] = {
            "friendly_name": "Veilleuse",
            "domain": "switch",
            "state": "on",
            "attributes": {},
            "area_id": "chambre_charlie",
            "area_name": "Chambre Charlie",
        }
        hints = ["light", "switch", "media_player"]
        eids = self.client.resolve_all_entities("lumières", room="enfants", domain_hints=hints)
        self.assertEqual(sorted(eids), ["light.plafonnier_charlie", "light.plafonnier_zoe"])
        eids = self.client.resolve_all_entities("tout", room="enfants", domain_hints=hints)
        self.assertIn("switch.veilleuse_charlie", eids)

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

    def test_no_area_leak_by_name(self):
        """A speaker with no area must not answer for another speaker's name or room."""
        for eid, name, area in (("media_player.google_home", "Google Home", "Living Room"),
                                ("media_player.sonos", "Sonos Paris", "")):  # fmt: skip
            self.client.entities[eid] = {
                "friendly_name": name, "domain": "media_player", "state": "idle",
                "attributes": {}, "area_id": None, "area_name": area,
            }  # fmt: skip
        for room in ("Google Home", "Living Room"):
            eids = self.client.resolve_all_entities(room, room=room, domain_hints=["media_player"])
            self.assertEqual(eids, ["media_player.google_home"], room)
        eids = self.client.resolve_all_entities("Sonos", room="Sonos", domain_hints=["media_player"])
        self.assertEqual(eids, ["media_player.sonos"])


# ---------------------------------------------------------------------------
# Tests: cover sub-kinds (roller shutters vs roof windows, both `cover`)
# ---------------------------------------------------------------------------


class TestCoverKinds(unittest.TestCase):
    def setUp(self):
        self.client = make_client()
        # A room with both a roof window and its shutter, as Velux exposes them:
        # device_class set on the window, absent on the shutter.
        self.client.entities["cover.velux_window_cuisine"] = {
            "friendly_name": "Fenêtre Cuisine",
            "domain": "cover",
            "state": "closed",
            "attributes": {"device_class": "window"},
            "area_id": "cuisine",
            "area_name": "Cuisine",
        }
        self.client.entities["cover.velux_shutter_cuisine"] = {
            "friendly_name": "Volet Cuisine",
            "domain": "cover",
            "state": "closed",
            "attributes": {},
            "area_id": "cuisine",
            "area_name": "Cuisine",
        }
        self.client.areas["cuisine"] = "Cuisine"

    def test_device_class_wins_over_name(self):
        """An unhelpfully named cover is still classified by its device_class."""
        self.assertEqual(
            cover_kind({"friendly_name": "Toiture 3", "attributes": {"device_class": "window"}}),
            "fenetre",
        )
        self.assertEqual(
            cover_kind({"friendly_name": "Toiture 3", "attributes": {"device_class": "shutter"}}),
            "volet",
        )

    def test_name_fallback_when_no_device_class(self):
        self.assertEqual(cover_kind({"friendly_name": "Volet Cuisine", "attributes": {}}), "volet")
        self.assertEqual(cover_kind({"friendly_name": "Fenêtre Cuisine", "attributes": {}}), "fenetre")
        self.assertIsNone(cover_kind({"friendly_name": "Portail", "attributes": {}}))

    def test_volet_does_not_match_window(self):
        eids = self.client.resolve_all_entities("volets", room="Cuisine")
        self.assertEqual(eids, ["cover.velux_shutter_cuisine"])

    def test_fenetre_does_not_match_shutter(self):
        eids = self.client.resolve_all_entities("fenêtres", room="Cuisine")
        self.assertEqual(eids, ["cover.velux_window_cuisine"])

    def test_resolve_entity_scoped_by_kind(self):
        eid = self.client.resolve_entity("volet", room="Cuisine", domain_hints=["cover"])
        self.assertEqual(eid, "cover.velux_shutter_cuisine")

    def test_undistinguished_house_keeps_all_covers(self):
        """A house whose covers carry no kind at all must behave as before."""
        eids = self.client.resolve_all_entities("volet", room="Chambre invités")
        self.assertEqual(eids, ["cover.volet_invites"])

    def test_prompt_lists_windows_apart(self):
        prompt = self.client.get_entity_list_for_prompt()
        self.assertIn("fenêtres: Fenêtre Cuisine", prompt)
        self.assertIn("volets: Volet Cuisine", prompt)


# ---------------------------------------------------------------------------
# Tests: detecting an action claimed without any function call
# ---------------------------------------------------------------------------


class TestActionClaimDetection(unittest.TestCase):
    def test_claims(self):
        for text in [
            "C'est fait, j'ai ouvert toutes les fenêtres à l'exception de celle de Charlie.",
            "C'est entendu, je ferme tous les volets sauf celui de la chambre de Charlie.",
            "Voilà, la lumière du salon est allumée.",
            "J'ai bien réglé le thermostat à 20 degrés.",
            "J'ai éteint les lumières.",
            "J'ai éteint les lumières. Autre chose ?",
        ]:
            self.assertTrue(claims_completed_action(text), text)

    def test_does_not_flag_state_reports(self):
        """Reading a state out loud is not claiming to have changed it."""
        for text in [
            "Oui, le volet de la salle de jeux est fermé.",
            "Non, les volets de la cuisine sont actuellement fermés.",
            "L'eau de la piscine est à 29,8 degrés.",
            "Il est actuellement 20h32.",
            "Lundi 27 juillet, tu as la livraison du piano.",
            "Je n'ai pas accès à cette information.",
        ]:
            self.assertFalse(claims_completed_action(text), text)

    def test_does_not_flag_questions(self):
        """Offering an action is not claiming it (2026-10-10: the challenge made it play music)."""
        for text in [
            "Je ne suis pas sûr de comprendre ce que vous voulez dire par cette dernière phrase. "
            "Souhaitez-vous que je lance de la musique ou que j'affiche une image d'une ville ?",
            "Voulez-vous que je ferme les volets du salon?",
        ]:
            self.assertFalse(claims_completed_action(text), text)

    def test_empty(self):
        self.assertFalse(claims_completed_action(""))
        self.assertFalse(claims_completed_action(None))


# ---------------------------------------------------------------------------
# Tests: a named room must never leak into another room
# ---------------------------------------------------------------------------


class TestNoCrossRoomFallback(unittest.TestCase):
    def setUp(self):
        self.client = make_client()
        # Mirror Charras: Zoé's room has no cover at all, Charlie's next door does.
        del self.client.entities["cover.volet_zoe"]

    def test_existing_room_without_the_device_resolves_to_nothing(self):
        eid = self.client.resolve_entity("Volet Chambre de Zoé", room="Chambre Zoé", domain_hints=["cover"])
        self.assertIsNone(eid)

    def test_does_not_actuate_a_neighbouring_room(self):
        eids = self.client.resolve_all_entities("volet", room="Chambre Zoé", domain_hints=["cover"])
        self.assertEqual(eids, [])
        self.assertNotIn("cover.volet_charlie", eids)

    def test_unknown_room_still_falls_back(self):
        """The rescue path stays: HA has no 'Garage' area, so the filter is dropped."""
        eid = self.client.resolve_entity("Appliques salon", room="Garage")
        self.assertEqual(eid, "light.appliques_salon")


# ---------------------------------------------------------------------------
# Tests: sensor search (read-only, kept out of the system prompt)
# ---------------------------------------------------------------------------


def sensor_state(entity_id, name, state, unit=None):
    attrs = {"friendly_name": name}
    if unit:
        attrs["unit_of_measurement"] = unit
    return {"entity_id": entity_id, "state": state, "attributes": attrs}


FAKE_SENSOR_STATES = [
    sensor_state("sensor.piscine_water_temperature", "Piscine Water Temperature", "29.8", "°C"),
    sensor_state("sensor.piscine_air_temperature", "Piscine Air Temperature", "23", "°C"),
    sensor_state("sensor.piscine_ph_level", "Piscine pH Level", "7.2"),
    sensor_state("sensor.piscine_pump_state", "Piscine Pump State", "unknown"),
    sensor_state("sensor.piscine_cover_state", "Piscine Pool Cover State", "unavailable"),
    sensor_state("sensor.chaudiere_temperature", "Chaudière température", "51.4", "°C"),
    # Not a sensor: must never surface, whatever it is called.
    {
        "entity_id": "light.piscine",
        "state": "on",
        "attributes": {"friendly_name": "Lumière piscine"},
    },
]


class TestSensorSearch(unittest.TestCase):
    def setUp(self):
        self.client = make_client()

    def search(self, query, limit=15):
        return self.client.match_sensors(FAKE_SENSOR_STATES, query, limit)

    def test_finds_by_keyword(self):
        out = self.search("piscine")
        self.assertIn("Piscine Water Temperature: 29.8 °C", out)
        self.assertIn("Piscine pH Level: 7.2", out)

    def test_skips_unknown_and_unavailable(self):
        out = self.search("piscine")
        self.assertNotIn("Pump State", out)
        self.assertNotIn("Pool Cover State", out)

    def test_ignores_non_sensor_domains(self):
        self.assertNotIn("Lumière piscine", self.search("piscine"))

    def test_more_specific_query_ranks_first(self):
        """Both words matching beats one — 'piscine ph' must not lead with a temperature."""
        self.assertTrue(self.search("piscine ph").startswith("Piscine pH Level"))

    def test_accent_and_case_insensitive(self):
        self.assertIn("Chaudière température: 51.4 °C", self.search("CHAUDIERE"))

    def test_no_match(self):
        self.assertIn("Aucun capteur trouvé", self.search("tracteur"))

    def test_truncates_and_says_so(self):
        out = self.search("piscine", limit=2)
        self.assertEqual(len(out.split("\n")), 3)
        self.assertIn("1 autres capteurs non affichés", out)


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
# Tests: pictures for the screen
# ---------------------------------------------------------------------------


class TestIsAbout(unittest.TestCase):
    """Top Wikipedia hits measured on 2026-10-10 for spoken requests."""

    def test_article_about_the_query(self):
        self.assertTrue(is_about("tour Eiffel", "Tour Eiffel"))
        self.assertTrue(is_about("la Tour Eiffel", "Tour Eiffel"))
        self.assertTrue(is_about("Zinedine Zidane", "Zinédine Zidane"))

    def test_title_may_say_more(self):
        self.assertTrue(is_about("Arc de Triomphe", "Arc de triomphe de l'Étoile"))
        self.assertTrue(is_about("chat roux", "Roux (robe de chat)"))

    def test_article_merely_related(self):
        self.assertFalse(is_about("coucher de soleil sur la mer", "Le Coucher du soleil"))
        self.assertFalse(is_about("Daft Punk", "Discovery (album)"))

    def test_empty_query(self):
        self.assertFalse(is_about("", "Tour Eiffel"))


class TestScreenTool(unittest.TestCase):
    def test_offered_only_with_a_screen(self):
        client = make_client()
        names = {t["name"] for t in get_tool_definitions(client)}
        self.assertNotIn("show_image", names)
        client.screens = ["media_player.google_home"]
        names = {t["name"] for t in get_tool_definitions(client)}
        self.assertIn("show_image", names)


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

    def test_volume_phrase_with_history(self):
        self.assertEqual(parse_text_tool_call("Plus fort !", True), ("change_volume", {"direction": "up"}))
        self.assertEqual(parse_text_tool_call("baisse le son", True), ("change_volume", {"direction": "down"}))

    def test_reply_mentioning_volume_is_not_a_command(self):
        fn, _ = parse_text_tool_call("Je ne peux pas parler plus fort, désolé.", True)
        self.assertIsNone(fn)


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


class TestResolveStation(unittest.TestCase):
    def test_names_and_aliases(self):
        from voice_server import _resolve_station

        self.assertEqual(_resolve_station("France Inter")[0], "France Inter")
        self.assertEqual(_resolve_station("Rire & Chansons")[0], "Rire et Chansons")
        self.assertEqual(_resolve_station("la radio RTL")[0], "RTL")
        self.assertEqual(_resolve_station("France Bleu")[0], "ici Paris")
        self.assertEqual(_resolve_station("france inter sur FM")[0], "France Inter")

    def test_unknown_or_ambiguous(self):
        from voice_server import _resolve_station

        for name in ("", "la radio", "musique classique", "Radio Nova"):
            self.assertIsNone(_resolve_station(name), name)


class TestSpeakerLeftOn(unittest.TestCase):
    STREAM = "http://192.168.1.189:8889/spotify/stream.mp3"
    FIP = "http://icecast.radiofrance.fr/fip-midfi.mp3"

    @staticmethod
    def player(state, content_id):
        return {"state": state, "attributes": {"media_content_id": content_id}}

    def test_finds_the_speaker_left_on_a_known_url(self):
        from voice_server import _speaker_left_on

        entities = {
            "media_player.google_home": self.player("paused", "https://thumb.wikimedia.org/okapi.jpg"),
            "media_player.sonos_paris": self.player("paused", self.STREAM),
            "light.salon": {"state": "on", "attributes": {}},
        }
        found = _speaker_left_on(entities, {self.STREAM, self.FIP})
        self.assertEqual(found, ("media_player.sonos_paris", self.STREAM))

    def test_none_when_stopped_or_elsewhere(self):
        from voice_server import _speaker_left_on

        entities = {
            "media_player.a": self.player("idle", self.STREAM),
            "media_player.b": self.player("playing", "http://example.com/podcast.mp3"),
        }
        self.assertIsNone(_speaker_left_on(entities, {self.STREAM, self.FIP}))


class TestMatchPlaylist(unittest.TestCase):
    PLAYLISTS = [{"name": n, "uri": n} for n in ("Matin tranquille ☀️", "Soirée", "Endormissement Zoé et Charlie")]

    def test_spoken_names(self):
        from spotify import match_playlist

        self.assertEqual(match_playlist(self.PLAYLISTS, "ma playlist du matin")["name"], "Matin tranquille ☀️")
        self.assertEqual(match_playlist(self.PLAYLISTS, "soirée")["name"], "Soirée")
        self.assertEqual(match_playlist(self.PLAYLISTS, "endormissement")["name"], "Endormissement Zoé et Charlie")

    def test_no_match(self):
        from spotify import match_playlist

        for query in ("", "ma playlist", "jazz"):
            self.assertIsNone(match_playlist(self.PLAYLISTS, query), query)


class TestSpotifyStreamRead(unittest.TestCase):
    """The FIFO is read in whole stereo frames: a split frame would garble the rest of the stream."""

    def test_whole_frames_only(self):
        import os
        import tempfile

        from spotify import SpotifyStream

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "pcm")
            os.mkfifo(path)
            stream = SpotifyStream(path)
            self.assertEqual(stream._read(), b"")  # no writer yet
            writer = os.open(path, os.O_WRONLY)
            try:
                os.write(writer, bytes(range(10)))
                self.assertEqual(stream._read(), bytes(range(8)))
                os.write(writer, bytes(range(10, 12)))
                self.assertEqual(stream._read(), bytes(range(8, 12)))
                self.assertEqual(stream._read(), b"")  # writer attached, nothing to read
            finally:
                os.close(writer)
                os.close(stream._fd)


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


class TestSpellTimes(unittest.TestCase):
    def test_clock_times(self):
        from tts import spell_times

        self.assertEqual(spell_times("Il est actuellement 17h34."), "Il est actuellement 17 heures 34.")
        self.assertEqual(spell_times("Réveil à 7h, pas 07:05"), "Réveil à 7 heures, pas 7 heures 5")
        self.assertEqual(spell_times("Rendez-vous à 12h00"), "Rendez-vous à 12 heures")

    def test_feminine_and_singular(self):
        from tts import spell_times

        self.assertEqual(spell_times("1h21"), "une heure vingt et une")
        self.assertEqual(spell_times("21h"), "vingt et une heures")
        self.assertEqual(spell_times("0h30"), "0 heure 30")

    def test_leaves_other_text_alone(self):
        from tts import spell_times

        for text in ("Score 3 : 2", "4h2o", "25h", "17:34:12", "2026"):
            self.assertEqual(spell_times(text), text)


# ---------------------------------------------------------------------------
# Tests: calendar_client helpers
# ---------------------------------------------------------------------------


class TestCalendarFormatEvent(unittest.TestCase):
    def test_timed_event(self):
        from calendar_client import _format_event

        item = {
            "summary": "Dentiste",
            "start": {"dateTime": "2026-03-25T14:30:00+01:00"},
            "end": {"dateTime": "2026-03-25T15:30:00+01:00"},
        }
        result = _format_event(item)
        self.assertEqual(result["title"], "Dentiste")
        self.assertEqual(result["start"], "14:30")
        self.assertEqual(result["end"], "15:30")
        self.assertEqual(result["date"], "2026-03-25")

    def test_all_day_event(self):
        from calendar_client import _format_event

        item = {
            "summary": "Vacances",
            "start": {"date": "2026-04-01"},
            "end": {"date": "2026-04-08"},
        }
        result = _format_event(item)
        self.assertEqual(result["title"], "Vacances")
        self.assertEqual(result["start"], "toute la journée")
        self.assertEqual(result["date"], "2026-04-01")
        self.assertNotIn("end", result)

    def test_no_title(self):
        from calendar_client import _format_event

        item = {
            "start": {"dateTime": "2026-03-25T10:00:00+01:00"},
            "end": {"dateTime": "2026-03-25T11:00:00+01:00"},
        }
        result = _format_event(item)
        self.assertEqual(result["title"], "(sans titre)")


class TestCalendarFormatDate(unittest.TestCase):
    def test_format_date_fr(self):
        from datetime import datetime
        from zoneinfo import ZoneInfo

        from calendar_client import _format_date_fr

        TZ = ZoneInfo("Europe/Paris")

        dt = datetime(2026, 3, 25, tzinfo=TZ)  # Wednesday
        self.assertEqual(_format_date_fr(dt), "mercredi 25 mars 2026")

    def test_format_datetime_fr(self):
        from datetime import datetime
        from zoneinfo import ZoneInfo

        from calendar_client import _format_datetime_fr

        TZ = ZoneInfo("Europe/Paris")

        dt = datetime(2026, 3, 25, 14, 30, tzinfo=TZ)
        self.assertEqual(_format_datetime_fr(dt), "mercredi 25 mars 2026 à 14:30")


class TestCalendarQueryValidation(unittest.IsolatedAsyncioTestCase):
    async def test_query_without_service(self):
        import json

        from calendar_client import CalendarClient

        client = CalendarClient.__new__(CalendarClient)
        client._service = None
        result = await client.query_events("not-a-date")
        data = json.loads(result)
        self.assertEqual(data["error"], "Agenda non disponible")

    async def test_create_without_service(self):
        import json

        from calendar_client import CalendarClient

        client = CalendarClient.__new__(CalendarClient)
        client._service = None
        result = await client.create_event("Test", "bad-datetime")
        data = json.loads(result)
        self.assertEqual(data["error"], "Agenda non disponible")

    async def test_service_unavailable_valid_date(self):
        import json

        from calendar_client import CalendarClient

        client = CalendarClient.__new__(CalendarClient)
        client._service = None
        result = await client.query_events("2026-03-25")
        data = json.loads(result)
        self.assertEqual(data["error"], "Agenda non disponible")


if __name__ == "__main__":
    unittest.main()
