#!/usr/bin/env python3
"""The requests DECISIONS.md records the LLM getting wrong, replayed as dry runs (AD-081).

Like test_e2e.py, through /api/dry-run: the LLM's calls are judged, none is run, each gets « OK »
(AD-047). Every case runs several times, from a fresh conversation, at the home it happened in, and
counts passes: these are the requests a model gets right some of the time. A reply is judged only
where « OK » cannot mislead it.

Usage:
    python test_hard_cases.py                 # every case, 3 times
    python test_hard_cases.py -k AD-071 -n 5  # cases matching "AD-071", 5 times
    VOICE_SERVER=http://localhost:8889 python test_hard_cases.py -o runs.jsonl   # another server, each turn kept
"""

import argparse
import contextlib
import json
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.request

from test_e2e import SERVER, post, tool_names


def norm(s) -> str:
    """Lower case, accents stripped: ha_client.fold, which this stdlib-only script does not import."""
    s = unicodedata.normalize("NFD", str(s or "").lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def calls(d: dict, *functions: str) -> list[dict]:
    return [tc for tc in d.get("tool_calls", []) if tc["function"] in functions]


def arg(tc: dict, key: str) -> str:
    return norm(tc.get("args", {}).get(key, ""))


def reply(d: dict) -> str:
    return d.get("response") or ""


# Calls that act, or speak, or make something; reads are left out
ACTIONS = {
    "turn_on", "turn_off", "open_cover", "close_cover", "set_cover_position", "set_heating", "lock", "unlock",
    "play_radio", "play_music", "move_media", "music_control", "change_volume", "set_volume", "stop_media",
    "set_timer", "set_alarm", "cancel_timer", "create_event", "update_event", "show_image",
}  # fmt: skip
ENGLISH = re.compile(r"\b(the|you|i'm|i am|sorry|what|are|doing|have|nothing)\b", re.I)


def acts(d: dict) -> bool:
    return bool(set(tool_names(d)) & ACTIONS)


def at_charras(tc: dict) -> bool:
    return any("charras" in arg(tc, key) for key in ("home", "entity", "query"))


# A check takes the dry run's answer and says whether it passed, and what it expected


def has(function: str, cond=lambda tc: True, expect: str = ""):
    return lambda d: (any(cond(tc) for tc in calls(d, function)), expect or function)


def in_room(function: str, word: str):
    return has(function, lambda tc: word in arg(tc, "room"), f"{function} {word}")


def instead(want: str, never: str):
    return lambda d: (want in tool_names(d) and never not in tool_names(d), f"{want}, no {never}")


heating_auto_at_charras = has(
    "set_heating", lambda tc: at_charras(tc) and arg(tc, "mode") == "auto", "set_heating Auto at Charras"
)


def reads_charras(d):
    return any(at_charras(tc) for tc in calls(d, "get_sensor", "get_state")), "a read at Charras"


def small_talk(d):
    return not acts(d) and bool(reply(d)) and len(ENGLISH.findall(reply(d))) < 2, "no action, a French reply"


def goodbye(d):
    return "end_conversation" in tool_names(d) and bool(reply(d).strip()), "end_conversation, and words"


def never_heats_charras(d):
    return not any(at_charras(tc) for tc in calls(d, "set_heating")), "set_heating never at Charras"


def garbled_shutters(d):
    # A room the request did not say: the whole house, which the dry run answers as the server does (AD-058,
    # « which room? »), or one made up, which it answers « OK »
    no_room_made_up = all(not arg(tc, "room") or tc.get("result") != "OK" for tc in calls(d, "close_cover"))
    return no_room_made_up and not calls(d, "open_cover") and "?" in reply(d), "no room made up, a question"


def zoe_closed_or_asked(d):
    closed = any("zoe" in arg(tc, "room") for tc in calls(d, "close_cover"))
    asked = "?" in reply(d) and not calls(d, "close_cover")
    return not calls(d, "open_cover") and (closed or asked), "close Zoé's shutters or ask, never open"


def france_inter_or_asked(d):
    played = any("inter" in arg(tc, "station") for tc in calls(d, "play_radio"))
    return played or (not acts(d) and "?" in reply(d)), "play_radio France Inter, or a question"


# (name, site, [(request, check), …]): the turns of one conversation
CASES = [
    ("AD-058 garbled shutters", "paris", [("Tu peux fermer les volets là, je suis un désolé.", garbled_shutters)]),
    ("AD-062 Femme les volets", "paris", [
        ("Femme les volets dans la chambre de Zoé. Après, je n'en fais pas un puissance 4 ?", zoe_closed_or_asked),
    ]),
    ("AD-063 move to the Sonos", "paris", [
        ("Mets France Inter sur la Google Home", has("play_radio")),
        ("Passe le son sur la Sonos", in_room("move_media", "sonos")),
    ]),
    ("AD-041 close the door again", "paris", [
        ("Verrouille la porte d'entrée", has("lock")),
        ("Tu peux fermer la porte ?", has("lock")),
    ]),
    ("AD-041 unlock", "paris", [
        ("déverrouille la porte", instead("unlock", "lock")),
    ]),
    ("AD-046 a shop's hours", "paris", [
        ("Est-ce que le magasin Altermundi de la rue du Commerce il est ouvert le dimanche ?", has("find_place")),
    ]),
    ("AD-071 maison Achara", "paris", [
        ("Est-ce que tu peux activer le chauffage dans la maison Achara ?", heating_auto_at_charras),
        ("Combien il fait dans la maison là-bas ?", reads_charras),
        ("Tu parles de Paris ou de la maison en Charente ?",
         lambda d: (not acts(d) and "charras" in norm(reply(d)), "no action, says Charras")),
    ]),
    ("AD-071 heating over there", "paris", [
        ("Il fait combien à Charras dans la maison ?", reads_charras),
        ("Allume le chauffage là-bas", heating_auto_at_charras),
    ]),
    ("AD-077 heating at Paris", "paris", [("allume le chauffage", never_heats_charras)]),
    ("AD-077 heating off at Paris", "paris", [("coupe le chauffage", never_heats_charras)]),
    ("AD-072 add the place", "paris", [
        ("Ajoute à l'agenda pour demain treize heures déjeuner avec Clémence", has("create_event")),
        ("Tu peux rajouter sur ce rendez-vous qu'il a lieu au bistrot qui s'appelle le Récepteur",
         instead("update_event", "create_event")),
    ]),
    ("AD-072 move the dentist", "paris", [
        ("Ajoute un rendez-vous chez le dentiste demain à 10h", has("create_event")),
        ("en fait c'est à 11h", instead("update_event", "create_event")),
    ]),
    ("AD-075 the setpoint, three turns", "paris", [
        ("Il fait quelle température dans la maison à Chara là ?", reads_charras),
        ("Ben normalement il est déjà allumé hein", reads_charras),
        ("Ah bah oui, mais avec quelle température on consigne ?", reads_charras),
    ]),
    ("AD-075 setpoint asked", "paris", [("C'est quoi la consigne du chauffage à Charras ?", reads_charras)]),
    ("AD-076 Charras's door", "paris", [
        ("La porte d'entrée de Charras est fermée ?", has("get_state", at_charras, "get_state at Charras")),
        ("Et le chauffage là-bas ?", reads_charras),
    ]),
    ("AD-077 kitchen window", "paris", [("La fenêtre de la cuisine est fermée ?", has("get_state"))]),
    ("AD-077 games room shutter", "paris", [
        ("Le volet de la salle de jeux est fermé ?",
         has("get_state", lambda tc: "jeux" in arg(tc, "room") or at_charras(tc), "get_state salle de jeux")),
    ]),
    ("AD-078 open it again", "charras", [
        ("Ferme le volet des combles", has("close_cover")),
        ("Tu peux le rouvrir", has("open_cover")),
        ("il continue de se fermer, et il ne se rouvre pas", has("open_cover")),
    ]),
    ("AD-080 Merci, bonne journée", "charras", [("Merci, bonne journée.", goodbye)]),
    ("AD-080 Bonne nuit", "charras", [("Bonne nuit.", goodbye)]),
    ("AD-080 Merci", "charras", [("Merci.", goodbye)]),
    ("AD-068 Charras's door from Paris", "paris", [
        ("Ouvre la porte de Charras", has("unlock", at_charras, "unlock at Charras")),
    ]),
    ("AD-076 weather, not a sensor", "paris", [("Il fait combien dehors ?", has("get_weather"))]),
    ("AD-078 small talk", "charras", [("Tu vas bien ?", small_talk)]),
    ("AD-074 sous-douce", "paris", [("France Inter sur la sous-douce", france_inter_or_asked)]),
    ("AD-069 an okapi", "charras", [
        ("montre-moi un okapi", has("show_image", lambda tc: not arg(tc, "room"), "show_image, no room")),
    ]),
    # Needs Spotify: play_music is offered only when go-librespot answers
    ("AD-057 Bach", "paris", [
        ("Mets le prélude numéro 2 en do mineur de Bach",
         has("play_music", lambda tc: arg(tc, "kind") == "track", "play_music track")),
    ]),
    ("AD-049 English", "paris", [("What are you doing?", small_talk)]),
    ("AD-059 the time", "paris", [("Quelle heure est-il ?", small_talk)]),
    ("AD-061 and the kitchen too", "charras", [
        ("éteins le salon", in_room("turn_off", "salon")),
        ("et dans la cuisine aussi ?", in_room("turn_off", "cuisine")),
    ]),
    ("AD-058 close the door", "paris", [("Ferme la porte", instead("lock", "close_cover"))]),
    ("AD-041 is the door locked", "paris", [("est-ce que la porte est verrouillée ?", has("get_state"))]),
    ("AD-073 the radio", "paris", [("Tu peux mettre la radio France Inter", has("play_radio"))]),
    ("AD-015 children's lights", "charras", [("éteins les lumières des enfants", in_room("turn_off", "enfant"))]),
    ("AD-062 pasta timer", "paris", [
        ("Lance un minuteur de trois minutes pour les pâtes",
         has("set_timer", lambda tc: tc["args"].get("duration_minutes") == 3 and tc["args"].get("label"),
             "set_timer 3, with a label")),
    ]),
]  # fmt: skip


def main():
    parser = argparse.ArgumentParser(description="Hard cases from DECISIONS.md, as dry runs")
    parser.add_argument("-k", "--filter", help="Run only cases matching this string")
    parser.add_argument("-n", "--repeats", type=int, default=3, help="Runs of each case (default 3)")
    parser.add_argument("-o", "--output", help="Append each turn, as JSON lines, to this file")
    args = parser.parse_args()

    try:
        health = urllib.request.urlopen(f"{SERVER}/health", timeout=5)
    except urllib.error.HTTPError as e:
        health = e  # 503, a part not ready: it still says which model answers
    except OSError:
        sys.exit(f"ERROR: Server not reachable at {SERVER}")
    model = json.loads(health.read())["components"]["llm"]["model"]
    print(f"{model} at {SERVER}\n")

    cases = [c for c in CASES if not args.filter or args.filter.lower() in c[0].lower()]
    passed = total = 0
    failures = []
    with open(args.output, "a") if args.output else contextlib.nullcontext() as out:
        for name, site, turns in cases:
            passes = [0] * len(turns)
            for run in range(args.repeats):
                post("/api/reset-conversation", {"site": site}, 10)
                for i, (text, check) in enumerate(turns):
                    t0 = time.monotonic()
                    try:
                        d = post("/api/dry-run", {"text": text, "site": site}, 120)
                    except Exception as e:
                        # The turns after it would answer a conversation that lacks this one: they fail too
                        failures.append(f"{name} · {text} — {type(e).__name__}: {e}")
                        break
                    wall = round(time.monotonic() - t0, 2)
                    ok, expect = check(d)
                    passes[i] += ok
                    found = [(tc["function"], tc["args"]) for tc in d.get("tool_calls", [])]
                    if not ok:
                        got = ", ".join(f"{fn}({a})" for fn, a in found) or "no call"
                        failures.append(f"{name} · {text} — expected {expect}; got {got}, « {reply(d)[:100]} »")
                    if out:
                        out.write(json.dumps({
                            "model": model, "case": name, "site": site, "run": run, "turn": i, "text": text, "pass": ok,
                            "expect": expect, "wall": wall, "response": reply(d), "calls": found,
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
    main()
