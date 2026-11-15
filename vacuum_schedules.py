"""A robot vacuum's weekly schedules by voice (AD-108): the robot's own (`cleanSchedule2`, those iRobot's app shows
and sets), read and written through its Home Assistant's `roomba_schedule` services (Paris's custom integration,
home-infra paris/custom_components/). The robot cleans on its own, and a holiday ends there too: without this
server.

An entry: `{"enabled": bool, "type": 0, "start": {"day": [0-6, Sunday 0], "hour": h, "min": m}, "cmd": {"command":
"start"}}` for the whole home, or with the params of rooms_params for some rooms of its map."""

import copy
from datetime import date
from typing import Optional

from ha_client import (
    SCHEDULE_SERVICE,
    Refusal,
    fold,
    map_rooms,
    rooms_named,
    rooms_params,
    vacuum_map,
    wants_whole_home,
)
from places import DAYS as WEEK
from places import MONTHS
from timer import parse_time

DAYS = (WEEK[-1], *WEEK[:-1])  # the robot's: 0 is Sunday
SCHEDULE_ACTIONS = ("show", "add", "change", "remove", "disable", "enable")
_DONE = {"add": "ajoutée", "change": "changée", "remove": "supprimée", "disable": "désactivée", "enable": "activée"}


def day_of(text: Optional[str]) -> int:
    """The robot's day a word names: « mardi », « le mardi », « mardis »."""
    for word in fold(text or "").replace("'", " ").split():
        if (day := word.removesuffix("s")) in DAYS:
            return DAYS.index(day)
    raise Refusal(f"précise le jour ({', '.join(WEEK)})")


def time_of(text: Optional[str]) -> tuple[int, int]:
    try:
        return parse_time(str(text or ""))
    except ValueError:
        raise Refusal("précise l'heure (HH:MM)") from None


def clock(hour: int, minute: int) -> str:
    return f"{hour} h {minute:02}" if minute else f"{hour} h"


def days_words(days: list[int]) -> str:
    """Monday first, as a week is said: « lundi et jeudi »."""
    names = [DAYS[day] for day in sorted(days, key=lambda day: (day - 1) % 7)]
    return " et ".join(filter(None, (", ".join(names[:-1]), names[-1])))


def date_words(day: date | str) -> str:
    day = date.fromisoformat(day) if isinstance(day, str) else day
    return f"{day.day} {MONTHS[day.month - 1]}"


def regions(entry: dict) -> list[str]:
    return [region["region_id"] for region in entry["cmd"].get("regions", [])]


def entry_words(entry: dict, plan: dict, state: bool = True) -> str:
    """« lundi et jeudi à 8 h 30 : tout le logement », « mercredi à 9 h, désactivée : Chambre Charlie, Couloir »;
    without `state`, not whether it is on."""
    names = {region: name for name, region in plan.get("rooms", {}).items()}
    rooms = ", ".join(names.get(region, f"pièce n° {region}") for region in regions(entry)) or "tout le logement"
    start = entry["start"]
    off = ", désactivée" if state and not entry["enabled"] else ""
    return f"{days_words(start['day'])} à {clock(start['hour'], start['min'])}{off} : {rooms}"


def schedule_words(entries: list[dict], plan: dict) -> str:
    """« Lundi et jeudi à 8 h 30 : tout le logement. Vendredi à 9 h : Cuisine »."""
    said = [entry_words(entry, plan) for entry in entries]
    return ". ".join(words[:1].upper() + words[1:] for words in said) or "Aucune programmation"


def _command(entity_id: str, room: str) -> dict:
    """A start in the rooms said, the whole home for none (or « tout »)."""
    if wants_whole_home(room):
        return {"command": "start"}
    plan = vacuum_map(entity_id)
    found = map_rooms(entity_id, room, "l'aspirateur")
    return {"command": "start", **rooms_params(plan, [plan["rooms"][name] for name in found])}


def _without(entity_id: str, entry: dict, room: str) -> dict:
    """An entry's start without the rooms said: « pas la chambre de Zoé », of the whole home its map's other rooms."""
    plan = vacuum_map(entity_id)
    if not (rooms := plan.get("rooms")):
        raise Refusal("l'aspirateur ne connaît pas les pièces, il ne fait que tout le logement")
    names = {region: name for name, region in rooms.items()}
    kept = regions(entry) or list(rooms.values())
    out, unknown = rooms_named(room, [names[region] for region in kept if region in names])
    if unknown:
        raise Refusal(f"« {', '.join(unknown)} » n'est pas dans cette programmation")
    if not (kept := [region for region in kept if names.get(region) not in out]):
        raise Refusal("il ne resterait aucune pièce : supprime plutôt cette programmation")
    return {"command": "start", **rooms_params(plan, kept)}


def _the_day(entries: list[dict], plan: dict, day: int, at: Optional[str]) -> dict:
    """The entry of a day, split off one that covers others: « le lundi » of « lundi et jeudi »."""
    found = [entry for entry in entries if day in entry["start"]["day"]]
    if at and len(found) > 1:
        found = [entry for entry in found if (entry["start"]["hour"], entry["start"]["min"]) == time_of(at)]
    if not found:
        raise Refusal(f"pas de programmation le {DAYS[day]}. {schedule_words(entries, plan)}")
    if len(found) > 1:
        raise Refusal(f"plusieurs le {DAYS[day]} ({', '.join(entry_words(e, plan) for e in found)}) : laquelle ?")
    if len((entry := found[0])["start"]["day"]) == 1:
        return entry
    entry["start"]["day"].remove(day)
    split = copy.deepcopy(entry)
    split["start"]["day"] = [day]
    entries.insert(entries.index(entry) + 1, split)
    return split


def edit(entries: list[dict], entity_id: str, action: str, args: dict) -> tuple[list[dict], list[dict]]:
    """The schedule after `action`, and the entries it changed. Raises Refusal."""
    entries, plan = copy.deepcopy(entries), vacuum_map(entity_id)
    day, at = args.get("day"), args.get("time")
    if action == "add":
        hour, minute = time_of(at)
        start = {"day": [day_of(day)], "hour": hour, "min": minute}
        entry = {"enabled": True, "type": 0, "start": start, "cmd": _command(entity_id, str(args.get("room") or ""))}
        return [*entries, entry], [entry]
    if action in ("enable", "disable") and not day:
        on = action == "enable"
        changed = [entry for entry in entries if entry["enabled"] != on]
        for entry in changed:
            entry["enabled"] = on
        return entries, changed
    entry = _the_day(entries, plan, day_of(day), None if action == "change" else at)
    if action == "remove":
        entries.remove(entry)
    elif action in ("enable", "disable"):
        if entry["enabled"] == (action == "enable"):
            return entries, []
        entry["enabled"] = action == "enable"
    elif action == "change":
        room, without = str(args.get("room") or ""), str(args.get("except_room") or "")
        if not (at or room or without):
            raise Refusal("précise ce qui change : l'heure ou les pièces")
        if at:
            entry["start"]["hour"], entry["start"]["min"] = time_of(at)
        if room:
            entry["cmd"] = _command(entity_id, room)
        if without:
            entry["cmd"] = _without(entity_id, entry, without)
    return entries, [entry]


async def run(client, entity_id: str, args: dict) -> str:
    """A `vacuum_schedule` call on this robot: its result, said by the model."""
    if (action := str(args.get("action") or "")) not in SCHEDULE_ACTIONS:
        return f"Rien n'a été fait : précise l'action ({', '.join(SCHEDULE_ACTIONS)})"

    async def service(name: str, **data) -> Optional[dict]:
        return await client.service_response(SCHEDULE_SERVICE, name, entity_id=entity_id, **data)

    if (now := await service("get")) is None:
        return "Rien n'a été fait : les programmations de l'aspirateur ne se lisent pas"
    entries, plan = now.get("schedule", []), vacuum_map(entity_id)
    if action == "show":
        held = f" Suspendues jusqu'au {date_words(now['until'])}." if now.get("until") else ""
        return f"Programmations de l'aspirateur : {schedule_words(entries, plan)}.{held}"
    every = action in ("enable", "disable") and not args.get("day")
    try:
        until = date.fromisoformat(args["until"]) if action == "disable" and args.get("until") else None
    except ValueError:
        return "Rien n'a été fait : précise la date de retour (AAAA-MM-JJ)"
    if every and (action == "disable" or now.get("suspended")):
        # A holiday, which Home Assistant ends at its date (« jusqu'au 28 »); back, those it turned off alone
        data = {"until": str(until)} if until else {}
        answer = await service("suspend" if action == "disable" else "resume", **data)
        changed = (answer or {}).get("changed", [])
        if answer and answer.get("confirmed") and not changed and until:  # « finalement jusqu'au 30 »
            return f"Programmations déjà désactivées, remises le {date_words(until)}"
    else:
        try:
            new, changed = edit(entries, entity_id, action, args)
        except Refusal as refusal:
            return f"Rien n'a été fait : {refusal}"
        answer = await service("set", schedule=new) if changed else {"confirmed": True}
    if not answer or not answer.get("confirmed"):
        return "Pas de confirmation : le robot n'a pas repris la programmation"
    if not changed:
        return f"Rien n'a été fait : déjà ainsi. {schedule_words(entries, plan)}"
    words = "; ".join(entry_words(entry, plan, state=action not in ("enable", "disable")) for entry in changed)
    if len(changed) > 1 or every:
        back = f" jusqu'au {date_words(until)}, remises ce jour-là" if until else ""
        return f"Programmations {_DONE[action]}s{back} : {words}"
    return f"Programmation {_DONE[action]} : {words}"
