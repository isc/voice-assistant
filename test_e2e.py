#!/usr/bin/env python3
"""End-to-end integration tests for the voice assistant LLM pipeline.

Sends commands to the /api/dry-run endpoint which calls the LLM but
does NOT execute tool calls against Home Assistant. Verifies that
the LLM produces correct tool calls for each scenario.

Requires the server to be running (./ctl.sh start).

Usage:
    python test_e2e.py              # run all tests
    python test_e2e.py -k weather   # run tests matching "weather"
    python test_e2e.py -v           # verbose output
"""

import argparse
import json
import sys
import urllib.request

SERVER = "http://localhost:8888"


def send(text: str) -> dict:
    """Send a text command via dry-run (no HA execution)."""
    req = urllib.request.Request(
        f"{SERVER}/api/dry-run",
        data=json.dumps({"text": text}).encode(),
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req, timeout=30)
    return json.loads(resp.read())


def reset():
    """Clear conversation history between independent tests."""
    req = urllib.request.Request(
        f"{SERVER}/api/reset-conversation",
        data=b"",
        method="POST",
    )
    urllib.request.urlopen(req, timeout=5)


def tool_names(data: dict) -> list[str]:
    """Extract tool call function names from response."""
    return [tc["function"] for tc in data.get("tool_calls", [])]


def tool_args(data: dict, function: str) -> dict | None:
    """Extract args for the first matching tool call."""
    for tc in data.get("tool_calls", []):
        if tc["function"] == function:
            return tc["args"]
    return None


def has_tool(data: dict, function: str) -> bool:
    """Check if a specific tool was called."""
    return function in tool_names(data)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

TESTS = []


def test(name, multi_turn=False):
    """Decorator to register a test case. multi_turn=True skips reset."""

    def decorator(fn):
        TESTS.append((name, fn, multi_turn))
        return fn

    return decorator


@test("turn_on: simple light")
def test_turn_on_simple():
    data = send("allume le plafonnier dans la chambre invités")
    assert has_tool(data, "turn_on"), f"Expected turn_on, got {tool_names(data)}"
    args = tool_args(data, "turn_on")
    room = args.get("room", "").lower()
    assert "invit" in room or "chambre" in room, f"Expected room with 'invités', got {args}"


@test("turn_off: simple light")
def test_turn_off_simple():
    data = send("éteins la lumière du salon")
    assert has_tool(data, "turn_off"), f"Expected turn_off, got {tool_names(data)}"


@test("turn_on: with brightness")
def test_brightness():
    data = send("allume le plafonnier chambre invités à 30%")
    assert has_tool(data, "turn_on"), f"Expected turn_on, got {tool_names(data)}"
    args = tool_args(data, "turn_on")
    assert "brightness" in args, f"Expected brightness in args, got {args}"
    assert args["brightness"] <= 50, f"Expected brightness <= 50, got {args['brightness']}"


@test("close_cover: single room")
def test_close_cover():
    data = send("ferme le volet de la chambre invités")
    assert has_tool(data, "close_cover"), f"Expected close_cover, got {tool_names(data)}"


@test("close_cover: room group 'enfants'")
def test_close_cover_group():
    data = send("ferme les volets des enfants")
    assert has_tool(data, "close_cover"), f"Expected close_cover, got {tool_names(data)}"
    args = tool_args(data, "close_cover")
    assert "enfant" in args.get("room", "").lower(), f"Expected room with 'enfants', got {args}"


@test("open_cover: all shutters")
def test_open_cover_all():
    data = send("ouvre tous les volets")
    assert has_tool(data, "open_cover"), f"Expected open_cover, got {tool_names(data)}"


@test("get_weather: default city")
def test_weather_default():
    data = send("quel temps fait-il")
    assert has_tool(data, "get_weather"), f"Expected get_weather, got {tool_names(data)}"


@test("get_weather: specific city")
def test_weather_city():
    data = send("quel temps fait-il à Lyon")
    assert has_tool(data, "get_weather"), f"Expected get_weather, got {tool_names(data)}"
    args = tool_args(data, "get_weather")
    assert "lyon" in args.get("location", "").lower(), f"Expected location with 'Lyon', got {args}"


@test("get_state: light status")
def test_get_state():
    data = send("est-ce que la lumière du salon est allumée")
    assert has_tool(data, "get_state"), f"Expected get_state, got {tool_names(data)}"


@test("set_temperature: thermostat")
def test_set_temperature():
    data = send("mets le chauffage à 21 degrés")
    assert has_tool(data, "set_temperature"), f"Expected set_temperature, got {tool_names(data)}"
    args = tool_args(data, "set_temperature")
    assert args.get("temperature") == 21, f"Expected temperature=21, got {args}"


@test("multi-action: light + weather")
def test_multi_action():
    data = send("éteins la lumière du salon et dis-moi la météo")
    names = tool_names(data)
    assert "turn_off" in names, f"Expected turn_off in {names}"
    assert "get_weather" in names, f"Expected get_weather in {names}"


@test("end_conversation: goodbye")
def test_end_conversation():
    data = send("merci bonne nuit")
    assert has_tool(data, "end_conversation"), f"Expected end_conversation, got {tool_names(data)}"


@test("no tool call: general question")
def test_no_tool_general():
    data = send("raconte-moi une blague courte")
    ha_tools = {
        "turn_on",
        "turn_off",
        "open_cover",
        "close_cover",
        "set_temperature",
        "get_state",
    }
    called = set(tool_names(data))
    assert not (called & ha_tools), f"Expected no HA tool calls, got {called}"
    assert data.get("response"), "Expected a text response"


@test("turn_off: all lights with 'partout'")
def test_turn_off_all():
    data = send("éteins toutes les lumières de la maison")
    names = tool_names(data)
    assert "turn_off" in names, f"Expected turn_off in {names}"


@test("multi-turn: follow-up uses context", multi_turn=True)
def test_multi_turn():
    reset()  # clean slate for this conversation
    data1 = send("allume le plafonnier chambre invités")
    assert has_tool(data1, "turn_on"), f"Turn 1: expected turn_on, got {tool_names(data1)}"
    # Follow-up without specifying the room — LLM should use context
    data2 = send("et baisse à 30%")
    assert has_tool(data2, "turn_on"), f"Turn 2: expected turn_on, got {tool_names(data2)}"
    args = tool_args(data2, "turn_on")
    assert "brightness" in args, f"Turn 2: expected brightness in args, got {args}"


@test("set_timer: 5 minutes")
def test_set_timer():
    data = send("mets un timer de 5 minutes")
    assert has_tool(data, "set_timer"), f"Expected set_timer, got {tool_names(data)}"
    args = tool_args(data, "set_timer")
    assert args.get("duration_minutes") == 5, f"Expected duration_minutes=5, got {args}"


@test("set_timer: with label")
def test_set_timer_label():
    data = send("mets un timer de 10 minutes pour les pâtes")
    assert has_tool(data, "set_timer"), f"Expected set_timer, got {tool_names(data)}"
    args = tool_args(data, "set_timer")
    assert args.get("duration_minutes") == 10, f"Expected duration_minutes=10, got {args}"
    assert args.get("label"), f"Expected label in args, got {args}"


@test("set_alarm: specific time")
def test_set_alarm():
    data = send("réveille-moi à 7 heures")
    assert has_tool(data, "set_alarm"), f"Expected set_alarm, got {tool_names(data)}"
    args = tool_args(data, "set_alarm")
    assert "07" in args.get("time", ""), f"Expected time with '07', got {args}"


@test("cancel_timer")
def test_cancel_timer():
    data = send("annule le timer")
    assert has_tool(data, "cancel_timer"), f"Expected cancel_timer, got {tool_names(data)}"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="E2E tests for voice assistant LLM pipeline")
    parser.add_argument("-k", "--filter", help="Run only tests matching this string")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show tool call details")
    args = parser.parse_args()

    # Check server is running
    try:
        urllib.request.urlopen(f"{SERVER}/api/exchanges", timeout=5)
    except Exception:
        print(f"ERROR: Server not reachable at {SERVER}")
        print("Start it with: ./ctl.sh start")
        sys.exit(1)

    tests = TESTS
    if args.filter:
        tests = [(name, fn, mt) for name, fn, mt in tests if args.filter.lower() in name.lower()]

    passed = 0
    failed = 0
    errors = []

    for name, fn, multi_turn in tests:
        if not multi_turn:
            reset()
        try:
            fn()
            passed += 1
            print(f"  PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}")
            errors.append((name, str(e)))
        except Exception as e:
            failed += 1
            print(f"  ERROR {name}")
            errors.append((name, f"{type(e).__name__}: {e}"))

    print(f"\n{'=' * 50}")
    print(f"{passed} passed, {failed} failed out of {passed + failed}")

    if errors:
        print("\nFailures:")
        for name, msg in errors:
            print(f"  {name}: {msg}")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
