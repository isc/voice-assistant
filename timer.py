"""Timer manager — in-memory timers with asyncio scheduling and ESP event forwarding."""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Coroutine

from aioesphomeapi import VoiceAssistantTimerEventType

logger = logging.getLogger(__name__)

# Callback signature: (event_type, timer) -> None (async)
TimerCallback = Callable[
    ["VoiceAssistantTimerEventType", "Timer"],
    Coroutine,
]


@dataclass
class Timer:
    """A single timer instance."""

    id: str
    name: str | None
    total_seconds: int
    fire_at: float
    is_active: bool = True
    is_alarm: bool = False  # True for set_alarm (has a target time)
    target_time: str | None = None  # "HH:MM" for alarms
    client: str | None = None  # the web app's browser that set it, told when it rings (AD-070)
    satellite: bool = True  # shown and rung by its site's satellite: not when set from a phone away (AD-070)
    task: asyncio.Task | None = field(default=None, repr=False)

    @property
    def seconds_left(self) -> int:
        return max(0, int(self.fire_at - time.time()))


class TimerManager:
    """Manages in-memory timers with asyncio scheduling."""

    def __init__(self):
        self.timers: dict[str, Timer] = {}

    def start_timer(
        self,
        total_seconds: int,
        name: str | None,
        on_event: TimerCallback,
        client: str | None = None,
        satellite: bool = True,
    ) -> Timer:
        """Create and start a new timer. Fires STARTED event immediately."""
        timer_id = uuid.uuid4().hex[:8]
        timer = Timer(
            id=timer_id,
            name=name,
            total_seconds=total_seconds,
            fire_at=time.time() + total_seconds,
            client=client,
            satellite=satellite,
        )
        self.timers[timer_id] = timer
        timer.task = asyncio.create_task(self._run_timer(timer, on_event))
        logger.info(f"Timer started: id={timer_id} name={name} duration={total_seconds}s")
        return timer

    def start_alarm(
        self,
        time_str: str,
        name: str | None,
        on_event: TimerCallback,
        client: str | None = None,
        satellite: bool = True,
    ) -> Timer:
        """Create a timer that fires at a specific time (HH:MM). Wraps to next day if past."""
        hour, minute = _parse_time(time_str)
        now = datetime.now()
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)
        # From the target's own timestamp: subtracting naive datetimes counts wall-clock hours, and
        # « réveille-moi à 7h » said on 24 October at 23:00 rang at 6:00 (the clocks go back)
        total_seconds = int(target.timestamp() - time.time())
        target_time_str = f"{hour:02d}:{minute:02d}"
        logger.info(f"Alarm for {target_time_str} -> {total_seconds}s from now")
        timer = self.start_timer(total_seconds, name, on_event, client=client, satellite=satellite)
        timer.is_alarm = True
        timer.target_time = target_time_str
        return timer

    def cancel_timer(
        self, timer_id: str | None = None, name: str | None = None, client: str | None = None
    ) -> Timer | None:
        """Cancel an active timer by ID, or by name: a spoken one, so with neither or a name that
        matches none, the only active timer (« annule le minuteur ») that `client` sees (get_timers).
        Returns the cancelled timer or None."""
        timer = None
        if timer_id:
            timer = self.timers.get(timer_id)
        else:
            active = self.get_timers(client)
            if name:
                timer = next((t for t in active if t.name and t.name.lower() == name.lower()), None)
            if not timer and len(active) == 1:
                timer = active[0]

        if timer and timer.is_active:
            timer.is_active = False
            if timer.task and not timer.task.done():
                timer.task.cancel()
            del self.timers[timer.id]
            logger.info(f"Timer cancelled: id={timer.id} name={timer.name}")
            return timer
        return None

    def get_timers(self, client: str | None = None) -> list[Timer]:
        """Active timers by fire time, those `client` sees: its site's satellite's, and the ones that
        browser of the web app set away from home (AD-070); with no client, the satellite's alone."""
        return sorted(
            [t for t in self.timers.values() if t.is_active and (t.satellite or t.client == client)],
            key=lambda t: t.fire_at,
        )

    def cancel_all(self):
        """Cancel all active timers (cleanup on shutdown)."""
        for timer in list(self.timers.values()):
            timer.is_active = False
            if timer.task and not timer.task.done():
                timer.task.cancel()
        self.timers.clear()

    async def _run_timer(self, timer: Timer, on_event: TimerCallback):
        """Run a timer: send STARTED, then FINISHED when it fires."""
        try:
            await on_event(VoiceAssistantTimerEventType.VOICE_ASSISTANT_TIMER_STARTED, timer)
            # The satellite counts down from STARTED itself, as with Home Assistant: an UPDATED every
            # second sent 28,800 events for an 8-hour alarm
            await asyncio.sleep(max(0.0, timer.fire_at - time.time()))

            if timer.is_active:
                timer.is_active = False
                await on_event(VoiceAssistantTimerEventType.VOICE_ASSISTANT_TIMER_FINISHED, timer)
                self.timers.pop(timer.id, None)
                logger.info(f"Timer finished: id={timer.id} name={timer.name}")

        except asyncio.CancelledError:
            logger.debug(f"Timer task cancelled: id={timer.id}")
        except Exception as e:
            logger.error(f"Timer error: {e}")


def _parse_time(time_str: str) -> tuple[int, int]:
    """Parse a time string like '7:00', '07:00', '7h', '7h30' into (hour, minute)."""
    time_str = time_str.strip().lower()

    # "7h30", "07h00", "7h"
    if "h" in time_str:
        parts = time_str.split("h")
        hour = int(parts[0])
        minute = int(parts[1]) if parts[1] else 0
        return hour, minute

    # "7:00", "07:00"
    if ":" in time_str:
        parts = time_str.split(":")
        return int(parts[0]), int(parts[1])

    # "700", "0700" — unlikely but handle
    raise ValueError(f"Cannot parse time: {time_str}")


def _format_remaining(seconds: int) -> str:
    """Format remaining seconds into a French human-readable string."""
    if seconds >= 3600:
        h = seconds // 3600
        m = (seconds % 3600) // 60
        return f"{h}h{m:02d}" if m else f"{h}h"
    if seconds >= 60:
        m = seconds // 60
        s = seconds % 60
        return f"{m}min{s:02d}s" if s else f"{m}min"
    return f"{seconds}s"


def format_timers_for_prompt(timers: list[Timer]) -> str:
    """Format active timers for inclusion in the LLM system prompt."""
    if not timers:
        return ""
    parts = []
    for t in timers:
        remaining = _format_remaining(t.seconds_left)
        if t.is_alarm and t.target_time:
            label = f'"{t.name}" ' if t.name else ""
            parts.append(f"alarme {label}pour {t.target_time} ({remaining} restantes)")
        else:
            label = f'"{t.name}" ' if t.name else ""
            total = _format_remaining(t.total_seconds)
            parts.append(f"timer {label}de {total} ({remaining} restantes)")
    return "Minuteurs/alarmes en cours (tu les as programmés, donne ces infos si l'utilisateur demande): " + ", ".join(
        parts
    )
