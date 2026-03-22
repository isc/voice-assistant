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
    task: asyncio.Task | None = field(default=None, repr=False)

    @property
    def seconds_left(self) -> int:
        return max(0, int(self.fire_at - time.time()))


class TimerManager:
    """Manages in-memory timers with asyncio scheduling."""

    def __init__(self):
        self.timers: dict[str, Timer] = {}

    def start_timer(self, total_seconds: int, name: str | None, on_event: TimerCallback) -> Timer:
        """Create and start a new timer. Fires STARTED event immediately."""
        timer_id = uuid.uuid4().hex[:8]
        timer = Timer(
            id=timer_id,
            name=name,
            total_seconds=total_seconds,
            fire_at=time.time() + total_seconds,
        )
        self.timers[timer_id] = timer
        timer.task = asyncio.create_task(self._run_timer(timer, on_event))
        logger.info(f"Timer started: id={timer_id} name={name} duration={total_seconds}s")
        return timer

    def start_alarm(self, time_str: str, name: str | None, on_event: TimerCallback) -> Timer:
        """Create a timer that fires at a specific time (HH:MM). Wraps to next day if past."""
        hour, minute = _parse_time(time_str)
        now = datetime.now()
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)
        total_seconds = int((target - now).total_seconds())
        target_time_str = f"{hour:02d}:{minute:02d}"
        logger.info(f"Alarm for {target_time_str} -> {total_seconds}s from now")
        timer = self.start_timer(total_seconds, name, on_event)
        timer.is_alarm = True
        timer.target_time = target_time_str
        return timer

    def cancel_timer(self, timer_id: str | None = None, name: str | None = None) -> Timer | None:
        """Cancel an active timer by ID or name. Returns the cancelled timer or None."""
        timer = None
        if timer_id and timer_id in self.timers:
            timer = self.timers[timer_id]
        elif name:
            name_lower = name.lower()
            for t in self.timers.values():
                if t.is_active and t.name and t.name.lower() == name_lower:
                    timer = t
                    break
        if not timer:
            # If only one active timer, cancel it regardless of name
            active = [t for t in self.timers.values() if t.is_active]
            if len(active) == 1 and name:
                timer = active[0]

        if timer and timer.is_active:
            timer.is_active = False
            if timer.task and not timer.task.done():
                timer.task.cancel()
            del self.timers[timer.id]
            logger.info(f"Timer cancelled: id={timer.id} name={timer.name}")
            return timer
        return None

    def get_timers(self) -> list[Timer]:
        """Return all active timers sorted by fire time."""
        return sorted(
            [t for t in self.timers.values() if t.is_active],
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
        """Run a timer: send STARTED, tick every 1s with UPDATED, then FINISHED."""
        try:
            await on_event(VoiceAssistantTimerEventType.VOICE_ASSISTANT_TIMER_STARTED, timer)

            while timer.is_active and timer.seconds_left > 0:
                await asyncio.sleep(1)
                if timer.is_active:
                    await on_event(VoiceAssistantTimerEventType.VOICE_ASSISTANT_TIMER_UPDATED, timer)

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
    return (
        "Minuteurs/alarmes en cours (tu les as programmés, "
        "donne ces infos si l'utilisateur demande): " + ", ".join(parts)
    )
