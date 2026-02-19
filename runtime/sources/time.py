# runtime/sources/time.py

"""
TimeSource — Clock-based event emitter.

Emits structured time events at configurable intervals.
No OS dependencies beyond stdlib.

Events emitted:
  time_tick       — periodic (default 60s), carries hour/minute/day_of_week
  hour_changed    — at hour boundaries
  date_changed    — at midnight
"""

import time
from datetime import datetime
from typing import Dict, List, Optional

from runtime.sources.base import EventSource
from world.timeline import WorldEvent


class TimeSource(EventSource):
    """
    Emits clock events at configurable intervals.
    Tracks last-emitted state to prevent duplicates.
    """

    def __init__(
        self,
        tick_interval: float = 60.0,
    ):
        self._tick_interval = tick_interval

        # Last emission timestamps
        self._last_tick = 0.0

        # Previous state (for diffing)
        self._last_hour: Optional[int] = None
        self._last_date: Optional[str] = None

    def bootstrap(self) -> List[WorldEvent]:
        """Emit current clock state immediately.

        Sets diffing baselines so first poll doesn't duplicate.
        """
        now = time.time()
        dt = datetime.now()
        self._last_tick = now
        self._last_hour = dt.hour
        self._last_date = dt.strftime("%Y-%m-%d")
        return [WorldEvent(
            timestamp=now,
            source="time",
            type="time_tick",
            payload={
                "domain": "time",
                "hour": dt.hour,
                "minute": dt.minute,
                "day_of_week": dt.strftime("%A"),
                "date": dt.strftime("%Y-%m-%d"),
                "severity": "background",
            },
        )]

    def poll(self) -> List[WorldEvent]:
        events: List[WorldEvent] = []
        now = time.time()
        dt = datetime.now()

        # ── Tick ──
        if now - self._last_tick >= self._tick_interval:
            self._last_tick = now
            events.append(WorldEvent(
                timestamp=now,
                source="time",
                type="time_tick",
                payload={
                    "domain": "time",
                    "hour": dt.hour,
                    "minute": dt.minute,
                    "day_of_week": dt.strftime("%A"),
                    "date": dt.strftime("%Y-%m-%d"),
                    "severity": "background",
                },
            ))

        # ── Hour changed ──
        if self._last_hour is not None and dt.hour != self._last_hour:
            events.append(WorldEvent(
                timestamp=now,
                source="time",
                type="hour_changed",
                payload={
                    "domain": "time",
                    "hour": dt.hour,
                    "severity": "background",
                },
            ))
        self._last_hour = dt.hour

        # ── Date changed ──
        today = dt.strftime("%Y-%m-%d")
        if self._last_date is not None and today != self._last_date:
            events.append(WorldEvent(
                timestamp=now,
                source="time",
                type="date_changed",
                payload={
                    "domain": "time",
                    "date": today,
                    "day_of_week": dt.strftime("%A"),
                    "severity": "background",
                },
            ))
        self._last_date = today

        return events
