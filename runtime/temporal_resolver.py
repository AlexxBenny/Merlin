# runtime/temporal_resolver.py

"""
TemporalResolver — Deterministic time expression → UTC epoch conversion.

The LLM NEVER computes epochs. It returns structured intent like:
    {"kind": "delay", "expression": "10 seconds"}
    {"kind": "absolute_time", "expression": "4 PM"}

This module deterministically converts those into UTC epoch floats.

This is INFRASTRUCTURE, not cognition.
No LLM. No cortex imports. No IR imports. No models imports.

Supported trigger kinds:
    delay          → "10 seconds", "2 minutes", "1 hour"
    absolute_time  → "4 PM", "3:30 PM", "23:30", "16:00"

Phase 1 scope:
    - delay (relative seconds)
    - absolute_time (same-day or next occurrence)

Phase 2+:
    - "tomorrow at 9 AM" (date + time)
    - recurring interval alignment
    - cron-like expressions

Safety:
    - All times resolved in local timezone, stored with timezone name.
    - Never raises — returns None on parse failure.
    - Deterministic: same input + same now → same output.
"""

import logging
import re
import time as _time
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Time unit multipliers (seconds)
# ─────────────────────────────────────────────────────────────

_UNIT_SECONDS: Dict[str, int] = {
    "second": 1, "seconds": 1, "sec": 1, "secs": 1, "s": 1,
    "minute": 60, "minutes": 60, "min": 60, "mins": 60, "m": 60,
    "hour": 60 * 60, "hours": 60 * 60, "hr": 60 * 60, "hrs": 60 * 60, "h": 60 * 60,
}

# ─────────────────────────────────────────────────────────────
# Delay pattern: "10 seconds", "2 minutes", "1 hour", "30s", "5m"
# ─────────────────────────────────────────────────────────────

_DELAY_PATTERN = re.compile(
    r'(\d+)\s*'
    r'(seconds?|secs?|minutes?|mins?|hours?|hrs?|[smh])\b',
    re.IGNORECASE,
)

# ─────────────────────────────────────────────────────────────
# Absolute time patterns
# ─────────────────────────────────────────────────────────────

# "4 PM", "4PM", "4:30 PM", "4:30PM", "16:00", "16:30", "3 am"
_TIME_12H_PATTERN = re.compile(
    r'(\d{1,2})(?::(\d{2}))?\s*(am|pm)',
    re.IGNORECASE,
)
_TIME_24H_PATTERN = re.compile(
    r'(\d{1,2}):(\d{2})(?::(\d{2}))?$',
)

# ─────────────────────────────────────────────────────────────
# Date modifier patterns (Phase 1 stretch)
# ─────────────────────────────────────────────────────────────

_TOMORROW_PATTERN = re.compile(r'\btomorrow\b', re.IGNORECASE)


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

class TemporalResolver:
    """Deterministic time expression resolver.

    Pure infrastructure. No LLM. No cognition.

    Usage:
        resolver = TemporalResolver()
        epoch = resolver.resolve({"kind": "delay", "expression": "10 seconds"})
        # epoch = now + 10

        epoch = resolver.resolve({"kind": "absolute_time", "expression": "4 PM"})
        # epoch = today's 4 PM (or tomorrow if already past)

    Thread-safe: no mutable state.
    """

    @staticmethod
    def resolve(
        trigger_spec: Dict[str, Any],
        now: Optional[float] = None,
    ) -> Optional[float]:
        """Convert trigger spec → UTC epoch float.

        Args:
            trigger_spec: {"kind": "delay|absolute_time", "expression": "..."}
            now: Optional override for current time (for testing).

        Returns:
            UTC epoch float, or None if expression cannot be parsed.
            Never raises.
        """
        if not trigger_spec or not isinstance(trigger_spec, dict):
            return None

        kind = trigger_spec.get("kind", "").lower().strip()
        expression = str(trigger_spec.get("expression", "")).strip()

        if not kind or not expression:
            return None

        if now is None:
            now = _time.time()

        try:
            if kind == "delay":
                return _resolve_delay(expression, now)
            elif kind == "absolute_time":
                return _resolve_absolute_time(expression, now)
            else:
                logger.warning(
                    "[TEMPORAL] Unknown trigger kind: '%s'", kind,
                )
                return None
        except Exception as e:
            logger.warning(
                "[TEMPORAL] Failed to resolve '%s' (%s): %s",
                expression, kind, e,
            )
            return None

    @staticmethod
    def resolve_delay_seconds(expression: str) -> Optional[int]:
        """Parse a delay expression into total seconds.

        Used by TickSchedulerManager for retry backoff calculation.

        Args:
            expression: e.g. "10 seconds", "2 minutes", "1 hour"

        Returns:
            Total seconds as int, or None if unparseable.
        """
        return _parse_delay_seconds(expression)

    @staticmethod
    def get_local_timezone() -> str:
        """Return the IANA timezone name for the local system.

        Falls back to UTC offset string if IANA name unavailable.
        Stored with scheduled jobs for DST safety.
        """
        try:
            # Try to get IANA name via datetime
            local_tz = datetime.now().astimezone().tzinfo
            tz_name = str(local_tz)

            # Python's tzinfo str might return 'UTC+05:30' style
            # Try to get a proper name
            if hasattr(local_tz, 'key'):
                return local_tz.key  # Python 3.9+ zoneinfo
            if hasattr(local_tz, 'zone'):
                return local_tz.zone  # pytz

            return tz_name
        except Exception:
            return "UTC"


# ─────────────────────────────────────────────────────────────
# Internal resolvers
# ─────────────────────────────────────────────────────────────

def _parse_delay_seconds(expression: str) -> Optional[int]:
    """Parse delay expression → total seconds.

    Supports:
        "10 seconds", "2 minutes", "1 hour"
        "30s", "5m", "1h"
        "10 secs", "2 mins", "1 hr"

    Returns None if unparseable.
    """
    match = _DELAY_PATTERN.search(expression)
    if not match:
        return None

    value = int(match.group(1))
    unit = match.group(2).lower()

    multiplier = _UNIT_SECONDS.get(unit)
    if multiplier is None:
        return None

    return value * multiplier


def _resolve_delay(expression: str, now: float) -> Optional[float]:
    """Resolve delay expression → epoch.

    "10 seconds" → now + 10
    """
    seconds = _parse_delay_seconds(expression)
    if seconds is None:
        return None
    if seconds <= 0:
        return None
    return now + seconds


def _resolve_absolute_time(expression: str, now: float) -> Optional[float]:
    """Resolve absolute time expression → epoch.

    "4 PM" → today's 16:00 local time (or tomorrow if already past)
    "3:30 PM" → today's 15:30 local time
    "23:30" → today's 23:30 local time
    "tomorrow at 4 PM" → tomorrow's 16:00

    Smart scheduling: if the time has already passed today,
    it's automatically scheduled for tomorrow.
    """
    # Check for "tomorrow" modifier
    is_tomorrow = bool(_TOMORROW_PATTERN.search(expression))

    # Strip "tomorrow" and common prepositions for time parsing
    time_text = _TOMORROW_PATTERN.sub("", expression).strip()
    time_text = re.sub(r'^(at|by|around)\s+', '', time_text, flags=re.IGNORECASE).strip()

    hour, minute = _parse_time_expression(time_text)
    if hour is None:
        return None

    # Build target datetime in local timezone
    now_dt = datetime.fromtimestamp(now)
    target_dt = now_dt.replace(hour=hour, minute=minute, second=0, microsecond=0)

    if is_tomorrow:
        target_dt += timedelta(days=1)
    elif target_dt <= now_dt:
        # Already passed today → schedule for tomorrow
        target_dt += timedelta(days=1)

    return target_dt.timestamp()


def _parse_time_expression(text: str) -> Tuple[Optional[int], int]:
    """Parse time string → (hour_24h, minute).

    Supports:
        "4 PM" → (16, 0)
        "4:30 PM" → (16, 30)
        "4PM" → (16, 0)
        "23:30" → (23, 30)
        "16:00" → (16, 0)
        "3 am" → (3, 0)

    Returns (None, 0) if unparseable.
    """
    text = text.strip()

    # Try 12-hour format first: "4 PM", "4:30PM", "3 am"
    match = _TIME_12H_PATTERN.search(text)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2)) if match.group(2) else 0
        period = match.group(3).lower()

        if hour < 1 or hour > 12:
            return None, 0
        if minute < 0 or minute > 59:
            return None, 0

        if period == "pm" and hour != 12:
            hour += 12
        elif period == "am" and hour == 12:
            hour = 0

        return hour, minute

    # Try 24-hour format: "23:30", "16:00"
    match = _TIME_24H_PATTERN.match(text)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2))

        if hour < 0 or hour > 23:
            return None, 0
        if minute < 0 or minute > 59:
            return None, 0

        return hour, minute

    return None, 0
