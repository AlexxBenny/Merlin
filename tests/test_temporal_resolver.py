# tests/test_temporal_resolver.py

"""
Tests for runtime/temporal_resolver.py — Deterministic time resolution.

Validates:
- Delay expressions: "10 seconds", "2 minutes", "1 hour", "30s", "5m"
- Absolute time: "4 PM", "3:30 PM", "23:30", "16:00", "4PM", "3 am"
- Tomorrow modifier: "tomorrow at 4 PM"
- Smart next-day: past times auto-schedule for tomorrow
- Edge cases: invalid input, empty spec, zero delay
- Timezone awareness: get_local_timezone returns a string
"""

import time

import pytest

from runtime.temporal_resolver import TemporalResolver


# Fixed "now" for deterministic tests
# 2026-03-04 12:00:00 local time (noon)
_NOON = 1741067400.0  # approximate epoch — tests use relative offsets


@pytest.fixture
def resolver():
    return TemporalResolver()


# ─────────────────────────────────────────────────────────────
# Delay resolution
# ─────────────────────────────────────────────────────────────

class TestDelayResolution:
    """Delay expressions: relative offsets from now."""

    def test_10_seconds(self, resolver):
        now = 1000.0
        result = resolver.resolve(
            {"kind": "delay", "expression": "10 seconds"}, now=now,
        )
        assert result == 1010.0

    def test_2_minutes(self, resolver):
        now = 1000.0
        result = resolver.resolve(
            {"kind": "delay", "expression": "2 minutes"}, now=now,
        )
        assert result == 1120.0

    def test_1_hour(self, resolver):
        now = 1000.0
        result = resolver.resolve(
            {"kind": "delay", "expression": "1 hour"}, now=now,
        )
        assert result == 4600.0

    def test_short_form_30s(self, resolver):
        now = 1000.0
        result = resolver.resolve(
            {"kind": "delay", "expression": "30s"}, now=now,
        )
        assert result == 1030.0

    def test_short_form_5m(self, resolver):
        now = 1000.0
        result = resolver.resolve(
            {"kind": "delay", "expression": "5m"}, now=now,
        )
        assert result == 1300.0

    def test_short_form_1h(self, resolver):
        now = 1000.0
        result = resolver.resolve(
            {"kind": "delay", "expression": "1h"}, now=now,
        )
        assert result == 4600.0

    def test_abbreviation_secs(self, resolver):
        now = 1000.0
        result = resolver.resolve(
            {"kind": "delay", "expression": "15 secs"}, now=now,
        )
        assert result == 1015.0

    def test_abbreviation_mins(self, resolver):
        now = 1000.0
        result = resolver.resolve(
            {"kind": "delay", "expression": "3 mins"}, now=now,
        )
        assert result == 1180.0

    def test_abbreviation_hrs(self, resolver):
        now = 1000.0
        result = resolver.resolve(
            {"kind": "delay", "expression": "2 hrs"}, now=now,
        )
        assert result == 8200.0

    def test_no_space_before_unit(self, resolver):
        now = 1000.0
        result = resolver.resolve(
            {"kind": "delay", "expression": "10seconds"}, now=now,
        )
        assert result == 1010.0


# ─────────────────────────────────────────────────────────────
# Absolute time resolution
# ─────────────────────────────────────────────────────────────

class TestAbsoluteTimeResolution:
    """Absolute time: same-day or auto next-day."""

    def test_future_time_today(self, resolver):
        """If time is in the future today, schedule for today."""
        from datetime import datetime
        # Use real "now" and set target 2 hours ahead
        now = time.time()
        now_dt = datetime.fromtimestamp(now)
        future_hour = (now_dt.hour + 2) % 24
        period = "AM" if future_hour < 12 else "PM"
        display_hour = future_hour if future_hour <= 12 else future_hour - 12
        if display_hour == 0:
            display_hour = 12

        result = resolver.resolve(
            {"kind": "absolute_time", "expression": f"{display_hour} {period}"},
            now=now,
        )
        assert result is not None
        assert result > now

    def test_past_time_schedules_tomorrow(self, resolver):
        """If time has already passed today, schedule for tomorrow."""
        from datetime import datetime
        now = time.time()
        now_dt = datetime.fromtimestamp(now)
        # Use 1 hour ago
        past_hour = (now_dt.hour - 1) % 24
        period = "AM" if past_hour < 12 else "PM"
        display_hour = past_hour if past_hour <= 12 else past_hour - 12
        if display_hour == 0:
            display_hour = 12

        result = resolver.resolve(
            {"kind": "absolute_time", "expression": f"{display_hour} {period}"},
            now=now,
        )
        assert result is not None
        # Should be ~23 hours from now (tomorrow)
        assert result > now
        assert result > now + 3600  # at least 1 hour in the future

    def test_4pm_format(self, resolver):
        """'4 PM' parses correctly."""
        from datetime import datetime
        now = time.time()
        result = resolver.resolve(
            {"kind": "absolute_time", "expression": "4 PM"}, now=now,
        )
        assert result is not None
        target_dt = datetime.fromtimestamp(result)
        assert target_dt.hour == 16
        assert target_dt.minute == 0

    def test_4pm_no_space(self, resolver):
        """'4PM' parses correctly."""
        from datetime import datetime
        now = time.time()
        result = resolver.resolve(
            {"kind": "absolute_time", "expression": "4PM"}, now=now,
        )
        assert result is not None
        target_dt = datetime.fromtimestamp(result)
        assert target_dt.hour == 16

    def test_330pm(self, resolver):
        """'3:30 PM' parses correctly."""
        from datetime import datetime
        now = time.time()
        result = resolver.resolve(
            {"kind": "absolute_time", "expression": "3:30 PM"}, now=now,
        )
        assert result is not None
        target_dt = datetime.fromtimestamp(result)
        assert target_dt.hour == 15
        assert target_dt.minute == 30

    def test_24h_format(self, resolver):
        """'23:30' parses correctly."""
        from datetime import datetime
        now = time.time()
        result = resolver.resolve(
            {"kind": "absolute_time", "expression": "23:30"}, now=now,
        )
        assert result is not None
        target_dt = datetime.fromtimestamp(result)
        assert target_dt.hour == 23
        assert target_dt.minute == 30

    def test_3am(self, resolver):
        """'3 am' parses correctly."""
        from datetime import datetime
        now = time.time()
        result = resolver.resolve(
            {"kind": "absolute_time", "expression": "3 am"}, now=now,
        )
        assert result is not None
        target_dt = datetime.fromtimestamp(result)
        assert target_dt.hour == 3

    def test_12pm_is_noon(self, resolver):
        """'12 PM' = noon (12:00)."""
        from datetime import datetime
        now = time.time()
        result = resolver.resolve(
            {"kind": "absolute_time", "expression": "12 PM"}, now=now,
        )
        assert result is not None
        target_dt = datetime.fromtimestamp(result)
        assert target_dt.hour == 12

    def test_12am_is_midnight(self, resolver):
        """'12 AM' = midnight (0:00)."""
        from datetime import datetime
        now = time.time()
        result = resolver.resolve(
            {"kind": "absolute_time", "expression": "12 AM"}, now=now,
        )
        assert result is not None
        target_dt = datetime.fromtimestamp(result)
        assert target_dt.hour == 0

    def test_with_at_prefix(self, resolver):
        """'at 4 PM' strips the 'at' prefix."""
        from datetime import datetime
        now = time.time()
        result = resolver.resolve(
            {"kind": "absolute_time", "expression": "at 4 PM"}, now=now,
        )
        assert result is not None
        target_dt = datetime.fromtimestamp(result)
        assert target_dt.hour == 16


# ─────────────────────────────────────────────────────────────
# Tomorrow modifier
# ─────────────────────────────────────────────────────────────

class TestTomorrowModifier:
    """'tomorrow at X' always schedules for next day."""

    def test_tomorrow_at_4pm(self, resolver):
        from datetime import datetime
        now = time.time()
        now_dt = datetime.fromtimestamp(now)
        result = resolver.resolve(
            {"kind": "absolute_time", "expression": "tomorrow at 4 PM"},
            now=now,
        )
        assert result is not None
        target_dt = datetime.fromtimestamp(result)
        assert target_dt.hour == 16
        assert target_dt.day != now_dt.day or target_dt.month != now_dt.month

    def test_tomorrow_always_future(self, resolver):
        """Tomorrow is always > 12 hours away."""
        now = time.time()
        result = resolver.resolve(
            {"kind": "absolute_time", "expression": "tomorrow at 9 AM"},
            now=now,
        )
        assert result is not None
        assert result > now


# ─────────────────────────────────────────────────────────────
# resolve_delay_seconds utility
# ─────────────────────────────────────────────────────────────

class TestResolveDelaySeconds:
    def test_basic(self):
        assert TemporalResolver.resolve_delay_seconds("10 seconds") == 10

    def test_minutes(self):
        assert TemporalResolver.resolve_delay_seconds("2 minutes") == 120

    def test_hours(self):
        assert TemporalResolver.resolve_delay_seconds("1 hour") == 3600

    def test_short_form(self):
        assert TemporalResolver.resolve_delay_seconds("30s") == 30

    def test_invalid(self):
        assert TemporalResolver.resolve_delay_seconds("banana") is None


# ─────────────────────────────────────────────────────────────
# Edge cases and error handling
# ─────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_none_spec(self, resolver):
        assert resolver.resolve(None) is None

    def test_empty_dict(self, resolver):
        assert resolver.resolve({}) is None

    def test_missing_kind(self, resolver):
        assert resolver.resolve({"expression": "10 seconds"}) is None

    def test_missing_expression(self, resolver):
        assert resolver.resolve({"kind": "delay"}) is None

    def test_unknown_kind(self, resolver):
        result = resolver.resolve(
            {"kind": "cron", "expression": "* * * * *"},
        )
        assert result is None

    def test_invalid_delay_text(self, resolver):
        result = resolver.resolve(
            {"kind": "delay", "expression": "not a time"},
        )
        assert result is None

    def test_invalid_absolute_text(self, resolver):
        result = resolver.resolve(
            {"kind": "absolute_time", "expression": "not a time"},
        )
        assert result is None

    def test_zero_delay_rejected(self, resolver):
        """Zero-second delay should be rejected."""
        result = resolver.resolve(
            {"kind": "delay", "expression": "0 seconds"}, now=1000.0,
        )
        assert result is None

    def test_invalid_hour_13pm(self, resolver):
        """13 PM is invalid in 12-hour format."""
        result = resolver.resolve(
            {"kind": "absolute_time", "expression": "13 PM"},
        )
        assert result is None

    def test_invalid_hour_25(self, resolver):
        """25:00 is invalid."""
        result = resolver.resolve(
            {"kind": "absolute_time", "expression": "25:00"},
        )
        assert result is None

    def test_invalid_minute_61(self, resolver):
        """4:61 PM is invalid."""
        result = resolver.resolve(
            {"kind": "absolute_time", "expression": "4:61 PM"},
        )
        assert result is None


# ─────────────────────────────────────────────────────────────
# Timezone awareness
# ─────────────────────────────────────────────────────────────

class TestTimezoneAwareness:
    def test_get_local_timezone_returns_string(self):
        tz = TemporalResolver.get_local_timezone()
        assert isinstance(tz, str)
        assert len(tz) > 0
