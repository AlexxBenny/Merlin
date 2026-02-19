# runtime/sources/system.py

"""
SystemSource — OS-level system awareness via threshold-based polling.

Emits structured events ONLY when meaningful state changes occur.
All thresholds are config-driven. No hardcoding.

Events emitted:
  cpu_high / cpu_normal          (hysteresis pair)
  memory_pressure / memory_normal (hysteresis pair)
  battery_low / battery_critical / battery_charging / battery_normal
  disk_high
  foreground_window_changed
  idle_detected / idle_ended

Design rules:
  - Diff-based: only emits on actual change
  - Hysteresis: prevents flapping (e.g. cpu_high at 85%, normal at 70%)
  - Source-level polling intervals independent of event loop tick
  - All OS interaction is contained HERE — nothing leaks
"""

import time
import logging
from typing import Any, Dict, List, Optional

from runtime.sources.base import EventSource
from world.timeline import WorldEvent

logger = logging.getLogger(__name__)

# Optional imports — graceful degradation
try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False
    logger.warning("psutil not installed — SystemSource resource polling disabled")

try:
    import pygetwindow as gw
    _HAS_PYGETWINDOW = True
except ImportError:
    _HAS_PYGETWINDOW = False
    logger.warning("pygetwindow not installed — window tracking disabled")


# ─────────────────────────────────────────────────────────────
# Default thresholds (overridden by config/execution.yaml)
# ─────────────────────────────────────────────────────────────

DEFAULT_THRESHOLDS = {
    "cpu_high": 85,
    "cpu_normal": 70,
    "ram_high": 90,
    "ram_normal": 80,
    "battery_low": 20,
    "battery_critical": 10,
    "disk_high": 95,
    "idle_seconds": 300,
}

DEFAULT_INTERVALS = {
    "resource_poll_interval": 2.0,
    "window_poll_interval": 0.5,
    "idle_poll_interval": 5.0,
}


class SystemSource(EventSource):
    """
    Threshold-based, diff-only OS system source.

    Polls psutil + pygetwindow at configurable intervals.
    Emits events only when thresholds are crossed or state changes.
    """

    def __init__(
        self,
        thresholds: Optional[Dict[str, float]] = None,
        intervals: Optional[Dict[str, float]] = None,
        system_controller: Optional[Any] = None,
    ):
        t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
        i = {**DEFAULT_INTERVALS, **(intervals or {})}

        # Thresholds
        self._cpu_high = t["cpu_high"]
        self._cpu_normal = t["cpu_normal"]
        self._ram_high = t["ram_high"]
        self._ram_normal = t["ram_normal"]
        self._battery_low = t["battery_low"]
        self._battery_critical = t["battery_critical"]
        self._disk_high = t["disk_high"]
        self._idle_seconds = t["idle_seconds"]

        # Per-domain poll intervals
        self._resource_interval = i["resource_poll_interval"]
        self._window_interval = i["window_poll_interval"]
        self._idle_interval = i["idle_poll_interval"]

        # Last poll timestamps
        self._last_resource_poll = 0.0
        self._last_window_poll = 0.0
        self._last_idle_poll = 0.0

        # Previous state (for diffing)
        self._cpu_state: Optional[str] = None       # "high" | "normal" | None
        self._ram_state: Optional[str] = None       # "high" | "normal" | None
        self._battery_state: Optional[str] = None   # "critical" | "low" | "charging" | "normal"
        self._disk_alerted: bool = False
        self._fg_window: Optional[str] = None
        self._fg_app: Optional[str] = None
        self._is_idle: bool = False

        # Infrastructure: hardware reads for bootstrap
        self._system_controller = system_controller

    # ─────────────────────────────────────────────────────────────
    # Bootstrap — snapshot-style authoritative initialization
    # ─────────────────────────────────────────────────────────────

    def bootstrap(self) -> List[WorldEvent]:
        """Read current system state and emit single snapshot event.

        Reads hardware (brightness, volume, mute) via SystemController,
        resources (CPU, RAM, battery) via psutil, and foreground window
        via pygetwindow. All values packed into one system_state_snapshot.

        Also initializes diff state so first poll doesn't re-emit.
        """
        now = time.time()
        payload: Dict[str, Any] = {"domain": "system", "severity": "info"}

        # Hardware from SystemController
        if self._system_controller:
            try:
                brightness = self._system_controller.get_brightness()
                if brightness is not None:
                    payload["brightness"] = brightness
            except Exception as e:
                logger.warning("SystemSource bootstrap: brightness read failed: %s", e)

            try:
                vol, muted = self._system_controller.get_volume()
                if vol is not None:
                    payload["volume"] = vol
                if muted is not None:
                    payload["muted"] = muted
            except Exception as e:
                logger.warning("SystemSource bootstrap: volume read failed: %s", e)

        # Resources from psutil
        if _HAS_PSUTIL:
            try:
                cpu = psutil.cpu_percent(interval=0)
                cpu_status = "high" if cpu >= self._cpu_high else "normal"
                payload["cpu"] = round(cpu, 1)
                payload["cpu_status"] = cpu_status
                self._cpu_state = cpu_status
            except Exception as e:
                logger.warning("SystemSource bootstrap: CPU read failed: %s", e)

            try:
                mem = psutil.virtual_memory()
                mem_status = "high" if mem.percent >= self._ram_high else "normal"
                payload["memory"] = round(mem.percent, 1)
                payload["memory_status"] = mem_status
                self._ram_state = mem_status
            except Exception as e:
                logger.warning("SystemSource bootstrap: RAM read failed: %s", e)

            try:
                battery = psutil.sensors_battery()
                if battery is not None:
                    payload["battery_percent"] = round(battery.percent, 1)
                    payload["battery_charging"] = battery.power_plugged
                    if battery.power_plugged:
                        batt_status = "charging"
                    elif battery.percent <= self._battery_critical:
                        batt_status = "critical"
                    elif battery.percent <= self._battery_low:
                        batt_status = "low"
                    else:
                        batt_status = "normal"
                    payload["battery_status"] = batt_status
                    self._battery_state = batt_status
            except Exception as e:
                logger.warning("SystemSource bootstrap: battery read failed: %s", e)

        # Foreground window
        if _HAS_PYGETWINDOW:
            try:
                active = gw.getActiveWindow()
                if active and active.title:
                    title = active.title
                    parts = title.rsplit(" - ", 1)
                    app = parts[-1].strip() if len(parts) > 1 else title.strip()
                    payload["foreground_app"] = app
                    payload["foreground_window"] = title
                    self._fg_window = title
                    self._fg_app = app
            except Exception as e:
                logger.warning("SystemSource bootstrap: window read failed: %s", e)

        return [self._make_event(now, "system_state_snapshot", **payload)]

    # ─────────────────────────────────────────────────────────
    # EventSource contract
    # ─────────────────────────────────────────────────────────

    def poll(self) -> List[WorldEvent]:
        """
        Return new events since last poll.
        Respects per-domain intervals. Non-blocking.
        """
        events: List[WorldEvent] = []
        now = time.time()

        # Resource polling (CPU, RAM, battery, disk)
        if now - self._last_resource_poll >= self._resource_interval:
            self._last_resource_poll = now
            events.extend(self._poll_resources(now))

        # Window polling (foreground app/title)
        if now - self._last_window_poll >= self._window_interval:
            self._last_window_poll = now
            events.extend(self._poll_window(now))

        # Idle polling
        if now - self._last_idle_poll >= self._idle_interval:
            self._last_idle_poll = now
            events.extend(self._poll_idle(now))

        return events

    # ─────────────────────────────────────────────────────────
    # Resource polling (CPU, RAM, battery, disk)
    # ─────────────────────────────────────────────────────────

    def _poll_resources(self, now: float) -> List[WorldEvent]:
        if not _HAS_PSUTIL:
            return []

        events: List[WorldEvent] = []

        # ── CPU ──
        cpu = psutil.cpu_percent(interval=0)  # non-blocking
        events.extend(self._check_cpu(cpu, now))

        # ── RAM ──
        mem = psutil.virtual_memory()
        events.extend(self._check_ram(mem.percent, now))

        # ── Battery ──
        battery = psutil.sensors_battery()
        if battery is not None:
            events.extend(self._check_battery(
                percent=battery.percent,
                plugged=battery.power_plugged,
                now=now,
            ))

        # ── Disk ──
        try:
            disk = psutil.disk_usage("/")
            events.extend(self._check_disk(disk.percent, now))
        except OSError:
            pass

        return events

    def _check_cpu(self, percent: float, now: float) -> List[WorldEvent]:
        """Hysteresis: high at threshold, normal below lower threshold."""
        events: List[WorldEvent] = []

        if percent >= self._cpu_high and self._cpu_state != "high":
            self._cpu_state = "high"
            events.append(self._make_event(
                now, "cpu_high",
                value=round(percent, 1), severity="info",
            ))
        elif percent <= self._cpu_normal and self._cpu_state == "high":
            self._cpu_state = "normal"
            events.append(self._make_event(
                now, "cpu_normal",
                value=round(percent, 1), severity="info",
            ))

        return events

    def _check_ram(self, percent: float, now: float) -> List[WorldEvent]:
        """Hysteresis: high at threshold, normal below lower threshold."""
        events: List[WorldEvent] = []

        if percent >= self._ram_high and self._ram_state != "high":
            self._ram_state = "high"
            events.append(self._make_event(
                now, "memory_pressure",
                value=round(percent, 1), severity="info",
            ))
        elif percent <= self._ram_normal and self._ram_state == "high":
            self._ram_state = "normal"
            events.append(self._make_event(
                now, "memory_normal",
                value=round(percent, 1), severity="info",
            ))

        return events

    def _check_battery(
        self, percent: float, plugged: bool, now: float,
    ) -> List[WorldEvent]:
        """
        Battery state machine:
          charging → plugged in
          critical → ≤ critical threshold
          low → ≤ low threshold
          normal → above low threshold, not plugged
        """
        events: List[WorldEvent] = []

        if plugged:
            new_state = "charging"
        elif percent <= self._battery_critical:
            new_state = "critical"
        elif percent <= self._battery_low:
            new_state = "low"
        else:
            new_state = "normal"

        if new_state != self._battery_state:
            self._battery_state = new_state

            if new_state == "critical":
                events.append(self._make_event(
                    now, "battery_critical",
                    value=round(percent, 1), severity="critical",
                ))
            elif new_state == "low":
                events.append(self._make_event(
                    now, "battery_low",
                    value=round(percent, 1), severity="critical",
                ))
            elif new_state == "charging":
                events.append(self._make_event(
                    now, "battery_charging",
                    value=round(percent, 1), severity="info",
                ))
            # "normal" transition is silent — no event needed

        return events

    def _check_disk(self, percent: float, now: float) -> List[WorldEvent]:
        events: List[WorldEvent] = []

        if percent >= self._disk_high and not self._disk_alerted:
            self._disk_alerted = True
            events.append(self._make_event(
                now, "disk_high",
                value=round(percent, 1), severity="info",
            ))
        elif percent < self._disk_high and self._disk_alerted:
            self._disk_alerted = False

        return events

    # ─────────────────────────────────────────────────────────
    # Window polling
    # ─────────────────────────────────────────────────────────

    def _poll_window(self, now: float) -> List[WorldEvent]:
        if not _HAS_PYGETWINDOW:
            return []

        events: List[WorldEvent] = []

        try:
            active = gw.getActiveWindow()
            if active is None:
                return []

            title = active.title or ""
            # Extract app name from window title (last segment after " - ")
            parts = title.rsplit(" - ", 1)
            app = parts[-1].strip() if len(parts) > 1 else title.strip()

            if title != self._fg_window:
                self._fg_window = title
                self._fg_app = app
                events.append(self._make_event(
                    now, "foreground_window_changed",
                    app=app, window=title, severity="background",
                ))
        except Exception as e:
            # Window detection must never crash — but don't swallow silently
            logger.warning("SystemSource: window poll failed: %s", e)

        return events

    # ─────────────────────────────────────────────────────────
    # Idle polling
    # ─────────────────────────────────────────────────────────

    def _poll_idle(self, now: float) -> List[WorldEvent]:
        """
        Track user idle time via psutil boot_time as fallback.
        Full idle detection via win32api.GetLastInputInfo is
        platform-specific — deferred to Phase 6B.
        """
        # Placeholder: idle detection requires win32api
        # For now, returns empty — wired for future hook
        return []

    # ─────────────────────────────────────────────────────────
    # Event factory
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _make_event(
        timestamp: float,
        event_type: str,
        **payload: Any,
    ) -> WorldEvent:
        return WorldEvent(
            timestamp=timestamp,
            source="system",
            type=event_type,
            payload={"domain": "system", **payload},
        )
