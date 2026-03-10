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
import random
import logging
from typing import Any, Dict, List, Optional, Set

from runtime.sources.base import EventSource
from world.timeline import WorldEvent, WorldTimeline

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

try:
    import win32gui
    import win32process
    _HAS_WIN32 = True
except ImportError:
    _HAS_WIN32 = False
    logger.info("win32gui/win32process not installed — process-based window ID disabled")

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
    "process_poll_interval": 5.0,    # App lifecycle validation
    "refresh_interval": 30.0,       # Full state refresh (absolute values)
}

# Grace period: don't declare a process dead within this window after launch
PROCESS_LAUNCH_GRACE_SECONDS = 5.0


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
        timeline: Optional[WorldTimeline] = None,
        app_registry: Optional[Any] = None,
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
        self._process_interval = i["process_poll_interval"]
        self._refresh_interval = i["refresh_interval"]

        # Last poll timestamps
        self._last_resource_poll = 0.0
        self._last_window_poll = 0.0
        self._last_idle_poll = 0.0
        self._last_process_poll = 0.0
        self._last_refresh_poll = 0.0

        # Previous state (for diffing)
        self._cpu_state: Optional[str] = None       # "high" | "normal" | None
        self._ram_state: Optional[str] = None       # "high" | "normal" | None
        self._battery_state: Optional[str] = None   # "critical" | "low" | "charging" | "normal"
        self._disk_alerted: bool = False
        self._fg_window: Optional[str] = None
        self._fg_app: Optional[str] = None
        self._is_idle: bool = False

        # Process tracking state
        # Populated from WorldState snapshot (O(active_apps), not O(events)).
        # Dict: app_key -> {pid, launch_time}
        self._tracked_pids: Dict[str, Dict[str, Any]] = {}

        # Canonical process name cache: app_id -> [name1, name2, ...]
        # Built once, avoids O(entities) lookup on every poll tick
        self._canonical_names_cache: Dict[str, List[str]] = {}
        self._canonical_cache_built = False

        # Infrastructure references
        self._system_controller = system_controller
        self._timeline = timeline
        self._app_registry = app_registry

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

        # Foreground window — process-based detection
        fg_info = self._read_foreground_process()
        if fg_info:
            payload["foreground_app"] = fg_info["app"]
            payload["foreground_window"] = fg_info["window"]
            self._fg_window = fg_info["window"]
            self._fg_app = fg_info["app"]

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

        # Process lifecycle validation (app liveness)
        # Jitter ±0.5s prevents CPU spike synchronization across subsystems
        jittered_interval = self._process_interval + random.uniform(-0.5, 0.5)
        if now - self._last_process_poll >= jittered_interval:
            self._last_process_poll = now
            events.extend(self._poll_processes(now))

        # Periodic full state refresh (absolute values, not thresholds)
        if now - self._last_refresh_poll >= self._refresh_interval:
            self._last_refresh_poll = now
            events.extend(self._poll_refresh(now))

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
        events: List[WorldEvent] = []

        try:
            fg_info = self._read_foreground_process()
            if fg_info is None:
                return []

            app = fg_info["app"]
            title = fg_info["window"]

            if title != self._fg_window:
                self._fg_window = title
                self._fg_app = app
                events.append(self._make_event(
                    now, "foreground_window_changed",
                    app=app, window=title, severity="background",
                ))
        except Exception as e:
            # Window detection must never crash
            logger.warning("SystemSource: window poll failed: %s", e)

        return events

    def _read_foreground_process(self) -> Optional[Dict[str, str]]:
        """Read foreground window info using process-based detection.

        Returns {app: process_name, window: title} or None.
        Uses win32gui + psutil for deterministic process identification.
        Falls back to title parsing only if win32 is unavailable.

        Handles NoSuchProcess, AccessDenied, ZombieProcess gracefully
        (window may close between GetForegroundWindow and Process()).
        """
        # Primary: process-based detection (deterministic)
        if _HAS_WIN32 and _HAS_PSUTIL:
            try:
                hwnd = win32gui.GetForegroundWindow()
                if not hwnd:
                    return None
                title = win32gui.GetWindowText(hwnd)
                if not title:
                    return None
                _, pid = win32process.GetWindowThreadProcessId(hwnd)
                proc = psutil.Process(pid)
                return {"app": proc.name(), "window": title}
            except (psutil.NoSuchProcess, psutil.AccessDenied,
                    psutil.ZombieProcess):
                # Window closed between API calls — not an error
                return None
            except Exception:
                return None

        # Fallback: pygetwindow + title parsing (legacy, less reliable)
        if _HAS_PYGETWINDOW:
            try:
                active = gw.getActiveWindow()
                if active and active.title:
                    title = active.title
                    parts = title.rsplit(" - ", 1)
                    app = parts[-1].strip() if len(parts) > 1 else title.strip()
                    return {"app": app, "window": title}
            except Exception:
                pass

        return None

    # ─────────────────────────────────────────────────────────
    # Process lifecycle polling
    # ─────────────────────────────────────────────────────────

    def _poll_processes(self, now: float) -> List[WorldEvent]:
        """Validate liveness of tracked application processes.

        Scans timeline for app_launched events to discover tracked PIDs.
        For each tracked PID:
        - If alive → emit app_heartbeat
        - If dead AND past grace period → emit process_stopped
        - If dead AND in grace → skip (allow time for process replacement)

        Uses canonical_process_names from ApplicationRegistry to detect
        process replacement (child PID inherits parent's app identity).
        """
        if not _HAS_PSUTIL:
            return []

        events: List[WorldEvent] = []

        # Discover new tracked PIDs from timeline events
        self._refresh_tracked_pids()

        # Get canonical process names for ancestry matching
        canonical_names = self._get_canonical_names()

        dead_keys: List[str] = []
        for app_key, info in self._tracked_pids.items():
            pid = info["pid"]
            launch_time = info["launch_time"]

            if pid is None:
                continue

            alive = psutil.pid_exists(pid)

            if alive:
                events.append(self._make_event(
                    now, "app_heartbeat",
                    app=app_key, pid=pid, severity="background",
                ))
            else:
                # Process gone — check for replacement via ancestry
                replacement_pid = self._find_replacement_process(
                    app_key, canonical_names,
                )
                if replacement_pid:
                    # Process replaced (launcher exited, child took over)
                    info["pid"] = replacement_pid
                    events.append(self._make_event(
                        now, "app_heartbeat",
                        app=app_key, pid=replacement_pid,
                        severity="background",
                    ))
                elif (now - launch_time) < PROCESS_LAUNCH_GRACE_SECONDS:
                    # Still in grace period — skip
                    logger.debug(
                        "Process %d (%s) gone but in grace period, skipping",
                        pid, app_key,
                    )
                else:
                    # Process truly dead
                    events.append(self._make_event(
                        now, "process_stopped",
                        app=app_key, pid=pid, severity="info",
                    ))
                    dead_keys.append(app_key)

        # Remove dead entries from tracking
        for key in dead_keys:
            self._tracked_pids.pop(key, None)

        return events

    def _refresh_tracked_pids(self) -> None:
        """Sync tracked PIDs from WorldState snapshot.

        Uses O(active_apps) WorldState.tracked_apps instead of
        scanning the full O(total_events) timeline.
        """
        if not self._timeline:
            return

        from world.state import WorldState
        ws = WorldState.from_events(self._timeline.all_events())
        snapshot_apps = ws.system.session.tracked_apps

        # Add newly tracked apps
        for key, app_state in snapshot_apps.items():
            if app_state.running and key not in self._tracked_pids:
                self._tracked_pids[key] = {
                    "pid": app_state.pid,
                    "launch_time": app_state.launch_time,
                }

        # Remove apps no longer tracked or no longer running
        dead_keys = [
            k for k in self._tracked_pids
            if k not in snapshot_apps or not snapshot_apps[k].running
        ]
        for k in dead_keys:
            self._tracked_pids.pop(k, None)

    def _get_canonical_names(self) -> Dict[str, List[str]]:
        """Get canonical process names from ApplicationRegistry (cached)."""
        if self._canonical_cache_built:
            return self._canonical_names_cache

        if not self._app_registry:
            return {}

        try:
            for entity in self._app_registry.all_entities():
                names = getattr(entity, 'canonical_process_names', [])
                if names:
                    self._canonical_names_cache[entity.app_id.lower()] = [
                        n.lower() for n in names
                    ]
            self._canonical_cache_built = True
        except Exception:
            pass

        return self._canonical_names_cache

    def _find_replacement_process(
        self,
        app_key: str,
        canonical_names: Dict[str, List[str]],
    ) -> Optional[int]:
        """Find a replacement process matching the app's canonical names.

        Handles process replacement (launcher exits, child inherits).
        Only matches processes whose name matches canonical_process_names.
        """
        if not _HAS_PSUTIL:
            return None

        names = canonical_names.get(app_key)
        if not names:
            # Fallback: use app_key as process name heuristic
            names = [app_key]

        try:
            for proc in psutil.process_iter(['name', 'pid']):
                pname = (proc.info.get('name') or '').lower()
                # Match against any canonical name
                for cname in names:
                    if cname in pname:
                        return proc.info['pid']
        except Exception:
            pass

        return None

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
    # Periodic full-state refresh
    # ─────────────────────────────────────────────────────────

    def _poll_refresh(self, now: float) -> List[WorldEvent]:
        """
        Periodic full-state refresh — re-reads all absolute sensor values.

        Unlike threshold-based events (cpu_high, battery_low), this emits
        current read values unconditionally so WorldState never carries
        stale bootstrap-era data.

        This event is IGNORED by NotificationPolicy (silent update).
        """
        if not _HAS_PSUTIL:
            return []

        payload: Dict[str, Any] = {
            "domain": "system",
            "severity": "background",
        }

        try:
            payload["cpu"] = round(psutil.cpu_percent(interval=0), 1)
        except Exception:
            pass

        try:
            mem = psutil.virtual_memory()
            payload["memory"] = round(mem.percent, 1)
        except Exception:
            pass

        try:
            battery = psutil.sensors_battery()
            if battery is not None:
                payload["battery_percent"] = round(battery.percent, 1)
                payload["battery_charging"] = battery.power_plugged
        except Exception:
            pass

        try:
            disk = psutil.disk_usage("/")
            payload["disk"] = round(disk.percent, 1)
        except Exception:
            pass

        return [self._make_event(now, "system_state_refresh", **payload)]

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
