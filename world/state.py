# world/state.py

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, ConfigDict, Field

from world.timeline import WorldEvent


class VisibleList(BaseModel):
    """
    Represents ordered, indexable UI lists
    (email inbox, search results, playlists, etc.)
    """
    id: str
    items: List[Dict[str, Any]]


class MediaState(BaseModel):
    platform: Optional[str] = None
    title: Optional[str] = None
    artist: Optional[str] = None
    is_playing: bool = False
    is_ad: bool = False


# ─────────────────────────────────────────────────────────────
# Nested system domain models
# ─────────────────────────────────────────────────────────────

class ResourceState(BaseModel):
    """CPU, RAM, disk — compute resources."""
    cpu_percent: Optional[float] = None
    cpu_status: Optional[str] = None           # "high" | "normal"
    memory_percent: Optional[float] = None
    memory_status: Optional[str] = None        # "high" | "normal"
    disk_percent: Optional[float] = None
    # Future: gpu_percent, network_bandwidth, thermal


class HardwareState(BaseModel):
    """Physical hardware state — battery, peripherals, display, audio."""
    battery_percent: Optional[float] = None
    battery_charging: Optional[bool] = None
    battery_status: Optional[str] = None       # "critical" | "low" | "charging" | "normal"
    brightness_percent: Optional[int] = None
    volume_percent: Optional[int] = None
    muted: Optional[bool] = None
    nightlight_enabled: Optional[bool] = None


class SessionState(BaseModel):
    """User session — foreground app, idle, open apps."""
    foreground_app: Optional[str] = None
    foreground_window: Optional[str] = None
    idle_seconds: Optional[float] = None
    open_apps: List[str] = Field(default_factory=list)
    # Future: open_windows: List[WindowInfo]


class SystemState(BaseModel):
    """
    Aggregated system state — nested by concern.
    Non-optional: always exists, sub-models default to empty.
    """
    resources: ResourceState = Field(default_factory=ResourceState)
    hardware: HardwareState = Field(default_factory=HardwareState)
    session: SessionState = Field(default_factory=SessionState)


class TimeState(BaseModel):
    """
    Current clock state. Updated by TimeSource events.
    """
    hour: Optional[int] = None
    minute: Optional[int] = None
    day_of_week: Optional[str] = None
    date: Optional[str] = None


class WorldState(BaseModel):
    """
    Deterministic projection of the world at a point in time.

    Validity flags (known flags):
    - media_known: True after bootstrap reads media state. If False,
      media=None means 'not yet polled', not 'no session'.
    - time_known: True after bootstrap reads clock.

    Semantic contract:
    - media is None AND NOT media_known → pre-bootstrap (unknown)
    - media is not None AND media_known → authoritative
    - media.is_playing / media.title → real OS values
    """

    active_app: Optional[str] = None
    active_window: Optional[str] = None
    cwd: Optional[str] = None  # Current working directory (backs WORKSPACE anchor)

    visible_lists: Dict[str, VisibleList] = Field(default_factory=dict)
    media: Optional[MediaState] = None
    media_known: bool = False

    # Structured domain sub-models (non-optional)
    system: SystemState = Field(default_factory=SystemState)
    time: Optional[TimeState] = None
    time_known: bool = False

    last_user_focus: Optional[str] = None

    model_config = ConfigDict(extra="forbid")

    @staticmethod
    def from_events(events: List[WorldEvent]) -> "WorldState":
        """
        Pure reducer: events → state
        """
        state = WorldState()

        for event in events:
            t = event.type
            p = event.payload

            # ── App / Window ──
            if t == "app_focused":
                state.active_app = p.get("app")
                state.active_window = p.get("window")

            elif t == "list_rendered":
                state.visible_lists[p["list_id"]] = VisibleList(
                    id=p["list_id"],
                    items=p.get("items", []),
                )

            # ── Media ──
            elif t == "media_started":
                state.media = MediaState(
                    platform=p.get("platform") or p.get("source_app"),
                    title=p.get("title"),
                    artist=p.get("artist"),
                    is_playing=True,
                    is_ad=p.get("is_ad", False),
                )

            elif t == "media_stopped":
                if state.media:
                    state.media.is_playing = False

            elif t == "media_track_changed":
                if state.media:
                    state.media.title = p.get("title")
                    state.media.artist = p.get("artist")
                else:
                    state.media = MediaState(
                        platform=p.get("source_app"),
                        title=p.get("title"),
                        artist=p.get("artist"),
                        is_playing=True,
                    )

            elif t == "ad_detected":
                if state.media:
                    state.media.is_ad = True

            # ── System: Resources ──
            elif t == "cpu_high":
                state.system.resources.cpu_percent = p.get("value")
                state.system.resources.cpu_status = "high"

            elif t == "cpu_normal":
                state.system.resources.cpu_percent = p.get("value")
                state.system.resources.cpu_status = "normal"

            elif t == "memory_pressure":
                state.system.resources.memory_percent = p.get("value")
                state.system.resources.memory_status = "high"

            elif t == "memory_normal":
                state.system.resources.memory_percent = p.get("value")
                state.system.resources.memory_status = "normal"

            elif t == "disk_high":
                state.system.resources.disk_percent = p.get("value")

            # ── System: Hardware ──
            elif t == "battery_low":
                state.system.hardware.battery_percent = p.get("value")
                state.system.hardware.battery_charging = False
                state.system.hardware.battery_status = "low"

            elif t == "battery_critical":
                state.system.hardware.battery_percent = p.get("value")
                state.system.hardware.battery_charging = False
                state.system.hardware.battery_status = "critical"

            elif t == "battery_charging":
                state.system.hardware.battery_percent = p.get("value")
                state.system.hardware.battery_charging = True
                state.system.hardware.battery_status = "charging"

            # ── System: Hardware actuation ──
            elif t == "brightness_changed":
                state.system.hardware.brightness_percent = p.get("value")

            elif t == "volume_changed":
                state.system.hardware.volume_percent = p.get("value")

            elif t == "mute_toggled":
                state.system.hardware.muted = p.get("muted")

            elif t == "nightlight_toggled":
                state.system.hardware.nightlight_enabled = p.get("enabled")

            # ── System: Session ──
            elif t == "foreground_window_changed":
                state.active_app = p.get("app")
                state.active_window = p.get("window")
                state.system.session.foreground_app = p.get("app")
                state.system.session.foreground_window = p.get("window")

            elif t == "idle_detected":
                state.system.session.idle_seconds = p.get("seconds")

            elif t == "idle_ended":
                state.system.session.idle_seconds = 0.0

            # ── System: App lifecycle (actuation tracking) ──
            elif t == "app_launched":
                app = p.get("app")
                if app and app not in state.system.session.open_apps:
                    state.system.session.open_apps.append(app)

            elif t == "app_closed":
                app = p.get("app")
                if app and app in state.system.session.open_apps:
                    state.system.session.open_apps.remove(app)

            # ── Time ──
            elif t == "time_tick":
                state.time_known = True
                state.time = TimeState(
                    hour=p.get("hour"),
                    minute=p.get("minute"),
                    day_of_week=p.get("day_of_week"),
                    date=p.get("date"),
                )

            elif t == "hour_changed":
                state.time_known = True
                state.time = state.time or TimeState()
                state.time.hour = p.get("hour")

            elif t == "date_changed":
                state.time_known = True
                state.time = state.time or TimeState()
                state.time.date = p.get("date")
                state.time.day_of_week = p.get("day_of_week")

            # ── User focus ──
            elif t == "user_focus_changed":
                state.last_user_focus = p.get("target")

            # ── Bootstrap snapshot events ──
            elif t == "media_state_snapshot":
                state.media_known = True
                if p.get("has_session"):
                    state.media = MediaState(
                        platform=p.get("source_app"),
                        title=p.get("title"),
                        artist=p.get("artist"),
                        is_playing=p.get("is_playing", False),
                        is_ad=p.get("is_ad", False),
                    )
                else:
                    # No session — explicit empty, not None
                    state.media = MediaState()

            elif t == "system_state_snapshot":
                if p.get("brightness") is not None:
                    state.system.hardware.brightness_percent = p["brightness"]
                if p.get("volume") is not None:
                    state.system.hardware.volume_percent = p["volume"]
                if p.get("muted") is not None:
                    state.system.hardware.muted = p["muted"]
                if p.get("cpu") is not None:
                    state.system.resources.cpu_percent = p["cpu"]
                    state.system.resources.cpu_status = p.get("cpu_status", "normal")
                if p.get("memory") is not None:
                    state.system.resources.memory_percent = p["memory"]
                    state.system.resources.memory_status = p.get("memory_status", "normal")
                if p.get("battery_percent") is not None:
                    state.system.hardware.battery_percent = p["battery_percent"]
                    state.system.hardware.battery_charging = p.get("battery_charging")
                    state.system.hardware.battery_status = p.get("battery_status")
                if p.get("foreground_app") is not None:
                    state.active_app = p["foreground_app"]
                    state.active_window = p.get("foreground_window")
                    state.system.session.foreground_app = p["foreground_app"]
                    state.system.session.foreground_window = p.get("foreground_window")

        return state
