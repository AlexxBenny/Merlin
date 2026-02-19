# runtime/sources/media.py

"""
MediaSource — Media playback state tracker.

Polls OS media session state and emits events on change.
Graceful degradation: returns [] if no media API available.

Events emitted (diff-based, during polling):
  media_started       — playback began
  media_stopped       — playback ended
  media_paused        — playback paused
  media_track_changed — track/title changed while playing
  media_source_changed — media app changed

Events emitted (bootstrap — snapshot-style):
  media_state_snapshot — full current media truth at startup

Events emitted (health transitions):
  media_source_unhealthy — consecutive poll failures exceeded threshold
  media_source_recovered — polls succeeding again after unhealthy
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from runtime.sources.base import EventSource
from world.timeline import WorldEvent


logger = logging.getLogger(__name__)

# Optional: Windows media session via winsdk
_HAS_MEDIA_API = False
try:
    # winsdk is optional — graceful fallback
    from winsdk.windows.media.control import (
        GlobalSystemMediaTransportControlsSessionManager as MediaManager,
    )
    _HAS_MEDIA_API = True
except ImportError:
    logger.info("winsdk not installed — MediaSource using fallback (no media tracking)")


@dataclass
class _MediaSnapshot:
    """Internal snapshot of current media state for diffing."""
    is_playing: bool = False
    title: Optional[str] = None
    artist: Optional[str] = None
    source_app: Optional[str] = None


class MediaSource(EventSource):
    """
    Diff-based media state source with bootstrap and resilience.

    - bootstrap(): reads current OS media state, emits snapshot event
    - poll(): diff-based, emits transition events
    - Failure tracking: counts consecutive failures, emits health events
    """

    def __init__(
        self,
        poll_interval: float = 1.0,
        max_consecutive_failures: int = 5,
    ):
        self._poll_interval = poll_interval
        self._last_poll = 0.0
        self._prev = _MediaSnapshot()

        # Resilience: failure tracking
        self._consecutive_failures = 0
        self._max_failures = max_consecutive_failures
        self._unhealthy = False

    # ─────────────────────────────────────────────────────────
    # Bootstrap — snapshot-style, not fake transitions
    # ─────────────────────────────────────────────────────────

    def bootstrap(self) -> List[WorldEvent]:
        """Read current media state. Emit snapshot event.

        Always emits exactly one media_state_snapshot event.
        Sets has_session=True if a media session is active.
        Sets has_session=False if no session or API unavailable.

        The reducer sets media_known=True for ALL cases,
        ensuring known flag is set even when API is missing.
        """
        current = self._read_media_state()
        now = time.time()

        if current is None:
            # API failed — still emit snapshot so media_known becomes True
            return [self._make_event(now, "media_state_snapshot",
                has_session=False)]

        has_session = current.title is not None or current.is_playing
        return [self._make_event(now, "media_state_snapshot",
            has_session=has_session,
            is_playing=current.is_playing,
            title=current.title,
            artist=current.artist,
            source_app=current.source_app,
        )]

    # ─────────────────────────────────────────────────────────
    # Polling — diff-based with health transitions
    # ─────────────────────────────────────────────────────────

    def poll(self) -> List[WorldEvent]:
        now = time.time()
        if now - self._last_poll < self._poll_interval:
            return []
        self._last_poll = now

        # Already degraded — stop retrying until recovery trigger
        if self._unhealthy:
            return []

        was_unhealthy = self._unhealthy
        current = self._read_media_state()

        events: List[WorldEvent] = []

        # Health: crossed unhealthy threshold
        if (current is None
                and self._consecutive_failures >= self._max_failures
                and not self._unhealthy):
            self._unhealthy = True
            events.append(self._make_event(now, "media_source_unhealthy",
                consecutive_failures=self._consecutive_failures,
                severity="critical"))
            return events

        # Health: recovered from unhealthy
        if was_unhealthy and not self._unhealthy:
            events.append(self._make_event(now, "media_source_recovered",
                severity="info"))

        if current is None:
            return events

        # Normal diff
        events.extend(self._diff(self._prev, current, now))
        self._prev = current
        return events

    # ─────────────────────────────────────────────────────────
    # OS media state reading (with failure counting)
    # ─────────────────────────────────────────────────────────

    def _read_media_state(self) -> Optional[_MediaSnapshot]:
        """Read current media playback state from OS.

        Returns None if API unavailable or read fails.
        Counts consecutive failures. Resets on success.
        """
        if not _HAS_MEDIA_API:
            return _MediaSnapshot()  # No API: empty snapshot, no changes

        try:
            result = self._read_winsdk_media()
            # Success — reset failure counter
            if self._consecutive_failures > 0:
                was_unhealthy = self._unhealthy
                self._consecutive_failures = 0
                if was_unhealthy:
                    self._unhealthy = False
                    # Recovery will be emitted by poll()
            return result
        except Exception as e:
            self._consecutive_failures += 1
            logger.warning(
                "MediaSource: poll failed (%d/%d): %s",
                self._consecutive_failures, self._max_failures, e,
            )
            return None

    def _read_winsdk_media(self) -> _MediaSnapshot:
        """Read media state via Windows SDK (async bridge).

        Raises on failure — caller handles counting.
        """
        import asyncio

        async def _get():
            manager = await MediaManager.request_async()
            session = manager.get_current_session()
            if session is None:
                return _MediaSnapshot()

            info = await session.try_get_media_properties_async()
            playback = session.get_playback_info()

            # Playback status: 4 = Playing, 5 = Paused
            is_playing = (playback.playback_status == 4) if playback else False

            return _MediaSnapshot(
                is_playing=is_playing,
                title=info.title if info else None,
                artist=info.artist if info else None,
                source_app=session.source_app_user_model_id or None,
            )

        try:
            # Background threads (RuntimeEventLoop) have no asyncio loop.
            # Create a dedicated loop for WinRT async calls.
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(_get())
            finally:
                loop.close()
        except Exception as e:
            logger.warning("MediaSource: WinRT async transport failed: %s", e)
            raise  # Let outer handler count the failure

    # ─────────────────────────────────────────────────────────
    # State diffing (unchanged)
    # ─────────────────────────────────────────────────────────

    def _diff(
        self,
        prev: _MediaSnapshot,
        curr: _MediaSnapshot,
        now: float,
    ) -> List[WorldEvent]:
        events: List[WorldEvent] = []

        # Playback state change
        if not prev.is_playing and curr.is_playing:
            events.append(self._make_event(now, "media_started",
                title=curr.title, artist=curr.artist,
                source_app=curr.source_app,
            ))
        elif prev.is_playing and not curr.is_playing:
            events.append(self._make_event(now, "media_stopped",
                title=prev.title, source_app=prev.source_app,
            ))

        # Track changed while playing
        if (curr.is_playing and prev.is_playing
                and curr.title != prev.title
                and curr.title is not None):
            events.append(self._make_event(now, "media_track_changed",
                title=curr.title, artist=curr.artist,
                source_app=curr.source_app,
            ))

        # Source app changed
        if (curr.source_app != prev.source_app
                and curr.source_app is not None
                and prev.source_app is not None):
            events.append(self._make_event(now, "media_source_changed",
                source_app=curr.source_app,
                previous_app=prev.source_app,
            ))

        return events

    @staticmethod
    def _make_event(
        timestamp: float,
        event_type: str,
        **payload: Any,
    ) -> WorldEvent:
        return WorldEvent(
            timestamp=timestamp,
            source="media",
            type=event_type,
            payload={"domain": "media", "severity": "background", **payload},
        )

