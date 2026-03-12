# runtime/sources/browser.py

"""
BrowserSource — Proactive browser state polling.

Follows the SystemSource pattern: configurable polling interval,
diff-based event emission, no event unless state changes.

Events emitted:
  browser_state_snapshot      (bootstrap — initial state)
  browser_page_changed        (poll — URL or title changed)
  browser_disconnected        (poll — browser connection lost)

Design rules:
  - poll_interval = 30s (not aggressive — avoids anti-automation heuristics)
  - Controller actions already emit fresh state; source catches manual changes
  - Diff-based: only emits when URL or title changes
  - Graceful degradation: swallows all controller exceptions
"""

import logging
import time
from typing import Any, Dict, List, Optional

from runtime.sources.base import EventSource
from world.timeline import WorldEvent

logger = logging.getLogger(__name__)


class BrowserSource(EventSource):
    """Proactive browser state polling source.

    Polls the BrowserController at configurable intervals and emits
    events when the browser state changes (URL, title, tabs).

    Only active when a browser connection exists.
    """

    def __init__(
        self,
        browser_controller=None,
        poll_interval: float = 30.0,
    ):
        self._controller = browser_controller
        self._poll_interval = poll_interval
        self._last_poll = 0.0

        # Previous state for diffing
        self._last_url: Optional[str] = None
        self._last_title: Optional[str] = None
        self._last_tab_count: int = 0
        self._last_entity_count: int = 0
        self._was_alive: bool = False

    def bootstrap(self) -> List[WorldEvent]:
        """Read initial browser state if browser is alive."""
        if not self._controller:
            return []

        try:
            if not self._controller.is_alive():
                return []

            snapshot = self._controller.get_snapshot(cached=False)
            self._last_url = snapshot.url
            self._last_title = snapshot.title
            self._last_tab_count = snapshot.tab_count
            self._last_entity_count = len(snapshot.entities)
            self._was_alive = True

            return [self._make_event(
                time.time(), "browser_state_snapshot",
                url=snapshot.url,
                title=snapshot.title,
                entity_count=len(snapshot.entities),
                tab_count=snapshot.tab_count,
                active_tab_id=(
                    snapshot.tabs[0].tab_id if snapshot.tabs else None
                ),
                tab_urls=[t.url for t in snapshot.tabs],
                top_entities=self._extract_top_entities(snapshot),
            )]

        except Exception as e:
            logger.debug("BrowserSource bootstrap failed: %s", e)
            return []

    def poll(self) -> List[WorldEvent]:
        """Poll browser state. Emit events on change."""
        now = time.time()
        if now - self._last_poll < self._poll_interval:
            return []
        self._last_poll = now

        if not self._controller:
            return []

        events: List[WorldEvent] = []

        # Check liveness
        try:
            alive = self._controller.is_alive()
        except Exception:
            alive = False

        if not alive:
            if self._was_alive:
                # Browser disconnected
                self._was_alive = False
                events.append(self._make_event(
                    now, "browser_disconnected",
                    prev_url=self._last_url,
                    severity="info",
                ))
                self._last_url = None
                self._last_title = None
                self._last_tab_count = 0
            return events

        if not self._was_alive:
            self._was_alive = True

        # Read current state
        try:
            snapshot = self._controller.get_snapshot(cached=False)
        except Exception as e:
            logger.debug("BrowserSource poll failed: %s", e)
            return events

        # Diff-based: emit only on change
        url_changed = snapshot.url != self._last_url
        title_changed = snapshot.title != self._last_title
        tab_changed = snapshot.tab_count != self._last_tab_count
        entity_count = len(snapshot.entities)
        entity_changed = entity_count != self._last_entity_count

        if url_changed or title_changed or tab_changed:
            events.append(self._make_event(
                now, "browser_page_changed",
                url=snapshot.url,
                title=snapshot.title,
                prev_url=self._last_url,
                entity_count=entity_count,
                tab_count=snapshot.tab_count,
                active_tab_id=(
                    snapshot.tabs[0].tab_id if snapshot.tabs else None
                ),
                tab_urls=[t.url for t in snapshot.tabs],
                top_entities=self._extract_top_entities(snapshot),
            ))
        elif entity_changed:
            # Entity count changed without URL change (scroll, dynamic load)
            # Emit entity refresh so WorldState.browser.top_entities is updated
            events.append(self._make_event(
                now, "browser_entities_refreshed",
                url=snapshot.url,
                title=snapshot.title,
                entity_count=entity_count,
                tab_count=snapshot.tab_count,
                active_tab_id=(
                    snapshot.tabs[0].tab_id if snapshot.tabs else None
                ),
                tab_urls=[t.url for t in snapshot.tabs],
                top_entities=self._extract_top_entities(snapshot),
            ))

        self._last_url = snapshot.url
        self._last_title = snapshot.title
        self._last_tab_count = snapshot.tab_count
        self._last_entity_count = entity_count

        return events

    @staticmethod
    def _extract_top_entities(snapshot, limit: int = 10) -> list:
        """Extract top N entities as lightweight dicts for WorldState.

        Returns [{index, type, text}, ...] capped at `limit`.
        These propagate through WorldState → coordinator prompt
        via _extract_world_facts() model_dump().
        """
        entities = []
        for e in snapshot.entities[:limit]:
            entities.append({
                "index": e.index,
                "type": e.entity_type,
                "text": (e.text[:60] + "...") if len(e.text) > 60 else e.text,
            })
        return entities

    @staticmethod
    def _make_event(
        timestamp: float,
        event_type: str,
        **payload: Any,
    ) -> WorldEvent:
        return WorldEvent(
            timestamp=timestamp,
            source="browser",
            type=event_type,
            payload={"domain": "browser", **payload},
        )
