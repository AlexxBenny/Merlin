# tests/test_browser_context_propagation.py

"""
Tests for browser context propagation — Phase 1.

Covers:
1. BrowserWorldState.top_entities field (model layer)
2. BrowserSource._extract_top_entities (extraction logic)
3. WorldState.from_events — browser event → top_entities storage
4. Coordinator prompt visibility (entities in _extract_world_facts)
5. Autonomous task context injection
"""

import pytest
import time
from dataclasses import dataclass
from typing import Optional
from unittest.mock import MagicMock, patch, PropertyMock

from world.state import WorldState, BrowserWorldState
from world.timeline import WorldEvent, WorldTimeline
from world.snapshot import WorldSnapshot
from runtime.sources.browser import BrowserSource


# ─────────────────────────────────────────────────────────────
# Fixtures — mock DOMEntity and PageSnapshot
# ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MockDOMEntity:
    """Minimal DOMEntity mock for testing."""
    index: int
    backend_node_id: int
    entity_type: str
    text: str
    url: Optional[str] = None
    ax_role: Optional[str] = None


@dataclass(frozen=True)
class MockTabInfo:
    tab_id: str
    url: str
    title: str


@dataclass(frozen=True)
class MockPageSnapshot:
    """Minimal PageSnapshot mock for testing."""
    snapshot_id: str
    url: str
    title: str
    entities: tuple
    entity_count: int
    tab_count: int
    tabs: tuple = ()
    scroll_pct: Optional[float] = None
    timestamp: float = 0.0


def _sample_entities(n=5):
    """Generate n sample DOMEntity mocks."""
    types = ["link", "button", "input", "link", "link",
             "button", "link", "link", "input", "link",
             "link", "link"]
    texts = [
        "iPhone 14 Pro", "Add to Cart", "Search Amazon",
        "MacBook Air M3", "Samsung Galaxy S24",
        "Buy Now", "Dell XPS 15", "Lenovo ThinkPad",
        "Filter Results", "HP Spectre x360",
        "ASUS ROG", "Acer Swift Go",
    ]
    return tuple(
        MockDOMEntity(
            index=i + 1,
            backend_node_id=1000 + i,
            entity_type=types[i % len(types)],
            text=texts[i % len(texts)],
        )
        for i in range(n)
    )


def _sample_snapshot(entities=None, url="https://amazon.com/s?k=iphone",
                     title="Amazon.com: iphone"):
    """Build a mock PageSnapshot."""
    if entities is None:
        entities = _sample_entities()
    return MockPageSnapshot(
        snapshot_id="abc123",
        url=url,
        title=title,
        entities=entities,
        entity_count=len(entities),
        tab_count=1,
        tabs=(MockTabInfo("tab1", url, title),),
    )


# ─────────────────────────────────────────────────────────────
# 1. BrowserWorldState model tests
# ─────────────────────────────────────────────────────────────

class TestBrowserWorldStateModel:
    """Test BrowserWorldState includes top_entities field."""

    def test_default_empty_entities(self):
        """top_entities defaults to empty list."""
        state = BrowserWorldState()
        assert state.top_entities == []
        assert state.active is False

    def test_stores_entities(self):
        """top_entities accepts list of dicts."""
        entities = [
            {"index": 1, "type": "link", "text": "iPhone 14"},
            {"index": 2, "type": "button", "text": "Buy Now"},
        ]
        state = BrowserWorldState(
            active=True,
            url="https://amazon.com",
            title="Amazon",
            entity_count=2,
            top_entities=entities,
        )
        assert len(state.top_entities) == 2
        assert state.top_entities[0]["index"] == 1
        assert state.top_entities[1]["type"] == "button"

    def test_model_dump_includes_entities(self):
        """model_dump() must include top_entities — this is how
        _extract_world_facts() propagates them to the coordinator."""
        entities = [{"index": 1, "type": "link", "text": "Result 1"}]
        state = BrowserWorldState(
            active=True,
            url="https://test.com",
            top_entities=entities,
        )
        dumped = state.model_dump(exclude_none=True)
        assert "top_entities" in dumped
        assert dumped["top_entities"] == entities

    def test_empty_entities_in_dump(self):
        """Empty top_entities still serializes (not excluded by exclude_none)."""
        state = BrowserWorldState(active=True, url="https://test.com")
        dumped = state.model_dump(exclude_none=True)
        assert "top_entities" in dumped
        assert dumped["top_entities"] == []

    def test_disconnected_clears_entities(self):
        """active=False state should have empty entities."""
        state = BrowserWorldState(active=False)
        assert state.top_entities == []
        assert state.entity_count == 0


# ─────────────────────────────────────────────────────────────
# 2. BrowserSource._extract_top_entities tests
# ─────────────────────────────────────────────────────────────

class TestExtractTopEntities:
    """Test BrowserSource entity extraction logic."""

    def test_extracts_correct_fields(self):
        """Each entity dict has index, type, text."""
        snapshot = _sample_snapshot(entities=_sample_entities(3))
        result = BrowserSource._extract_top_entities(snapshot)

        assert len(result) == 3
        for e in result:
            assert "index" in e
            assert "type" in e
            assert "text" in e
        assert result[0]["index"] == 1
        assert result[0]["type"] == "link"
        assert result[0]["text"] == "iPhone 14 Pro"

    def test_caps_at_25(self):
        """Should return at most 25 entities even if more exist."""
        snapshot = _sample_snapshot(entities=_sample_entities(30))
        result = BrowserSource._extract_top_entities(snapshot)
        assert len(result) == 25

    def test_custom_limit(self):
        """Custom limit is respected."""
        snapshot = _sample_snapshot(entities=_sample_entities(12))
        result = BrowserSource._extract_top_entities(snapshot, limit=5)
        assert len(result) == 5

    def test_empty_entities(self):
        """No entities → empty list."""
        snapshot = _sample_snapshot(entities=())
        result = BrowserSource._extract_top_entities(snapshot)
        assert result == []

    def test_long_text_truncated(self):
        """Entity text longer than 60 chars is truncated with '...'."""
        long_text = "A" * 100
        entities = (MockDOMEntity(
            index=1, backend_node_id=1000,
            entity_type="link", text=long_text,
        ),)
        snapshot = _sample_snapshot(entities=entities)
        result = BrowserSource._extract_top_entities(snapshot)

        assert len(result) == 1
        assert result[0]["text"].endswith("...")
        assert len(result[0]["text"]) == 63  # 60 + "..."

    def test_short_text_not_truncated(self):
        """Text <= 60 chars is NOT truncated."""
        text = "Short title"
        entities = (MockDOMEntity(
            index=1, backend_node_id=1000,
            entity_type="button", text=text,
        ),)
        snapshot = _sample_snapshot(entities=entities)
        result = BrowserSource._extract_top_entities(snapshot)
        assert result[0]["text"] == text


# ─────────────────────────────────────────────────────────────
# 3. WorldState.from_events — browser entity storage
# ─────────────────────────────────────────────────────────────

class TestWorldStateEntityStorage:
    """Test that WorldState.from_events stores top_entities from browser events."""

    def _browser_event(self, event_type="browser_state_snapshot",
                       entities=None, **extra):
        """Build a browser WorldEvent with optional entities."""
        payload = {
            "domain": "browser",
            "url": "https://amazon.com/s?k=iphone",
            "title": "Amazon: iphone",
            "entity_count": 5,
            "tab_count": 1,
            "active_tab_id": "tab1",
            "tab_urls": ["https://amazon.com/s?k=iphone"],
            "top_entities": entities or [],
            **extra,
        }
        return WorldEvent(
            timestamp=time.time(),
            source="browser",
            type=event_type,
            payload=payload,
        )

    def test_snapshot_event_stores_entities(self):
        """browser_state_snapshot event stores top_entities in WorldState."""
        entities = [
            {"index": 1, "type": "link", "text": "iPhone 14"},
            {"index": 2, "type": "button", "text": "Add to Cart"},
        ]
        event = self._browser_event(
            "browser_state_snapshot", entities=entities,
        )
        state = WorldState.from_events([event])

        assert state.browser.active is True
        assert len(state.browser.top_entities) == 2
        assert state.browser.top_entities[0]["text"] == "iPhone 14"
        assert state.browser.top_entities[1]["type"] == "button"

    def test_page_changed_event_stores_entities(self):
        """browser_page_changed also stores top_entities."""
        entities = [
            {"index": 1, "type": "link", "text": "MacBook Air"},
        ]
        event = self._browser_event(
            "browser_page_changed", entities=entities,
        )
        state = WorldState.from_events([event])

        assert state.browser.active is True
        assert len(state.browser.top_entities) == 1
        assert state.browser.top_entities[0]["text"] == "MacBook Air"

    def test_no_entities_key_defaults_empty(self):
        """If event payload has no top_entities key, defaults to []."""
        event = WorldEvent(
            timestamp=time.time(),
            source="browser",
            type="browser_state_snapshot",
            payload={
                "domain": "browser",
                "url": "https://test.com",
                "title": "Test",
                "entity_count": 0,
                "tab_count": 1,
                # No top_entities key
            },
        )
        state = WorldState.from_events([event])
        assert state.browser.active is True
        assert state.browser.top_entities == []

    def test_disconnect_clears_entities(self):
        """browser_disconnected clears all browser state including entities."""
        # First: set up a browser with entities
        entities = [{"index": 1, "type": "link", "text": "Test"}]
        snapshot_event = self._browser_event(
            "browser_state_snapshot", entities=entities,
        )
        disconnect_event = WorldEvent(
            timestamp=time.time(),
            source="browser",
            type="browser_disconnected",
            payload={"domain": "browser", "prev_url": "https://test.com"},
        )
        state = WorldState.from_events([snapshot_event, disconnect_event])

        assert state.browser.active is False
        assert state.browser.top_entities == []

    def test_page_change_replaces_entities(self):
        """When page changes, entities from old page are replaced."""
        old_entities = [{"index": 1, "type": "link", "text": "Old Link"}]
        new_entities = [
            {"index": 1, "type": "button", "text": "New Button"},
            {"index": 2, "type": "input", "text": "Search"},
        ]
        events = [
            self._browser_event(
                "browser_state_snapshot", entities=old_entities,
            ),
            self._browser_event(
                "browser_page_changed", entities=new_entities,
                url="https://youtube.com", title="YouTube",
            ),
        ]
        state = WorldState.from_events(events)

        assert state.browser.url == "https://youtube.com"
        assert len(state.browser.top_entities) == 2
        assert state.browser.top_entities[0]["text"] == "New Button"


# ─────────────────────────────────────────────────────────────
# 4. Coordinator visibility — entities in world_facts
# ─────────────────────────────────────────────────────────────

class TestCoordinatorEntityVisibility:
    """Verify that entities appear in the coordinator's world_facts."""

    def test_entities_in_flattened_world_facts(self):
        """_extract_world_facts should include top_entities when present."""
        from cortex.cognitive_coordinator import LLMCognitiveCoordinator

        state = WorldState(
            browser=BrowserWorldState(
                active=True,
                url="https://amazon.com",
                title="Amazon",
                entity_count=2,
                top_entities=[
                    {"index": 1, "type": "link", "text": "iPhone 14"},
                    {"index": 2, "type": "button", "text": "Buy Now"},
                ],
            ),
        )
        snapshot = WorldSnapshot.build(state, [])
        facts = LLMCognitiveCoordinator._extract_world_facts(snapshot)

        # The flattened output must contain entity data
        assert "top_entities" in facts
        assert "iPhone 14" in facts or "top_entities" in facts

    def test_no_entities_when_browser_inactive(self):
        """No browser → no browser section in world_facts."""
        from cortex.cognitive_coordinator import LLMCognitiveCoordinator

        state = WorldState()  # No browser
        snapshot = WorldSnapshot.build(state, [])
        facts = LLMCognitiveCoordinator._extract_world_facts(snapshot)

        assert "top_entities" not in facts


# ─────────────────────────────────────────────────────────────
# 5. Autonomous task context injection
# ─────────────────────────────────────────────────────────────

class TestAutonomousTaskContextInjection:
    """Test that autonomous_task prepends browser context to agent task."""

    def _make_skill(self, controller=None, adapter=None):
        """Create a BrowserAutonomousTaskSkill with mocked dependencies."""
        from skills.browser.autonomous_task import BrowserAutonomousTaskSkill
        if adapter is None:
            adapter = MagicMock()
            adapter.is_available.return_value = True
            adapter.run_task.return_value = {
                "success": True,
                "final_url": "https://youtube.com/results",
                "page_title": "YouTube Results",
            }
        return BrowserAutonomousTaskSkill(
            browser_adapter=adapter,
            session_manager=None,
            browser_controller=controller,
        )

    def test_context_prepended_when_browser_alive(self):
        """When browser is on a page, task string is enriched with context."""
        controller = MagicMock()
        controller.is_alive.return_value = True
        controller.get_snapshot.return_value = _sample_snapshot(
            url="https://youtube.com",
            title="YouTube",
            entities=(),
        )

        adapter = MagicMock()
        adapter.is_available.return_value = True
        adapter.run_task.return_value = {"success": True, "final_url": ""}
        adapter._config = None  # Skip safety gate

        skill = self._make_skill(controller=controller, adapter=adapter)
        world = MagicMock(spec=WorldTimeline)
        world.emit = MagicMock()

        skill.execute({"task": "search marvel trailer"}, world)

        # Verify run_task was called with enriched prompt
        call_args = adapter.run_task.call_args
        enriched_task = call_args[0][0]
        assert "youtube.com" in enriched_task
        assert "search marvel trailer" in enriched_task
        assert "Context:" in enriched_task

    def test_no_context_when_browser_not_alive(self):
        """When browser is dead, raw task string is passed."""
        controller = MagicMock()
        controller.is_alive.return_value = False

        adapter = MagicMock()
        adapter.is_available.return_value = True
        adapter.run_task.return_value = {"success": True, "final_url": ""}
        adapter._config = None

        skill = self._make_skill(controller=controller, adapter=adapter)
        world = MagicMock(spec=WorldTimeline)
        world.emit = MagicMock()

        skill.execute({"task": "search laptops"}, world)

        call_args = adapter.run_task.call_args
        task_str = call_args[0][0]
        assert task_str == "search laptops"

    def test_no_context_when_on_about_blank(self):
        """about:blank URLs are not injected as context."""
        controller = MagicMock()
        controller.is_alive.return_value = True
        controller.get_snapshot.return_value = _sample_snapshot(
            url="about:blank", title="", entities=(),
        )

        adapter = MagicMock()
        adapter.is_available.return_value = True
        adapter.run_task.return_value = {"success": True, "final_url": ""}
        adapter._config = None

        skill = self._make_skill(controller=controller, adapter=adapter)
        world = MagicMock(spec=WorldTimeline)
        world.emit = MagicMock()

        skill.execute({"task": "open amazon"}, world)

        call_args = adapter.run_task.call_args
        task_str = call_args[0][0]
        assert task_str == "open amazon"

    def test_graceful_degradation_on_controller_error(self):
        """Controller exception → falls back to raw task, no crash."""
        controller = MagicMock()
        controller.is_alive.side_effect = RuntimeError("dead")

        adapter = MagicMock()
        adapter.is_available.return_value = True
        adapter.run_task.return_value = {"success": True, "final_url": ""}
        adapter._config = None

        skill = self._make_skill(controller=controller, adapter=adapter)
        world = MagicMock(spec=WorldTimeline)
        world.emit = MagicMock()

        skill.execute({"task": "search test"}, world)

        call_args = adapter.run_task.call_args
        task_str = call_args[0][0]
        assert task_str == "search test"


# ─────────────────────────────────────────────────────────────
# 6. BrowserSource event emission tests
# ─────────────────────────────────────────────────────────────

class TestBrowserSourceEmitsEntities:
    """Test that BrowserSource includes top_entities in emitted events."""

    def _make_source(self, snapshot=None):
        """Create a BrowserSource with mocked controller."""
        controller = MagicMock()
        controller.is_alive.return_value = True
        controller.get_snapshot.return_value = (
            snapshot or _sample_snapshot()
        )
        return BrowserSource(
            browser_controller=controller,
            poll_interval=0,  # Always polls
        )

    def test_bootstrap_includes_entities(self):
        """Bootstrap event payload contains top_entities."""
        source = self._make_source()
        events = source.bootstrap()

        assert len(events) == 1
        payload = events[0].payload
        assert "top_entities" in payload
        assert len(payload["top_entities"]) == 5
        assert payload["top_entities"][0]["index"] == 1
        assert payload["top_entities"][0]["type"] == "link"

    def test_poll_includes_entities_on_change(self):
        """Poll event includes entities when page changes."""
        source = self._make_source()
        # Bootstrap first to set baseline
        source.bootstrap()

        # Change URL to trigger poll event
        new_snapshot = _sample_snapshot(
            url="https://youtube.com",
            title="YouTube",
            entities=_sample_entities(3),
        )
        source._controller.get_snapshot.return_value = new_snapshot

        events = source.poll()
        assert len(events) == 1
        payload = events[0].payload
        assert "top_entities" in payload
        assert len(payload["top_entities"]) == 3

    def test_entity_count_matches(self):
        """entity_count in payload matches actual entity len."""
        source = self._make_source(
            snapshot=_sample_snapshot(entities=_sample_entities(7)),
        )
        events = source.bootstrap()
        payload = events[0].payload
        assert payload["entity_count"] == 7
        assert len(payload["top_entities"]) == 7
