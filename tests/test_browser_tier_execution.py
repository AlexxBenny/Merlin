# tests/test_browser_tier_execution.py

"""
Tests for three-tier browser execution architecture.

Covers:
- EntityResolver browser entity resolution (cosine scoring)
- Scroll direction normalization
- BrowserSource entity refresh on scroll
- WorldState handling of browser_entities_refreshed events
- EntityResolutionError user_message for not_found_browser
"""

import math
import pytest
from unittest.mock import MagicMock, patch

from ir.mission import MissionPlan, MissionNode, IR_VERSION


# ─────────────────────────────────────────────────────────────
# EntityResolver — cosine scoring
# ─────────────────────────────────────────────────────────────

class TestCosineScoring:
    """Pure cosine similarity on token lists."""

    @pytest.fixture
    def resolver(self):
        from cortex.entity_resolver import EntityResolver
        return EntityResolver(registry=MagicMock(), skill_registry=MagicMock())

    def test_identical_tokens(self, resolver):
        """Identical token lists → score 1.0."""
        score = resolver._cosine_similarity(
            ["howard", "stark"],
            ["howard", "stark"],
        )
        assert abs(score - 1.0) < 0.01

    def test_no_overlap(self, resolver):
        """Zero overlap → score 0.0."""
        score = resolver._cosine_similarity(
            ["howard", "stark"],
            ["ironman", "scene"],
        )
        assert score == 0.0

    def test_partial_overlap(self, resolver):
        """Partial overlap → score between 0 and 1."""
        score = resolver._cosine_similarity(
            ["howard", "stark"],
            ["howard", "stark", "speech"],
        )
        assert 0.5 < score < 1.0

    def test_empty_query(self, resolver):
        """Empty query → 0.0."""
        assert resolver._cosine_similarity([], ["a", "b"]) == 0.0

    def test_empty_entity(self, resolver):
        """Empty entity → 0.0."""
        assert resolver._cosine_similarity(["a"], []) == 0.0

    def test_single_token_match(self, resolver):
        """Single token against multi-token entity."""
        score = resolver._cosine_similarity(
            ["stark"],
            ["tony", "stark", "lab", "scene"],
        )
        # 1 match out of 4 tokens entity-side
        assert 0.2 < score < 0.6

    def test_howard_stark_example(self, resolver):
        """The canonical user example:
        query=howard stark
        entities: Howard Stark Speech, Tony Stark Lab, Ironman Scene
        Expected: Howard Stark Speech scores highest.
        """
        q = resolver._tokenize("howard stark")
        e1 = resolver._tokenize("Howard Stark Speech")
        e2 = resolver._tokenize("Tony Stark Lab")
        e3 = resolver._tokenize("Ironman Scene")

        s1 = resolver._cosine_similarity(q, e1)
        s2 = resolver._cosine_similarity(q, e2)
        s3 = resolver._cosine_similarity(q, e3)

        assert s1 > s2 > s3
        assert s1 > 0.55  # Should be RESOLVED
        assert s3 < 0.55  # Should be NOT_FOUND for this query


class TestTokenizer:
    """Tokenizer strips stopwords and lowercases."""

    @pytest.fixture
    def resolver(self):
        from cortex.entity_resolver import EntityResolver
        return EntityResolver(registry=MagicMock(), skill_registry=MagicMock())

    def test_lowercase(self, resolver):
        assert resolver._tokenize("Howard STARK") == ["howard", "stark"]

    def test_stopword_removal(self, resolver):
        tokens = resolver._tokenize("the Howard and Stark")
        assert "the" not in tokens
        assert "and" not in tokens
        assert "howard" in tokens

    def test_empty_string(self, resolver):
        assert resolver._tokenize("") == []


# ─────────────────────────────────────────────────────────────
# EntityResolver — browser entity resolution (Phase 9D)
# ─────────────────────────────────────────────────────────────

class TestBrowserEntityResolution:
    """Resolve entity_ref → entity_index in browser.click nodes."""

    @pytest.fixture
    def resolver(self):
        from cortex.entity_resolver import EntityResolver
        return EntityResolver(registry=MagicMock(), skill_registry=MagicMock())

    def _make_plan(self, entity_ref=None, entity_index=None):
        inputs = {}
        if entity_ref is not None:
            inputs["entity_ref"] = entity_ref
        if entity_index is not None:
            inputs["entity_index"] = entity_index
        return MissionPlan(
            id="test_plan",
            nodes=[MissionNode(
                id="n0", skill="browser.click", inputs=inputs,
            )],
            metadata={"ir_version": IR_VERSION},
        )

    def _make_world(self, entities):
        ws = MagicMock()
        ws.browser = MagicMock()
        ws.browser.top_entities = entities
        return ws

    def test_resolved_single_match(self, resolver):
        """Clear entity match → entity_ref replaced with entity_index."""
        from cortex.entity_resolver import EntityResolutionError
        plan = self._make_plan(entity_ref="howard stark")
        ws = self._make_world([
            {"index": 1, "type": "link", "text": "Ironman Scene"},
            {"index": 2, "type": "link", "text": "Howard Stark Speech"},
            {"index": 3, "type": "link", "text": "Tony Stark Lab"},
        ])

        resolved = resolver.resolve_plan(plan, world_snapshot=ws)
        assert "entity_index" in resolved.nodes[0].inputs
        assert resolved.nodes[0].inputs["entity_index"] == 2
        assert "entity_ref" not in resolved.nodes[0].inputs

    def test_not_found_browser(self, resolver):
        """No entity matches → not_found_browser violation."""
        from cortex.entity_resolver import EntityResolutionError
        plan = self._make_plan(entity_ref="captain america")
        ws = self._make_world([
            {"index": 1, "type": "link", "text": "Ironman Scene"},
            {"index": 2, "type": "link", "text": "Howard Stark Speech"},
        ])

        with pytest.raises(EntityResolutionError) as exc:
            resolver.resolve_plan(plan, world_snapshot=ws)

        violations = exc.value.violations
        assert len(violations) == 1
        assert violations[0].resolution_type == "not_found_browser"
        assert violations[0].raw_value == "captain america"

    def test_not_found_browser_user_message(self, resolver):
        """not_found_browser violation generates clean user message."""
        from cortex.entity_resolver import EntityResolutionError
        plan = self._make_plan(entity_ref="captain america")
        ws = self._make_world([
            {"index": 1, "type": "link", "text": "Ironman Scene"},
            {"index": 2, "type": "link", "text": "Howard Stark Speech"},
        ])

        with pytest.raises(EntityResolutionError) as exc:
            resolver.resolve_plan(plan, world_snapshot=ws)

        msg = exc.value.user_message()
        assert "captain america" in msg
        assert "I can't find" in msg

    def test_ambiguous_match(self, resolver):
        """Multiple close matches → ambiguous violation."""
        from cortex.entity_resolver import EntityResolutionError
        plan = self._make_plan(entity_ref="stark")
        ws = self._make_world([
            {"index": 1, "type": "link", "text": "Tony Stark Lab"},
            {"index": 2, "type": "link", "text": "Howard Stark Speech"},
            {"index": 3, "type": "link", "text": "Ironman Battle Scene"},
        ])

        with pytest.raises(EntityResolutionError) as exc:
            resolver.resolve_plan(plan, world_snapshot=ws)

        violations = exc.value.violations
        assert len(violations) == 1
        assert violations[0].resolution_type == "ambiguous"

    def test_entity_index_passthrough(self, resolver):
        """entity_index (not entity_ref) → no resolution attempted."""
        plan = self._make_plan(entity_index=5)
        ws = self._make_world([])

        resolved = resolver.resolve_plan(plan, world_snapshot=ws)
        assert resolved.nodes[0].inputs["entity_index"] == 5

    def test_no_entities_available(self, resolver):
        """No entities on page → not_found_browser."""
        from cortex.entity_resolver import EntityResolutionError
        plan = self._make_plan(entity_ref="something")
        ws = self._make_world([])

        with pytest.raises(EntityResolutionError) as exc:
            resolver.resolve_plan(plan, world_snapshot=ws)

        assert exc.value.violations[0].resolution_type == "not_found_browser"

    def test_no_world_snapshot(self, resolver):
        """No world_snapshot → not_found_browser."""
        from cortex.entity_resolver import EntityResolutionError
        plan = self._make_plan(entity_ref="something")

        with pytest.raises(EntityResolutionError) as exc:
            resolver.resolve_plan(plan, world_snapshot=None)

        assert exc.value.violations[0].resolution_type == "not_found_browser"


# ─────────────────────────────────────────────────────────────
# Scroll direction normalization
# ─────────────────────────────────────────────────────────────

class TestScrollNormalization:
    """BrowserScrollSkill normalizes direction synonyms."""

    def _make_skill(self):
        from skills.browser.browser_scroll import BrowserScrollSkill
        controller = MagicMock()
        controller.scroll_page.return_value = MagicMock(success=True)
        return BrowserScrollSkill(controller), controller

    def test_down_literal(self):
        skill, ctrl = self._make_skill()
        skill.execute({"direction": "down"}, MagicMock())
        ctrl.scroll_page.assert_called_once_with("down")

    def test_up_literal(self):
        skill, ctrl = self._make_skill()
        skill.execute({"direction": "up"}, MagicMock())
        ctrl.scroll_page.assert_called_once_with("up")

    def test_little_becomes_down(self):
        """'scroll down a little' → direction='little' → normalized to 'down'."""
        skill, ctrl = self._make_skill()
        skill.execute({"direction": "little"}, MagicMock())
        ctrl.scroll_page.assert_called_once_with("down")

    def test_above_becomes_up(self):
        skill, ctrl = self._make_skill()
        skill.execute({"direction": "above"}, MagicMock())
        ctrl.scroll_page.assert_called_once_with("up")

    def test_top_becomes_up(self):
        skill, ctrl = self._make_skill()
        skill.execute({"direction": "top"}, MagicMock())
        ctrl.scroll_page.assert_called_once_with("up")

    def test_bottom_becomes_down(self):
        skill, ctrl = self._make_skill()
        skill.execute({"direction": "bottom"}, MagicMock())
        ctrl.scroll_page.assert_called_once_with("down")

    def test_default_is_down(self):
        skill, ctrl = self._make_skill()
        skill.execute({}, MagicMock())
        ctrl.scroll_page.assert_called_once_with("down")


# ─────────────────────────────────────────────────────────────
# BrowserSource entity refresh
# ─────────────────────────────────────────────────────────────

class TestBrowserSourceEntityRefresh:
    """BrowserSource emits browser_entities_refreshed on entity count change."""

    def _make_entity_mock(self, index, entity_type="link", text="Entity"):
        e = MagicMock()
        e.index = index
        e.entity_type = entity_type
        e.text = text
        return e

    def test_entity_count_change_emits_event(self):
        """Entity count change without URL change → browser_entities_refreshed."""
        from runtime.sources.browser import BrowserSource

        controller = MagicMock()
        source = BrowserSource(browser_controller=controller, poll_interval=0)

        # Set up initial state
        source._was_alive = True
        source._last_url = "https://youtube.com"
        source._last_title = "YouTube"
        source._last_tab_count = 1
        source._last_entity_count = 10
        source._last_poll = 0  # Force poll

        # Simulate scroll: same URL but entity count changed
        snapshot = MagicMock()
        snapshot.url = "https://youtube.com"
        snapshot.title = "YouTube"
        snapshot.tab_count = 1
        snapshot.entities = [
            self._make_entity_mock(i, text=f"Video {i}")
            for i in range(15)
        ]
        snapshot.tabs = [MagicMock(tab_id="t1", url="https://youtube.com")]
        controller.is_alive.return_value = True
        controller.get_snapshot.return_value = snapshot

        events = source.poll()
        assert len(events) == 1
        assert events[0].type == "browser_entities_refreshed"

    def test_no_event_when_nothing_changed(self):
        """No state change → no event."""
        from runtime.sources.browser import BrowserSource

        controller = MagicMock()
        source = BrowserSource(browser_controller=controller, poll_interval=0)

        source._was_alive = True
        source._last_url = "https://youtube.com"
        source._last_title = "YouTube"
        source._last_tab_count = 1
        source._last_entity_count = 10
        source._last_poll = 0

        snapshot = MagicMock()
        snapshot.url = "https://youtube.com"
        snapshot.title = "YouTube"
        snapshot.tab_count = 1
        snapshot.entities = [
            self._make_entity_mock(i, text=f"Video {i}")
            for i in range(10)
        ]
        snapshot.tabs = [MagicMock(tab_id="t1", url="https://youtube.com")]
        controller.is_alive.return_value = True
        controller.get_snapshot.return_value = snapshot

        events = source.poll()
        assert len(events) == 0


# ─────────────────────────────────────────────────────────────
# WorldState — browser_entities_refreshed handler
# ─────────────────────────────────────────────────────────────

class TestWorldStateBrowserEntitiesRefreshed:
    """WorldState processes browser_entities_refreshed events."""

    def test_entities_refreshed_updates_top_entities(self):
        from world.state import WorldState
        from world.timeline import WorldEvent

        events = [WorldEvent(
            source="browser",
            timestamp=1.0,
            type="browser_entities_refreshed",
            payload={
                "url": "https://youtube.com",
                "title": "YouTube",
                "entity_count": 15,
                "tab_count": 1,
                "active_tab_id": "t1",
                "tab_urls": ["https://youtube.com"],
                "top_entities": [
                    {"index": 1, "type": "link", "text": "Video A"},
                    {"index": 2, "type": "link", "text": "Video B"},
                ],
            },
        )]

        state = WorldState.from_events(events)
        assert state.browser.active is True
        assert len(state.browser.top_entities) == 2
        assert state.browser.top_entities[0]["text"] == "Video A"
        assert state.browser.entity_count == 15


# ─────────────────────────────────────────────────────────────
# browser.click contract — entity_ref / entity_index alternatives
# ─────────────────────────────────────────────────────────────

class TestBrowserClickEntityRef:
    """browser.click contract accepts entity_ref as optional input."""

    def test_entity_ref_in_optional_inputs(self):
        from skills.browser.browser_click import BrowserClickSkill
        assert "entity_ref" in BrowserClickSkill.contract.optional_inputs
        assert BrowserClickSkill.contract.optional_inputs["entity_ref"] == "entity_ref"

    def test_entity_index_in_optional_inputs(self):
        from skills.browser.browser_click import BrowserClickSkill
        assert "entity_index" in BrowserClickSkill.contract.optional_inputs

    def test_input_groups_defined(self):
        from skills.browser.browser_click import BrowserClickSkill
        groups = BrowserClickSkill.contract.input_groups
        assert len(groups) == 1
        assert {"entity_index", "entity_ref"} in groups

    def test_no_required_inputs(self):
        """browser.click has no required inputs — all via optional + groups."""
        from skills.browser.browser_click import BrowserClickSkill
        assert BrowserClickSkill.contract.inputs == {}


# ─────────────────────────────────────────────────────────────
# browser.click — index drift protection
# ─────────────────────────────────────────────────────────────

class TestBrowserClickIndexDrift:
    """browser.click verifies entity text and falls back on drift."""

    def _make_entity(self, index, text, backend_node_id=100):
        e = MagicMock()
        e.index = index
        e.text = text
        e.backend_node_id = backend_node_id
        return e

    def _make_skill(self, entities):
        from skills.browser.browser_click import BrowserClickSkill
        controller = MagicMock()
        snapshot = MagicMock()
        snapshot.entities = entities
        controller.get_snapshot.return_value = snapshot
        click_result = MagicMock(success=True)
        click_result.snapshot = MagicMock(url="https://x.com", title="X")
        controller.click.return_value = click_result
        return BrowserClickSkill(controller), controller

    def test_fresh_snapshot_used(self):
        """Skill uses cached=False for fresh DOM (called twice: entity + post-click event)."""
        entities = [self._make_entity(1, "Video A")]
        skill, ctrl = self._make_skill(entities)
        skill.execute({"entity_index": 1}, MagicMock())
        # Called twice: once for entity resolution, once for post-click world state
        assert ctrl.get_snapshot.call_count == 2
        ctrl.get_snapshot.assert_called_with(cached=False)

    def test_text_match_no_drift(self):
        """Entity text matches resolved text → click proceeds normally."""
        entities = [self._make_entity(1, "Howard Stark Speech", 42)]
        skill, ctrl = self._make_skill(entities)
        skill.execute({
            "entity_index": 1,
            "_resolved_entity_text": "Howard Stark Speech",
        }, MagicMock())
        ctrl.click.assert_called_once_with(42)

    def test_index_drifted_fallback_by_text(self):
        """Index shifted but text matches another entity → click correct one."""
        entities = [
            self._make_entity(1, "New Element Here", 10),
            self._make_entity(2, "Howard Stark Speech", 42),
        ]
        skill, ctrl = self._make_skill(entities)
        # Resolver said index=1 but DOM shifted; "Howard Stark Speech"
        # is now at index=2
        skill.execute({
            "entity_index": 1,
            "_resolved_entity_text": "Howard Stark Speech",
        }, MagicMock())
        # Should click backend_node_id=42 (the correct element)
        ctrl.click.assert_called_once_with(42)

    def test_no_expected_text_skips_verification(self):
        """Compiler-set entity_index (no resolver) → no text check."""
        entities = [self._make_entity(3, "Some Button", 99)]
        skill, ctrl = self._make_skill(entities)
        skill.execute({"entity_index": 3}, MagicMock())
        ctrl.click.assert_called_once_with(99)

    def test_resolver_passes_resolved_text(self):
        """EntityResolver sets _resolved_entity_text on resolution."""
        from cortex.entity_resolver import EntityResolver
        resolver = EntityResolver(
            registry=MagicMock(), skill_registry=MagicMock(),
        )
        node = MagicMock()
        node.id = "n0"
        node.skill = "browser.click"
        node.inputs = {"entity_ref": "howard stark"}
        entities = [
            {"index": 1, "type": "link", "text": "Howard Stark Speech"},
            {"index": 2, "type": "link", "text": "Ironman Scene"},
        ]
        violations = resolver._resolve_browser_entity(node, "howard stark", entities)
        assert violations == []
        assert "_resolved_entity_text" in node.inputs
        assert node.inputs["_resolved_entity_text"] == "Howard Stark Speech"
