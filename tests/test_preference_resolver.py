# tests/test_preference_resolver.py

"""
Tests for PreferenceResolver and PreferenceMemory.

Verifies:
- Preference detection patterns
- Key derivation from skill+param
- Memory lookup integration
- Graceful degradation (no memory, no match)
- Plan immutability
"""

import pytest
from unittest.mock import MagicMock

from cortex.preference_resolver import (
    PreferenceResolver,
    PreferenceMemory,
    _PREFERENCE_PATTERNS,
)
from ir.mission import MissionPlan, MissionNode


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _node(skill="system.set_volume", inputs=None, node_id="n0"):
    return MissionNode(
        id=node_id,
        skill=skill,
        inputs=inputs or {},
        outputs={},
        depends_on=[],
    )


def _plan(*nodes):
    return MissionPlan(
        id="test-plan",
        nodes=list(nodes),
        metadata={"ir_version": "1.0"},
    )


# ─────────────────────────────────────────────────────────────
# Tests: Preference detection patterns
# ─────────────────────────────────────────────────────────────

class TestPreferencePatterns:
    """Regex patterns correctly detect preference references."""

    @pytest.mark.parametrize("value", [
        "my preferred level",
        "preferred",
        "my preferred setting",
        "set to preferred",
    ])
    def test_preferred_detected(self, value):
        assert any(p.search(value) for p in _PREFERENCE_PATTERNS)

    @pytest.mark.parametrize("value", [
        "my favourite editor",
        "favorite music",
        "my favorite",
    ])
    def test_favourite_detected(self, value):
        assert any(p.search(value) for p in _PREFERENCE_PATTERNS)

    @pytest.mark.parametrize("value", [
        "my usual level",
        "the default",
        "normal setting",
    ])
    def test_usual_default_detected(self, value):
        assert any(p.search(value) for p in _PREFERENCE_PATTERNS)

    @pytest.mark.parametrize("value", [
        "50",
        "notepad",
        "hello world",
        "set volume to 75",
    ])
    def test_non_preference_not_detected(self, value):
        assert not any(p.search(value) for p in _PREFERENCE_PATTERNS)


# ─────────────────────────────────────────────────────────────
# Tests: Key derivation
# ─────────────────────────────────────────────────────────────

class TestKeyDerivation:
    """Preference keys derived from skill+param context."""

    def test_set_volume(self):
        key = PreferenceResolver._derive_preference_key("level", "system.set_volume")
        assert key == "preferred_volume"

    def test_open_app(self):
        key = PreferenceResolver._derive_preference_key("app_name", "system.open_app")
        assert key == "preferred_open_app"

    def test_toggle_nightlight(self):
        key = PreferenceResolver._derive_preference_key("state", "system.toggle_nightlight")
        assert key == "preferred_nightlight"

    def test_no_prefix_skill(self):
        key = PreferenceResolver._derive_preference_key("path", "create_folder")
        assert key == "preferred_create_folder"


# ─────────────────────────────────────────────────────────────
# Tests: PreferenceMemory
# ─────────────────────────────────────────────────────────────

class TestPreferenceMemory:
    """PreferenceMemory wraps MemoryStore for key-based lookup."""

    def test_no_store_returns_none(self):
        mem = PreferenceMemory(memory_store=None)
        assert mem.lookup("preferred_volume") is None

    def test_cache_works(self):
        mem = PreferenceMemory(memory_store=None)
        mem.store("preferred_volume", 35)
        assert mem.lookup("preferred_volume") == 35

    def test_lookup_from_episode_metadata(self):
        """Finds preference value in episode metadata."""
        mock_store = MagicMock()
        mock_store.retrieve_relevant.return_value = [
            {
                "mission_id": "m1",
                "query": "set volume to 35",
                "outcome_summary": "Volume set to 35",
                "metadata": {
                    "preferences": {"preferred_volume": 35},
                },
            }
        ]
        mem = PreferenceMemory(memory_store=mock_store)
        assert mem.lookup("preferred_volume") == 35

    def test_lookup_caches_after_first_hit(self):
        mock_store = MagicMock()
        mock_store.retrieve_relevant.return_value = [
            {"metadata": {"preferences": {"preferred_volume": 35}}},
        ]
        mem = PreferenceMemory(memory_store=mock_store)

        # First lookup — queries store
        assert mem.lookup("preferred_volume") == 35
        assert mock_store.retrieve_relevant.call_count == 1

        # Second lookup — uses cache
        assert mem.lookup("preferred_volume") == 35
        assert mock_store.retrieve_relevant.call_count == 1

    def test_lookup_returns_none_when_no_match(self):
        mock_store = MagicMock()
        mock_store.retrieve_relevant.return_value = [
            {"metadata": {}},
        ]
        mem = PreferenceMemory(memory_store=mock_store)
        assert mem.lookup("preferred_volume") is None

    def test_lookup_handles_store_exception(self):
        mock_store = MagicMock()
        mock_store.retrieve_relevant.side_effect = RuntimeError("db error")
        mem = PreferenceMemory(memory_store=mock_store)
        assert mem.lookup("preferred_volume") is None


# ─────────────────────────────────────────────────────────────
# Tests: PreferenceResolver
# ─────────────────────────────────────────────────────────────

class TestPreferenceResolver:
    """PreferenceResolver resolves preference tokens in plan inputs."""

    def test_non_preference_passthrough(self):
        """Non-preference values are not modified."""
        resolver = PreferenceResolver(memory=PreferenceMemory())
        plan = _plan(_node(inputs={"level": "50"}))

        result = resolver.resolve_plan(plan)
        assert result is plan  # Same object — no changes

    def test_preference_resolved_from_memory(self):
        """Preference token resolved to stored value."""
        mem = PreferenceMemory()
        mem.store("preferred_volume", 35)

        resolver = PreferenceResolver(memory=mem)
        plan = _plan(_node(
            skill="system.set_volume",
            inputs={"level": "my preferred level"},
        ))

        result = resolver.resolve_plan(plan)
        assert result is not plan  # New plan
        assert result.nodes[0].inputs["level"] == 35

    def test_unresolvable_preference_passthrough(self):
        """Preference detected but no memory match — value unchanged."""
        resolver = PreferenceResolver(memory=PreferenceMemory())
        plan = _plan(_node(inputs={"level": "my preferred level"}))

        result = resolver.resolve_plan(plan)
        # Returns same plan because nothing actually changed value
        assert result.nodes[0].inputs["level"] == "my preferred level"

    def test_numeric_inputs_untouched(self):
        """Non-string inputs are never checked for preferences."""
        mem = PreferenceMemory()
        mem.store("preferred_volume", 35)

        resolver = PreferenceResolver(memory=mem)
        plan = _plan(_node(inputs={"level": 50}))

        result = resolver.resolve_plan(plan)
        assert result is plan  # No changes
        assert result.nodes[0].inputs["level"] == 50

    def test_plan_immutability(self):
        """Original plan is never mutated."""
        mem = PreferenceMemory()
        mem.store("preferred_volume", 35)

        resolver = PreferenceResolver(memory=mem)
        original_plan = _plan(_node(
            skill="system.set_volume",
            inputs={"level": "preferred"},
        ))

        result = resolver.resolve_plan(original_plan)

        # Original unchanged
        assert original_plan.nodes[0].inputs["level"] == "preferred"
        # Result resolved
        assert result.nodes[0].inputs["level"] == 35

    def test_multiple_nodes_resolved(self):
        """All nodes in a multi-node plan are checked."""
        mem = PreferenceMemory()
        mem.store("preferred_volume", 35)
        mem.store("preferred_brightness", 70)

        resolver = PreferenceResolver(memory=mem)
        plan = _plan(
            _node(
                skill="system.set_volume",
                inputs={"level": "preferred"},
                node_id="n0",
            ),
            _node(
                skill="system.set_brightness",
                inputs={"level": "my usual"},
                node_id="n1",
            ),
        )

        result = resolver.resolve_plan(plan)
        assert result.nodes[0].inputs["level"] == 35
        assert result.nodes[1].inputs["level"] == 70

    def test_default_memory_no_crash(self):
        """PreferenceResolver with default (empty) memory doesn't crash."""
        resolver = PreferenceResolver()
        plan = _plan(_node(inputs={"level": "my preferred level"}))

        # Should not raise
        result = resolver.resolve_plan(plan)
        assert result.nodes[0].inputs["level"] == "my preferred level"
