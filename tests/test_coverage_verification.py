# tests/test_coverage_verification.py

"""
Tests for capability-based intent coverage verification.

Validates:
- Full coverage returns (True, [])
- Partial coverage returns (False, uncovered_intents)
- Node consumption: each node covers at most one intent
- Capability matching: intent.action == node skill action
- Parameter disambiguation for same-action nodes
- Empty plan / empty intents edge cases
- IntentMatcher protocol satisfaction
"""

import pytest
from unittest.mock import MagicMock

from ir.mission import MissionPlan, MissionNode, ExecutionMode
from cortex.validators import (
    verify_intent_coverage,
    CapabilityIntentMatcher,
    HeuristicIntentMatcher,
    IntentMatcher,
)


# ── Helpers ──────────────────────────────────────────────────

def _make_node(node_id, skill, inputs=None):
    """Create a MissionNode with specified skill and literal inputs."""
    return MissionNode(
        id=node_id,
        skill=skill,
        inputs=inputs or {},
        outputs={},
        depends_on=[],
        mode=ExecutionMode.foreground,
    )


def _make_plan(*nodes):
    """Create a MissionPlan from a list of MissionNodes."""
    return MissionPlan(
        id="test_plan",
        nodes=list(nodes),
        metadata={"ir_version": "1.0"},
    )


def _make_intent(action, parameters=None):
    return {
        "action": action,
        "parameters": parameters or {},
    }


# ── Protocol Tests ───────────────────────────────────────────

class TestIntentMatcherProtocol:
    def test_capability_satisfies_protocol(self):
        matcher = CapabilityIntentMatcher()
        assert isinstance(matcher, IntentMatcher)

    def test_heuristic_alias_satisfies_protocol(self):
        matcher = HeuristicIntentMatcher()
        assert isinstance(matcher, IntentMatcher)


# ── Full Coverage Tests ──────────────────────────────────────

class TestFullCoverage:
    def test_single_intent_single_node(self):
        """One intent, one matching node → full coverage."""
        plan = _make_plan(
            _make_node("node_0", "fs.create_folder", {"name": "docs", "anchor": "DESKTOP"}),
        )
        intents = [_make_intent("create_folder", {"name": "docs"})]

        covered, uncovered = verify_intent_coverage(plan, intents, None)
        assert covered is True
        assert uncovered == []

    def test_multi_intent_multi_node(self):
        """Three intents, three matching nodes → full coverage."""
        plan = _make_plan(
            _make_node("node_0", "system.set_volume", {"level": 70}),
            _make_node("node_1", "system.set_brightness", {"level": 10}),
            _make_node("node_2", "system.media_play", {}),
        )
        intents = [
            _make_intent("set_volume", {"level": 70}),
            _make_intent("set_brightness", {"level": 10}),
            _make_intent("media_play"),
        ]

        covered, uncovered = verify_intent_coverage(plan, intents, None)
        assert covered is True
        assert uncovered == []


# ── Partial Coverage Tests ───────────────────────────────────

class TestPartialCoverage:
    def test_missing_node_for_intent(self):
        """Two intents but only one matching node → partial coverage."""
        plan = _make_plan(
            _make_node("node_0", "fs.create_folder", {"name": "docs"}),
        )
        intents = [
            _make_intent("create_folder", {"name": "docs"}),
            _make_intent("media_play"),  # No matching node
        ]

        covered, uncovered = verify_intent_coverage(plan, intents, None)
        assert covered is False
        assert len(uncovered) == 1
        assert uncovered[0]["action"] == "media_play"

    def test_all_uncovered(self):
        """No nodes match any intent → all uncovered."""
        plan = _make_plan(
            _make_node("node_0", "system.set_brightness", {"level": 80}),
        )
        intents = [
            _make_intent("create_folder", {"name": "docs"}),
            _make_intent("open_app", {"app_name": "notepad"}),
        ]

        covered, uncovered = verify_intent_coverage(plan, intents, None)
        assert covered is False
        assert len(uncovered) == 2


# ── Node Consumption Tests ───────────────────────────────────

class TestNodeConsumption:
    def test_one_node_cannot_cover_two_intents(self):
        """Single node consumed by first match cannot cover second intent."""
        plan = _make_plan(
            _make_node("node_0", "system.media_play", {}),
        )
        intents = [
            _make_intent("media_play"),  # Matches node_0
            _make_intent("media_play"),  # Same, but node consumed
        ]

        covered, uncovered = verify_intent_coverage(plan, intents, None)
        assert covered is False
        assert len(uncovered) == 1

    def test_two_similar_nodes_cover_two_intents(self):
        """Two similar nodes can each cover one intent (parameter disambiguation)."""
        plan = _make_plan(
            _make_node("node_0", "fs.create_folder", {"name": "A"}),
            _make_node("node_1", "fs.create_folder", {"name": "B"}),
        )
        intents = [
            _make_intent("create_folder", {"name": "A"}),
            _make_intent("create_folder", {"name": "B"}),
        ]

        covered, uncovered = verify_intent_coverage(plan, intents, None)
        assert covered is True
        assert uncovered == []


# ── Capability Matching Tests ────────────────────────────────

class TestCapabilityMatching:
    def test_action_match_exact(self):
        """Exact action match: intent.action == node skill action."""
        plan = _make_plan(
            _make_node("node_0", "system.open_app", {"app_name": "Spotify"}),
        )
        intents = [_make_intent("open_app", {"app_name": "Spotify"})]

        covered, uncovered = verify_intent_coverage(plan, intents, None)
        assert covered is True

    def test_action_mismatch(self):
        """Different action → no match."""
        plan = _make_plan(
            _make_node("node_0", "system.open_app", {"app_name": "Chrome"}),
        )
        intents = [_make_intent("close_app", {"app_name": "Chrome"})]

        covered, uncovered = verify_intent_coverage(plan, intents, None)
        assert covered is False

    def test_no_params_still_matches(self):
        """Skills with empty inputs: action match alone is sufficient."""
        plan = _make_plan(
            _make_node("node_0", "system.media_play", {}),
        )
        intents = [_make_intent("media_play")]

        covered, uncovered = verify_intent_coverage(plan, intents, None)
        assert covered is True

    def test_parameter_disambiguation(self):
        """When two nodes share the same action, parameter values disambiguate."""
        plan = _make_plan(
            _make_node("node_0", "system.open_app", {"app_name": "Chrome"}),
            _make_node("node_1", "system.open_app", {"app_name": "Spotify"}),
        )
        intents = [
            _make_intent("open_app", {"app_name": "Spotify"}),
            _make_intent("open_app", {"app_name": "Chrome"}),
        ]

        covered, uncovered = verify_intent_coverage(plan, intents, None)
        assert covered is True
        assert uncovered == []


# ── Edge Cases ───────────────────────────────────────────────

class TestCoverageEdgeCases:
    def test_empty_plan(self):
        """Empty plan → all intents uncovered."""
        plan = _make_plan()
        intents = [_make_intent("create_folder", {"name": "docs"})]

        covered, uncovered = verify_intent_coverage(plan, intents, None)
        assert covered is False
        assert len(uncovered) == 1

    def test_empty_intents(self):
        """No intents → trivially covered."""
        plan = _make_plan(
            _make_node("node_0", "fs.create_folder", {"name": "docs"}),
        )

        covered, uncovered = verify_intent_coverage(plan, [], None)
        assert covered is True
        assert uncovered == []

    def test_empty_action_in_intent(self):
        """Intent with empty action → no match."""
        plan = _make_plan(
            _make_node("node_0", "system.media_play", {}),
        )
        intents = [_make_intent("")]

        covered, uncovered = verify_intent_coverage(plan, intents, None)
        assert covered is False

    def test_custom_matcher(self):
        """Custom IntentMatcher can be injected."""
        class AlwaysMatchMatcher:
            def match(self, intent, candidate_nodes, registry):
                return candidate_nodes[0].id if candidate_nodes else None

        plan = _make_plan(
            _make_node("node_0", "any.skill", {}),
        )
        intents = [_make_intent("whatever")]

        covered, uncovered = verify_intent_coverage(
            plan, intents, None, matcher=AlwaysMatchMatcher(),
        )
        assert covered is True
