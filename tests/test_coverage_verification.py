# tests/test_coverage_verification.py

"""
Tests for Phase 5A: verify_intent_coverage() and HeuristicIntentMatcher.

Validates:
- Full coverage returns (True, [])
- Partial coverage returns (False, uncovered_intents)
- Node consumption: each node covers at most one intent
- Domain-biased filtering with cross-domain fallback
- Argument-level matching (intent.object vs node.inputs)
- Verb-to-skill token matching
- Empty plan / empty intents edge cases
- IntentMatcher protocol satisfaction
"""

import pytest
from unittest.mock import MagicMock

from ir.mission import MissionPlan, MissionNode, ExecutionMode
from cortex.validators import (
    verify_intent_coverage,
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


def _make_intent(verb, obj, domain_hint="", modifiers=""):
    return {
        "verb": verb,
        "object": obj,
        "domain_hint": domain_hint,
        "modifiers": modifiers,
    }


# ── Protocol Tests ───────────────────────────────────────────

class TestIntentMatcherProtocol:
    def test_heuristic_satisfies_protocol(self):
        matcher = HeuristicIntentMatcher()
        assert isinstance(matcher, IntentMatcher)


# ── Full Coverage Tests ──────────────────────────────────────

class TestFullCoverage:
    def test_single_intent_single_node(self):
        """One intent, one matching node → full coverage."""
        plan = _make_plan(
            _make_node("node_0", "fs.create_folder", {"name": "docs", "anchor": "DESKTOP"}),
        )
        intents = [_make_intent("create", "folder docs", "fs")]

        covered, uncovered = verify_intent_coverage(plan, intents, None)
        assert covered is True
        assert uncovered == []

    def test_multi_intent_multi_node(self):
        """Three intents, three matching nodes → full coverage."""
        plan = _make_plan(
            _make_node("node_0", "fs.create_folder", {"name": "docs", "anchor": "DESKTOP"}),
            _make_node("node_1", "system.open_app", {"app_name": "notepad"}),
            _make_node("node_2", "system.media_play", {}),
        )
        intents = [
            _make_intent("create", "docs", "fs"),
            _make_intent("open", "notepad", "system"),
            _make_intent("play", "", "system"),
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
            _make_intent("create", "docs", "fs"),
            _make_intent("play", "music", "system"),  # No matching node
        ]

        covered, uncovered = verify_intent_coverage(plan, intents, None)
        assert covered is False
        assert len(uncovered) == 1
        assert uncovered[0]["verb"] == "play"

    def test_all_uncovered(self):
        """No nodes match any intent → all uncovered."""
        plan = _make_plan(
            _make_node("node_0", "system.set_brightness", {"level": 80}),
        )
        intents = [
            _make_intent("create", "folder", "fs"),
            _make_intent("open", "notepad", "system"),
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
            _make_intent("play", "", "system"),  # Matches node_0
            _make_intent("play", "", "system"),  # Same pattern, but node consumed
        ]

        covered, uncovered = verify_intent_coverage(plan, intents, None)
        assert covered is False
        assert len(uncovered) == 1

    def test_two_similar_nodes_cover_two_intents(self):
        """Two similar nodes can each cover one intent."""
        plan = _make_plan(
            _make_node("node_0", "fs.create_folder", {"name": "A"}),
            _make_node("node_1", "fs.create_folder", {"name": "B"}),
        )
        intents = [
            _make_intent("create", "A", "fs"),
            _make_intent("create", "B", "fs"),
        ]

        covered, uncovered = verify_intent_coverage(plan, intents, None)
        assert covered is True
        assert uncovered == []


# ── Domain Filtering Tests ───────────────────────────────────

class TestDomainFiltering:
    def test_domain_bias_preferential(self):
        """Domain hint biases toward same-domain nodes."""
        plan = _make_plan(
            _make_node("node_0", "system.open_app", {"app_name": "notepad"}),
            _make_node("node_1", "fs.create_folder", {"name": "docs"}),
        )
        intents = [_make_intent("create", "docs", "fs")]

        covered, uncovered = verify_intent_coverage(plan, intents, None)
        assert covered is True
        assert uncovered == []

    def test_domain_fallback_on_mismatch(self):
        """If domain hint has no match, fallback to all nodes (soft)."""
        plan = _make_plan(
            _make_node("node_0", "system.open_app", {"app_name": "notepad"}),
        )
        # Intent says domain "browser" but only system nodes exist
        intents = [_make_intent("open", "notepad", "browser")]

        covered, uncovered = verify_intent_coverage(plan, intents, None)
        assert covered is True  # Soft fallback — still matches by verb+object


# ── Argument Alignment Tests ─────────────────────────────────

class TestArgumentAlignment:
    def test_substring_match(self):
        """intent.object as substring of node input value → match."""
        plan = _make_plan(
            _make_node("node_0", "system.open_app", {"app_name": "Spotify"}),
        )
        intents = [_make_intent("open", "spotify", "system")]

        covered, uncovered = verify_intent_coverage(plan, intents, None)
        assert covered is True

    def test_no_argument_match(self):
        """Verb matches but object doesn't appear in inputs → no match."""
        plan = _make_plan(
            _make_node("node_0", "system.open_app", {"app_name": "Chrome"}),
        )
        intents = [_make_intent("open", "firefox", "system")]

        covered, uncovered = verify_intent_coverage(plan, intents, None)
        assert covered is False

    def test_empty_inputs_verb_match_sufficient(self):
        """Skills with empty inputs: verb match alone is sufficient."""
        plan = _make_plan(
            _make_node("node_0", "system.media_play", {}),
        )
        intents = [_make_intent("play", "music", "system")]

        covered, uncovered = verify_intent_coverage(plan, intents, None)
        assert covered is True


# ── Edge Cases ───────────────────────────────────────────────

class TestCoverageEdgeCases:
    def test_empty_plan(self):
        """Empty plan → all intents uncovered."""
        plan = _make_plan()
        intents = [_make_intent("create", "folder", "fs")]

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

    def test_custom_matcher(self):
        """Custom IntentMatcher can be injected."""
        class AlwaysMatchMatcher:
            def match(self, intent, candidate_nodes, registry):
                return candidate_nodes[0].id if candidate_nodes else None

        plan = _make_plan(
            _make_node("node_0", "any.skill", {}),
        )
        intents = [_make_intent("whatever", "anything")]

        covered, uncovered = verify_intent_coverage(
            plan, intents, None, matcher=AlwaysMatchMatcher(),
        )
        assert covered is True
