# tests/test_escalation_features.py

"""
Tests for EscalationPolicy feature-based decision path.

Validates:
- Feature-based path preferred over substring fallback
- intra_query_coreference → MISSION (not CLARIFY)
- Single-clause context with no external context → CLARIFY
- Single-clause context with external context → MISSION
- Fallback path (no features) still works
- No redundant linguistic heuristics in escalation
"""

import pytest
from unittest.mock import MagicMock

from brain.escalation_policy import EscalationPolicy, EscalationDecision
from brain.structural_classifier import QueryFeatures


@pytest.fixture
def policy():
    return EscalationPolicy(
        referential_markers=["it", "that", "this one", "the first"],
    )


def _make_snapshot(has_lists=False):
    snap = MagicMock()
    snap.state.visible_lists = ["result1"] if has_lists else []
    return snap


def _make_frame(has_context=False):
    frame = MagicMock()
    frame.context_frames = [{"domain": "fs"}] if has_context else []
    return frame


# ─────────────────────────────────────────────────────────────
# Feature-based path
# ─────────────────────────────────────────────────────────────

class TestFeatureBasedEscalation:
    """Tests for the preferred feature-based decision path."""

    def test_intra_query_coreference_returns_mission(self, policy):
        """Multi-clause + context = intra-query coref → MISSION."""
        features = QueryFeatures(requires_context=True, is_multi_clause=True)
        decision = policy.decide_for_user_input(
            user_text="create folder alex. inside it create man",
            snapshot=_make_snapshot(),
            frame=_make_frame(),
            features=features,
        )
        assert decision == EscalationDecision.MISSION

    def test_single_clause_no_context_clarify(self, policy):
        """Single clause + context + no external context → CLARIFY."""
        features = QueryFeatures(requires_context=True, is_multi_clause=False)
        decision = policy.decide_for_user_input(
            user_text="delete it",
            snapshot=_make_snapshot(has_lists=False),
            frame=_make_frame(has_context=False),
            features=features,
        )
        assert decision == EscalationDecision.CLARIFY

    def test_single_clause_with_visible_lists_mission(self, policy):
        """Single clause + context + visible lists → MISSION."""
        features = QueryFeatures(requires_context=True, is_multi_clause=False)
        decision = policy.decide_for_user_input(
            user_text="delete it",
            snapshot=_make_snapshot(has_lists=True),
            frame=_make_frame(),
            features=features,
        )
        assert decision == EscalationDecision.MISSION

    def test_single_clause_with_conversation_context_mission(self, policy):
        """Single clause + context + conversation frames → MISSION."""
        features = QueryFeatures(requires_context=True, is_multi_clause=False)
        decision = policy.decide_for_user_input(
            user_text="open it",
            snapshot=_make_snapshot(),
            frame=_make_frame(has_context=True),
            features=features,
        )
        assert decision == EscalationDecision.MISSION

    def test_no_context_features_returns_mission(self, policy):
        """No context requirement → always MISSION."""
        features = QueryFeatures(requires_context=False)
        decision = policy.decide_for_user_input(
            user_text="set brightness to 50",
            snapshot=_make_snapshot(),
            frame=_make_frame(),
            features=features,
        )
        assert decision == EscalationDecision.MISSION

    def test_scheduling_only_returns_mission(self, policy):
        """Scheduling without context → MISSION."""
        features = QueryFeatures(requires_scheduling=True)
        decision = policy.decide_for_user_input(
            user_text="pause after 10 seconds",
            snapshot=_make_snapshot(),
            frame=_make_frame(),
            features=features,
        )
        assert decision == EscalationDecision.MISSION


# ─────────────────────────────────────────────────────────────
# Fallback path (no features)
# ─────────────────────────────────────────────────────────────

class TestFallbackEscalation:
    """When features=None, substring matching still works."""

    def test_fallback_referential_no_context_clarify(self, policy):
        decision = policy.decide_for_user_input(
            user_text="delete it",
            snapshot=_make_snapshot(),
            frame=_make_frame(),
            features=None,
        )
        assert decision == EscalationDecision.CLARIFY

    def test_fallback_no_referential_mission(self, policy):
        decision = policy.decide_for_user_input(
            user_text="set brightness to 50",
            snapshot=_make_snapshot(),
            frame=_make_frame(),
            features=None,
        )
        assert decision == EscalationDecision.MISSION


# ─────────────────────────────────────────────────────────────
# The exact failing scenario
# ─────────────────────────────────────────────────────────────

class TestOriginalBugScenario:
    """Regression test for the exact query that triggered the bug."""

    def test_original_failing_query(self, policy):
        """The query that caused the session bug:
        'create a folder named alex. Inside it create two folders
        man and drake. Meanwhile set brightness to 10 and play music'

        Before fix: CLARIFY (false positive from 'it' substring)
        After fix: MISSION (intra_query_coreference detected)
        """
        features = QueryFeatures(requires_context=True, is_multi_clause=True)
        assert features.intra_query_coreference is True
        decision = policy.decide_for_user_input(
            user_text=(
                "create a folder named alex. inside it create two folders "
                "man and drake. meanwhile set brightness to 10 and play music"
            ),
            snapshot=_make_snapshot(),
            frame=_make_frame(),
            features=features,
        )
        assert decision == EscalationDecision.MISSION
