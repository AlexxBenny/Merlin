# tests/test_pending_mission.py

"""
Tests for P0: Suspend/Resume (PendingMission lifecycle).

Covers:
- PendingMission dataclass construction
- Unified _handle_pending_response routing
- Clarification resume (query merging)
- Partial capability backward compatibility
- Decline/cancel behavior
- New-query-during-pending routing
"""

import time
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

from brain.core import Percept, BrainCore, CognitiveRoute
from brain.escalation_policy import CognitiveTier
from world.snapshot import WorldSnapshot
from world.state import WorldState
from merlin import Merlin, PendingMission


def _make_percept(text: str = "test query") -> Percept:
    return Percept(
        modality="text", payload=text,
        confidence=1.0, timestamp=time.time(),
    )


def _make_snapshot() -> WorldSnapshot:
    return WorldSnapshot.build(WorldState(), [])


def _make_merlin(**kwargs) -> Merlin:
    """Build a minimal Merlin for testing pending mission flows."""
    brain = MagicMock(spec=BrainCore)
    brain.route.return_value = CognitiveRoute.MISSION
    brain.last_features = None

    escalation = MagicMock()
    cortex = MagicMock()
    registry = MagicMock()
    registry.all_names.return_value = set()
    timeline = MagicMock()
    timeline.all_events.return_value = []
    reflex = MagicMock()
    report_builder = MagicMock()
    output_channel = MagicMock()
    notification = MagicMock()
    event_sources = []

    return Merlin(
        brain=brain,
        escalation_policy=escalation,
        cortex=cortex,
        registry=registry,
        timeline=timeline,
        reflex_engine=reflex,
        report_builder=report_builder,
        output_channel=output_channel,
        notification_policy=notification,
        event_sources=event_sources,
        **kwargs,
    )


class TestPendingMissionDataclass(unittest.TestCase):
    """PendingMission construction and field defaults."""

    def test_clarification_construction(self):
        percept = _make_percept("create folder")
        snapshot = _make_snapshot()
        pm = PendingMission(
            kind="clarification",
            original_percept=percept,
            snapshot=snapshot,
            question="Which drive?",
        )
        assert pm.kind == "clarification"
        assert pm.original_percept is percept
        assert pm.snapshot is snapshot
        assert pm.question == "Which drive?"
        assert pm.tier is None
        assert pm.valid_intents is None
        assert pm.unsupported_intents is None
        assert pm.created_at > 0

    def test_partial_construction(self):
        percept = _make_percept("do X and Y")
        snapshot = _make_snapshot()
        pm = PendingMission(
            kind="partial",
            original_percept=percept,
            snapshot=snapshot,
            question="Should I proceed?",
            tier=CognitiveTier.SIMPLE,
            valid_intents=[{"action": "open_app"}],
            unsupported_intents=[{"action": "fly", "description": "fly to moon"}],
        )
        assert pm.kind == "partial"
        assert pm.tier == CognitiveTier.SIMPLE
        assert len(pm.valid_intents) == 1
        assert len(pm.unsupported_intents) == 1

    def test_created_at_auto_populated(self):
        before = time.time()
        pm = PendingMission(
            kind="clarification",
            original_percept=_make_percept(),
            snapshot=_make_snapshot(),
            question="?",
        )
        after = time.time()
        assert before <= pm.created_at <= after


class TestHandlePendingResponse(unittest.TestCase):
    """Unified _handle_pending_response routing."""

    def setUp(self):
        self.merlin = _make_merlin()

    def test_decline_cancels_clarification(self):
        """'no' during clarification pending → cancel."""
        self.merlin._pending_mission = PendingMission(
            kind="clarification",
            original_percept=_make_percept("create folder"),
            snapshot=_make_snapshot(),
            question="Which drive?",
        )
        result = self.merlin.handle_percept(
            _make_percept("no")
        )
        assert "cancelled" in result.lower()
        assert self.merlin._pending_mission is None

    def test_decline_cancels_partial(self):
        """'cancel' during partial pending → cancel."""
        self.merlin._pending_mission = PendingMission(
            kind="partial",
            original_percept=_make_percept("do X and Y"),
            snapshot=_make_snapshot(),
            question="Should I proceed?",
            tier=CognitiveTier.SIMPLE,
            valid_intents=[{"action": "open_app"}],
            unsupported_intents=[{"action": "fly", "description": "fly"}],
        )
        result = self.merlin.handle_percept(
            _make_percept("cancel")
        )
        assert "cancelled" in result.lower()
        assert self.merlin._pending_mission is None

    def test_confirm_resumes_partial(self):
        """'yes' during partial pending → resumes via orchestrator."""
        percept = _make_percept("open app and fly")
        snapshot = _make_snapshot()
        self.merlin._pending_mission = PendingMission(
            kind="partial",
            original_percept=percept,
            snapshot=snapshot,
            question="Should I proceed?",
            tier=CognitiveTier.SIMPLE,
            valid_intents=[{"action": "open_app", "parameters": {"name": "chrome"}}],
            unsupported_intents=[{"action": "fly", "description": "fly"}],
        )
        self.merlin.orchestrator.handle_user_input = MagicMock(return_value="Done")

        result = self.merlin.handle_percept(_make_percept("yes"))
        assert result is not None
        self.merlin.orchestrator.handle_user_input.assert_called_once()
        assert self.merlin._pending_mission is None

    def test_clarification_answer_resumes_mission(self):
        """Non-decline response during clarification → merges query and calls _handle_mission."""
        percept = _make_percept("create folder research")
        snapshot = _make_snapshot()
        self.merlin._pending_mission = PendingMission(
            kind="clarification",
            original_percept=percept,
            snapshot=snapshot,
            question="Which drive?",
        )

        # Mock _handle_mission to capture the merged percept
        captured = {}
        original_handle_mission = self.merlin._handle_mission

        def mock_handle_mission(p, s, **kwargs):
            captured["percept"] = p
            captured["snapshot"] = s
            return "Mission completed"

        self.merlin._handle_mission = mock_handle_mission

        result = self.merlin.handle_percept(_make_percept("D drive"))

        assert result == "Mission completed"
        assert "create folder research" in captured["percept"].payload
        assert "D drive" in captured["percept"].payload
        assert captured["snapshot"] is snapshot  # Original snapshot preserved
        assert self.merlin._pending_mission is None

    def test_new_query_during_partial_reroutes(self):
        """Non-confirm, non-decline during partial → reroute as new query."""
        self.merlin._pending_mission = PendingMission(
            kind="partial",
            original_percept=_make_percept("do X and Y"),
            snapshot=_make_snapshot(),
            question="Should I proceed?",
            tier=CognitiveTier.SIMPLE,
            valid_intents=[{"action": "open_app"}],
            unsupported_intents=[{"action": "fly", "description": "fly"}],
        )

        # New query "set volume to 50" should be routed normally
        self.merlin.brain.route.return_value = CognitiveRoute.REFUSE
        result = self.merlin.handle_percept(
            _make_percept("set volume to 50")
        )
        # Should have been routed to REFUSE path
        assert self.merlin._pending_mission is None

    def test_pending_cleared_after_consumption(self):
        """Pending is always cleared after handling, regardless of outcome."""
        self.merlin._pending_mission = PendingMission(
            kind="clarification",
            original_percept=_make_percept("test"),
            snapshot=_make_snapshot(),
            question="?",
        )
        self.merlin._handle_mission = MagicMock(return_value="ok")

        self.merlin.handle_percept(_make_percept("answer"))
        assert self.merlin._pending_mission is None


class TestHandleClarify(unittest.TestCase):
    """_handle_clarify stores PendingMission for resume."""

    def test_stores_pending_mission(self):
        """_handle_clarify creates a clarification PendingMission."""
        merlin = _make_merlin()
        percept = _make_percept("create folder")
        snapshot = _make_snapshot()

        merlin._handle_clarify(percept, snapshot)

        assert merlin._pending_mission is not None
        assert merlin._pending_mission.kind == "clarification"
        assert merlin._pending_mission.original_percept is percept
        assert merlin._pending_mission.snapshot is snapshot

    def test_uses_clarifier_llm_when_available(self):
        """If clarifier LLM is provided, question comes from LLM."""
        llm = MagicMock()
        llm.complete.return_value = "Where exactly?"
        merlin = _make_merlin(clarifier_llm=llm)

        result = merlin._handle_clarify(_make_percept(), _make_snapshot())

        assert result == "Where exactly?"
        assert merlin._pending_mission.question == "Where exactly?"

    def test_falls_back_to_default_response(self):
        """Without clarifier LLM, uses default response."""
        merlin = _make_merlin()

        result = merlin._handle_clarify(_make_percept(), _make_snapshot())

        assert "clarify" in result.lower()
        assert merlin._pending_mission.question == result


class TestResumeFromClarification(unittest.TestCase):
    """_resume_from_clarification merges query correctly."""

    def test_merged_query_format(self):
        """Merged query has the form: 'original (answer)'."""
        merlin = _make_merlin()
        pending = PendingMission(
            kind="clarification",
            original_percept=_make_percept("create folder research"),
            snapshot=_make_snapshot(),
            question="Which drive?",
        )

        captured = {}
        def mock_handle_mission(p, s, **kwargs):
            captured["percept"] = p
            return "done"

        merlin._handle_mission = mock_handle_mission
        merlin._resume_from_clarification(pending, "D drive")

        assert captured["percept"].payload == "create folder research (D drive)"

    def test_preserves_original_percept_metadata(self):
        """Merged percept preserves modality, confidence, timestamp."""
        merlin = _make_merlin()
        original = Percept(
            modality="speech", payload="create folder",
            confidence=0.85, timestamp=12345.0,
        )
        pending = PendingMission(
            kind="clarification",
            original_percept=original,
            snapshot=_make_snapshot(),
            question="?",
        )

        captured = {}
        def mock_handle_mission(p, s, **kwargs):
            captured["percept"] = p
            return "done"

        merlin._handle_mission = mock_handle_mission
        merlin._resume_from_clarification(pending, "D drive")

        assert captured["percept"].modality == "speech"
        assert captured["percept"].confidence == 0.85
        assert captured["percept"].timestamp == 12345.0

    def test_uses_original_snapshot(self):
        """Resume uses the stored snapshot, not a fresh one."""
        merlin = _make_merlin()
        original_snapshot = _make_snapshot()
        pending = PendingMission(
            kind="clarification",
            original_percept=_make_percept("test"),
            snapshot=original_snapshot,
            question="?",
        )

        captured = {}
        def mock_handle_mission(p, s, **kwargs):
            captured["snapshot"] = s
            return "done"

        merlin._handle_mission = mock_handle_mission
        merlin._resume_from_clarification(pending, "answer")

        assert captured["snapshot"] is original_snapshot


if __name__ == "__main__":
    unittest.main()
