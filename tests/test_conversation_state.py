# tests/test_conversation_state.py

"""
Tests for Structured ConversationState (v2).

Validates:
- EntityRecord schema enforcement
- GoalState schema enforcement and lifecycle
- ConversationFrame entity_registry operations
- ConversationFrame intent tracking with cap
- ConversationFrame goal lifecycle with cap
- ConversationFrame last_results storage (preserves structure)
- Backward compatibility — existing fields unaffected
"""

import time
import pytest
from pydantic import ValidationError

from conversation.frame import (
    ConversationFrame,
    ConversationTurn,
    EntityRecord,
    GoalState,
    ENTITY_CAP,
    GOAL_CAP,
    INTENT_CAP,
)


# ---------------------------------------------------------------
# EntityRecord schema
# ---------------------------------------------------------------

class TestEntityRecordSchema:
    """Validate EntityRecord type enforcement."""

    def test_valid_entity_record(self):
        record = EntityRecord(
            type="list",
            value=[{"title": "Python tutorial"}],
            source_mission="m_1",
        )
        assert record.type == "list"
        assert record.source_mission == "m_1"
        assert record.created_at > 0

    def test_missing_required_fields_rejected(self):
        with pytest.raises(ValidationError):
            EntityRecord(type="list", value="x")  # missing source_mission

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            EntityRecord(
                type="list",
                value=[],
                source_mission="m_1",
                garbage="nope",
            )

    def test_scalar_value(self):
        record = EntityRecord(
            type="path",
            value="/Desktop/hello",
            source_mission="m_2",
        )
        assert record.value == "/Desktop/hello"

    def test_timestamp_auto_set(self):
        before = time.time()
        record = EntityRecord(type="scalar", value="x", source_mission="m_1")
        assert record.created_at >= before


# ---------------------------------------------------------------
# GoalState schema + lifecycle
# ---------------------------------------------------------------

class TestGoalStateSchema:
    """Validate GoalState type enforcement."""

    def test_valid_goal(self):
        goal = GoalState(id="g_1", description="Research AI")
        assert goal.id == "g_1"
        assert goal.status == "active"
        assert goal.mission_ids == []
        assert goal.created_at > 0

    def test_status_values(self):
        for status in ("active", "completed", "failed", "stalled"):
            goal = GoalState(id="g_1", description="test", status=status)
            assert goal.status == status

    def test_invalid_status_rejected(self):
        with pytest.raises(ValidationError):
            GoalState(id="g_1", description="test", status="invalid")

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            GoalState(id="g_1", description="test", garbage="nope")

    def test_mission_ids_append(self):
        goal = GoalState(id="g_1", description="test")
        goal.mission_ids.append("m_1")
        goal.mission_ids.append("m_2")
        assert goal.mission_ids == ["m_1", "m_2"]


# ---------------------------------------------------------------
# Entity registry operations
# ---------------------------------------------------------------

class TestEntityRegistry:
    """Validate ConversationFrame entity registry."""

    def test_register_entity_stores(self):
        frame = ConversationFrame()
        frame.register_entity("results", [1, 2, 3], "list", "m_1")
        assert "results" in frame.entity_registry
        assert frame.entity_registry["results"].value == [1, 2, 3]
        assert frame.entity_registry["results"].type == "list"

    def test_register_overwrites(self):
        frame = ConversationFrame()
        frame.register_entity("path", "/old", "path", "m_1")
        frame.register_entity("path", "/new", "path", "m_2")
        assert frame.entity_registry["path"].value == "/new"
        assert frame.entity_registry["path"].source_mission == "m_2"

    def test_get_entity_returns_record(self):
        frame = ConversationFrame()
        frame.register_entity("file", "report.pdf", "path", "m_1")
        record = frame.get_entity("file")
        assert record is not None
        assert record.value == "report.pdf"

    def test_get_entity_missing_returns_none(self):
        frame = ConversationFrame()
        assert frame.get_entity("nonexistent") is None

    def test_entity_cap_enforced(self):
        frame = ConversationFrame()
        for i in range(ENTITY_CAP + 10):
            frame.register_entity(f"entity_{i}", f"val_{i}", "scalar", f"m_{i}")
        assert len(frame.entity_registry) == ENTITY_CAP

    def test_entity_cap_drops_oldest(self):
        frame = ConversationFrame()
        # Register with explicit ordering via small sleep simulation
        for i in range(ENTITY_CAP + 5):
            frame.register_entity(f"entity_{i}", f"val_{i}", "scalar", f"m_{i}")
        # Oldest (entity_0 through entity_4) should be evicted
        for i in range(5):
            assert f"entity_{i}" not in frame.entity_registry
        # Newest should survive
        assert f"entity_{ENTITY_CAP + 4}" in frame.entity_registry


# ---------------------------------------------------------------
# Intent tracking
# ---------------------------------------------------------------

class TestIntentTracking:
    """Validate ConversationFrame intent tracking."""

    def test_push_intent_appends(self):
        frame = ConversationFrame()
        frame.push_intent("search YouTube")
        frame.push_intent("create folder")
        assert frame.recent_intents == ["search YouTube", "create folder"]

    def test_push_intent_order_preserved(self):
        frame = ConversationFrame()
        for i in range(5):
            frame.push_intent(f"intent_{i}")
        assert frame.recent_intents == [f"intent_{i}" for i in range(5)]

    def test_push_intent_enforces_cap(self):
        frame = ConversationFrame()
        for i in range(INTENT_CAP + 5):
            frame.push_intent(f"intent_{i}")
        assert len(frame.recent_intents) == INTENT_CAP
        # Oldest dropped
        assert frame.recent_intents[0] == "intent_5"
        assert frame.recent_intents[-1] == f"intent_{INTENT_CAP + 4}"


# ---------------------------------------------------------------
# Goal lifecycle
# ---------------------------------------------------------------

class TestGoalLifecycle:
    """Validate ConversationFrame goal lifecycle."""

    def test_add_goal_creates_active(self):
        frame = ConversationFrame()
        goal = frame.add_goal("g_1", "Research AI")
        assert goal.status == "active"
        assert len(frame.active_goals) == 1

    def test_complete_goal_updates_status(self):
        frame = ConversationFrame()
        frame.add_goal("g_1", "Research AI")
        frame.complete_goal("g_1")
        assert frame.active_goals[0].status == "completed"

    def test_fail_goal_updates_status(self):
        frame = ConversationFrame()
        frame.add_goal("g_1", "Research AI")
        frame.fail_goal("g_1")
        assert frame.active_goals[0].status == "failed"

    def test_get_active_goals_filters(self):
        frame = ConversationFrame()
        frame.add_goal("g_1", "Active goal")
        frame.add_goal("g_2", "Will complete")
        frame.complete_goal("g_2")
        active = frame.get_active_goals()
        assert len(active) == 1
        assert active[0].id == "g_1"

    def test_goal_cap_enforced(self):
        frame = ConversationFrame()
        for i in range(GOAL_CAP + 3):
            frame.add_goal(f"g_{i}", f"Goal {i}")
        assert len(frame.active_goals) <= GOAL_CAP

    def test_complete_nonexistent_goal_is_noop(self):
        frame = ConversationFrame()
        frame.add_goal("g_1", "Real goal")
        frame.complete_goal("g_nonexistent")  # Should not raise
        assert frame.active_goals[0].status == "active"


# ---------------------------------------------------------------
# Last results storage
# ---------------------------------------------------------------

class TestLastResults:
    """Validate ConversationFrame last_results storage."""

    def test_store_results_sets(self):
        frame = ConversationFrame()
        results = {"n_0": {"path": "/tmp/hello"}, "n_1": {"results": [1, 2]}}
        frame.store_results(results)
        assert frame.last_results == results

    def test_store_results_replaces(self):
        frame = ConversationFrame()
        frame.store_results({"n_0": {"old": True}})
        frame.store_results({"n_1": {"new": True}})
        assert "n_0" not in frame.last_results
        assert frame.last_results == {"n_1": {"new": True}}

    def test_store_results_preserves_structure(self):
        """Results must preserve full node→output structure — no flattening."""
        frame = ConversationFrame()
        nested = {
            "n_0": {"path": "/Desktop/hello", "status": "created"},
            "n_1": {"results": [{"title": "a"}, {"title": "b"}]},
        }
        frame.store_results(nested)
        assert frame.last_results["n_0"]["path"] == "/Desktop/hello"
        assert len(frame.last_results["n_1"]["results"]) == 2


# ---------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------

class TestBackwardCompatibility:
    """Verify existing ConversationFrame fields still work."""

    def test_existing_fields_unaffected(self):
        frame = ConversationFrame()
        assert frame.active_domain is None
        assert frame.active_entity is None
        assert frame.last_mission_id is None
        assert frame.context_frames == {}
        assert frame.unresolved_references == {}
        assert frame.history == []
        assert frame.outcomes == []

    def test_append_turn_still_works(self):
        frame = ConversationFrame()
        frame.append_turn("user", "hello")
        frame.append_turn("assistant", "hi there")
        assert len(frame.history) == 2
        assert frame.history[0].role == "user"

    def test_new_fields_default_empty(self):
        frame = ConversationFrame()
        assert frame.entity_registry == {}
        assert frame.recent_intents == []
        assert frame.active_goals == []
        assert frame.last_results == {}

    def test_extra_fields_still_rejected(self):
        """ConversationFrame still enforces extra='forbid'."""
        with pytest.raises(ValidationError):
            ConversationFrame(garbage="nope")
