# tests/test_goal_hierarchy.py

"""
Tests for Phase 5A: GoalState hierarchy fields and ConversationFrame helpers.

Validates:
- GoalState hierarchy field defaults (parent_goal, subgoal_ids, priority, progress)
- GoalState serialization/deserialization with new fields
- attach_mission_to_goal() helper
- update_goal_progress() helper
- Backward compatibility (existing GoalState usage unaffected)
"""

import pytest

from conversation.frame import ConversationFrame, GoalState


# ── GoalState Field Tests ────────────────────────────────────

class TestGoalStateHierarchyFields:
    def test_defaults(self):
        """New hierarchy fields have safe defaults."""
        goal = GoalState(
            id="goal_1",
            description="Test goal",
            status="active",
        )
        assert goal.parent_goal is None
        assert goal.subgoal_ids == []
        assert goal.priority == 0
        assert goal.progress == 0.0

    def test_explicit_values(self):
        """Hierarchy fields accept explicit values."""
        goal = GoalState(
            id="goal_1",
            description="Test goal",
            status="active",
            parent_goal="goal_0",
            subgoal_ids=["goal_2", "goal_3"],
            priority=5,
            progress=0.75,
        )
        assert goal.parent_goal == "goal_0"
        assert goal.subgoal_ids == ["goal_2", "goal_3"]
        assert goal.priority == 5
        assert goal.progress == 0.75

    def test_serialization_roundtrip(self):
        """GoalState with hierarchy fields survives JSON roundtrip."""
        goal = GoalState(
            id="goal_1",
            description="Parent goal",
            status="active",
            parent_goal="root",
            subgoal_ids=["sub_1"],
            priority=3,
            progress=0.5,
        )
        data = goal.model_dump()
        restored = GoalState(**data)
        assert restored.parent_goal == "root"
        assert restored.subgoal_ids == ["sub_1"]
        assert restored.priority == 3
        assert restored.progress == 0.5

    def test_backward_compatibility(self):
        """Existing GoalState creation without new fields still works."""
        goal = GoalState(
            id="legacy_goal",
            description="Old-style goal",
            status="completed",
        )
        # All original fields work
        assert goal.id == "legacy_goal"
        assert goal.description == "Old-style goal"
        assert goal.status == "completed"
        # New fields have defaults
        assert goal.parent_goal is None


# ── ConversationFrame Helper Tests ───────────────────────────

class TestAttachMissionToGoal:
    def test_attach_to_existing_goal(self):
        """Attaching a mission to an existing goal succeeds."""
        frame = ConversationFrame()
        frame.active_goals.append(
            GoalState(id="goal_1", description="Test", status="active")
        )

        result = frame.attach_mission_to_goal("mission_abc", "goal_1")
        assert result is True

    def test_attach_to_nonexistent_goal(self):
        """Attaching to a non-existent goal returns False."""
        frame = ConversationFrame()
        result = frame.attach_mission_to_goal("mission_abc", "no_such_goal")
        assert result is False

    def test_attach_multiple_missions(self):
        """Multiple missions can be attached to the same goal."""
        frame = ConversationFrame()
        frame.active_goals.append(
            GoalState(id="goal_1", description="Test", status="active")
        )

        result1 = frame.attach_mission_to_goal("mission_1", "goal_1")
        result2 = frame.attach_mission_to_goal("mission_2", "goal_1")
        assert result1 is True
        assert result2 is True


class TestUpdateGoalProgress:
    def test_no_subgoals(self):
        """Goal with no subgoals — progress stays at 0."""
        frame = ConversationFrame()
        frame.active_goals.append(
            GoalState(id="parent", description="Parent", status="active")
        )
        frame.update_goal_progress("parent")
        goal = next(g for g in frame.active_goals if g.id == "parent")
        assert goal.progress == 0.0

    def test_all_subgoals_completed(self):
        """Goal with all completed subgoals → progress = 1.0."""
        frame = ConversationFrame()
        frame.active_goals.append(
            GoalState(
                id="parent", description="Parent", status="active",
                subgoal_ids=["sub_1", "sub_2"],
            )
        )
        frame.active_goals.append(
            GoalState(id="sub_1", description="Sub 1", status="completed")
        )
        frame.active_goals.append(
            GoalState(id="sub_2", description="Sub 2", status="completed")
        )
        frame.update_goal_progress("parent")
        parent = next(g for g in frame.active_goals if g.id == "parent")
        assert parent.progress == 1.0

    def test_partial_completion(self):
        """One of two subgoals completed → progress = 0.5."""
        frame = ConversationFrame()
        frame.active_goals.append(
            GoalState(
                id="parent", description="Parent", status="active",
                subgoal_ids=["sub_1", "sub_2"],
            )
        )
        frame.active_goals.append(
            GoalState(id="sub_1", description="Sub 1", status="completed")
        )
        frame.active_goals.append(
            GoalState(id="sub_2", description="Sub 2", status="active")
        )
        frame.update_goal_progress("parent")
        parent = next(g for g in frame.active_goals if g.id == "parent")
        assert parent.progress == 0.5

    def test_nonexistent_goal(self):
        """Updating a non-existent goal is a no-op (no crash)."""
        frame = ConversationFrame()
        frame.update_goal_progress("no_such_goal")  # Should not raise
