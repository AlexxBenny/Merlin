# tests/test_executor_type_guard.py

"""
Tests for executor runtime type boundary enforcement.

Ensures:
- Executor rejects dict input with TypeError
- Executor accepts WorldSnapshot input
- Executor accepts None input (backward compat)
- Condition evaluation works with WorldSnapshot for world.* paths
- Condition evaluation returns False when snapshot is None
"""

import pytest

from ir.mission import (
    IR_VERSION,
    MissionPlan,
    MissionNode,
    ConditionExpr,
)
from execution.executor import MissionExecutor
from execution.registry import SkillRegistry
from skills.base import Skill
from skills.contract import SkillContract, FailurePolicy
from skills.skill_result import SkillResult
from ir.mission import ExecutionMode
from world.timeline import WorldTimeline
from world.state import WorldState, MediaState, HardwareState, SystemState
from world.snapshot import WorldSnapshot
from typing import Any, Dict, Optional


# ─────────────────────────────────────────────────────────────
# Test skill
# ─────────────────────────────────────────────────────────────

class PassthroughSkill(Skill):
    contract = SkillContract(
        name="test.passthrough",
        description="Returns input as output",
        domain="test",
        requires_focus=False,
        resource_cost="low",
        inputs={"value": "text"},
        outputs={"result": "text"},
        allowed_modes={ExecutionMode.foreground},
        failure_policy={ExecutionMode.foreground: FailurePolicy.FAIL},
        emits_events=[],
        mutates_world=False,
        idempotent=True,
    )

    def execute(self, inputs, world, snapshot=None):
        return SkillResult(
            outputs={"result": inputs.get("value", "ok")},
            metadata={},
        )


class SnapshotReadingSkill(Skill):
    """Skill that reads snapshot.state — would crash on dict."""
    contract = SkillContract(
        name="test.snapshot_reader",
        description="Reads snapshot state",
        domain="test",
        requires_focus=False,
        resource_cost="low",
        inputs={},
        outputs={"media_playing": "text"},
        allowed_modes={ExecutionMode.foreground},
        failure_policy={ExecutionMode.foreground: FailurePolicy.FAIL},
        emits_events=[],
        mutates_world=False,
        idempotent=True,
    )

    def execute(self, inputs, world, snapshot=None):
        media = snapshot.state.media if snapshot and snapshot.state else None
        is_playing = media.is_playing if media else False
        return SkillResult(
            outputs={"media_playing": is_playing},
            metadata={},
        )


def _make_executor(*skills):
    timeline = WorldTimeline()
    registry = SkillRegistry()
    for skill in skills:
        registry.register(skill, validate_types=False)
    return MissionExecutor(
        registry=registry,
        timeline=timeline,
    ), timeline


def _make_plan(nodes):
    return MissionPlan(
        id="type-guard-test",
        nodes=nodes,
        metadata={"ir_version": IR_VERSION},
    )


def _make_snapshot(**kwargs):
    state = WorldState(**kwargs)
    return WorldSnapshot.build(state, [])


# ─────────────────────────────────────────────────────────────
# Type Guard: Reject dict
# ─────────────────────────────────────────────────────────────

class TestTypeGuard:
    """Executor must reject dict and accept WorldSnapshot or None."""

    def test_rejects_dict(self):
        """Passing a dict must raise TypeError — not silently corrupt."""
        executor, _ = _make_executor(PassthroughSkill())
        plan = _make_plan([
            MissionNode(
                id="n1", skill="test.passthrough",
                inputs={"value": "hello"},
            ),
        ])

        with pytest.raises(TypeError, match="WorldSnapshot"):
            executor.run(plan, world_snapshot={"some": "dict"})

    def test_rejects_model_dump(self):
        """Passing model_dump() result must raise TypeError."""
        executor, _ = _make_executor(PassthroughSkill())
        plan = _make_plan([
            MissionNode(
                id="n1", skill="test.passthrough",
                inputs={"value": "hello"},
            ),
        ])
        snapshot = _make_snapshot()

        with pytest.raises(TypeError, match="WorldSnapshot"):
            executor.run(plan, world_snapshot=snapshot.state.model_dump())

    def test_accepts_world_snapshot(self):
        """Passing WorldSnapshot must work."""
        executor, _ = _make_executor(PassthroughSkill())
        plan = _make_plan([
            MissionNode(
                id="n1", skill="test.passthrough",
                inputs={"value": "hello"},
            ),
        ])
        snapshot = _make_snapshot()

        er = executor.run(plan, world_snapshot=snapshot)
        assert er.results["n1"]["result"] == "hello"

    def test_accepts_none(self):
        """Passing None (backward compat) must work."""
        executor, _ = _make_executor(PassthroughSkill())
        plan = _make_plan([
            MissionNode(
                id="n1", skill="test.passthrough",
                inputs={"value": "hello"},
            ),
        ])

        er = executor.run(plan, world_snapshot=None)
        assert er.results["n1"]["result"] == "hello"

    def test_accepts_no_argument(self):
        """Omitting world_snapshot entirely must work (default None)."""
        executor, _ = _make_executor(PassthroughSkill())
        plan = _make_plan([
            MissionNode(
                id="n1", skill="test.passthrough",
                inputs={"value": "hello"},
            ),
        ])

        er = executor.run(plan)
        assert er.results["n1"]["result"] == "hello"


# ─────────────────────────────────────────────────────────────
# State-aware skills receive WorldSnapshot
# ─────────────────────────────────────────────────────────────

class TestSnapshotFlowToSkills:
    """Verify WorldSnapshot flows correctly from executor to skill."""

    def test_skill_receives_snapshot_with_media_state(self):
        """Skill accessing snapshot.state.media must work via mission path."""
        executor, _ = _make_executor(SnapshotReadingSkill())
        plan = _make_plan([
            MissionNode(
                id="n1", skill="test.snapshot_reader",
            ),
        ])
        snapshot = _make_snapshot(
            media=MediaState(is_playing=True, platform="Spotify"),
        )

        er = executor.run(plan, world_snapshot=snapshot)
        assert er.results["n1"]["media_playing"] is True

    def test_skill_receives_snapshot_no_media(self):
        """Skill with no media state handles None gracefully."""
        executor, _ = _make_executor(SnapshotReadingSkill())
        plan = _make_plan([
            MissionNode(
                id="n1", skill="test.snapshot_reader",
            ),
        ])
        snapshot = _make_snapshot()

        er = executor.run(plan, world_snapshot=snapshot)
        assert er.results["n1"]["media_playing"] is False

    def test_skill_receives_none_snapshot(self):
        """Skill handles None snapshot gracefully."""
        executor, _ = _make_executor(SnapshotReadingSkill())
        plan = _make_plan([
            MissionNode(
                id="n1", skill="test.snapshot_reader",
            ),
        ])

        er = executor.run(plan, world_snapshot=None)
        assert er.results["n1"]["media_playing"] is False


# ─────────────────────────────────────────────────────────────
# Condition evaluation with WorldSnapshot
# ─────────────────────────────────────────────────────────────

class TestConditionWithWorldSnapshot:
    """Condition evaluation on world.* paths works with WorldSnapshot."""

    def test_condition_true_runs_node(self):
        """world.media.is_playing == True when state matches → node runs."""
        executor, _ = _make_executor(PassthroughSkill())
        plan = _make_plan([
            MissionNode(
                id="gated", skill="test.passthrough",
                inputs={"value": "hello"},
                condition_on=ConditionExpr(
                    source="world.media.is_playing",
                    equals=True,
                ),
            ),
        ])
        snapshot = _make_snapshot(
            media=MediaState(is_playing=True),
        )

        er = executor.run(plan, world_snapshot=snapshot)
        assert "gated" in er.results
        assert er.results["gated"]["result"] == "hello"

    def test_condition_false_skips_node(self):
        """world.media.is_playing == True when state is False → node skipped."""
        executor, _ = _make_executor(PassthroughSkill())
        plan = _make_plan([
            MissionNode(
                id="gated", skill="test.passthrough",
                inputs={"value": "hello"},
                condition_on=ConditionExpr(
                    source="world.media.is_playing",
                    equals=True,
                ),
            ),
        ])
        snapshot = _make_snapshot(
            media=MediaState(is_playing=False),
        )

        er = executor.run(plan, world_snapshot=snapshot)
        assert "gated" not in er.results

    def test_condition_none_snapshot_skips(self):
        """world.* condition with None snapshot → node skipped."""
        executor, _ = _make_executor(PassthroughSkill())
        plan = _make_plan([
            MissionNode(
                id="gated", skill="test.passthrough",
                inputs={"value": "hello"},
                condition_on=ConditionExpr(
                    source="world.media.is_playing",
                    equals=True,
                ),
            ),
        ])

        er = executor.run(plan, world_snapshot=None)
        assert "gated" not in er.results

    def test_condition_nested_path(self):
        """world.system.hardware.brightness_percent navigates nested state."""
        executor, _ = _make_executor(PassthroughSkill())
        plan = _make_plan([
            MissionNode(
                id="gated", skill="test.passthrough",
                inputs={"value": "bright"},
                condition_on=ConditionExpr(
                    source="world.system.hardware.brightness_percent",
                    equals=90,
                ),
            ),
        ])
        snapshot = _make_snapshot(
            system=SystemState(
                hardware=HardwareState(brightness_percent=90),
            ),
        )

        er = executor.run(plan, world_snapshot=snapshot)
        assert "gated" in er.results
        assert er.results["gated"]["result"] == "bright"
