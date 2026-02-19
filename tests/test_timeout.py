# tests/test_timeout.py

"""
Tests for Phase 3D: Timeout Enforcement.

Validates:
- Hung skill triggers timeout
- Outcome reflects TIMED_OUT status
- Execution continues safely for independent nodes
- Dependents of timed-out nodes are skipped
"""

import time
import pytest

from ir.mission import IR_VERSION, MissionNode, MissionPlan, ExecutionMode
from execution.executor import MissionExecutor, NodeStatus
from execution.registry import SkillRegistry
from skills.contract import SkillContract, FailurePolicy
from skills.skill_result import SkillResult
from world.timeline import WorldTimeline


class HangingSkill:
    """A skill that sleeps forever (simulates a hang)."""
    name = "test.hang"
    input_keys = set()
    output_keys = set()
    contract = SkillContract(
        name="test.hang",
        inputs={},
        outputs={},
        allowed_modes={ExecutionMode.foreground},
        failure_policy={ExecutionMode.foreground: FailurePolicy.CONTINUE},
        emits_events=[],
        mutates_world=False,
    )

    def execute(self, inputs, timeline, snapshot=None):
        time.sleep(5)  # Will be interrupted by timeout
        return SkillResult(outputs={})


class FastSkill:
    """A skill that completes immediately."""
    name = "test.fast"
    input_keys = set()
    output_keys = {"result"}
    contract = SkillContract(
        name="test.fast",
        inputs={},
        outputs={"result": "string"},
        allowed_modes={ExecutionMode.foreground},
        failure_policy={},
        emits_events=[],
        mutates_world=False,
    )

    def execute(self, inputs, timeline, snapshot=None):
        return SkillResult(outputs={"result": "done"})


def _make_plan(nodes_spec):
    nodes = []
    for spec in nodes_spec:
        nid, skill = spec[0], spec[1]
        deps = spec[2] if len(spec) > 2 else []
        nodes.append(MissionNode(
            id=nid,
            skill=skill,
            depends_on=deps,
            mode=ExecutionMode.foreground,
        ))
    return MissionPlan(
        id="timeout_test",
        nodes=nodes,
        metadata={"ir_version": IR_VERSION},
    )


class TestTimeoutSingleNode:
    """Timeout on a single-node plan."""

    def test_hung_skill_times_out(self):
        registry = SkillRegistry()
        registry.register(HangingSkill(), validate_types=False)
        timeline = WorldTimeline()

        executor = MissionExecutor(
            registry, timeline,
            node_timeout=0.5,  # 500ms timeout
        )

        plan = _make_plan([("hang_node", "test.hang")])
        er = executor.run(plan)

        # Node should not be in results (it timed out)
        assert "hang_node" not in er.results
        # Typed status preserved
        assert er.node_statuses["hang_node"] == NodeStatus.TIMED_OUT

    def test_fast_skill_completes_within_timeout(self):
        registry = SkillRegistry()
        registry.register(FastSkill(), validate_types=False)
        timeline = WorldTimeline()

        executor = MissionExecutor(
            registry, timeline,
            node_timeout=5.0,
        )

        plan = _make_plan([("fast_node", "test.fast")])
        er = executor.run(plan)

        assert "fast_node" in er.results
        assert er.results["fast_node"]["result"] == "done"


class TestTimeoutDependentNodes:
    """Verify dependents of timed-out nodes are skipped."""

    def test_dependent_skipped_after_timeout(self):
        registry = SkillRegistry()
        registry.register(HangingSkill(), validate_types=False)
        registry.register(FastSkill(), validate_types=False)
        timeline = WorldTimeline()

        executor = MissionExecutor(
            registry, timeline,
            node_timeout=0.5,
        )

        # hang_node → fast_dependent
        plan = _make_plan([
            ("hang_node", "test.hang"),
            ("fast_dependent", "test.fast", ["hang_node"]),
        ])
        er = executor.run(plan)

        # hang_node timed out → fast_dependent should be skipped
        assert "hang_node" not in er.results
        assert "fast_dependent" not in er.results
        assert er.node_statuses["hang_node"] == NodeStatus.TIMED_OUT
        assert er.node_statuses["fast_dependent"] == NodeStatus.SKIPPED


class TestTimeoutWithParallelNodes:
    """Timeout in parallel layer."""

    def test_one_hangs_others_complete(self):
        registry = SkillRegistry()
        registry.register(HangingSkill(), validate_types=False)
        registry.register(FastSkill(), validate_types=False)
        timeline = WorldTimeline()

        executor = MissionExecutor(
            registry, timeline,
            node_timeout=0.5,
        )

        # Two independent nodes: one hangs, one fast
        plan = _make_plan([
            ("hang_node", "test.hang"),
            ("fast_node", "test.fast"),
        ])
        er = executor.run(plan)

        # fast_node should complete fine
        assert "fast_node" in er.results
        assert er.results["fast_node"]["result"] == "done"

        # hang_node should have timed out
        assert "hang_node" not in er.results
        assert er.node_statuses["hang_node"] == NodeStatus.TIMED_OUT


class TestNoTimeout:
    """Without timeout configured, skills run indefinitely (tested with fast skill)."""

    def test_no_timeout_completes_normally(self):
        registry = SkillRegistry()
        registry.register(FastSkill(), validate_types=False)
        timeline = WorldTimeline()

        # No timeout set
        executor = MissionExecutor(registry, timeline)

        plan = _make_plan([("fast_node", "test.fast")])
        er = executor.run(plan)

        assert er.results["fast_node"]["result"] == "done"
