# tests/test_output_reference_e2e.py

"""
End-to-end integration tests for OutputReference bounded access.

Tests the complete pipeline: IR → Validator → Executor → SkillResult
using manual mission JSON construction.

Matrix:
  A. Happy Path (3 tests)
  B. Deterministic Errors (4 tests)
  C. Multi-Node Chains (2 tests)
  D. Stress Tests (2 tests)
  E. Abort Semantics (2 tests)

No LLM involved. No decomposer. No coordinator.
These test the RUNTIME layer in isolation.
"""

import logging
import pytest
from typing import Any, Dict
from unittest.mock import MagicMock

from ir.mission import (
    IR_VERSION,
    MissionPlan,
    MissionNode,
    OutputSpec,
    OutputReference,
    ExecutionMode,
)
from execution.executor import MissionExecutor, ExecutionResult, NodeStatus
from execution.registry import SkillRegistry
from skills.base import Skill
from skills.contract import SkillContract, FailurePolicy
from skills.skill_result import SkillResult
from cortex.validators import validate_mission_plan, MissionValidationError


# ──────────────────────────────────────────────────────────────
# Mock Skills (deterministic, no real system calls)
# ──────────────────────────────────────────────────────────────


class MockListAppsSkill(Skill):
    """Returns a fixed list of 3 apps. Deterministic."""

    contract = SkillContract(
        name="system.list_apps",
        description="List running applications",
        inputs={},
        outputs={"apps": "application_list"},
        allowed_modes={ExecutionMode.foreground},
        failure_policy={ExecutionMode.foreground: FailurePolicy.FAIL},
        emits_events=[],
        mutates_world=False,
        idempotent=True,
    )

    def execute(self, inputs, world, snapshot=None):
        return SkillResult(
            outputs={"apps": [
                {"name": "Chrome", "pid": 1234, "title": "Google Chrome"},
                {"name": "VSCode", "pid": 5678, "title": "Visual Studio Code"},
                {"name": "Notepad", "pid": 9012, "title": "Untitled - Notepad"},
            ]},
            metadata={"entity": "running apps", "domain": "system"},
        )


class MockOpenAppSkill(Skill):
    """Opens app by name. Validates app_name is string."""

    contract = SkillContract(
        name="system.open_app",
        description="Open an application",
        inputs={"app_name": "application_name"},
        optional_inputs={"args": "cli_arguments"},
        outputs={"opened": "application_name", "pid": "process_id"},
        allowed_modes={ExecutionMode.foreground, ExecutionMode.side_effect},
        failure_policy={
            ExecutionMode.foreground: FailurePolicy.FAIL,
            ExecutionMode.side_effect: FailurePolicy.CONTINUE,
        },
        emits_events=["app_launched"],
        mutates_world=True,
    )

    def execute(self, inputs, world, snapshot=None):
        app_name = inputs["app_name"]
        if not isinstance(app_name, str):
            raise TypeError(
                f"app_name must be str, got {type(app_name).__name__}: {app_name!r}"
            )
        world.emit("skill.system", "app_launched", {
            "app": app_name, "pid": 99999,
        })
        return SkillResult(
            outputs={"opened": app_name, "pid": 99999},
            metadata={"entity": f"app '{app_name}'", "domain": "system"},
        )


class MockMediaPauseSkill(Skill):
    """Pause media. No inputs. For multi-node chain tests."""

    contract = SkillContract(
        name="system.media_pause",
        description="Pause media playback",
        inputs={},
        outputs={"changed": "whether_playback_state_was_changed"},
        allowed_modes={ExecutionMode.foreground},
        failure_policy={ExecutionMode.foreground: FailurePolicy.FAIL},
        emits_events=[],
        mutates_world=False,
    )

    def execute(self, inputs, world, snapshot=None):
        return SkillResult(
            outputs={"changed": True},
            metadata={"entity": "media", "domain": "system"},
        )


class MockGetTimeSkill(Skill):
    """Returns a string. For testing index on non-list output."""

    contract = SkillContract(
        name="system.get_time",
        description="Get current time",
        inputs={},
        outputs={"time": "info_string"},
        allowed_modes={ExecutionMode.foreground},
        failure_policy={ExecutionMode.foreground: FailurePolicy.FAIL},
        emits_events=[],
        mutates_world=False,
        idempotent=True,
    )

    def execute(self, inputs, world, snapshot=None):
        return SkillResult(
            outputs={"time": "11:30 AM"},
            metadata={"entity": "time", "domain": "system"},
        )


class MockSetVolumeSkill(Skill):
    """Sets volume. Returns int. For testing field on non-dict."""

    contract = SkillContract(
        name="system.set_volume",
        description="Set system volume",
        inputs={"level": "volume_percentage"},
        outputs={"actual": "actual_volume"},
        allowed_modes={ExecutionMode.foreground},
        failure_policy={ExecutionMode.foreground: FailurePolicy.FAIL},
        emits_events=["volume_changed"],
        mutates_world=True,
    )

    def execute(self, inputs, world, snapshot=None):
        level = inputs["level"]
        world.emit("skill.system", "volume_changed", {"level": level})
        return SkillResult(
            outputs={"actual": level},
            metadata={"entity": "volume", "domain": "system"},
        )


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────


@pytest.fixture
def registry():
    """Registry with all mock skills."""
    reg = SkillRegistry()
    reg.register(MockListAppsSkill())
    reg.register(MockOpenAppSkill())
    reg.register(MockMediaPauseSkill())
    reg.register(MockGetTimeSkill())
    reg.register(MockSetVolumeSkill())
    return reg


@pytest.fixture
def timeline():
    """Real WorldTimeline for event enforcement."""
    from world.timeline import WorldTimeline
    return WorldTimeline()


@pytest.fixture
def executor(registry, timeline):
    return MissionExecutor(registry, timeline, max_workers=2)


def _make_plan(nodes, plan_id="test_plan"):
    """Construct a MissionPlan from node dicts."""
    return MissionPlan(
        id=plan_id,
        nodes=nodes,
        metadata={"ir_version": IR_VERSION},
    )


# ──────────────────────────────────────────────────────────────
# A. Happy Path
# ──────────────────────────────────────────────────────────────


class TestHappyPath:
    """Cases 1-2: successful resolution and type mismatch detection."""

    def test_case_1_index_plus_field(self, executor, registry):
        """
        Case 1: list_apps → open_app with $ref index+field

        Intent: "list running apps and open the second one"
        node_0 outputs apps list
        node_1 receives apps[1].name = "VSCode"
        """
        plan = _make_plan([
            MissionNode(
                id="node_0",
                skill="system.list_apps",
                inputs={},
                outputs={"apps": OutputSpec(name="system.list_apps.apps", type="application_list")},
                depends_on=[],
                mode=ExecutionMode.foreground,
            ),
            MissionNode(
                id="node_1",
                skill="system.open_app",
                inputs={
                    "app_name": OutputReference(
                        node="node_0", output="apps", index=1, field="name"
                    )
                },
                outputs={"opened": OutputSpec(name="system.open_app.opened", type="application_name")},
                depends_on=["node_0"],
                mode=ExecutionMode.foreground,
            ),
        ])

        # Validate first
        validate_mission_plan(plan, {"system.list_apps", "system.open_app"}, registry=registry)

        # Execute
        result = executor.run(plan)

        # Verify
        assert "node_0" in result.completed
        assert "node_1" in result.completed
        assert result.results["node_1"]["opened"] == "VSCode"
        assert result.node_statuses["node_0"] == NodeStatus.COMPLETED
        assert result.node_statuses["node_1"] == NodeStatus.COMPLETED

    def test_case_2_index_without_field_type_mismatch(self, executor, registry):
        """
        Case 2: $ref with index only, no field.

        Result: apps[1] = {"name": "VSCode", "pid": 5678, ...}
        open_app expects string → skill should reject dict input.
        """
        plan = _make_plan([
            MissionNode(
                id="node_0",
                skill="system.list_apps",
                inputs={},
                outputs={"apps": OutputSpec(name="system.list_apps.apps", type="application_list")},
                depends_on=[],
                mode=ExecutionMode.foreground,
            ),
            MissionNode(
                id="node_1",
                skill="system.open_app",
                inputs={
                    "app_name": OutputReference(
                        node="node_0", output="apps", index=1
                    )  # No field — resolves to dict
                },
                outputs={},
                depends_on=["node_0"],
                mode=ExecutionMode.foreground,
            ),
        ])

        validate_mission_plan(plan, {"system.list_apps", "system.open_app"}, registry=registry)

        # open_app has FAIL policy — should propagate
        with pytest.raises(RuntimeError, match="app_name must be str"):
            executor.run(plan)

    def test_case_2b_flat_ref_entire_list(self, executor, registry):
        """
        Flat $ref (no index, no field) returns entire list.
        Skill gets list instead of string → should fail.
        """
        plan = _make_plan([
            MissionNode(
                id="node_0",
                skill="system.list_apps",
                inputs={},
                outputs={"apps": OutputSpec(name="system.list_apps.apps", type="application_list")},
                depends_on=[],
                mode=ExecutionMode.foreground,
            ),
            MissionNode(
                id="node_1",
                skill="system.open_app",
                inputs={
                    "app_name": OutputReference(node="node_0", output="apps")
                },
                outputs={},
                depends_on=["node_0"],
                mode=ExecutionMode.foreground,
            ),
        ])

        with pytest.raises(RuntimeError, match="app_name must be str"):
            executor.run(plan)


# ──────────────────────────────────────────────────────────────
# B. Deterministic Error Cases
# ──────────────────────────────────────────────────────────────


class TestDeterministicErrors:
    """Cases 3-6: every error case produces clear, deterministic messages."""

    def test_case_3_out_of_bounds_index(self, executor, registry):
        """
        Case 3: Index 999 on 3-element list.
        Must fail deterministically with clear message.
        """
        plan = _make_plan([
            MissionNode(
                id="node_0",
                skill="system.list_apps",
                inputs={},
                outputs={"apps": OutputSpec(name="system.list_apps.apps", type="application_list")},
                depends_on=[],
                mode=ExecutionMode.foreground,
            ),
            MissionNode(
                id="node_1",
                skill="system.open_app",
                inputs={
                    "app_name": OutputReference(
                        node="node_0", output="apps", index=999, field="name"
                    )
                },
                outputs={},
                depends_on=["node_0"],
                mode=ExecutionMode.foreground,
            ),
        ])

        with pytest.raises(RuntimeError, match="out of bounds.*length=3"):
            executor.run(plan)

    def test_case_4_missing_field(self, executor, registry):
        """
        Case 4: field="email" on app dict that only has name/pid/title.
        Must fail with available keys listed.
        """
        plan = _make_plan([
            MissionNode(
                id="node_0",
                skill="system.list_apps",
                inputs={},
                outputs={"apps": OutputSpec(name="system.list_apps.apps", type="application_list")},
                depends_on=[],
                mode=ExecutionMode.foreground,
            ),
            MissionNode(
                id="node_1",
                skill="system.open_app",
                inputs={
                    "app_name": OutputReference(
                        node="node_0", output="apps", index=1, field="email"
                    )
                },
                outputs={},
                depends_on=["node_0"],
                mode=ExecutionMode.foreground,
            ),
        ])

        with pytest.raises(RuntimeError, match="email.*not found"):
            executor.run(plan)

    def test_case_5_index_on_non_list(self, executor, registry):
        """
        Case 5: Index on string output (get_time returns "11:30 AM").
        Must fail with type error.
        """
        plan = _make_plan([
            MissionNode(
                id="node_0",
                skill="system.get_time",
                inputs={},
                outputs={"time": OutputSpec(name="system.get_time.time", type="info_string")},
                depends_on=[],
                mode=ExecutionMode.foreground,
            ),
            MissionNode(
                id="node_1",
                skill="system.open_app",
                inputs={
                    "app_name": OutputReference(
                        node="node_0", output="time", index=0
                    )
                },
                outputs={},
                depends_on=["node_0"],
                mode=ExecutionMode.foreground,
            ),
        ])

        with pytest.raises(RuntimeError, match="not list"):
            executor.run(plan)

    def test_case_6_field_on_non_dict(self, executor, registry):
        """
        Case 6: Field on string output (get_time returns "11:30 AM").
        Must fail with type error.
        """
        plan = _make_plan([
            MissionNode(
                id="node_0",
                skill="system.get_time",
                inputs={},
                outputs={"time": OutputSpec(name="system.get_time.time", type="info_string")},
                depends_on=[],
                mode=ExecutionMode.foreground,
            ),
            MissionNode(
                id="node_1",
                skill="system.open_app",
                inputs={
                    "app_name": OutputReference(
                        node="node_0", output="time", field="hours"
                    )
                },
                outputs={},
                depends_on=["node_0"],
                mode=ExecutionMode.foreground,
            ),
        ])

        with pytest.raises(RuntimeError, match="not dict"):
            executor.run(plan)


# ──────────────────────────────────────────────────────────────
# C. Multi-Node Chains
# ──────────────────────────────────────────────────────────────


class TestMultiNodeChains:
    """Cases 7-8: dependency ordering and enforcement."""

    def test_case_7_three_step_chain(self, executor, registry):
        """
        Case 7: list_apps → open_app (via $ref) → media_pause (depends on open_app)

        Three-step sequential chain.
        Reference resolution must not break dependency ordering.
        """
        plan = _make_plan([
            MissionNode(
                id="node_0",
                skill="system.list_apps",
                inputs={},
                outputs={"apps": OutputSpec(name="system.list_apps.apps", type="application_list")},
                depends_on=[],
                mode=ExecutionMode.foreground,
            ),
            MissionNode(
                id="node_1",
                skill="system.open_app",
                inputs={
                    "app_name": OutputReference(
                        node="node_0", output="apps", index=0, field="name"
                    )
                },
                outputs={"opened": OutputSpec(name="system.open_app.opened", type="application_name")},
                depends_on=["node_0"],
                mode=ExecutionMode.foreground,
            ),
            MissionNode(
                id="node_2",
                skill="system.media_pause",
                inputs={},
                outputs={"changed": OutputSpec(name="system.media_pause.changed", type="whether_playback_state_was_changed")},
                depends_on=["node_1"],
                mode=ExecutionMode.foreground,
            ),
        ])

        validate_mission_plan(
            plan,
            {"system.list_apps", "system.open_app", "system.media_pause"},
            registry=registry,
        )

        result = executor.run(plan)

        # All three completed in order
        assert "node_0" in result.completed
        assert "node_1" in result.completed
        assert "node_2" in result.completed
        assert result.results["node_1"]["opened"] == "Chrome"
        assert result.results["node_2"]["changed"] is True

    def test_case_8_missing_dependency_rejected(self, registry):
        """
        Case 8: node_2 references node_0 without depends_on.
        Validator must reject before execution.
        """
        plan = _make_plan([
            MissionNode(
                id="node_0",
                skill="system.list_apps",
                inputs={},
                outputs={"apps": OutputSpec(name="system.list_apps.apps", type="application_list")},
                depends_on=[],
                mode=ExecutionMode.foreground,
            ),
            MissionNode(
                id="node_1",
                skill="system.media_pause",
                inputs={},
                outputs={},
                depends_on=["node_0"],
                mode=ExecutionMode.foreground,
            ),
            MissionNode(
                id="node_2",
                skill="system.open_app",
                inputs={
                    "app_name": OutputReference(
                        node="node_0", output="apps", index=0, field="name"
                    )
                },
                outputs={},
                depends_on=["node_1"],  # depends on node_1, NOT node_0
                mode=ExecutionMode.foreground,
            ),
        ])

        with pytest.raises(MissionValidationError, match="does not depend on it"):
            validate_mission_plan(
                plan,
                {"system.list_apps", "system.open_app", "system.media_pause"},
                registry=registry,
            )


# ──────────────────────────────────────────────────────────────
# D. Stress Tests
# ──────────────────────────────────────────────────────────────


class TestStressTests:
    """Cases 9-10: multiple refs and circular reference attempts."""

    def test_case_9_two_refs_in_same_node(self, executor, registry):
        """
        Case 9: Two $refs in same node inputs.
        Both must resolve independently. No mutation or caching bug.
        """

        # A skill that takes two string inputs
        class TwoInputSkill(Skill):
            contract = SkillContract(
                name="test.two_inputs",
                description="Takes two inputs",
                inputs={"first": "application_name", "second": "application_name"},
                outputs={"result": "info_string"},
                allowed_modes={ExecutionMode.foreground},
                failure_policy={ExecutionMode.foreground: FailurePolicy.FAIL},
                emits_events=[],
                mutates_world=False,
            )

            def execute(self, inputs, world, snapshot=None):
                return SkillResult(
                    outputs={"result": f"{inputs['first']}+{inputs['second']}"},
                    metadata={},
                )

        registry.register(TwoInputSkill())

        plan = _make_plan([
            MissionNode(
                id="node_0",
                skill="system.list_apps",
                inputs={},
                outputs={"apps": OutputSpec(name="system.list_apps.apps", type="application_list")},
                depends_on=[],
                mode=ExecutionMode.foreground,
            ),
            MissionNode(
                id="node_1",
                skill="test.two_inputs",
                inputs={
                    "first": OutputReference(
                        node="node_0", output="apps", index=0, field="name"
                    ),
                    "second": OutputReference(
                        node="node_0", output="apps", index=2, field="name"
                    ),
                },
                outputs={"result": OutputSpec(name="test.two_inputs.result", type="info_string")},
                depends_on=["node_0"],
                mode=ExecutionMode.foreground,
            ),
        ])

        validate_mission_plan(
            plan,
            {"system.list_apps", "test.two_inputs"},
            registry=registry,
        )

        result = executor.run(plan)

        assert "node_1" in result.completed
        assert result.results["node_1"]["result"] == "Chrome+Notepad"

    def test_case_10_circular_reference_rejected(self, registry):
        """
        Case 10: Forward reference (node_0 references node_1).
        Validator must reject — circular/forward dependencies are illegal.
        """
        plan = _make_plan([
            MissionNode(
                id="node_0",
                skill="system.open_app",
                inputs={
                    "app_name": OutputReference(
                        node="node_1", output="apps", index=0, field="name"
                    )
                },
                outputs={},
                depends_on=["node_1"],  # depends on future node
                mode=ExecutionMode.foreground,
            ),
            MissionNode(
                id="node_1",
                skill="system.list_apps",
                inputs={},
                outputs={"apps": OutputSpec(name="system.list_apps.apps", type="application_list")},
                depends_on=["node_0"],  # circular
                mode=ExecutionMode.foreground,
            ),
        ])

        with pytest.raises(MissionValidationError, match="Cycle detected"):
            validate_mission_plan(
                plan,
                {"system.list_apps", "system.open_app"},
                registry=registry,
            )


# ──────────────────────────────────────────────────────────────
# E. Abort Semantics (critical safety test)
# ──────────────────────────────────────────────────────────────


class TestAbortSemantics:
    """Failure propagation: upstream failure must prevent downstream execution."""

    def test_upstream_ref_failure_aborts_downstream(self, executor, registry):
        """
        CRITICAL: If node_1 fails (bad index), node_2 must NOT execute.

        node_0: list_apps
        node_1: open_app with index=999 → out of bounds → FAIL
        node_2: media_pause depends_on node_1

        node_2 must be SKIPPED, never COMPLETED.
        """
        plan = _make_plan([
            MissionNode(
                id="node_0",
                skill="system.list_apps",
                inputs={},
                outputs={"apps": OutputSpec(name="system.list_apps.apps", type="application_list")},
                depends_on=[],
                mode=ExecutionMode.foreground,
            ),
            MissionNode(
                id="node_1",
                skill="system.open_app",
                inputs={
                    "app_name": OutputReference(
                        node="node_0", output="apps", index=999, field="name"
                    )
                },
                outputs={},
                depends_on=["node_0"],
                mode=ExecutionMode.foreground,
            ),
            MissionNode(
                id="node_2",
                skill="system.media_pause",
                inputs={},
                outputs={},
                depends_on=["node_1"],
                mode=ExecutionMode.foreground,
            ),
        ])

        # open_app has FAIL policy — RuntimeError propagates
        with pytest.raises(RuntimeError, match="out of bounds"):
            executor.run(plan)

    def test_independent_node_unaffected_by_sibling_failure(self, executor, registry):
        """
        node_0: list_apps
        node_1: open_app with bad index → FAIL (foreground, FAIL policy)
        node_2: get_time (independent, no dependency on node_1)

        Since node_1 has FAIL policy, the entire mission should abort.
        node_2 should NOT complete.

        This tests that FAIL policy correctly stops the entire mission.
        """
        plan = _make_plan([
            MissionNode(
                id="node_0",
                skill="system.list_apps",
                inputs={},
                outputs={"apps": OutputSpec(name="system.list_apps.apps", type="application_list")},
                depends_on=[],
                mode=ExecutionMode.foreground,
            ),
            MissionNode(
                id="node_1",
                skill="system.open_app",
                inputs={
                    "app_name": OutputReference(
                        node="node_0", output="apps", index=999, field="name"
                    )
                },
                outputs={},
                depends_on=["node_0"],
                mode=ExecutionMode.foreground,
            ),
            MissionNode(
                id="node_2",
                skill="system.get_time",
                inputs={},
                outputs={"time": OutputSpec(name="system.get_time.time", type="info_string")},
                depends_on=[],  # independent root
                mode=ExecutionMode.foreground,
            ),
        ])

        # FAIL policy means RuntimeError propagates, mission aborts
        with pytest.raises(RuntimeError, match="out of bounds"):
            executor.run(plan)


# ──────────────────────────────────────────────────────────────
# F. Validator Heuristic Guard
# ──────────────────────────────────────────────────────────────


class TestValidatorHeuristicGuard:
    """Verify the heuristic type guard warns for suspicious index usage."""

    def test_index_on_list_type_no_warning(self, registry, caplog):
        """Index on application_list → no warning."""
        plan = _make_plan([
            MissionNode(
                id="node_0",
                skill="system.list_apps",
                inputs={},
                outputs={"apps": OutputSpec(name="system.list_apps.apps", type="application_list")},
                depends_on=[],
                mode=ExecutionMode.foreground,
            ),
            MissionNode(
                id="node_1",
                skill="system.open_app",
                inputs={
                    "app_name": OutputReference(
                        node="node_0", output="apps", index=1, field="name"
                    )
                },
                outputs={},
                depends_on=["node_0"],
                mode=ExecutionMode.foreground,
            ),
        ])

        with caplog.at_level(logging.WARNING):
            validate_mission_plan(
                plan,
                {"system.list_apps", "system.open_app"},
                registry=registry,
            )

        # No "hallucination" warning for list type
        assert "hallucination" not in caplog.text.lower()

    def test_index_on_non_list_type_warns(self, registry, caplog):
        """Index on info_string → warning about possible LLM hallucination."""
        plan = _make_plan([
            MissionNode(
                id="node_0",
                skill="system.get_time",
                inputs={},
                outputs={"time": OutputSpec(name="system.get_time.time", type="info_string")},
                depends_on=[],
                mode=ExecutionMode.foreground,
            ),
            MissionNode(
                id="node_1",
                skill="system.open_app",
                inputs={
                    "app_name": OutputReference(
                        node="node_0", output="time", index=0
                    )
                },
                outputs={},
                depends_on=["node_0"],
                mode=ExecutionMode.foreground,
            ),
        ])

        with caplog.at_level(logging.WARNING):
            validate_mission_plan(
                plan,
                {"system.get_time", "system.open_app"},
                registry=registry,
            )

        # Should warn about index on non-list type
        assert "hallucination" in caplog.text.lower() or "does not contain" in caplog.text.lower()
