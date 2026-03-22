"""
IR v1 + Skill Contract v1 Compliance Tests

Tests every freeze-integrity constraint:
- IR model validation (§1)
- Executor enforcement (§2)
- DAG validator (§3)
- Skill Contract enforcement (§4)
"""

import pytest
from pydantic import ValidationError

from ir.mission import (
    IR_VERSION,
    ConditionExpr,
    ExecutionMode,
    MissionNode,
    MissionPlan,
    OutputReference,
    OutputSpec,
)
from cortex.validators import MissionValidationError, validate_mission_plan
from execution.executor import MissionExecutor
from execution.registry import SkillRegistry
from skills.base import Skill
from skills.skill_result import SkillResult
from skills.contract import SkillContract, FailurePolicy
from world.timeline import WorldTimeline


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

class DummySkill(Skill):
    """Minimal foreground-only skill for testing."""
    contract = SkillContract(
        name="test.dummy",
        description="Test skill",
        inputs={"input_a": "text"},
        outputs={"result": "text"},
        allowed_modes={ExecutionMode.foreground},
        failure_policy={ExecutionMode.foreground: FailurePolicy.FAIL},
    )

    def execute(self, inputs, world, snapshot=None):
        return SkillResult(outputs={"result": inputs.get("input_a", "ok")})


class FailingSkill(Skill):
    """Skill that always fails. Allows all modes for testing."""
    contract = SkillContract(
        name="test.failing",
        description="Always fails",
        inputs={},
        outputs={},
        allowed_modes={
            ExecutionMode.foreground,
            ExecutionMode.background,
            ExecutionMode.side_effect,
        },
        failure_policy={
            ExecutionMode.foreground: FailurePolicy.FAIL,
            ExecutionMode.background: FailurePolicy.CONTINUE,
            ExecutionMode.side_effect: FailurePolicy.IGNORE,
        },
    )

    def execute(self, inputs, world, snapshot=None):
        raise RuntimeError("Intentional failure")


class EventEmittingSkill(Skill):
    """Skill that emits a declared world event."""
    contract = SkillContract(
        name="test.emitter",
        description="Emits events",
        inputs={"msg": "text"},
        outputs={"status": "text"},
        allowed_modes={ExecutionMode.foreground},
        failure_policy={ExecutionMode.foreground: FailurePolicy.FAIL},
        emits_events=["thing_happened"],
        mutates_world=True,
    )

    def execute(self, inputs, world, snapshot=None):
        world.emit(self.contract.name, "thing_happened", {"msg": inputs["msg"]})
        return SkillResult(outputs={"status": "done"})


class UndeclaredEventSkill(Skill):
    """Skill that emits an event NOT declared in its contract."""
    contract = SkillContract(
        name="test.sneaky",
        description="Emits undeclared events",
        inputs={},
        outputs={"status": "text"},
        allowed_modes={ExecutionMode.foreground},
        failure_policy={ExecutionMode.foreground: FailurePolicy.FAIL},
        emits_events=["declared_event"],
        mutates_world=True,
    )

    def execute(self, inputs, world, snapshot=None):
        world.emit(self.contract.name, "undeclared_event", {})
        return SkillResult(outputs={"status": "done"})


class NoMutateButEmitsSkill(Skill):
    """Skill that claims mutates_world=False but emits anyway."""
    contract = SkillContract(
        name="test.liar",
        description="Claims no mutation",
        inputs={},
        outputs={"status": "text"},
        allowed_modes={ExecutionMode.foreground},
        failure_policy={ExecutionMode.foreground: FailurePolicy.FAIL},
        mutates_world=False,
    )

    def execute(self, inputs, world, snapshot=None):
        world.emit(self.contract.name, "illegal_event", {})
        return SkillResult(outputs={"status": "done"})


class ForegroundOnlySkill(Skill):
    """Skill that only allows foreground mode."""
    contract = SkillContract(
        name="test.fgonly",
        description="Foreground only",
        inputs={},
        outputs={"status": "text"},
        allowed_modes={ExecutionMode.foreground},
        failure_policy={ExecutionMode.foreground: FailurePolicy.FAIL},
    )

    def execute(self, inputs, world, snapshot=None):
        return SkillResult(outputs={"status": "ok"})


def _make_registry(*skills) -> SkillRegistry:
    reg = SkillRegistry()
    for s in skills:
        reg.register(s, validate_types=False)
    return reg


def _make_executor(*skills) -> tuple:
    """Returns (executor, timeline) tuple."""
    reg = _make_registry(*skills)
    tl = WorldTimeline()
    return MissionExecutor(reg, tl), tl


def _make_plan(
    nodes,
    plan_id="test_plan",
    ir_version=IR_VERSION,
) -> MissionPlan:
    return MissionPlan(
        id=plan_id,
        nodes=nodes,
        metadata={"ir_version": ir_version},
    )


# ==================================================================
# 1. IR Model Validation Tests
# ==================================================================


class TestOutputReferenceRejectsDollarStrings:
    """
    Verify IR model validation rejects $-prefixed strings in inputs.
    Enforcement point 1 of 2.
    """

    def test_dollar_string_in_input_rejected(self):
        with pytest.raises(ValidationError, match=r"banned \$-prefixed"):
            MissionNode(
                id="n1",
                skill="test.dummy",
                inputs={"key": "$node.output"},
            )

    def test_dollar_ref_string_rejected(self):
        with pytest.raises(ValidationError, match=r"banned \$-prefixed"):
            MissionNode(
                id="n1",
                skill="test.dummy",
                inputs={"key": "$ref.something"},
            )

    def test_literal_string_accepted(self):
        node = MissionNode(
            id="n1",
            skill="test.dummy",
            inputs={"key": "normal string"},
        )
        assert node.inputs["key"] == "normal string"

    def test_output_reference_accepted(self):
        ref = OutputReference(node="other", output="result")
        node = MissionNode(
            id="n1",
            skill="test.dummy",
            inputs={"key": ref},
        )
        assert isinstance(node.inputs["key"], OutputReference)


class TestMissionPlanRequiresIrVersion:
    """Verify MissionPlan raises if ir_version is missing."""

    def test_missing_ir_version_raises(self):
        with pytest.raises(ValidationError, match="ir_version"):
            MissionPlan(
                id="plan_1",
                nodes=[],
                metadata={},
            )

    def test_wrong_ir_version_raises(self):
        with pytest.raises(ValidationError, match="Unsupported IR version"):
            MissionPlan(
                id="plan_1",
                nodes=[],
                metadata={"ir_version": "2.0"},
            )

    def test_correct_ir_version_passes(self):
        plan = MissionPlan(
            id="plan_1",
            nodes=[],
            metadata={"ir_version": "1.0"},
        )
        assert plan.ir_version == "1.0"


class TestMissionPlanRequiresId:
    """Verify MissionPlan requires id field."""

    def test_missing_id_raises(self):
        with pytest.raises(ValidationError):
            MissionPlan(
                nodes=[],
                metadata={"ir_version": "1.0"},
            )


class TestMissionNodeSkillFormat:
    """Verify skills must match the frozen regex."""

    def test_valid_two_part(self):
        node = MissionNode(id="n1", skill="browser.search")
        assert node.skill == "browser.search"

    def test_valid_three_part(self):
        node = MissionNode(id="n1", skill="media.play.youtube")
        assert node.skill == "media.play.youtube"

    def test_single_part_rejected(self):
        with pytest.raises(ValidationError, match="domain.action"):
            MissionNode(id="n1", skill="browser")

    def test_uppercase_rejected(self):
        with pytest.raises(ValidationError, match="domain.action"):
            MissionNode(id="n1", skill="Browser.Search")

    def test_hyphen_rejected(self):
        with pytest.raises(ValidationError, match="domain.action"):
            MissionNode(id="n1", skill="browser.web-search")

    def test_four_parts_rejected(self):
        with pytest.raises(ValidationError, match="domain.action"):
            MissionNode(id="n1", skill="a.b.c.d")


class TestIrRejectsUnknownFields:
    """
    Verify extra fields on MissionNode/MissionPlan cause validation failure.
    Prevents silent ABI drift.
    """

    def test_extra_field_on_node_rejected(self):
        with pytest.raises(ValidationError):
            MissionNode(
                id="n1",
                skill="test.dummy",
                extra_field="should_fail",
            )

    def test_extra_field_on_plan_rejected(self):
        with pytest.raises(ValidationError):
            MissionPlan(
                id="plan_1",
                nodes=[],
                metadata={"ir_version": "1.0"},
                extra_field="should_fail",
            )

    def test_extra_field_on_output_spec_rejected(self):
        with pytest.raises(ValidationError):
            OutputSpec(
                name="test.output.v1",
                type="text",
                extra="should_fail",
            )

    def test_extra_field_on_output_reference_rejected(self):
        with pytest.raises(ValidationError):
            OutputReference(
                node="n1",
                output="result",
                extra="should_fail",
            )


class TestConditionExprSourceValidation:
    """Verify ConditionExpr.source namespace enforcement."""

    def test_valid_node_id(self):
        cond = ConditionExpr(source="check_calendar", equals=True)
        assert cond.source == "check_calendar"

    def test_valid_world_namespace(self):
        cond = ConditionExpr(source="world.calendar.today.busy", equals=True)
        assert cond.source == "world.calendar.today.busy"

    def test_invalid_source_rejected(self):
        with pytest.raises(ValidationError, match="ConditionExpr.source"):
            ConditionExpr(source="Invalid-Source!", equals=True)


# ==================================================================
# 2. Executor Tests (IR v1 + Skill Contract enforcement)
# ==================================================================


class TestExecutorVersionGate:
    """Verify executor rejects unknown IR versions."""

    def test_wrong_version_rejected(self):
        executor, _ = _make_executor(DummySkill())
        plan = MissionPlan(
            id="p1",
            nodes=[],
            metadata={"ir_version": "1.0"},
        )
        # Manually override for test
        plan.metadata["ir_version"] = "99.0"

        with pytest.raises(RuntimeError, match="Unsupported IR version"):
            executor.run(plan)


class TestExecutorRejectsDollarStrings:
    """
    Verify executor's _resolve_input rejects $ strings.
    Defense-in-depth, enforcement point 2 of 2.
    """

    def test_dollar_string_rejected_at_runtime(self):
        executor, _ = _make_executor(DummySkill())

        ok, _, err, failure_class = executor._resolve_input("$node.output", {})
        assert not ok
        assert "Banned $-prefixed string" in err
        assert failure_class == "INVALID_REFERENCE"


class TestExecutorResolvesOutputReference:
    """Verify OutputReference objects resolve correctly."""

    def test_valid_reference_resolved(self):
        executor, _ = _make_executor(DummySkill())

        results = {"node_a": {"result": "hello"}}
        ref = OutputReference(node="node_a", output="result")

        ok, value, err, _ = executor._resolve_input(ref, results)
        assert ok
        assert value == "hello"
        assert err == ""

    def test_missing_node_fails(self):
        executor, _ = _make_executor(DummySkill())

        ref = OutputReference(node="missing", output="result")
        ok, _, err, failure_class = executor._resolve_input(ref, {})
        assert not ok
        assert "missing" in err
        assert failure_class == "INVALID_REFERENCE"

    def test_missing_output_fails(self):
        executor, _ = _make_executor(DummySkill())

        results = {"node_a": {"other_key": "hello"}}
        ref = OutputReference(node="node_a", output="result")

        ok, _, err, failure_class = executor._resolve_input(ref, results)
        assert not ok
        assert "result" in err
        assert failure_class == "INVALID_REFERENCE"

    def test_literal_passes_through(self):
        executor, _ = _make_executor(DummySkill())

        ok, value, _, _ = executor._resolve_input("hello", {})
        assert ok
        assert value == "hello"

        ok, value, _, _ = executor._resolve_input(42, {})
        assert ok
        assert value == 42


class TestValidatorListNodes:
    """Verify validators work with List[MissionNode]."""

    def test_valid_plan_passes(self):
        plan = _make_plan([
            MissionNode(id="n1", skill="test.dummy"),
        ])
        validate_mission_plan(plan, {"test.dummy"})

    def test_unknown_skill_fails(self):
        plan = _make_plan([
            MissionNode(id="n1", skill="test.dummy"),
        ])
        with pytest.raises(MissionValidationError, match="Unknown skill"):
            validate_mission_plan(plan, {"other.skill"})

    def test_duplicate_ids_rejected(self):
        with pytest.raises(MissionValidationError, match="Duplicate"):
            plan = _make_plan([
                MissionNode(id="n1", skill="test.dummy"),
                MissionNode(id="n1", skill="test.dummy"),
            ])
            validate_mission_plan(plan, {"test.dummy"})

    def test_missing_dependency_rejected(self):
        plan = _make_plan([
            MissionNode(
                id="n1",
                skill="test.dummy",
                depends_on=["nonexistent"],
            ),
        ])
        with pytest.raises(MissionValidationError, match="missing node"):
            validate_mission_plan(plan, {"test.dummy"})

    def test_cycle_detected(self):
        plan = _make_plan([
            MissionNode(
                id="n1",
                skill="test.dummy",
                depends_on=["n2"],
            ),
            MissionNode(
                id="n2",
                skill="test.dummy",
                depends_on=["n1"],
            ),
        ])
        with pytest.raises(MissionValidationError, match="Cycle"):
            validate_mission_plan(plan, {"test.dummy"})


class TestConditionSkipSemantics:
    """Verify skipped nodes produce no output and don't fail."""

    def test_condition_false_skips_node(self):
        executor, _ = _make_executor(DummySkill())

        plan = _make_plan([
            MissionNode(
                id="gated",
                skill="test.dummy",
                inputs={"input_a": "hello"},
                condition_on=ConditionExpr(
                    source="world.media.is_playing",
                    equals=True,
                ),
            ),
        ])

        # World snapshot says is_playing=False → condition fails → node skipped
        from world.state import WorldState, MediaState
        from world.snapshot import WorldSnapshot
        snapshot = WorldSnapshot.build(
            state=WorldState(media=MediaState(is_playing=False)),
            recent_events=[],
        )
        er = executor.run(
            plan,
            world_snapshot=snapshot,
        )

        # Skipped node produces NO outputs
        assert "gated" not in er.results


class TestBackgroundFailureDoesNotFailMission:
    """Verify failure policy drives behavior, not hardcoded mode checks."""

    def test_background_failure_continues(self):
        executor, _ = _make_executor(DummySkill(), FailingSkill())

        plan = _make_plan([
            MissionNode(
                id="bg_fail",
                skill="test.failing",
                mode=ExecutionMode.background,
            ),
            MissionNode(
                id="fg_pass",
                skill="test.dummy",
                inputs={"input_a": "hello"},
            ),
        ])

        # Mission should complete despite background failure
        er = executor.run(plan)
        assert "fg_pass" in er.results
        assert er.results["fg_pass"]["result"] == "hello"

    def test_foreground_failure_fails_mission(self):
        executor, _ = _make_executor(FailingSkill())

        plan = _make_plan([
            MissionNode(
                id="fg_fail",
                skill="test.failing",
                mode=ExecutionMode.foreground,
            ),
        ])

        with pytest.raises(RuntimeError, match="policy=FAIL"):
            executor.run(plan)

    def test_side_effect_failure_ignored(self):
        executor, _ = _make_executor(DummySkill(), FailingSkill())

        plan = _make_plan([
            MissionNode(
                id="se_fail",
                skill="test.failing",
                mode=ExecutionMode.side_effect,
            ),
            MissionNode(
                id="fg_pass",
                skill="test.dummy",
                inputs={"input_a": "hello"},
            ),
        ])

        er = executor.run(plan)
        assert "fg_pass" in er.results


class TestDependencyOnSkippedNodeBlocksExecution:
    """
    Verify that if a dependency is skipped, the dependent node
    never runs (not just fails — never starts).
    """

    def test_skipped_dependency_cascades(self):
        executor, _ = _make_executor(DummySkill())

        plan = _make_plan([
            MissionNode(
                id="gated_parent",
                skill="test.dummy",
                inputs={"input_a": "hello"},
                outputs={
                    "result": OutputSpec(
                        name="test.result.v1",
                        type="text",
                    )
                },
                condition_on=ConditionExpr(
                    source="world.media.is_playing",
                    equals=True,
                ),
            ),
            MissionNode(
                id="dependent_child",
                skill="test.dummy",
                inputs={"input_a": "world"},
                depends_on=["gated_parent"],
            ),
        ])

        # Parent skipped → child should never run
        from world.state import WorldState, MediaState
        from world.snapshot import WorldSnapshot
        snapshot = WorldSnapshot.build(
            state=WorldState(media=MediaState(is_playing=False)),
            recent_events=[],
        )
        er = executor.run(
            plan,
            world_snapshot=snapshot,
        )

        assert "gated_parent" not in er.results
        assert "dependent_child" not in er.results

    def test_failed_dependency_cascades(self):
        executor, _ = _make_executor(DummySkill(), FailingSkill())

        plan = _make_plan([
            MissionNode(
                id="bg_parent",
                skill="test.failing",
                mode=ExecutionMode.background,
            ),
            MissionNode(
                id="dependent_child",
                skill="test.dummy",
                inputs={"input_a": "hello"},
                depends_on=["bg_parent"],
            ),
        ])

        er = executor.run(plan)

        # Background parent failed → child never runs
        assert "dependent_child" not in er.results


# ==================================================================
# 4. Skill Contract Enforcement Tests
# ==================================================================


class TestContractAllowedModeEnforcement:
    """Verify executor rejects nodes using disallowed execution modes."""

    def test_disallowed_mode_rejected(self):
        executor, _ = _make_executor(ForegroundOnlySkill())

        plan = _make_plan([
            MissionNode(
                id="n1",
                skill="test.fgonly",
                mode=ExecutionMode.background,  # not in allowed_modes
            ),
        ])

        with pytest.raises(RuntimeError, match="only allows"):
            executor.run(plan)

    def test_allowed_mode_accepted(self):
        executor, _ = _make_executor(ForegroundOnlySkill())

        plan = _make_plan([
            MissionNode(
                id="n1",
                skill="test.fgonly",
                mode=ExecutionMode.foreground,
            ),
        ])

        er = executor.run(plan)
        assert er.results["n1"]["status"] == "ok"


class TestContractEventEmissionEnforcement:
    """Verify executor enforces emits_events and mutates_world."""

    def test_declared_events_accepted(self):
        executor, tl = _make_executor(EventEmittingSkill())

        plan = _make_plan([
            MissionNode(
                id="n1",
                skill="test.emitter",
                inputs={"msg": "hello"},
            ),
        ])

        er = executor.run(plan)
        assert er.results["n1"]["status"] == "done"

        # Event should be in timeline
        events = tl.all_events()
        assert len(events) == 1
        assert events[0].type == "thing_happened"

    def test_undeclared_event_rejected(self):
        executor, _ = _make_executor(UndeclaredEventSkill())

        plan = _make_plan([
            MissionNode(
                id="n1",
                skill="test.sneaky",
            ),
        ])

        with pytest.raises(RuntimeError, match="undeclared event type"):
            executor.run(plan)

    def test_mutates_world_false_blocks_events(self):
        executor, _ = _make_executor(NoMutateButEmitsSkill())

        plan = _make_plan([
            MissionNode(
                id="n1",
                skill="test.liar",
            ),
        ])

        with pytest.raises(RuntimeError, match="mutates_world=False"):
            executor.run(plan)


class TestContractFailurePolicyDrivesBehavior:
    """Verify failure policy from contract drives executor behavior."""

    def test_fail_policy_raises(self):
        executor, _ = _make_executor(FailingSkill())

        plan = _make_plan([
            MissionNode(
                id="n1",
                skill="test.failing",
                mode=ExecutionMode.foreground,
            ),
        ])

        with pytest.raises(RuntimeError, match="policy=FAIL"):
            executor.run(plan)

    def test_continue_policy_logs_and_continues(self):
        executor, _ = _make_executor(DummySkill(), FailingSkill())

        plan = _make_plan([
            MissionNode(
                id="n1",
                skill="test.failing",
                mode=ExecutionMode.background,
            ),
            MissionNode(
                id="n2",
                skill="test.dummy",
                inputs={"input_a": "ok"},
            ),
        ])

        er = executor.run(plan)
        assert "n1" not in er.results
        assert "n2" in er.results

    def test_ignore_policy_silently_continues(self):
        executor, _ = _make_executor(DummySkill(), FailingSkill())

        plan = _make_plan([
            MissionNode(
                id="n1",
                skill="test.failing",
                mode=ExecutionMode.side_effect,
            ),
            MissionNode(
                id="n2",
                skill="test.dummy",
                inputs={"input_a": "ok"},
            ),
        ])

        er = executor.run(plan)
        assert "n2" in er.results


class TestContractPropertyDelegation:
    """Verify Skill.name/input_keys/output_keys delegate to contract."""

    def test_name_from_contract(self):
        s = DummySkill()
        assert s.name == "test.dummy"

    def test_input_keys_from_contract(self):
        s = DummySkill()
        assert s.input_keys == {"input_a"}

    def test_output_keys_from_contract(self):
        s = DummySkill()
        assert s.output_keys == {"result"}
