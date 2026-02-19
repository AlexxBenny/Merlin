# tests/test_input_validation.py

"""
Tests for validator checks #8 and #9 — compile-time input coverage.

Check #8: Missing required inputs → MissionValidationError
Check #9: Unexpected inputs → MissionValidationError

Edge cases:
- Zero required inputs (media_play) → passes
- Only optional provided → passes if no required
- Only required provided → passes even without optional
- Both required and optional provided → passes
"""

import pytest

from ir.mission import (
    MissionPlan,
    MissionNode,
    ExecutionMode,
    OutputReference,
    IR_VERSION,
)
from cortex.validators import validate_mission_plan, MissionValidationError
from skills.contract import SkillContract, FailurePolicy
from execution.registry import SkillRegistry


# ── Fake skills (use real semantic types) ──

class _SkillNoInputs:
    """Skill with zero required and zero optional inputs (like media_play)."""
    name = "test.no_inputs"
    contract = SkillContract(
        name="test.no_inputs",
        description="No inputs at all",
        inputs={},
        optional_inputs={},
        outputs={"changed": "boolean_flag"},
        allowed_modes={ExecutionMode.foreground},
        failure_policy={ExecutionMode.foreground: FailurePolicy.FAIL},
    )


class _SkillRequiredOnly:
    """Skill with only required inputs, no optional (like set_brightness)."""
    name = "test.required_only"
    contract = SkillContract(
        name="test.required_only",
        description="Required inputs only",
        inputs={"level": "brightness_percentage"},
        optional_inputs={},
        outputs={"brightness": "actual_brightness"},
        allowed_modes={ExecutionMode.foreground},
        failure_policy={ExecutionMode.foreground: FailurePolicy.FAIL},
    )


class _SkillMixed:
    """Skill with both required and optional (like create_folder)."""
    name = "test.mixed"
    contract = SkillContract(
        name="test.mixed",
        description="Required and optional inputs",
        inputs={"name": "folder_name"},
        optional_inputs={"anchor": "anchor_name", "parent": "relative_path"},
        outputs={"created": "filesystem_path"},
        allowed_modes={ExecutionMode.foreground},
        failure_policy={ExecutionMode.foreground: FailurePolicy.FAIL},
    )


class _SkillOptionalOnly:
    """Skill with only optional inputs, no required."""
    name = "test.optional_only"
    contract = SkillContract(
        name="test.optional_only",
        description="Only optional inputs",
        inputs={},
        optional_inputs={"anchor": "anchor_name"},
        outputs={"changed": "boolean_flag"},
        allowed_modes={ExecutionMode.foreground},
        failure_policy={ExecutionMode.foreground: FailurePolicy.FAIL},
    )


# ── Helpers ──

def _reg(*skills):
    r = SkillRegistry()
    for s in skills:
        r.register(s, validate_types=False)
    return r


def _plan(*nodes):
    return MissionPlan(
        id="test_plan",
        nodes=list(nodes),
        metadata={"ir_version": IR_VERSION},
    )


def _node(nid, skill, inputs=None, depends_on=None, mode=ExecutionMode.foreground):
    return MissionNode(
        id=nid,
        skill=skill,
        inputs=inputs or {},
        outputs={},
        depends_on=depends_on or [],
        mode=mode,
    )


# ── Check #8: Missing required ──

class TestMissingRequiredInputs:

    def test_missing_required_input_rejected(self):
        reg = _reg(_SkillRequiredOnly())
        plan = _plan(_node("n0", "test.required_only", inputs={}))
        with pytest.raises(MissionValidationError, match="requires inputs.*level"):
            validate_mission_plan(plan, {"test.required_only"}, registry=reg)

    def test_all_required_provided_passes(self):
        reg = _reg(_SkillRequiredOnly())
        plan = _plan(_node("n0", "test.required_only", inputs={"level": 50}))
        validate_mission_plan(plan, {"test.required_only"}, registry=reg)

    def test_mixed_missing_required_rejected(self):
        reg = _reg(_SkillMixed())
        plan = _plan(_node("n0", "test.mixed", inputs={"anchor": "DESKTOP"}))
        with pytest.raises(MissionValidationError, match="requires inputs.*name"):
            validate_mission_plan(plan, {"test.mixed"}, registry=reg)

    def test_mixed_required_only_passes(self):
        reg = _reg(_SkillMixed())
        plan = _plan(_node("n0", "test.mixed", inputs={"name": "myFolder"}))
        validate_mission_plan(plan, {"test.mixed"}, registry=reg)

    def test_mixed_all_provided_passes(self):
        reg = _reg(_SkillMixed())
        plan = _plan(_node("n0", "test.mixed", inputs={
            "name": "myFolder", "anchor": "DESKTOP", "parent": "projects",
        }))
        validate_mission_plan(plan, {"test.mixed"}, registry=reg)


# ── Check #9: Unexpected inputs ──

class TestUnexpectedInputs:

    def test_unexpected_input_rejected(self):
        reg = _reg(_SkillRequiredOnly())
        plan = _plan(_node("n0", "test.required_only", inputs={"level": 50, "foo": 123}))
        with pytest.raises(MissionValidationError, match="unexpected inputs.*foo"):
            validate_mission_plan(plan, {"test.required_only"}, registry=reg)

    def test_unexpected_on_no_input_skill(self):
        reg = _reg(_SkillNoInputs())
        plan = _plan(_node("n0", "test.no_inputs", inputs={"random_key": "value"}))
        with pytest.raises(MissionValidationError, match="unexpected inputs.*random_key"):
            validate_mission_plan(plan, {"test.no_inputs"}, registry=reg)

    def test_optional_input_not_unexpected(self):
        reg = _reg(_SkillOptionalOnly())
        plan = _plan(_node("n0", "test.optional_only", inputs={"anchor": "DESKTOP"}))
        validate_mission_plan(plan, {"test.optional_only"}, registry=reg)


# ── Edge case: zero required inputs ──

class TestZeroRequiredInputs:

    def test_no_inputs_empty_provided(self):
        """media_play-like: inputs={}, provided={} → passes."""
        reg = _reg(_SkillNoInputs())
        plan = _plan(_node("n0", "test.no_inputs", inputs={}))
        validate_mission_plan(plan, {"test.no_inputs"}, registry=reg)

    def test_optional_only_empty_provided(self):
        reg = _reg(_SkillOptionalOnly())
        plan = _plan(_node("n0", "test.optional_only", inputs={}))
        validate_mission_plan(plan, {"test.optional_only"}, registry=reg)


# ── Backward compatibility ──

class TestBackwardCompatibility:

    def test_no_registry_skips_input_checks(self):
        """When registry=None, checks #8-9 are skipped."""
        plan = _plan(_node("n0", "test.required_only", inputs={}))
        # Would fail with registry, should pass without
        validate_mission_plan(plan, {"test.required_only"})

    def test_no_registry_still_catches_structural(self):
        plan = _plan(_node("n0", "nonexistent.skill", inputs={}))
        with pytest.raises(MissionValidationError, match="Unknown skill"):
            validate_mission_plan(plan, {"test.required_only"})


# ── Multi-node ──

class TestMultiNodeValidation:

    def test_mixed_plan_valid(self):
        reg = _reg(_SkillNoInputs(), _SkillRequiredOnly(), _SkillMixed())
        plan = _plan(
            _node("n0", "test.no_inputs", inputs={}),
            _node("n1", "test.required_only", inputs={"level": 75}),
            _node("n2", "test.mixed", inputs={"name": "hello"}),
        )
        validate_mission_plan(
            plan,
            {"test.no_inputs", "test.required_only", "test.mixed"},
            registry=reg,
        )

    def test_mixed_plan_one_bad_node_fails(self):
        reg = _reg(_SkillNoInputs(), _SkillRequiredOnly())
        plan = _plan(
            _node("n0", "test.no_inputs", inputs={}),
            _node("n1", "test.required_only", inputs={}),  # missing level
        )
        with pytest.raises(MissionValidationError, match="requires inputs"):
            validate_mission_plan(
                plan,
                {"test.no_inputs", "test.required_only"},
                registry=reg,
            )
