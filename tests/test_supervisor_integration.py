# tests/test_supervisor_integration.py

"""
Tests for supervisor integration with CognitiveContext.

Covers:
- Assumption gate: skip when invalid, execute when valid
- _should_still_execute: guard mapping + invert
- Uncertainty updates after node execution
- _get_skill_domain: registry authority
- Backward compat: cognitive_ctx=None preserves existing behavior
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from execution.supervisor import (
    ExecutionSupervisor,
    ExecutionContext,
    GuardType,
    StepGuard,
)
from execution.executor import MissionExecutor, ExecutionResult, NodeStatus
from execution.cognitive_context import (
    Assumption,
    CognitiveContext,
    ExecutionState,
    GoalState,
)
from ir.mission import MissionNode, MissionPlan, ExecutionMode, IR_VERSION


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _make_node(node_id="n1", skill="fs.read_file", inputs=None):
    return MissionNode(
        id=node_id,
        skill=skill,
        inputs=inputs or {},
        depends_on=[],
        mode=ExecutionMode.foreground,
    )


def _make_plan(*nodes):
    if not nodes:
        nodes = [_make_node()]
    return MissionPlan(
        id="test_plan",
        nodes=list(nodes),
        metadata={"ir_version": IR_VERSION},
    )


def _make_cognitive_ctx(assumptions=None, uncertainty=None):
    gs = GoalState(original_query="test", required_outcomes=["read_file"])
    es = ExecutionState()
    if uncertainty:
        es.uncertainty.update(uncertainty)
    if assumptions:
        es.node_assumptions.update(assumptions)
    return CognitiveContext(goal=gs, execution=es)


def _make_supervisor(execute_returns=None):
    """Create supervisor with mocked executor."""
    executor = MagicMock(spec=MissionExecutor)
    executor.registry = MagicMock()

    # Default execute_node return
    if execute_returns is None:
        execute_returns = ("n1", NodeStatus.COMPLETED, {}, {})
    executor.execute_node.return_value = execute_returns

    # Mock _needs_focus and _has_conflicts to force guarded path
    executor._needs_focus.return_value = True
    executor._has_conflicts.return_value = False

    ctx = ExecutionContext()
    supervisor = ExecutionSupervisor(executor, ctx)
    return supervisor


# ─────────────────────────────────────────────────────────────
# _should_still_execute
# ─────────────────────────────────────────────────────────────

class TestShouldStillExecute:

    def test_no_guard_mapping_passes(self):
        """Assumption without guard_mapping is not checkable → passes."""
        sup = _make_supervisor()
        node = _make_node()
        assumption = Assumption(type="custom_check", guard_mapping=None)
        assert sup._should_still_execute(node, [assumption]) is True

    def test_invalid_guard_type_passes(self):
        """Unknown guard mapping string is skipped → passes."""
        sup = _make_supervisor()
        node = _make_node()
        assumption = Assumption(
            type="test", guard_mapping="nonexistent_guard_type",
        )
        assert sup._should_still_execute(node, [assumption]) is True

    def test_file_exists_with_invert_true(self):
        """invert=True: assumption holds when guard FAILS."""
        sup = _make_supervisor()
        node = _make_node()
        assumption = Assumption(
            type="file_not_in_index",
            params={"path": "/some/file"},
            guard_mapping="file_exists",
            invert=True,
        )
        # Guard returns True (file exists) → inverted → False → skip
        with patch.object(sup, '_evaluate_guard', return_value=True):
            assert sup._should_still_execute(node, [assumption]) is False

    def test_file_exists_with_invert_false(self):
        """invert=False: assumption holds when guard PASSES."""
        sup = _make_supervisor()
        node = _make_node()
        assumption = Assumption(
            type="file_found",
            params={"path": "/some/file"},
            guard_mapping="file_exists",
            invert=False,
        )
        with patch.object(sup, '_evaluate_guard', return_value=True):
            assert sup._should_still_execute(node, [assumption]) is True

    def test_multiple_assumptions_all_must_hold(self):
        sup = _make_supervisor()
        node = _make_node()
        assumptions = [
            Assumption(type="a", guard_mapping="file_exists", invert=True),
            Assumption(type="b", guard_mapping="app_running", invert=False),
        ]
        # file_exists returns False (inverted → True), app_running returns True
        guard_results = {
            GuardType.FILE_EXISTS: False,
            GuardType.APP_RUNNING: True,
        }
        with patch.object(
            sup, '_evaluate_guard',
            side_effect=lambda g: guard_results.get(g.type, True),
        ):
            assert sup._should_still_execute(node, assumptions) is True

    def test_one_failing_assumption_blocks(self):
        sup = _make_supervisor()
        node = _make_node()
        assumptions = [
            Assumption(type="a", guard_mapping="file_exists", invert=True),
            Assumption(type="b", guard_mapping="app_running", invert=False),
        ]
        # file_exists returns True (inverted → False) → BLOCK
        guard_results = {
            GuardType.FILE_EXISTS: True,
            GuardType.APP_RUNNING: True,
        }
        with patch.object(
            sup, '_evaluate_guard',
            side_effect=lambda g: guard_results.get(g.type, True),
        ):
            assert sup._should_still_execute(node, assumptions) is False


# ─────────────────────────────────────────────────────────────
# Assumption gate in _execute_guarded_node
# ─────────────────────────────────────────────────────────────

class TestAssumptionGate:

    def test_skips_node_when_assumption_invalid(self):
        """Node with invalid assumption is SKIPPED, not executed."""
        sup = _make_supervisor()
        node = _make_node()
        ctx = _make_cognitive_ctx(assumptions={
            "n1": [Assumption(
                type="file_not_in_index",
                guard_mapping="file_exists",
                invert=True,
            )],
        })
        sup._cognitive_ctx = ctx

        exec_result = ExecutionResult()
        # Guard returns True → inverted → False → assumption invalid → skip
        with patch.object(sup, '_evaluate_guard', return_value=True):
            sup._execute_guarded_node(node, exec_result, None)

        assert exec_result.node_statuses.get("n1") == NodeStatus.SKIPPED
        # Executor should NOT have been called
        sup._executor.execute_node.assert_not_called()

    def test_executes_when_no_cognitive_ctx(self):
        """Without cognitive_ctx, normal execution proceeds."""
        sup = _make_supervisor()
        sup._cognitive_ctx = None
        node = _make_node()
        exec_result = ExecutionResult()

        sup._execute_guarded_node(node, exec_result, None)

        sup._executor.execute_node.assert_called_once()

    def test_executes_when_no_assumptions(self):
        """With cognitive_ctx but no assumptions, normal execution."""
        sup = _make_supervisor()
        ctx = _make_cognitive_ctx()  # no assumptions
        sup._cognitive_ctx = ctx
        node = _make_node()
        exec_result = ExecutionResult()

        sup._execute_guarded_node(node, exec_result, None)

        sup._executor.execute_node.assert_called_once()


# ─────────────────────────────────────────────────────────────
# Uncertainty updates
# ─────────────────────────────────────────────────────────────

class TestUncertaintyUpdates:

    def test_success_reduces_uncertainty(self):
        sup = _make_supervisor(
            execute_returns=("n1", NodeStatus.COMPLETED, {}, {}),
        )
        ctx = _make_cognitive_ctx(uncertainty={"fs": 0.5})
        sup._cognitive_ctx = ctx
        node = _make_node(skill="fs.read_file")
        exec_result = ExecutionResult()

        # Mock _get_skill_domain to return "fs"
        with patch.object(sup, '_get_skill_domain', return_value="fs"):
            sup._execute_guarded_node(node, exec_result, None)

        # outcome_achieved reduces fs uncertainty by 0.2
        assert ctx.execution.uncertainty["fs"] < 0.5

    def test_not_found_increases_uncertainty(self):
        sup = _make_supervisor(
            execute_returns=("n1", NodeStatus.FAILED, {},
                             {"reason": "File not found: report.txt"}),
        )
        ctx = _make_cognitive_ctx(uncertainty={"fs": 0.0})
        sup._cognitive_ctx = ctx
        node = _make_node(skill="fs.read_file")
        exec_result = ExecutionResult()

        with patch.object(sup, '_get_skill_domain', return_value="fs"):
            sup._execute_guarded_node(node, exec_result, None)

        # file_not_found increases fs uncertainty by 0.2
        assert ctx.execution.uncertainty["fs"] > 0.0

    def test_no_update_without_cognitive_ctx(self):
        """Without cognitive_ctx, uncertainty stays untouched."""
        sup = _make_supervisor()
        sup._cognitive_ctx = None
        node = _make_node()
        exec_result = ExecutionResult()
        # Should not crash
        sup._execute_guarded_node(node, exec_result, None)


# ─────────────────────────────────────────────────────────────
# _get_skill_domain
# ─────────────────────────────────────────────────────────────

class TestGetSkillDomain:

    def test_uses_registry_contract(self):
        sup = _make_supervisor()
        mock_skill = MagicMock()
        mock_skill.contract.domain = "email"
        sup._executor.registry.get.return_value = mock_skill
        assert sup._get_skill_domain("email.send_message") == "email"

    def test_fallback_on_error(self):
        sup = _make_supervisor()
        sup._executor.registry.get.side_effect = KeyError("not found")
        assert sup._get_skill_domain("unknown.skill") == "general"

    def test_fallback_on_missing_contract(self):
        sup = _make_supervisor()
        sup._executor.registry.get.return_value = MagicMock(
            spec=[], contract=None,
        )
        sup._executor.registry.get.return_value.contract = None
        # Accessing .domain on None raises AttributeError
        assert sup._get_skill_domain("broken.skill") == "general"


# ─────────────────────────────────────────────────────────────
# run() backward compatibility
# ─────────────────────────────────────────────────────────────

class TestRunBackwardCompat:

    def test_cognitive_ctx_none_default(self):
        """run() with no cognitive_ctx defaults to None."""
        sup = _make_supervisor()
        plan = _make_plan()
        # Should not crash
        result = sup.run(plan)
        assert result is not None

    def test_cognitive_ctx_stored_on_run(self):
        """run() stores cognitive_ctx for the duration of the run."""
        sup = _make_supervisor()
        plan = _make_plan()
        ctx = _make_cognitive_ctx()
        sup.run(plan, cognitive_ctx=ctx)
        assert sup._cognitive_ctx is ctx


# ─────────────────────────────────────────────────────────────
# Inline recovery via DecisionEngine
# ─────────────────────────────────────────────────────────────

def _make_action_decision(skill="fs.search_file", inputs=None, strategy="effect_chain"):
    """Build an ActionDecision mock."""
    from execution.cognitive_context import ActionDecision, Assumption, DecisionExplanation
    return ActionDecision(
        skill=skill,
        inputs=inputs or {"query": "resume"},
        assumptions=[],
        score=0.3,
        explanation=DecisionExplanation(
            chosen_action=skill,
            final_score=0.3,
        ),
        strategy_source=strategy,
    )


def _make_escalation_decision(reason="Cannot resolve"):
    from execution.cognitive_context import EscalationDecision, EscalationLevel
    from execution.metacognition import FailureVerdict, FailureCategory, RecoveryAction
    verdict = FailureVerdict(
        category=FailureCategory.CAPABILITY_FAILURE,
        action=RecoveryAction.REPLAN,
        reason=reason,
        node_id="n1",
        skill_name="fs.read_file",
    )
    return EscalationDecision(
        level=EscalationLevel.GLOBAL,
        reason=reason,
        verdict=verdict,
    )


def _make_verdict(node_id="n1", skill="fs.read_file", reason="File not found"):
    from execution.metacognition import FailureVerdict, FailureCategory, RecoveryAction
    return FailureVerdict(
        category=FailureCategory.CAPABILITY_FAILURE,
        action=RecoveryAction.REPLAN,
        reason=reason,
        node_id=node_id,
        skill_name=skill,
    )


class TestInlineRecovery:
    """Tests for _attempt_inline_recovery — inline recovery at failure point."""

    def test_routes_through_executor(self):
        """Recovery builds MissionNode and calls executor.execute_node."""
        sup = _make_supervisor()
        ctx = _make_cognitive_ctx()
        sup._cognitive_ctx = ctx

        decision = _make_action_decision()
        de = MagicMock()
        de.decide.return_value = decision
        sup._decision_engine = de

        # Mock skill registry for safety gate
        mock_skill = MagicMock()
        mock_skill.contract.effect_type = "reveal"
        sup._executor.registry.get.return_value = mock_skill

        # Recovery succeeds
        sup._executor.execute_node.return_value = (
            "recovery_n1_0", NodeStatus.COMPLETED, {"matches": []}, {},
        )

        verdict = _make_verdict()
        exec_result = ExecutionResult()
        result = sup._attempt_inline_recovery(
            _make_node(), verdict, exec_result, None,
        )

        assert result is True
        # Must route through executor.execute_node, NOT raw skill.execute
        sup._executor.execute_node.assert_called_once()
        call_args = sup._executor.execute_node.call_args
        recovery_node = call_args[0][0]
        assert recovery_node.id == "recovery_n1_0"
        assert recovery_node.skill == "fs.search_file"

    def test_retries_after_success(self):
        """Recovery succeeds → returns True → caller retries original node."""
        sup = _make_supervisor()
        ctx = _make_cognitive_ctx()
        sup._cognitive_ctx = ctx

        de = MagicMock()
        de.decide.return_value = _make_action_decision()
        sup._decision_engine = de

        mock_skill = MagicMock()
        mock_skill.contract.effect_type = "reveal"
        sup._executor.registry.get.return_value = mock_skill

        sup._executor.execute_node.return_value = (
            "recovery_n1_0", NodeStatus.COMPLETED, {}, {},
        )

        result = sup._attempt_inline_recovery(
            _make_node(), _make_verdict(), ExecutionResult(), None,
        )
        assert result is True

    def test_respects_per_node_max(self):
        """After MAX_INLINE_RECOVERY, stops trying."""
        sup = _make_supervisor()
        ctx = _make_cognitive_ctx()
        ctx.execution.inline_recovery_count["n1"] = 2  # already at max
        sup._cognitive_ctx = ctx
        sup._decision_engine = MagicMock()

        result = sup._attempt_inline_recovery(
            _make_node(), _make_verdict(), ExecutionResult(), None,
        )
        assert result is False
        # DecisionEngine should NOT have been called
        sup._decision_engine.decide.assert_not_called()

    def test_dedup_same_action(self):
        """Same (skill, inputs) for same node → skipped on second attempt."""
        sup = _make_supervisor()
        ctx = _make_cognitive_ctx()
        sup._cognitive_ctx = ctx

        de = MagicMock()
        de.decide.return_value = _make_action_decision()
        sup._decision_engine = de

        mock_skill = MagicMock()
        mock_skill.contract.effect_type = "reveal"
        sup._executor.registry.get.return_value = mock_skill

        # First attempt succeeds
        sup._executor.execute_node.return_value = (
            "recovery_n1_0", NodeStatus.COMPLETED, {}, {},
        )

        node = _make_node()
        verdict = _make_verdict()

        r1 = sup._attempt_inline_recovery(node, verdict, ExecutionResult(), None)
        assert r1 is True

        # Second attempt with SAME action → deduped
        r2 = sup._attempt_inline_recovery(node, verdict, ExecutionResult(), None)
        assert r2 is False

    def test_skips_destructive(self):
        """effect_type not in safe set → recovery skipped."""
        sup = _make_supervisor()
        ctx = _make_cognitive_ctx()
        sup._cognitive_ctx = ctx

        de = MagicMock()
        de.decide.return_value = _make_action_decision(skill="fs.delete_file")
        sup._decision_engine = de

        mock_skill = MagicMock()
        mock_skill.contract.effect_type = "destroy"
        sup._executor.registry.get.return_value = mock_skill

        result = sup._attempt_inline_recovery(
            _make_node(), _make_verdict(), ExecutionResult(), None,
        )
        assert result is False
        sup._executor.execute_node.assert_not_called()

    def test_no_decision_engine_skips(self):
        """decision_engine=None → no inline recovery."""
        sup = _make_supervisor()
        ctx = _make_cognitive_ctx()
        sup._cognitive_ctx = ctx
        sup._decision_engine = None

        result = sup._attempt_inline_recovery(
            _make_node(), _make_verdict(), ExecutionResult(), None,
        )
        assert result is False

    def test_escalation_not_executed(self):
        """EscalationDecision → no inline execution."""
        sup = _make_supervisor()
        ctx = _make_cognitive_ctx()
        sup._cognitive_ctx = ctx

        de = MagicMock()
        de.decide.return_value = _make_escalation_decision()
        sup._decision_engine = de

        result = sup._attempt_inline_recovery(
            _make_node(), _make_verdict(), ExecutionResult(), None,
        )
        assert result is False
        sup._executor.execute_node.assert_not_called()
