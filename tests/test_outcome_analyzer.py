# tests/test_outcome_analyzer.py

"""
Tests for OutcomeAnalyzer, OutcomeSeverity, and outcome-aware replanning.

Covers:
- OutcomeAnalyzer classification (BENIGN, SOFT_FAILURE, HARD_FAILURE)
- Idempotent reason detection
- ExecutionResult.outcome_verdicts field
- ExecutionResult.recovery_explanation field
- Supervisor outcome classification for parallel nodes
- MissionCortex._build_failure_context_section
- MissionOrchestrator._plans_equivalent
- MissionOutcome.recovery_attempted field
"""

import pytest
from unittest.mock import MagicMock

from execution.metacognition import (
    OutcomeAnalyzer,
    OutcomeSeverity,
)
from execution.executor import ExecutionResult, NodeStatus
from conversation.outcome import MissionOutcome
from ir.mission import MissionPlan, MissionNode, ExecutionMode, IR_VERSION


# ─────────────────────────────────────────────────────────────
# OutcomeAnalyzer — classification
# ─────────────────────────────────────────────────────────────

class TestOutcomeAnalyzer:
    """Test OutcomeAnalyzer.classify() for all severity levels."""

    def setup_method(self):
        self.analyzer = OutcomeAnalyzer()

    def test_completed_is_benign(self):
        """COMPLETED status always classifies as BENIGN."""
        result = self.analyzer.classify("completed", {})
        assert result == OutcomeSeverity.BENIGN

    def test_completed_with_metadata_still_benign(self):
        """COMPLETED with arbitrary metadata is still BENIGN."""
        result = self.analyzer.classify("completed", {"reason": "whatever"})
        assert result == OutcomeSeverity.BENIGN

    def test_no_op_without_reason_is_soft_failure(self):
        """NO_OP without a reason is SOFT_FAILURE."""
        result = self.analyzer.classify("no_op", {})
        assert result == OutcomeSeverity.SOFT_FAILURE

    def test_no_op_with_unknown_reason_is_soft_failure(self):
        """NO_OP with a non-idempotent reason is SOFT_FAILURE."""
        result = self.analyzer.classify("no_op", {"reason": "no_media_session"})
        assert result == OutcomeSeverity.SOFT_FAILURE

    def test_no_op_already_playing_is_benign(self):
        """NO_OP with 'already_playing' is idempotent → BENIGN."""
        result = self.analyzer.classify("no_op", {"reason": "already_playing"})
        assert result == OutcomeSeverity.BENIGN

    def test_no_op_already_paused_is_benign(self):
        result = self.analyzer.classify("no_op", {"reason": "already_paused"})
        assert result == OutcomeSeverity.BENIGN

    def test_no_op_already_muted_is_benign(self):
        result = self.analyzer.classify("no_op", {"reason": "already_muted"})
        assert result == OutcomeSeverity.BENIGN

    def test_no_op_already_unmuted_is_benign(self):
        result = self.analyzer.classify("no_op", {"reason": "already_unmuted"})
        assert result == OutcomeSeverity.BENIGN

    def test_failed_is_hard_failure(self):
        """FAILED status classifies as HARD_FAILURE."""
        result = self.analyzer.classify("failed", {})
        assert result == OutcomeSeverity.HARD_FAILURE

    def test_timed_out_is_hard_failure(self):
        """TIMED_OUT status classifies as HARD_FAILURE."""
        result = self.analyzer.classify("timed_out", {})
        assert result == OutcomeSeverity.HARD_FAILURE

    def test_unknown_status_is_hard_failure(self):
        """Unknown status defaults to HARD_FAILURE."""
        result = self.analyzer.classify("something_weird", {})
        assert result == OutcomeSeverity.HARD_FAILURE

    def test_none_metadata_handled(self):
        """None metadata doesn't crash."""
        result = self.analyzer.classify("no_op", None)
        assert result == OutcomeSeverity.SOFT_FAILURE

    def test_skipped_is_hard_failure(self):
        """SKIPPED status classifies as HARD_FAILURE."""
        result = self.analyzer.classify("skipped", {})
        assert result == OutcomeSeverity.HARD_FAILURE


# ─────────────────────────────────────────────────────────────
# OutcomeSeverity — enum values
# ─────────────────────────────────────────────────────────────

class TestOutcomeSeverity:

    def test_enum_values(self):
        assert OutcomeSeverity.BENIGN == "benign"
        assert OutcomeSeverity.SOFT_FAILURE == "soft_failure"
        assert OutcomeSeverity.HARD_FAILURE == "hard_failure"

    def test_all_severities_exist(self):
        assert len(OutcomeSeverity) == 3


# ─────────────────────────────────────────────────────────────
# ExecutionResult — new fields
# ─────────────────────────────────────────────────────────────

class TestExecutionResultFields:

    def test_outcome_verdicts_default_empty(self):
        er = ExecutionResult()
        assert er.outcome_verdicts == []
        assert isinstance(er.outcome_verdicts, list)

    def test_recovery_explanation_default_none(self):
        er = ExecutionResult()
        assert er.recovery_explanation is None

    def test_outcome_verdicts_appendable(self):
        er = ExecutionResult()
        er.outcome_verdicts.append({
            "node_id": "node_0",
            "skill": "system.media_play",
            "status": "no_op",
            "reason": "no_media_session",
            "severity": "soft_failure",
        })
        assert len(er.outcome_verdicts) == 1
        assert er.outcome_verdicts[0]["severity"] == "soft_failure"

    def test_recovery_explanation_settable(self):
        er = ExecutionResult()
        er.recovery_explanation = "Recovery attempted: 2 node(s)"
        assert er.recovery_explanation == "Recovery attempted: 2 node(s)"


# ─────────────────────────────────────────────────────────────
# MissionOutcome — recovery_attempted field
# ─────────────────────────────────────────────────────────────

class TestMissionOutcomeRecovery:

    def test_recovery_attempted_default_false(self):
        outcome = MissionOutcome(
            mission_id="m_1",
            nodes_executed=[],
            nodes_skipped=[],
        )
        assert outcome.recovery_attempted is False

    def test_recovery_attempted_set_true(self):
        outcome = MissionOutcome(
            mission_id="m_1",
            nodes_executed=["n_0"],
            nodes_skipped=[],
            recovery_attempted=True,
        )
        assert outcome.recovery_attempted is True


# ─────────────────────────────────────────────────────────────
# MissionOrchestrator._plans_equivalent
# ─────────────────────────────────────────────────────────────

def _make_plan(*skill_input_pairs):
    """Build a MissionPlan from (skill, inputs) tuples."""
    nodes = []
    for i, (skill, inputs) in enumerate(skill_input_pairs):
        nodes.append(MissionNode(
            id=f"node_{i}",
            skill=skill,
            inputs=inputs,
            depends_on=[f"node_{i-1}"] if i > 0 else [],
            mode=ExecutionMode.foreground,
        ))
    return MissionPlan(
        id="test_plan",
        nodes=nodes,
        metadata={"ir_version": IR_VERSION},
    )


class TestPlansEquivalent:

    def test_identical_plans(self):
        from orchestrator.mission_orchestrator import MissionOrchestrator
        plan_a = _make_plan(("system.media_play", {}))
        plan_b = _make_plan(("system.media_play", {}))
        assert MissionOrchestrator._plans_equivalent(plan_a, plan_b) is True

    def test_different_skills(self):
        from orchestrator.mission_orchestrator import MissionOrchestrator
        plan_a = _make_plan(("system.media_play", {}))
        plan_b = _make_plan(("system.focus_app", {"app_name": "spotify"}))
        assert MissionOrchestrator._plans_equivalent(plan_a, plan_b) is False

    def test_different_node_count(self):
        from orchestrator.mission_orchestrator import MissionOrchestrator
        plan_a = _make_plan(("system.media_play", {}))
        plan_b = _make_plan(
            ("system.focus_app", {"app_name": "spotify"}),
            ("system.media_play", {}),
        )
        assert MissionOrchestrator._plans_equivalent(plan_a, plan_b) is False

    def test_same_skills_different_input_keys(self):
        from orchestrator.mission_orchestrator import MissionOrchestrator
        plan_a = _make_plan(("system.focus_app", {"app_name": "spotify"}))
        plan_b = _make_plan(("system.focus_app", {"app_id": "spotify"}))
        assert MissionOrchestrator._plans_equivalent(plan_a, plan_b) is False

    def test_same_skills_same_input_keys_different_values(self):
        """Same inputs keys but different values → equivalent
        (we only compare structure, not values)."""
        from orchestrator.mission_orchestrator import MissionOrchestrator
        plan_a = _make_plan(("system.focus_app", {"app_name": "spotify"}))
        plan_b = _make_plan(("system.focus_app", {"app_name": "chrome"}))
        assert MissionOrchestrator._plans_equivalent(plan_a, plan_b) is True

    def test_multi_node_identical(self):
        from orchestrator.mission_orchestrator import MissionOrchestrator
        plan_a = _make_plan(
            ("system.focus_app", {"app_name": "spotify"}),
            ("system.media_play", {}),
        )
        plan_b = _make_plan(
            ("system.focus_app", {"app_name": "spotify"}),
            ("system.media_play", {}),
        )
        assert MissionOrchestrator._plans_equivalent(plan_a, plan_b) is True


# ─────────────────────────────────────────────────────────────
# MissionCortex._build_failure_context_section
# ─────────────────────────────────────────────────────────────

class TestBuildFailureContextSection:

    def _make_cortex(self):
        from cortex.mission_cortex import MissionCortex
        registry = MagicMock()
        registry.all_names.return_value = []
        return MissionCortex(
            llm_client=None,
            registry=registry,
        )

    def test_none_returns_empty(self):
        cortex = self._make_cortex()
        result = cortex._build_failure_context_section(None)
        assert result == ""

    def test_empty_list_returns_empty(self):
        cortex = self._make_cortex()
        result = cortex._build_failure_context_section([])
        assert result == ""

    def test_single_failure_rendered(self):
        cortex = self._make_cortex()
        failures = [{
            "skill": "system.media_play",
            "status": "NO_OP",
            "reason": "no_media_session",
        }]
        result = cortex._build_failure_context_section(failures)
        assert "PREVIOUS EXECUTION FAILURES" in result
        assert "system.media_play" in result
        assert "NO_OP" in result
        assert "no_media_session" in result
        assert "RECOVERY plan" in result

    def test_multiple_failures_rendered(self):
        cortex = self._make_cortex()
        failures = [
            {"skill": "system.focus_app", "status": "FAILED", "reason": "window not found"},
            {"skill": "system.media_play", "status": "NO_OP", "reason": "no_media_session"},
        ]
        result = cortex._build_failure_context_section(failures)
        assert "system.focus_app" in result
        assert "system.media_play" in result

    def test_failure_without_reason(self):
        cortex = self._make_cortex()
        failures = [{"skill": "test.skill", "status": "FAILED", "reason": ""}]
        result = cortex._build_failure_context_section(failures)
        assert "test.skill" in result
        assert "reason:" not in result  # empty reason not rendered


# ─────────────────────────────────────────────────────────────
# Supervisor — outcome classification for parallel nodes
# ─────────────────────────────────────────────────────────────

class TestSupervisorOutcomeClassification:
    """Tests that supervisor classifies outcomes for both focus and parallel nodes."""

    def _make_supervisor(self, execute_node_return):
        from execution.supervisor import (
            ExecutionContext, ExecutionSupervisor,
        )
        executor = MagicMock()
        executor.timeline = MagicMock()
        executor.execute_node.return_value = execute_node_return
        executor._needs_focus.return_value = False  # parallel path
        executor._has_conflicts.return_value = False
        executor.registry = MagicMock()

        # Make _execute_parallel actually call execute_node and record
        def fake_parallel(nids, node_index, exec_result, world_snapshot):
            for nid in nids:
                node = node_index[nid]
                n_id, status, outputs, meta = executor.execute_node(
                    node, exec_result, world_snapshot,
                )
                exec_result.record(n_id, status, outputs, meta)
        executor._execute_parallel.side_effect = fake_parallel

        ctx = ExecutionContext()
        return ExecutionSupervisor(executor=executor, context=ctx)

    def test_parallel_no_op_classified_as_soft_failure(self):
        """Parallel node with NO_OP (non-idempotent) gets classified."""
        supervisor = self._make_supervisor(
            ("node_0", NodeStatus.NO_OP, {}, {"reason": "no_media_session"})
        )
        node = MissionNode(
            id="node_0", skill="system.media_play", inputs={},
            mode=ExecutionMode.foreground,
        )
        plan = MissionPlan(
            id="test", nodes=[node],
            metadata={"ir_version": IR_VERSION},
        )

        result = supervisor.run(plan)
        assert len(result.outcome_verdicts) == 1
        assert result.outcome_verdicts[0]["severity"] == "soft_failure"
        assert result.outcome_verdicts[0]["skill"] == "system.media_play"
        assert result.outcome_verdicts[0]["reason"] == "no_media_session"

    def test_parallel_completed_no_verdict(self):
        """Parallel node that completes successfully → no verdict stored."""
        supervisor = self._make_supervisor(
            ("node_0", NodeStatus.COMPLETED, {"played": True}, {})
        )
        node = MissionNode(
            id="node_0", skill="system.media_play", inputs={},
            mode=ExecutionMode.foreground,
        )
        plan = MissionPlan(
            id="test", nodes=[node],
            metadata={"ir_version": IR_VERSION},
        )

        result = supervisor.run(plan)
        assert len(result.outcome_verdicts) == 0

    def test_parallel_no_op_idempotent_no_verdict(self):
        """Parallel NO_OP with idempotent reason → BENIGN, no verdict."""
        supervisor = self._make_supervisor(
            ("node_0", NodeStatus.NO_OP, {}, {"reason": "already_playing"})
        )
        node = MissionNode(
            id="node_0", skill="system.media_play", inputs={},
            mode=ExecutionMode.foreground,
        )
        plan = MissionPlan(
            id="test", nodes=[node],
            metadata={"ir_version": IR_VERSION},
        )

        result = supervisor.run(plan)
        assert len(result.outcome_verdicts) == 0

    def test_build_outcome_sets_recovery_attempted(self):
        """_build_outcome reads recovery_explanation to set recovery_attempted."""
        from orchestrator.mission_orchestrator import MissionOrchestrator

        plan = MissionPlan(
            id="test",
            nodes=[MissionNode(id="n_0", skill="system.media_play", inputs={})],
            metadata={"ir_version": IR_VERSION},
        )

        # Without recovery
        er = ExecutionResult()
        er.record("n_0", NodeStatus.COMPLETED, {})
        outcome = MissionOrchestrator._build_outcome(plan, er)
        assert outcome.recovery_attempted is False

        # With recovery
        er2 = ExecutionResult()
        er2.record("n_0", NodeStatus.COMPLETED, {})
        er2.recovery_explanation = "Recovery attempted: 2 node(s)"
        outcome2 = MissionOrchestrator._build_outcome(plan, er2)
        assert outcome2.recovery_attempted is True
