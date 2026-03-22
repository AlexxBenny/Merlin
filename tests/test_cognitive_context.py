# tests/test_cognitive_context.py

"""
Tests for CognitiveContext, ExecutionState, GoalState, DecisionSnapshot,
and all data models in cognitive_context.py.

Tests cover:
- ExecutionState: uncertainty updates, decision trace, compaction, root cache
- GoalState: versioning, refinement, pending outcomes
- CognitiveContext: snapshot creation, world refresh, failure context
- DecisionSnapshot: immutability, correct field population
- Assumption, Commitment, DecisionRecord: model construction
- Budget enforcement
"""

import pytest

from execution.cognitive_context import (
    CognitiveContext,
    DecisionSnapshot,
    ExecutionState,
    GoalState,
    SimulatedState,
    Assumption,
    Commitment,
    DecisionRecord,
    DecisionExplanation,
    ActionDecision,
    EscalationDecision,
    EscalationLevel,
    FailureCause,
    FailureScope,
    MAX_TRACE,
    MAX_TOTAL_STEPS,
    MAX_RECOVERY_DEPTH,
    MAX_DYNAMIC_QUEUE,
    SCORING_WEIGHTS,
    COST_MAP,
    UNCERTAINTY_REDUCERS,
)


# ─────────────────────────────────────────────────────────────
# ExecutionState — uncertainty
# ─────────────────────────────────────────────────────────────

class TestExecutionStateUncertainty:

    def test_initial_uncertainty_all_zero(self):
        es = ExecutionState()
        assert all(v == 0.0 for v in es.uncertainty.values())
        assert "fs" in es.uncertainty
        assert "email" in es.uncertainty
        assert "system" in es.uncertainty
        assert "browser" in es.uncertainty
        assert "general" in es.uncertainty

    def test_increase_uncertainty(self):
        es = ExecutionState()
        es.update_uncertainty("multiple_matches", domain="fs")
        assert es.uncertainty["fs"] == pytest.approx(0.4)
        assert es.uncertainty["email"] == 0.0  # other domains unaffected

    def test_decrease_uncertainty(self):
        es = ExecutionState()
        es.update_uncertainty("multiple_matches", domain="fs")
        es.update_uncertainty("file_found", domain="fs")
        assert es.uncertainty["fs"] == pytest.approx(0.1)

    def test_uncertainty_clamped_at_zero(self):
        es = ExecutionState()
        es.update_uncertainty("file_found", domain="fs")
        assert es.uncertainty["fs"] == 0.0  # can't go below 0

    def test_uncertainty_clamped_at_one(self):
        es = ExecutionState()
        for _ in range(10):
            es.update_uncertainty("ambiguous_input", domain="fs")
        assert es.uncertainty["fs"] == 1.0

    def test_unknown_event_type_ignored(self):
        es = ExecutionState()
        es.update_uncertainty("totally_bogus_event", domain="fs")
        assert es.uncertainty["fs"] == 0.0

    def test_unknown_domain_falls_back_to_general(self):
        es = ExecutionState()
        es.update_uncertainty("multiple_matches", domain="quantum")
        assert es.uncertainty["general"] == pytest.approx(0.4)


# ─────────────────────────────────────────────────────────────
# ExecutionState — decision trace + compaction
# ─────────────────────────────────────────────────────────────

class TestExecutionStateTrace:

    def test_record_decision(self):
        es = ExecutionState()
        record = DecisionRecord(
            id="d_1", step=1, action_skill="fs.search_file",
        )
        es.record_decision(record)
        assert len(es.decision_trace) == 1
        assert es.decision_trace[0].id == "d_1"

    def test_root_cache_self(self):
        es = ExecutionState()
        record = DecisionRecord(id="d_1", step=1, action_skill="fs.search_file")
        es.record_decision(record)
        assert es.trace_root_cause("d_1") == "d_1"

    def test_root_cache_chain(self):
        es = ExecutionState()
        r1 = DecisionRecord(id="d_1", step=1, action_skill="fs.search_file")
        r2 = DecisionRecord(
            id="d_2", step=2, action_skill="fs.read_file",
            parent_ids=["d_1"],
        )
        r3 = DecisionRecord(
            id="d_3", step=3, action_skill="email.send_message",
            parent_ids=["d_2"],
        )
        es.record_decision(r1)
        es.record_decision(r2)
        es.record_decision(r3)
        # d_3 should trace back to d_1
        assert es.trace_root_cause("d_3") == "d_1"
        assert es.trace_root_cause("d_2") == "d_1"
        assert es.trace_root_cause("d_1") == "d_1"

    def test_compaction_at_max_trace(self):
        es = ExecutionState()
        # Fill beyond MAX_TRACE
        for i in range(MAX_TRACE + 5):
            r = DecisionRecord(
                id=f"d_{i}", step=i, action_skill="fs.search_file",
                outcome="success" if i % 2 == 0 else "failed",
            )
            es.record_decision(r)
        # Trace should be compacted
        assert len(es.decision_trace) <= MAX_TRACE
        assert len(es.decision_summaries) > 0
        assert "pattern" in es.decision_summaries[0]
        assert "count" in es.decision_summaries[0]

    def test_root_cache_survives_compaction(self):
        """Root cache persists even when trace entries are evicted."""
        es = ExecutionState()
        r1 = DecisionRecord(id="d_0", step=0, action_skill="fs.search_file")
        r2 = DecisionRecord(
            id="d_1", step=1, action_skill="fs.read_file",
            parent_ids=["d_0"],
        )
        es.record_decision(r1)
        es.record_decision(r2)
        # Force compaction by adding many more
        for i in range(2, MAX_TRACE + 15):
            es.record_decision(DecisionRecord(
                id=f"d_{i}", step=i, action_skill="test.skill",
            ))
        # Root cache should still know d_1's root is d_0
        assert es.trace_root_cause("d_1") == "d_0"


# ─────────────────────────────────────────────────────────────
# ExecutionState — attempt history
# ─────────────────────────────────────────────────────────────

class TestExecutionStateAttempts:

    def test_record_attempt(self):
        es = ExecutionState()
        es.step_count = 3
        es.record_attempt("fs.search_file", {"name": "report"}, "success")
        assert len(es.attempt_history) == 1
        assert es.attempt_history[0]["skill"] == "fs.search_file"
        assert es.attempt_history[0]["result"] == "success"
        assert es.attempt_history[0]["step"] == 3


# ─────────────────────────────────────────────────────────────
# ExecutionState — budget
# ─────────────────────────────────────────────────────────────

class TestExecutionStateBudget:

    def test_within_budget_initially(self):
        es = ExecutionState()
        assert es.within_budget is True

    def test_exceeds_step_budget(self):
        es = ExecutionState()
        es.step_count = MAX_TOTAL_STEPS
        assert es.within_budget is False

    def test_exceeds_recovery_depth(self):
        es = ExecutionState()
        es.recovery_depth = MAX_RECOVERY_DEPTH
        assert es.within_budget is False

    def test_exceeds_queue_size(self):
        es = ExecutionState()
        es.dynamic_queue = list(range(MAX_DYNAMIC_QUEUE))
        assert es.within_budget is False


# ─────────────────────────────────────────────────────────────
# GoalState
# ─────────────────────────────────────────────────────────────

class TestGoalState:

    def test_basic_creation(self):
        gs = GoalState(
            original_query="read report",
            required_outcomes=["read_file"],
        )
        assert gs.pending_outcomes == ["read_file"]
        assert gs.is_complete is False

    def test_complete_when_all_achieved(self):
        gs = GoalState(
            required_outcomes=["read_file", "send_email"],
            achieved_outcomes=["read_file", "send_email"],
        )
        assert gs.is_complete is True
        assert gs.pending_outcomes == []

    def test_refine_increments_version(self):
        gs = GoalState(
            original_query="send report",
            required_outcomes=["search", "send"],
        )
        removed = gs.refine("send latest report", ["search", "send", "verify"])
        assert gs.version == 2
        assert gs.original_query == "send latest report"
        assert removed == []  # nothing removed
        assert "verify" in gs.required_outcomes

    def test_refine_removes_outcomes(self):
        gs = GoalState(
            original_query="search and read",
            required_outcomes=["search", "read"],
            achieved_outcomes=["search"],
        )
        removed = gs.refine("just read", ["read"])
        assert removed == ["search"]
        assert gs.achieved_outcomes == []  # search was removed
        assert gs.required_outcomes == ["read"]

    def test_refine_preserves_valid_achieved(self):
        gs = GoalState(
            required_outcomes=["a", "b", "c"],
            achieved_outcomes=["a", "b"],
        )
        removed = gs.refine("refined", ["a", "c", "d"])
        assert "a" in gs.achieved_outcomes   # preserved (still required)
        assert "b" not in gs.achieved_outcomes  # removed (no longer required)
        assert removed == ["b"]

    def test_refinement_history_tracked(self):
        gs = GoalState(original_query="v1", required_outcomes=["a"])
        gs.refine("v2", ["a", "b"])
        gs.refine("v3", ["b"])
        assert len(gs.refinement_history) == 2
        assert gs.version == 3


# ─────────────────────────────────────────────────────────────
# CognitiveContext + DecisionSnapshot
# ─────────────────────────────────────────────────────────────

class TestCognitiveContext:

    def test_snapshot_is_frozen(self):
        gs = GoalState(original_query="test", required_outcomes=["read"])
        es = ExecutionState()
        ctx = CognitiveContext(goal=gs, execution=es)
        snap = ctx.snapshot_for_decision()

        # DecisionSnapshot is frozen
        with pytest.raises(AttributeError):
            snap.step_count = 99

    def test_snapshot_reflects_state(self):
        gs = GoalState(
            original_query="send report",
            required_outcomes=["search", "send"],
            achieved_outcomes=["search"],
        )
        es = ExecutionState()
        es.step_count = 5
        es.uncertainty["fs"] = 0.3
        ctx = CognitiveContext(goal=gs, execution=es)
        snap = ctx.snapshot_for_decision()

        assert snap.goal_original_query == "send report"
        assert snap.pending_outcomes == ("send",)
        assert "search" in snap.achieved_outcomes
        assert snap.step_count == 5
        assert snap.uncertainty["fs"] == pytest.approx(0.3)
        assert snap.within_budget is True

    def test_refresh_world_returns_new_context(self):
        gs = GoalState()
        es = ExecutionState()
        ctx = CognitiveContext(goal=gs, execution=es, world="old_world")
        new_ctx = ctx.refresh_world("new_world")

        assert new_ctx is not ctx
        assert new_ctx.world == "new_world"
        assert new_ctx.goal is ctx.goal     # same ref
        assert new_ctx.execution is ctx.execution  # same ref

    def test_to_failure_context(self):
        gs = GoalState(original_query="test", required_outcomes=["a"])
        es = ExecutionState()
        es.record_attempt("skill.x", {}, "failed")
        ctx = CognitiveContext(goal=gs, execution=es)
        fc = ctx.to_failure_context()

        assert fc["goal"] == "test"
        assert fc["required_outcomes"] == ["a"]
        assert len(fc["attempt_history"]) == 1


# ─────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────

class TestDataModels:

    def test_assumption_construction(self):
        a = Assumption(
            type="file_not_in_index",
            params={"name": "report"},
            guard_mapping="file_exists",
            invert=True,
        )
        assert a.type == "file_not_in_index"
        assert a.invert is True

    def test_commitment_construction(self):
        c = Commitment(
            key="selected_file",
            value="report.txt",
            alternatives=["report_v2.txt"],
            confidence=0.8,
        )
        assert c.key == "selected_file"
        assert len(c.alternatives) == 1

    def test_decision_explanation_construction(self):
        de = DecisionExplanation(
            chosen_action="fs.search_file",
            final_score=-0.35,
            components={"distance": -0.6, "cost": 0.2},
            weights={"distance": 0.3, "cost": 0.15},
        )
        assert de.final_score == -0.35
        assert de.components["distance"] == -0.6

    def test_action_decision_frozen(self):
        ad = ActionDecision(
            skill="fs.search_file",
            inputs={"name": "report"},
            assumptions=[],
            score=-0.5,
            explanation=DecisionExplanation(
                chosen_action="fs.search_file", final_score=-0.5,
            ),
        )
        assert ad.skill == "fs.search_file"
        with pytest.raises(AttributeError):
            ad.skill = "other"

    def test_escalation_decision(self):
        ed = EscalationDecision(
            level=EscalationLevel.GLOBAL,
            reason="test",
        )
        assert ed.level == EscalationLevel.GLOBAL

    def test_failure_cause_enum(self):
        assert FailureCause.MISSING_DATA == "missing_data"
        assert FailureCause.MISSING_STATE == "missing_state"
        assert FailureCause.INVALID_ASSUMPTION == "invalid_assumption"
        assert FailureCause.EXTERNAL_DEPENDENCY == "external_dependency"

    def test_failure_scope_enum(self):
        assert FailureScope.SINGLE_STEP == "single_step"
        assert FailureScope.MULTI_STEP == "multi_step"


# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

class TestConstants:

    def test_scoring_weights_sum_to_one(self):
        total = sum(SCORING_WEIGHTS.values())
        assert total == pytest.approx(1.0)

    def test_cost_map_has_three_levels(self):
        assert "low" in COST_MAP
        assert "medium" in COST_MAP
        assert "high" in COST_MAP
        assert COST_MAP["low"] < COST_MAP["medium"] < COST_MAP["high"]

    def test_uncertainty_reducers_not_empty(self):
        assert len(UNCERTAINTY_REDUCERS) > 0
        assert "fs.search_file" in UNCERTAINTY_REDUCERS
