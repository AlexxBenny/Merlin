# tests/test_decision_engine.py

"""
Tests for DecisionEngine — bounded adaptive recovery.

Tests cover:
- classify_complexity: 2-axis (cause × scope from DAG)
- Contract-driven recovery: _find_revealers (reveal effect_type), dedup
- _score_normalized: 7 components bounded [-1, +1]
- _goal_distance: direct/preparatory/irrelevant
- _contributes_to_outcome: uncertainty reducer check
- _contradicts_commitment: commitment conflict
- decide(): full pipeline (verdict → classification → scoring → decision)
- _build_explanation: DecisionExplanation construction + logging
- Budget enforcement, escalation routing
"""

import pytest

from execution.metacognition import (
    DecisionEngine,
    FailureCategory,
    FailureVerdict,
    RecoveryAction,
)
from execution.cognitive_context import (
    ActionDecision,
    Assumption,
    Commitment,
    CognitiveContext,
    DecisionExplanation,
    DecisionSnapshot,
    EscalationDecision,
    EscalationLevel,
    ExecutionState,
    FailureCause,
    FailureScope,
    GoalState,
    MAX_TOTAL_STEPS,
    SCORING_WEIGHTS,
)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _make_verdict(
    error="not found",
    category=FailureCategory.CAPABILITY_FAILURE,
    action=RecoveryAction.REPLAN,
    node_id="n1",
    skill_name="fs.read_file",
    original_inputs=None,
):
    return FailureVerdict(
        category=category,
        action=action,
        reason=f"Skill execution failed: {error}",
        node_id=node_id,
        skill_name=skill_name,
        context={
            "error": error,
            "original_inputs": original_inputs or {"path": "report.txt"},
        },
    )


def _make_snapshot(
    required=None,
    achieved=None,
    uncertainty=None,
    attempts=None,
    commitments=None,
    step_count=0,
    recovery_depth=0,
):
    gs = GoalState(
        original_query="test query",
        required_outcomes=required or ["read_file"],
        achieved_outcomes=achieved or [],
    )
    es = ExecutionState()
    es.step_count = step_count
    es.recovery_depth = recovery_depth
    if uncertainty:
        es.uncertainty.update(uncertainty)
    if attempts:
        for a in attempts:
            es.attempt_history.append(a)
    if commitments:
        for k, v in commitments.items():
            es.commitments[k] = v
    ctx = CognitiveContext(goal=gs, execution=es)
    return ctx.snapshot_for_decision()


# ─────────────────────────────────────────────────────────────
# classify_complexity
# ─────────────────────────────────────────────────────────────


def _make_contract_driven_de():
    """Build a DecisionEngine with contract-driven recovery mocks.

    Sets up:
    - Registry: fs.read_file (requires file_exists) + fs.search_file (produces file_reference, reveal)
    - LLM diagnosis: bridges file_exists → file_reference
    - Guard evaluation: confirms guard is unmet
    """
    from unittest.mock import MagicMock

    registry = MagicMock()

    read_skill = MagicMock()
    read_skill.contract.requires = ["file_exists"]
    read_skill.contract.produces = []
    read_skill.contract.effect_type = "maintain"
    read_skill.contract.inputs = {"path": "file_path_input"}
    read_skill.contract.optional_inputs = {}

    search_skill = MagicMock()
    search_skill.contract.requires = []
    search_skill.contract.produces = ["file_reference"]
    search_skill.contract.effect_type = "reveal"
    search_skill.contract.inputs = {"query": "file_search_query"}
    search_skill.contract.optional_inputs = {}

    registry.all_names.return_value = {"fs.read_file", "fs.search_file"}
    def _get(name):
        return {"fs.read_file": read_skill, "fs.search_file": search_skill}[name]
    registry.get.side_effect = _get

    de = DecisionEngine(registry=registry, llm_client=MagicMock())
    de._llm_diagnose = MagicMock(return_value={
        "state_type": "file_reference",
        "cause": "missing_file",
    })
    de._guard_is_unmet = MagicMock(return_value=True)
    return de


# ─────────────────────────────────────────────────────────────

class TestClassifyComplexity:

    def test_missing_data_from_missing_parameter(self):
        """MISSING_PARAMETER category → MISSING_DATA cause."""
        de = DecisionEngine()
        verdict = _make_verdict(
            error="not found",
            category=FailureCategory.MISSING_PARAMETER,
        )
        snap = _make_snapshot()
        cause, scope = de.classify_complexity(verdict, snap)
        assert cause == FailureCause.MISSING_DATA

    def test_missing_state_from_environment_mismatch(self):
        de = DecisionEngine()
        verdict = _make_verdict(
            error="guard failed",
            category=FailureCategory.ENVIRONMENT_MISMATCH,
        )
        snap = _make_snapshot()
        cause, scope = de.classify_complexity(verdict, snap)
        assert cause == FailureCause.MISSING_STATE
        assert scope == FailureScope.MULTI_STEP  # needs_prerequisites

    def test_capability_failure_from_timeout(self):
        """CAPABILITY_FAILURE category → INVALID_ASSUMPTION cause."""
        de = DecisionEngine()
        verdict = _make_verdict(error="connection timeout")
        snap = _make_snapshot()
        cause, _ = de.classify_complexity(verdict, snap)
        assert cause == FailureCause.INVALID_ASSUMPTION

    def test_invalid_assumption_from_permission(self):
        de = DecisionEngine()
        verdict = _make_verdict(error="permission denied")
        snap = _make_snapshot()
        cause, _ = de.classify_complexity(verdict, snap)
        assert cause == FailureCause.INVALID_ASSUMPTION

    def test_multi_step_when_has_dependents(self):
        """If failed node has dependents in DAG → multi_step."""
        from ir.mission import MissionNode, MissionPlan, ExecutionMode, IR_VERSION

        node1 = MissionNode(
            id="n1", skill="fs.read_file", inputs={},
            depends_on=[], mode=ExecutionMode.foreground,
        )
        node2 = MissionNode(
            id="n2", skill="email.send_message", inputs={},
            depends_on=["n1"], mode=ExecutionMode.foreground,
        )
        plan = MissionPlan(id="p1", nodes=[node1, node2],
                           metadata={"ir_version": IR_VERSION})

        de = DecisionEngine()
        verdict = _make_verdict(node_id="n1", error="not found")
        snap = _make_snapshot()
        cause, scope = de.classify_complexity(verdict, snap, plan)
        assert scope == FailureScope.MULTI_STEP  # n1 has dependents

    def test_single_step_when_no_dependents(self):
        from ir.mission import MissionNode, MissionPlan, ExecutionMode, IR_VERSION

        node1 = MissionNode(
            id="n1", skill="fs.read_file", inputs={},
            depends_on=[], mode=ExecutionMode.foreground,
        )
        plan = MissionPlan(id="p1", nodes=[node1],
                           metadata={"ir_version": IR_VERSION})

        de = DecisionEngine()
        verdict = _make_verdict(node_id="n1", error="not found")
        snap = _make_snapshot()
        cause, scope = de.classify_complexity(verdict, snap, plan)
        assert scope == FailureScope.SINGLE_STEP


# ─────────────────────────────────────────────────────────────
# Contract-driven recovery (replaces old hardcoded _try_heuristic)
# Recovery is now: _check_requires → _find_creators → _find_revealers → _find_enablers
# ─────────────────────────────────────────────────────────────

class TestContractDrivenRecovery:
    """Verify the principled recovery path works for cases
    previously handled by the hardcoded _HEURISTIC_TABLE."""

    def test_find_revealers_discovers_search_file(self):
        """file_reference guard → _find_revealers → search_file via contract."""
        from execution.supervisor import GuardType
        from unittest.mock import MagicMock

        # Create a registry with a mock search_file skill
        registry = MagicMock()
        mock_skill = MagicMock()
        mock_skill.contract.produces = ["file_reference"]
        mock_skill.contract.effect_type = "reveal"
        mock_skill.contract.inputs = {"query": "file_search_query"}
        mock_skill.contract.optional_inputs = {}
        registry.all_names.return_value = {"fs.search_file"}
        registry.get.return_value = mock_skill

        de = DecisionEngine(registry=registry)
        # Provide original_inputs with "query" so _infer_repair_inputs can match
        verdict = _make_verdict(
            error="not found",
            original_inputs={"query": "resume.pdf", "path": "reports/report.txt"},
        )
        snap = _make_snapshot()

        candidates = de._find_revealers(
            GuardType.FILE_REFERENCE, verdict, snap,
        )
        assert len(candidates) > 0
        assert candidates[0][0] == "fs.search_file"

    def test_find_revealers_skips_already_attempted(self):
        """Dedup: don't repeat a revealer already in attempt_history."""
        from execution.supervisor import GuardType
        from unittest.mock import MagicMock

        registry = MagicMock()
        mock_skill = MagicMock()
        mock_skill.contract.produces = ["file_reference"]
        mock_skill.contract.effect_type = "reveal"
        mock_skill.contract.inputs = {"query": "file_search_query"}
        mock_skill.contract.optional_inputs = {}
        registry.all_names.return_value = {"fs.search_file"}
        registry.get.return_value = mock_skill

        de = DecisionEngine(registry=registry)
        verdict = _make_verdict(
            error="not found",
            original_inputs={"query": "resume.pdf"},
        )
        snap = _make_snapshot(attempts=[
            {"skill": "fs.search_file", "inputs": {"query": "report"}, "result": "failed"},
        ])

        candidates = de._find_revealers(
            GuardType.FILE_REFERENCE, verdict, snap,
        )
        assert len(candidates) == 0  # skipped — already attempted

    def test_find_revealers_ignores_create_skills(self):
        """_find_revealers only returns effect_type='reveal', not 'create'."""
        from execution.supervisor import GuardType
        from unittest.mock import MagicMock

        registry = MagicMock()
        mock_skill = MagicMock()
        mock_skill.contract.produces = ["file_reference"]
        mock_skill.contract.effect_type = "create"  # not reveal!
        mock_skill.contract.inputs = {}
        mock_skill.contract.optional_inputs = {}
        registry.all_names.return_value = {"fs.write_file"}
        registry.get.return_value = mock_skill

        de = DecisionEngine(registry=registry)
        verdict = _make_verdict(error="not found")
        snap = _make_snapshot()

        candidates = de._find_revealers(
            GuardType.FILE_REFERENCE, verdict, snap,
        )
        assert len(candidates) == 0  # create skills excluded

    def test_no_registry_returns_empty(self):
        """No registry → no candidates."""
        from execution.supervisor import GuardType

        de = DecisionEngine()  # no registry
        verdict = _make_verdict(error="not found")
        snap = _make_snapshot()

        candidates = de._find_revealers(
            GuardType.FILE_REFERENCE, verdict, snap,
        )
        assert candidates == []


# ─────────────────────────────────────────────────────────────
# Normalized scoring
# ─────────────────────────────────────────────────────────────

class TestScoreNormalized:

    def test_all_components_bounded(self):
        de = DecisionEngine()
        snap = _make_snapshot()
        score, components, lookahead = de._score_normalized(
            "fs.search_file", {"name": "report"}, snap,
        )
        for name, val in components.items():
            assert -1.0 <= val <= 1.0, f"{name}={val} out of bounds"

    def test_lower_score_is_better(self):
        """Direct outcome match should score lower than irrelevant."""
        de = DecisionEngine()
        snap = _make_snapshot(required=["search_file"])
        s_search, _, _ = de._score_normalized(
            "fs.search_file", {"name": "report"}, snap,
        )
        s_random, _, _ = de._score_normalized(
            "email.send_message", {}, snap,
        )
        assert s_search < s_random

    def test_attempt_penalty_increases_score(self):
        de = DecisionEngine()
        snap_no_attempts = _make_snapshot()
        snap_with_attempts = _make_snapshot(attempts=[
            {"skill": "fs.search_file", "inputs": {}, "result": "failed"},
            {"skill": "fs.search_file", "inputs": {}, "result": "failed"},
        ])
        s1, c1, _ = de._score_normalized(
            "fs.search_file", {}, snap_no_attempts,
        )
        s2, c2, _ = de._score_normalized(
            "fs.search_file", {}, snap_with_attempts,
        )
        assert c2["penalty"] > c1["penalty"]
        assert s2 > s1

    def test_commitment_contradiction_penalized(self):
        de = DecisionEngine()
        snap = _make_snapshot(commitments={
            "selected_file": Commitment(
                key="selected_file", value="report_final.txt",
            ),
        })
        _, c_contradict, _ = de._score_normalized(
            "fs.search_file", {"name": "other_file"}, snap,
        )
        _, c_agree, _ = de._score_normalized(
            "fs.search_file", {"name": "report_final"}, snap,
        )
        assert c_contradict["commitment"] > c_agree["commitment"]

    def test_uncertainty_reducer_benefits_when_uncertain(self):
        de = DecisionEngine()
        snap_certain = _make_snapshot(uncertainty={"fs": 0.0})
        snap_uncertain = _make_snapshot(uncertainty={"fs": 0.8})
        _, c_certain, _ = de._score_normalized(
            "fs.search_file", {}, snap_certain,
        )
        _, c_uncertain, _ = de._score_normalized(
            "fs.search_file", {}, snap_uncertain,
        )
        assert c_uncertain["uncertainty"] < c_certain["uncertainty"]


# ─────────────────────────────────────────────────────────────
# _normalize
# ─────────────────────────────────────────────────────────────

class TestNormalize:

    def test_midpoint_is_zero(self):
        assert DecisionEngine._normalize(1.5, 0.0, 3.0) == pytest.approx(0.0)

    def test_lo_is_minus_one(self):
        assert DecisionEngine._normalize(0.0, 0.0, 3.0) == pytest.approx(-1.0)

    def test_hi_is_plus_one(self):
        assert DecisionEngine._normalize(3.0, 0.0, 3.0) == pytest.approx(1.0)

    def test_clamped_below(self):
        assert DecisionEngine._normalize(-5.0, 0.0, 3.0) == -1.0

    def test_clamped_above(self):
        assert DecisionEngine._normalize(10.0, 0.0, 3.0) == 1.0

    def test_equal_lo_hi_returns_zero(self):
        assert DecisionEngine._normalize(5.0, 3.0, 3.0) == 0.0


# ─────────────────────────────────────────────────────────────
# _goal_distance
# ─────────────────────────────────────────────────────────────

class TestGoalDistance:

    def test_direct_outcome_match(self):
        de = DecisionEngine()
        snap = _make_snapshot(required=["search_file"])
        dist = de._goal_distance("fs.search_file", snap)
        assert dist == -3.0

    def test_irrelevant_action(self):
        de = DecisionEngine()
        snap = _make_snapshot(required=["send_email"])
        dist = de._goal_distance("system.get_battery", snap)
        assert dist == 3.0

    def test_always_progressive(self):
        de = DecisionEngine()
        snap = _make_snapshot(required=["send_email"])
        dist = de._goal_distance("system.focus_app", snap)
        assert dist == -1.0

    def test_uncertainty_reducer_contributes_when_uncertain(self):
        de = DecisionEngine()
        snap = _make_snapshot(
            required=["send_email"],
            uncertainty={"fs": 0.5},
        )
        dist = de._goal_distance("fs.search_file", snap)
        assert dist < 3.0  # contributes


# ─────────────────────────────────────────────────────────────
# decide() — full pipeline
# ─────────────────────────────────────────────────────────────

class TestDecide:

    def test_returns_action_for_not_found(self):
        de = _make_contract_driven_de()
        verdict = _make_verdict(
            error="not found",
            original_inputs={"query": "report", "path": "report.txt"},
        )
        snap = _make_snapshot()
        result = de.decide(verdict, snap)
        assert isinstance(result, ActionDecision)
        assert result.skill == "fs.search_file"
        assert result.strategy_source == "effect_llm"

    def test_returns_escalation_when_budget_exhausted(self):
        de = DecisionEngine()
        verdict = _make_verdict(error="not found")
        snap = _make_snapshot(step_count=MAX_TOTAL_STEPS)
        result = de.decide(verdict, snap)
        assert isinstance(result, EscalationDecision)
        assert result.level == EscalationLevel.GLOBAL
        assert "Budget" in result.reason

    def test_escalates_for_timeout(self):
        """Timeout with default category → INVALID_ASSUMPTION → GLOBAL escalation."""
        de = DecisionEngine()
        verdict = _make_verdict(error="connection timeout")
        snap = _make_snapshot()
        result = de.decide(verdict, snap)
        assert isinstance(result, EscalationDecision)
        assert result.level == EscalationLevel.GLOBAL

    def test_escalates_for_multi_step_scope(self):
        de = DecisionEngine()
        verdict = _make_verdict(
            error="guard failed",
            category=FailureCategory.ENVIRONMENT_MISMATCH,
        )
        snap = _make_snapshot()
        result = de.decide(verdict, snap)
        assert isinstance(result, EscalationDecision)
        assert result.level == EscalationLevel.GLOBAL
        assert "Multi-step" in result.reason

    def test_explanation_has_components(self):
        de = _make_contract_driven_de()
        verdict = _make_verdict(
            error="not found",
            original_inputs={"query": "report", "path": "report.txt"},
        )
        snap = _make_snapshot()
        result = de.decide(verdict, snap)
        assert isinstance(result, ActionDecision)
        exp = result.explanation
        assert exp.chosen_action == "fs.search_file"
        assert len(exp.components) == 7
        assert sum(exp.weights.values()) == pytest.approx(1.0)
        assert len(exp.uncertainty_snapshot) > 0

    def test_explanation_includes_rejection_list(self):
        """When there's only 1 candidate, rejection list is empty."""
        de = _make_contract_driven_de()
        verdict = _make_verdict(
            error="not found",
            original_inputs={"query": "report", "path": "report.txt"},
        )
        snap = _make_snapshot()
        result = de.decide(verdict, snap)
        assert isinstance(result, ActionDecision)
        assert result.explanation.rejected == []

    def test_action_decision_has_assumptions(self):
        de = _make_contract_driven_de()
        verdict = _make_verdict(
            error="not found",
            original_inputs={"query": "report", "path": "report.txt"},
        )
        snap = _make_snapshot()
        result = de.decide(verdict, snap)
        assert isinstance(result, ActionDecision)
        assert len(result.assumptions) > 0
        assert result.assumptions[0].type == "reveal_repair"


# ─────────────────────────────────────────────────────────────
# _rejection_reason
# ─────────────────────────────────────────────────────────────

class TestRejectionReason:

    def test_cost_reason(self):
        reason = DecisionEngine._rejection_reason(
            {"cost": 0.8, "distance": -0.2, "penalty": 0.0,
             "uncertainty": 0.0, "exploration": 0.0,
             "commitment": 0.0, "future": 0.0},
        )
        assert reason == "higher cost"

    def test_penalty_reason(self):
        reason = DecisionEngine._rejection_reason(
            {"cost": 0.0, "distance": 0.0, "penalty": 0.9,
             "uncertainty": 0.0, "exploration": 0.0,
             "commitment": 0.0, "future": 0.0},
        )
        assert reason == "already attempted"


# ─────────────────────────────────────────────────────────────────
# Stage 3: Lookahead + Simulation
# ─────────────────────────────────────────────────────────────────

class TestSimulate:

    def test_produces_types_without_registry(self):
        """Without registry, produces inferred type from skill name."""
        de = DecisionEngine()
        snap = _make_snapshot(required=["read_file"])
        sim = de._simulate("fs.search_file", snap)
        assert len(sim.produced_types) > 0
        assert "fs_search_file_output" in sim.produced_types

    def test_reduces_uncertainty(self):
        de = DecisionEngine()
        snap = _make_snapshot(uncertainty={"fs": 0.8})
        sim = de._simulate("fs.search_file", snap)
        assert sim.uncertainty["fs"] < 0.8

    def test_achieves_matching_outcome(self):
        de = DecisionEngine()
        snap = _make_snapshot(required=["search_file"])
        sim = de._simulate("fs.search_file", snap)
        assert "search_file" in sim.achieved_outcomes
        assert len(sim.pending_outcomes) == 0

    def test_increments_step_count(self):
        de = DecisionEngine()
        snap = _make_snapshot(step_count=3)
        sim = de._simulate("fs.search_file", snap)
        assert sim.step_count == 4


class TestEstimateSuccess:

    def test_perfect_conditions(self):
        de = DecisionEngine()
        snap = _make_snapshot()
        p = de._estimate_success("fs.search_file", snap)
        assert p == pytest.approx(1.0)

    def test_uncertainty_reduces_probability(self):
        de = DecisionEngine()
        snap = _make_snapshot(uncertainty={"fs": 1.0})
        p = de._estimate_success("fs.search_file", snap)
        assert p < 1.0

    def test_prior_failures_reduce_probability(self):
        de = DecisionEngine()
        snap = _make_snapshot(attempts=[
            {"skill": "fs.search_file", "result": "failed"},
            {"skill": "fs.search_file", "result": "failed"},
        ])
        p = de._estimate_success("fs.search_file", snap)
        assert p < 1.0

    def test_clamped_above_minimum(self):
        de = DecisionEngine()
        snap = _make_snapshot(
            uncertainty={"fs": 1.0},
            attempts=[
                {"skill": "fs.search_file", "result": "failed"},
            ] * 10,
        )
        p = de._estimate_success("fs.search_file", snap)
        assert p >= 0.1


class TestGenerateFollowUps:

    def test_no_registry_returns_empty(self):
        de = DecisionEngine(registry=None)
        snap = _make_snapshot()
        sim = de._simulate("fs.search_file", snap)
        follow_ups = de._generate_follow_ups(sim, snap)
        assert follow_ups == []


class TestScoreWithLookahead:

    def test_returns_expected_structure(self):
        de = DecisionEngine()
        snap = _make_snapshot()
        result = de._score_with_lookahead("fs.search_file", snap)
        assert "p_success" in result
        assert "best_follow_up" in result
        assert "expected_future" in result
        assert "candidates" in result
        assert "produced_types" in result

    def test_p_success_in_range(self):
        de = DecisionEngine()
        snap = _make_snapshot()
        result = de._score_with_lookahead("fs.search_file", snap)
        assert 0.1 <= result["p_success"] <= 1.0

    def test_no_follow_ups_without_registry(self):
        de = DecisionEngine(registry=None)
        snap = _make_snapshot()
        result = de._score_with_lookahead("fs.search_file", snap)
        assert result["best_follow_up"] is None
        assert result["candidates"] == 0
        assert result["expected_future"] == 0.0

    def test_explanation_includes_lookahead(self):
        de = _make_contract_driven_de()
        verdict = _make_verdict(
            error="not found",
            original_inputs={"query": "report", "path": "report.txt"},
        )
        snap = _make_snapshot()
        result = de.decide(verdict, snap)
        assert isinstance(result, ActionDecision)
        assert "p_success" in result.explanation.lookahead
        assert "best_follow_up" in result.explanation.lookahead


# ─────────────────────────────────────────────────────────────────
# Stage 4: Intelligence Layer
# ─────────────────────────────────────────────────────────────────

class TestCreateCommitment:

    def test_creates_commitment(self):
        from execution.cognitive_context import ExecutionState
        es = ExecutionState()
        DecisionEngine.create_commitment(
            es, "selected_file", "report.txt",
            alternatives=["report_v2.txt", "notes.txt"],
            confidence=0.7, decision_id="d_abc",
        )
        assert "selected_file" in es.commitments
        c = es.commitments["selected_file"]
        assert c.value == "report.txt"
        assert len(c.alternatives) == 2
        assert c.confidence == 0.7
        assert c.source_decision_id == "d_abc"

    def test_overwrites_existing_commitment(self):
        from execution.cognitive_context import ExecutionState
        es = ExecutionState()
        DecisionEngine.create_commitment(
            es, "file", "v1.txt", alternatives=[], decision_id="d1",
        )
        DecisionEngine.create_commitment(
            es, "file", "v2.txt", alternatives=[], decision_id="d2",
        )
        assert es.commitments["file"].value == "v2.txt"
        assert es.commitments["file"].source_decision_id == "d2"


class TestReconsiderCommitment:

    def test_traces_to_commitment(self):
        from execution.cognitive_context import (
            ExecutionState, DecisionRecord, Commitment,
        )
        es = ExecutionState()
        # Decision d_root creates commitment
        r_root = DecisionRecord(id="d_root", step=1, action_skill="fs.search")
        es.record_decision(r_root)
        es.commitments["selected_file"] = Commitment(
            key="selected_file", value="report.txt",
            source_decision_id="d_root",
        )
        # Decision d_child is caused by d_root
        r_child = DecisionRecord(
            id="d_child", step=2, action_skill="fs.read",
            parent_ids=["d_root"],
        )
        es.record_decision(r_child)

        result = DecisionEngine.reconsider_commitment(es, "d_child")
        assert result is not None
        key, commitment = result
        assert key == "selected_file"
        assert commitment.value == "report.txt"

    def test_no_match_returns_none(self):
        from execution.cognitive_context import ExecutionState, DecisionRecord
        es = ExecutionState()
        r = DecisionRecord(id="d_1", step=1, action_skill="test")
        es.record_decision(r)
        result = DecisionEngine.reconsider_commitment(es, "d_1")
        assert result is None

    def test_no_commitments_returns_none(self):
        from execution.cognitive_context import ExecutionState
        es = ExecutionState()
        result = DecisionEngine.reconsider_commitment(es, "d_nonexist")
        assert result is None


class TestRecordDecisionWithCausalLink:

    def test_returns_decision_id(self):
        from execution.cognitive_context import ExecutionState
        es = ExecutionState()
        did = DecisionEngine.record_decision_with_causal_link(
            es, "fs.search_file", {"name": "report"},
        )
        assert did.startswith("d_")
        assert len(es.decision_trace) == 1

    def test_links_parent(self):
        from execution.cognitive_context import ExecutionState
        es = ExecutionState()
        d1 = DecisionEngine.record_decision_with_causal_link(
            es, "fs.search_file", {},
        )
        d2 = DecisionEngine.record_decision_with_causal_link(
            es, "fs.read_file", {},
            parent_decision_id=d1,
        )
        assert es.decision_trace[1].parent_ids == [d1]
        assert es.trace_root_cause(d2) == d1

    def test_caused_by_node(self):
        from execution.cognitive_context import ExecutionState
        es = ExecutionState()
        DecisionEngine.record_decision_with_causal_link(
            es, "fs.search_file", {},
            caused_by_node="node_read_01",
        )
        assert es.decision_trace[0].caused_by == "node_read_01"


class TestInvalidateCommitmentsForGoalChange:

    def test_removes_matching_commitments(self):
        from execution.cognitive_context import ExecutionState, Commitment
        es = ExecutionState()
        es.commitments["selected_report_file"] = Commitment(
            key="selected_report_file", value="report.txt",
        )
        es.commitments["email_recipient"] = Commitment(
            key="email_recipient", value="john@example.com",
        )

        invalidated = DecisionEngine.invalidate_commitments_for_goal_change(
            es, removed_outcomes=["report"],
        )
        assert "selected_report_file" in invalidated
        assert "email_recipient" not in invalidated
        assert "selected_report_file" not in es.commitments
        assert "email_recipient" in es.commitments

    def test_preserves_unrelated_commitments(self):
        from execution.cognitive_context import ExecutionState, Commitment
        es = ExecutionState()
        es.commitments["target_app"] = Commitment(
            key="target_app", value="notepad",
        )
        invalidated = DecisionEngine.invalidate_commitments_for_goal_change(
            es, removed_outcomes=["email"],
        )
        assert invalidated == []
        assert "target_app" in es.commitments

    def test_empty_removed_outcomes(self):
        from execution.cognitive_context import ExecutionState, Commitment
        es = ExecutionState()
        es.commitments["file"] = Commitment(key="file", value="x")
        invalidated = DecisionEngine.invalidate_commitments_for_goal_change(
            es, removed_outcomes=[],
        )
        assert invalidated == []
        assert "file" in es.commitments
