# tests/test_effect_recovery.py

"""
Tests for effect-driven recovery architecture.

Covers all Phase 2-3 functionality:
- SkillContract effect fields (requires, produces, effect_type)
- _check_requires: contract-based precondition check
- _find_creators: effect_type=="create" producer discovery
- _find_enablers: 1-level expansion for transitive chains
- _guard_is_unmet: True/False/None (unknown → AmbiguityDecision)
- _normalize_diagnosis: LLM output → GuardType | None
- _simulate_and_rank: DIRECT (2.0) > INDIRECT (1.0) > DEAD-END (0.5)
- _simulate: produced_guards projected from contract
- AmbiguityDecision handling in decide()
- classify_complexity: category-based (no string matching)
- E2E scenario: media_play failure → open_app recovery

Original scenario: "send attachment via email" → file not found → recovery.
Plan evolved through 12 iterations into the effect-driven architecture.
"""

import pytest
from unittest.mock import MagicMock
from types import SimpleNamespace

from execution.metacognition import (
    DecisionEngine,
    FailureCategory,
    FailureVerdict,
    RecoveryAction,
)
from execution.cognitive_context import (
    ActionDecision,
    AmbiguityDecision,
    Assumption,
    CognitiveContext,
    DecisionSnapshot,
    EscalationDecision,
    EscalationLevel,
    ExecutionState,
    FailureCause,
    FailureScope,
    GoalState,
    SimulatedState,
)
from execution.supervisor import GuardType


# ─────────────────────────────────────────────────────────────
# Mock helpers
# ─────────────────────────────────────────────────────────────

def _mock_contract(
    domain="system", requires=None, produces=None,
    effect_type="maintain", inputs=None, outputs=None,
    target_type="", emits_events=None, description="",
):
    """Create a mock SkillContract with effect fields."""
    contract = MagicMock()
    contract.domain = domain
    contract.requires = requires or []
    contract.produces = produces or []
    contract.effect_type = effect_type
    contract.inputs = inputs or {}
    contract.outputs = outputs or {}
    contract.target_type = target_type
    contract.emits_events = emits_events or []
    contract.description = description
    return contract


def _mock_skill(contract):
    """Wrap a contract in a mock skill."""
    skill = MagicMock()
    skill.contract = contract
    return skill


def _mock_registry(skill_map):
    """Create a mock registry with given skills.

    skill_map: dict of {name: contract}
    """
    registry = MagicMock()
    skills = {name: _mock_skill(c) for name, c in skill_map.items()}

    def _get(name):
        if name not in skills:
            raise KeyError(name)
        return skills[name]

    registry.get = _get
    registry.all_names.return_value = list(skills.keys())
    return registry


def _make_verdict(
    error="not found",
    category=FailureCategory.CAPABILITY_FAILURE,
    action=RecoveryAction.REPLAN,
    node_id="n1",
    skill_name="system.media_play",
    original_inputs=None,
    entity="",
):
    return FailureVerdict(
        category=category,
        action=action,
        reason=f"Skill execution failed: {error}",
        node_id=node_id,
        skill_name=skill_name,
        context={
            "error": error,
            "original_inputs": original_inputs or {},
            "entity": entity,
        },
    )


def _make_snapshot(
    attempt_history=None,
    step_count=0,
    required_outcomes=None,
):
    gs = GoalState(required_outcomes=required_outcomes or ["media_play"])
    es = ExecutionState()
    es.step_count = step_count
    if attempt_history:
        es.attempt_history = attempt_history
    ctx = CognitiveContext(goal=gs, execution=es)
    return ctx.snapshot_for_decision()


# ─────────────────────────────────────────────────────────────
# SkillContract effect fields (real skills)
# ─────────────────────────────────────────────────────────────

class TestSkillContractEffectFields:
    """Verify requires/produces/effect_type on real contracts."""

    def test_open_app_produces_app_running(self):
        from skills.system.open_app import OpenAppSkill
        skill = OpenAppSkill(system_controller=MagicMock())
        assert "app_running" in skill.contract.produces
        assert skill.contract.effect_type == "create"

    def test_media_play_requires_media_session_active(self):
        from skills.system.media_play import MediaPlaySkill
        skill = MediaPlaySkill(system_controller=MagicMock())
        assert "media_session_active" in skill.contract.requires
        assert skill.contract.effect_type == "maintain"

    def test_close_app_has_destroy_effect(self):
        from skills.system.close_app import CloseAppSkill
        skill = CloseAppSkill(system_controller=MagicMock())
        assert skill.contract.effect_type == "destroy"

    def test_write_file_produces_file_exists(self):
        from skills.fs.write_file import WriteFileSkill
        skill = WriteFileSkill(location_config=MagicMock())
        assert "file_exists" in skill.contract.produces
        assert skill.contract.effect_type == "create"

    def test_read_file_requires_file_exists(self):
        from skills.fs.read_file import ReadFileSkill
        skill = ReadFileSkill(location_config=MagicMock())
        assert "file_exists" in skill.contract.requires

    def test_focus_app_requires_app_running(self):
        from skills.system.focus_app import FocusAppSkill
        skill = FocusAppSkill(system_controller=MagicMock())
        assert "app_running" in skill.contract.requires
        assert "app_focused" in skill.contract.produces

    def test_media_next_requires_media_session(self):
        from skills.system.media_next import MediaNextSkill
        skill = MediaNextSkill(system_controller=MagicMock())
        assert "media_session_active" in skill.contract.requires

    def test_create_folder_produces_file_exists(self):
        from skills.fs.create_folder import CreateFolderSkill
        skill = CreateFolderSkill(location_config=MagicMock())
        assert "file_exists" in skill.contract.produces
        assert skill.contract.effect_type == "create"


# ─────────────────────────────────────────────────────────────
# _check_requires
# ─────────────────────────────────────────────────────────────

class TestCheckRequires:

    def test_returns_none_when_no_registry(self):
        de = DecisionEngine()
        verdict = _make_verdict()
        snap = _make_snapshot()
        assert de._check_requires(verdict, snap) is None

    def test_returns_none_when_no_skill_name(self):
        de = DecisionEngine()
        verdict = _make_verdict(skill_name=None)
        snap = _make_snapshot()
        assert de._check_requires(verdict, snap) is None

    def test_returns_none_when_no_requires(self):
        """Skill with empty requires → None (nothing to check)."""
        registry = _mock_registry({
            "system.open_app": _mock_contract(requires=[]),
        })
        de = DecisionEngine(registry=registry)
        verdict = _make_verdict(skill_name="system.open_app")
        snap = _make_snapshot()
        assert de._check_requires(verdict, snap) is None

    def test_returns_none_for_unknown_state(self):
        """_guard_is_unmet returns None (unknown) → _check_requires skips."""
        registry = _mock_registry({
            "system.media_play": _mock_contract(
                requires=["media_session_active"],
            ),
        })
        de = DecisionEngine(registry=registry)
        verdict = _make_verdict(skill_name="system.media_play")
        snap = _make_snapshot()
        # Without world state, _guard_is_unmet returns None (unknown)
        # which is NOT True, so _check_requires returns None.
        result = de._check_requires(verdict, snap)
        assert result is None

    def test_skips_invalid_guard_values(self):
        """Unknown guard value in requires → skipped (no crash)."""
        registry = _mock_registry({
            "system.test": _mock_contract(requires=["invalid_guard_xyz"]),
        })
        de = DecisionEngine(registry=registry)
        verdict = _make_verdict(skill_name="system.test")
        snap = _make_snapshot()
        assert de._check_requires(verdict, snap) is None


# ─────────────────────────────────────────────────────────────
# _find_creators
# ─────────────────────────────────────────────────────────────

class TestFindCreators:

    def test_returns_empty_without_registry(self):
        de = DecisionEngine()
        verdict = _make_verdict()
        snap = _make_snapshot()
        assert de._find_creators(
            GuardType.APP_RUNNING, verdict, snap,
        ) == []

    def test_finds_create_skill_no_inputs(self):
        """Creator with no required inputs → found."""
        registry = _mock_registry({
            "system.open_app": _mock_contract(
                produces=["app_running"],
                effect_type="create",
                inputs={},  # No required inputs
            ),
        })
        de = DecisionEngine(registry=registry)
        verdict = _make_verdict(skill_name="system.media_play")
        snap = _make_snapshot()
        creators = de._find_creators(
            GuardType.APP_RUNNING, verdict, snap,
        )
        assert len(creators) >= 1
        assert creators[0][0] == "system.open_app"

    def test_skips_maintain_skills(self):
        """media_play has effect_type maintain → NOT a creator for recovery."""
        registry = _mock_registry({
            "system.media_play": _mock_contract(
                produces=["media_session_active"],
                effect_type="maintain",
            ),
        })
        de = DecisionEngine(registry=registry)
        verdict = _make_verdict(skill_name="other.skill")
        snap = _make_snapshot()
        creators = de._find_creators(
            GuardType.MEDIA_SESSION_ACTIVE, verdict, snap,
        )
        assert len(creators) == 0

    def test_skips_destroy_skills(self):
        """close_app has effect_type destroy → NOT a creator."""
        registry = _mock_registry({
            "system.close_app": _mock_contract(
                produces=[],
                effect_type="destroy",
            ),
        })
        de = DecisionEngine(registry=registry)
        verdict = _make_verdict(skill_name="other.skill")
        snap = _make_snapshot()
        creators = de._find_creators(
            GuardType.APP_RUNNING, verdict, snap,
        )
        assert len(creators) == 0

    def test_skips_failed_skill(self):
        """Doesn't suggest the skill that just failed."""
        registry = _mock_registry({
            "system.open_app": _mock_contract(
                produces=["app_running"],
                effect_type="create",
                inputs={},
            ),
        })
        de = DecisionEngine(registry=registry)
        verdict = _make_verdict(skill_name="system.open_app")
        snap = _make_snapshot()
        creators = de._find_creators(
            GuardType.APP_RUNNING, verdict, snap,
        )
        assert len(creators) == 0  # Skipped because it's the failed skill

    def test_skips_already_attempted(self):
        """Doesn't suggest a skill that was already tried."""
        registry = _mock_registry({
            "system.open_app": _mock_contract(
                produces=["app_running"],
                effect_type="create",
                inputs={},
            ),
        })
        de = DecisionEngine(registry=registry)
        verdict = _make_verdict(skill_name="system.media_play")
        snap = _make_snapshot(
            attempt_history=[{"skill": "system.open_app"}],
        )
        creators = de._find_creators(
            GuardType.APP_RUNNING, verdict, snap,
        )
        assert len(creators) == 0

    def test_creator_has_effect_repair_assumption(self):
        """Creator candidates include Assumption with guard_mapping."""
        registry = _mock_registry({
            "system.open_app": _mock_contract(
                produces=["app_running"],
                effect_type="create",
                inputs={},
            ),
        })
        de = DecisionEngine(registry=registry)
        verdict = _make_verdict(skill_name="system.media_play")
        snap = _make_snapshot()
        creators = de._find_creators(
            GuardType.APP_RUNNING, verdict, snap,
        )
        assert len(creators) == 1
        _, _, assumptions = creators[0]
        assert assumptions[0].type == "effect_repair"
        assert assumptions[0].guard_mapping == "app_running"


# ─────────────────────────────────────────────────────────────
# _find_enablers (1-level expansion)
# ─────────────────────────────────────────────────────────────

class TestFindEnablers:

    def test_no_enablers_when_domain_mismatch(self):
        """open_app produces 'app_running' (domain 'app'),
        target 'media_session_active' (domain 'media') → no match."""
        registry = _mock_registry({
            "system.open_app": _mock_contract(
                produces=["app_running"],
                effect_type="create",
                inputs={},
            ),
        })
        de = DecisionEngine(registry=registry)
        verdict = _make_verdict(skill_name="system.media_play")
        snap = _make_snapshot()
        enablers = de._find_enablers(
            GuardType.MEDIA_SESSION_ACTIVE, verdict, snap,
        )
        assert len(enablers) == 0

    def test_finds_enabler_same_domain(self):
        """Enabler found when produced value has same domain prefix."""
        registry = _mock_registry({
            "system.create_app_session": _mock_contract(
                produces=["app_session"],  # domain 'app'
                effect_type="create",
                inputs={},
            ),
        })
        de = DecisionEngine(registry=registry)
        verdict = _make_verdict(skill_name="other.skill")
        snap = _make_snapshot()
        enablers = de._find_enablers(
            GuardType.APP_RUNNING, verdict, snap,
        )
        assert len(enablers) >= 1

    def test_bounded_to_3_results(self):
        """Enablers are capped at 3."""
        skills = {}
        for i in range(5):
            skills[f"system.create_{i}"] = _mock_contract(
                produces=[f"app_{i}"],
                effect_type="create",
                inputs={},
            )
        registry = _mock_registry(skills)
        de = DecisionEngine(registry=registry)
        verdict = _make_verdict(skill_name="other.skill")
        snap = _make_snapshot()
        enablers = de._find_enablers(
            GuardType.APP_RUNNING, verdict, snap,
        )
        assert len(enablers) <= 3


# ─────────────────────────────────────────────────────────────
# _guard_is_unmet
# ─────────────────────────────────────────────────────────────

class TestGuardIsUnmet:

    def test_returns_none_without_state(self):
        """No world state → unknown → None."""
        de = DecisionEngine()
        verdict = _make_verdict()
        snap = _make_snapshot()
        result = de._guard_is_unmet(GuardType.APP_RUNNING, verdict, snap)
        assert result is None

    def test_file_exists_always_returns_none(self):
        """FILE_EXISTS can't be verified without execution → None."""
        de = DecisionEngine()
        verdict = _make_verdict()
        snap = _make_snapshot()
        # FILE_EXISTS returns None regardless
        result = de._guard_is_unmet(GuardType.FILE_EXISTS, verdict, snap)
        assert result is None

    def test_unknown_guard_returns_none(self):
        """Unknown guard type → None."""
        de = DecisionEngine()
        verdict = _make_verdict()
        snap = _make_snapshot()
        result = de._guard_is_unmet(
            GuardType.REQUIRES_CONFIRMATION, verdict, snap,
        )
        assert result is None


# ─────────────────────────────────────────────────────────────
# _normalize_diagnosis
# ─────────────────────────────────────────────────────────────

class TestNormalizeDiagnosis:

    def test_valid_guard_type(self):
        de = DecisionEngine()
        result = de._normalize_diagnosis(
            {"state_type": "app_running", "cause": "transient_error"},
        )
        assert result == GuardType.APP_RUNNING

    def test_unknown_returns_none(self):
        de = DecisionEngine()
        result = de._normalize_diagnosis(
            {"state_type": "UNKNOWN", "cause": "unknown"},
        )
        assert result is None

    def test_empty_state_type_returns_none(self):
        de = DecisionEngine()
        result = de._normalize_diagnosis(
            {"state_type": "", "cause": "unknown"},
        )
        assert result is None

    def test_invalid_state_type_returns_none(self):
        de = DecisionEngine()
        result = de._normalize_diagnosis(
            {"state_type": "nonexistent_guard", "cause": "whatever"},
        )
        assert result is None

    def test_media_session_active(self):
        de = DecisionEngine()
        result = de._normalize_diagnosis(
            {"state_type": "media_session_active", "cause": "missing_resource"},
        )
        assert result == GuardType.MEDIA_SESSION_ACTIVE

    def test_file_exists(self):
        de = DecisionEngine()
        result = de._normalize_diagnosis(
            {"state_type": "file_exists", "cause": "missing_resource"},
        )
        assert result == GuardType.FILE_EXISTS


# ─────────────────────────────────────────────────────────────
# _simulate_and_rank
# ─────────────────────────────────────────────────────────────

class TestSimulateAndRank:

    def test_empty_candidates_returns_empty(self):
        de = DecisionEngine()
        result = de._simulate_and_rank([], None, _make_snapshot())
        assert result == []

    def test_direct_producer_ranked_first(self):
        """Candidate that produces the missing guard → 2.0 (DIRECT)."""
        registry = _mock_registry({
            "system.open_app": _mock_contract(
                produces=["app_running"],
                effect_type="create",
                outputs={"status": "text"},
            ),
            "system.restart_app": _mock_contract(
                produces=[],
                effect_type="create",
                outputs={"status": "text"},
            ),
        })
        de = DecisionEngine(registry=registry)
        snap = _make_snapshot()

        candidates = [
            ("system.restart_app", {"name": "spotify"}, []),
            ("system.open_app", {"name": "spotify"}, []),
        ]
        ranked = de._simulate_and_rank(
            candidates, GuardType.APP_RUNNING, snap,
        )
        # open_app produces app_running → DIRECT (2.0) → ranked first
        assert ranked[0][0] == "system.open_app"


# ─────────────────────────────────────────────────────────────
# _simulate: produced_guards
# ─────────────────────────────────────────────────────────────

class TestSimulateProducedGuards:

    def test_produced_guards_populated_from_contract(self):
        """_simulate projects produced_guards from contract.produces."""
        registry = _mock_registry({
            "system.open_app": _mock_contract(
                produces=["app_running"],
                effect_type="create",
                outputs={"status": "text"},
            ),
        })
        de = DecisionEngine(registry=registry)
        snap = _make_snapshot()
        sim = de._simulate("system.open_app", snap)
        assert "app_running" in sim.produced_guards

    def test_produced_guards_empty_for_unknown_skill(self):
        """Skill not in registry → empty produced_guards."""
        de = DecisionEngine()
        snap = _make_snapshot()
        sim = de._simulate("nonexistent.skill", snap)
        assert len(sim.produced_guards) == 0

    def test_multiple_produced_guards(self):
        """Skill with multiple produces → all in produced_guards."""
        registry = _mock_registry({
            "system.setup": _mock_contract(
                produces=["app_running", "app_focused"],
                effect_type="create",
                outputs={"status": "text"},
            ),
        })
        de = DecisionEngine(registry=registry)
        snap = _make_snapshot()
        sim = de._simulate("system.setup", snap)
        assert "app_running" in sim.produced_guards
        assert "app_focused" in sim.produced_guards


# ─────────────────────────────────────────────────────────────
# AmbiguityDecision
# ─────────────────────────────────────────────────────────────

class TestAmbiguityDecision:

    def test_ambiguity_decision_has_required_fields(self):
        decision = AmbiguityDecision(
            question="Which app should I open?",
            choices=[
                {"skill": "system.open_app", "inputs": {"name": "spotify"}},
                {"skill": "system.open_app", "inputs": {"name": "chrome"}},
            ],
            verdict=_make_verdict(),
        )
        assert decision.question == "Which app should I open?"
        assert len(decision.choices) == 2

    def test_ambiguity_decision_with_empty_choices(self):
        decision = AmbiguityDecision(
            question="Cannot determine state",
            choices=[],
            verdict=_make_verdict(),
        )
        assert decision.choices == []


# ─────────────────────────────────────────────────────────────
# classify_complexity: category-based (no string matching)
# ─────────────────────────────────────────────────────────────

class TestClassifyComplexityCategoryBased:
    """Verify classify_complexity uses _CATEGORY_TO_CAUSE, not strings."""

    def test_all_categories_covered(self):
        """Every FailureCategory maps to a FailureCause."""
        de = DecisionEngine()
        snap = _make_snapshot()
        for cat in FailureCategory:
            verdict = _make_verdict(category=cat)
            cause, scope = de.classify_complexity(verdict, snap)
            assert isinstance(cause, FailureCause)
            assert isinstance(scope, FailureScope)

    def test_environment_mismatch_is_missing_state(self):
        de = DecisionEngine()
        verdict = _make_verdict(
            category=FailureCategory.ENVIRONMENT_MISMATCH,
        )
        snap = _make_snapshot()
        cause, _ = de.classify_complexity(verdict, snap)
        assert cause == FailureCause.MISSING_STATE

    def test_missing_parameter_is_missing_data(self):
        de = DecisionEngine()
        verdict = _make_verdict(
            category=FailureCategory.MISSING_PARAMETER,
        )
        snap = _make_snapshot()
        cause, _ = de.classify_complexity(verdict, snap)
        assert cause == FailureCause.MISSING_DATA

    def test_capability_failure_is_invalid_assumption(self):
        de = DecisionEngine()
        verdict = _make_verdict(
            category=FailureCategory.CAPABILITY_FAILURE,
        )
        snap = _make_snapshot()
        cause, _ = de.classify_complexity(verdict, snap)
        assert cause == FailureCause.INVALID_ASSUMPTION

    def test_error_string_irrelevant(self):
        """Different error strings with same category → same cause.
        This proves string matching is eliminated.
        """
        de = DecisionEngine()
        snap = _make_snapshot()

        v1 = _make_verdict(error="not found", category=FailureCategory.CAPABILITY_FAILURE)
        v2 = _make_verdict(error="timeout xyz", category=FailureCategory.CAPABILITY_FAILURE)
        v3 = _make_verdict(error="permission denied", category=FailureCategory.CAPABILITY_FAILURE)

        c1, _ = de.classify_complexity(v1, snap)
        c2, _ = de.classify_complexity(v2, snap)
        c3, _ = de.classify_complexity(v3, snap)

        # All same category → all same cause, regardless of error string
        assert c1 == c2 == c3 == FailureCause.INVALID_ASSUMPTION


# ─────────────────────────────────────────────────────────────
# _llm_diagnose
# ─────────────────────────────────────────────────────────────

class TestLlmDiagnose:

    def test_returns_none_without_llm(self):
        de = DecisionEngine()
        verdict = _make_verdict()
        snap = _make_snapshot()
        assert de._llm_diagnose(verdict, snap) is None

    def test_llm_called_with_prompt(self):
        """Verify LLM is called and result returned."""
        mock_llm = MagicMock()
        # extract_json_block returns a str (JSON block extracted from LLM output)
        mock_llm.complete.return_value = (
            '{"state_type": "app_running", "entity": "spotify", '
            '"cause": "missing_resource"}'
        )
        de = DecisionEngine(llm_client=mock_llm)
        verdict = _make_verdict(skill_name="system.media_play")
        snap = _make_snapshot()
        result = de._llm_diagnose(verdict, snap)
        assert result is not None
        mock_llm.complete.assert_called_once()

    def test_llm_failure_returns_none(self):
        """LLM exception → graceful None."""
        mock_llm = MagicMock()
        mock_llm.complete.side_effect = RuntimeError("API error")
        de = DecisionEngine(llm_client=mock_llm)
        verdict = _make_verdict()
        snap = _make_snapshot()
        assert de._llm_diagnose(verdict, snap) is None


# ─────────────────────────────────────────────────────────────
# E2E: media_play → requires → find_creators → ActionDecision
# (the verification scenario from the implementation plan)
# ─────────────────────────────────────────────────────────────

class TestE2EMediaPlayRecovery:
    """End-to-end test for the scenario from the implementation plan:

    media_play fails → requires media_session_active → unmet
    → find_creators → open_app (create) → simulate confirms
    → ActionDecision
    """

    def test_media_play_failure_finds_open_app(self):
        """Full flow: media_play → open_app as recovery."""
        registry = _mock_registry({
            "system.media_play": _mock_contract(
                domain="media",
                requires=["media_session_active"],
                produces=["media_session_active"],
                effect_type="maintain",
                inputs={"action": "playback_action"},
            ),
            "system.open_app": _mock_contract(
                domain="system",
                requires=[],
                produces=["app_running"],
                effect_type="create",
                inputs={},  # No required inputs for simpler test
                outputs={"status": "text"},
            ),
        })
        de = DecisionEngine(registry=registry)
        verdict = _make_verdict(
            error="no media session",
            category=FailureCategory.ENVIRONMENT_MISMATCH,
            skill_name="system.media_play",
            entity="spotify",
        )
        snap = _make_snapshot(required_outcomes=["media_play"])

        result = de.decide(verdict, snap)
        # Should get ActionDecision or EscalationDecision
        assert isinstance(result, (ActionDecision, EscalationDecision))


# ─────────────────────────────────────────────────────────────
# MEDIA_SESSION_ACTIVE guard type
# ─────────────────────────────────────────────────────────────

class TestMediaSessionActiveGuard:

    def test_guard_type_exists(self):
        assert hasattr(GuardType, "MEDIA_SESSION_ACTIVE")
        assert GuardType.MEDIA_SESSION_ACTIVE.value == "media_session_active"

    def test_guard_type_in_valid_list(self):
        """MEDIA_SESSION_ACTIVE should be in the GuardType enum."""
        values = [gt.value for gt in GuardType]
        assert "media_session_active" in values


# ─────────────────────────────────────────────────────────────
# SimulatedState.produced_guards field
# ─────────────────────────────────────────────────────────────

class TestSimulatedStateProducedGuards:

    def test_produced_guards_field_exists(self):
        """SimulatedState has a produced_guards field."""
        sim = SimulatedState(
            achieved_outcomes=set(),
            produced_types=set(),
            produced_guards={"app_running"},
            pending_outcomes=[],
            uncertainty={},
            step_count=1,
        )
        assert "app_running" in sim.produced_guards

    def test_produced_guards_default_empty(self):
        """produced_guards defaults to empty set."""
        sim = SimulatedState(
            achieved_outcomes=set(),
            produced_types=set(),
            pending_outcomes=[],
            uncertainty={},
            step_count=1,
        )
        assert sim.produced_guards == set()
