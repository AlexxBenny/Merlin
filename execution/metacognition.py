# execution/metacognition.py

"""
MetaCognitionEngine — Deterministic failure classification.

Classifies execution failures into 3 actionable categories:

1. ENVIRONMENT_MISMATCH:
   Guard failed, focus changed, app closed — environment
   diverged from plan assumptions. Action: RETRY.

2. MISSING_PARAMETER:
   Skill needs input that wasn't in the plan (e.g. filename
   unknown when saving). Action: ASK_USER.

3. CAPABILITY_FAILURE:
   Skill itself failed (timeout, crash, permission denied).
   Action: REPLAN.

Architecture:
  ExecutionSupervisor
          ↓ (on failure)
  MetaCognitionEngine.classify()
          ↓
  FailureVerdict(category, action, context)

Integration point: _execute_guarded_node() calls classify()
after a node fails all retries, to decide whether to abort,
ask the user, or request replanning.
"""

import logging
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Failure categories
# ─────────────────────────────────────────────────────────────

class FailureCategory(str, Enum):
    """Deterministic failure categories.

    Each maps to exactly one recovery action.
    No ambiguity — the mapping is 1:1.
    """
    ENVIRONMENT_MISMATCH = "environment_mismatch"  # → retry
    MISSING_PARAMETER = "missing_parameter"        # → ask_user
    CAPABILITY_FAILURE = "capability_failure"       # → replan


class RecoveryAction(str, Enum):
    """What the supervisor should do after classification."""
    RETRY = "retry"       # Re-execute the same node
    ASK_USER = "ask_user" # Pause and request user input
    REPLAN = "replan"     # Return to planner for new plan
    ABORT = "abort"       # Stop execution entirely


# ─────────────────────────────────────────────────────────────
# FailureVerdict — structured output of classification
# ─────────────────────────────────────────────────────────────

class FailureVerdict(BaseModel):
    """Structured result of failure classification.

    Tells the supervisor exactly what to do and why.
    """
    model_config = ConfigDict(extra="forbid")

    category: FailureCategory
    action: RecoveryAction
    reason: str
    node_id: str
    skill_name: Optional[str] = None
    # Context for recovery: e.g. which param is missing, which app is gone
    context: Dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────────────────────
# Signal patterns — how we recognize each category
# ─────────────────────────────────────────────────────────────

# Guard types that indicate environment mismatch
_ENVIRONMENT_GUARD_TYPES = {
    "app_running", "app_focused", "window_visible", "active_window",
}

# Error messages / keywords that indicate missing parameters
_MISSING_PARAM_SIGNALS = [
    "missing required",
    "required parameter",
    "no value for",
    "parameter not provided",
    "filename unknown",
    "path not specified",
    "cannot determine",
]

# Error keywords that indicate hard capability failures
_CAPABILITY_FAILURE_SIGNALS = [
    "timeout",
    "timed out",
    "permission denied",
    "access denied",
    "not found",
    "crash",
    "exception",
    "skill failed",
    "execution error",
]


# ─────────────────────────────────────────────────────────────
# MetaCognitionEngine
# ─────────────────────────────────────────────────────────────

class MetaCognitionEngine:
    """Deterministic failure classifier for execution loop.

    Classifies failures using pattern matching against:
    - Guard failure types
    - Error messages from skill execution
    - Node status metadata

    Rules are evaluated in priority order:
    1. Environment mismatch (guard failures)
    2. Missing parameter (input errors)
    3. Capability failure (everything else)

    No LLM calls. Fully deterministic. O(1) classification.
    """

    def classify(
        self,
        node_id: str,
        skill_name: Optional[str] = None,
        failed_guards: Optional[List[str]] = None,
        error_message: Optional[str] = None,
        node_status: Optional[str] = None,
        retries_exhausted: bool = False,
        failure_class: Optional[str] = None,
    ) -> FailureVerdict:
        """Classify a node failure into a recovery action.

        Args:
            node_id: ID of the failed node
            skill_name: Name of the skill that failed
            failed_guards: Guard type values that failed (if any)
            error_message: Error string from execution
            node_status: Status string from executor
            retries_exhausted: Whether max retries have been exceeded
            failure_class: Structured failure class from input resolution
                (MISSING_DATA, INVALID_REFERENCE, TYPE_MISMATCH).
                When provided, bypasses string-match heuristics.

        Returns:
            FailureVerdict with category, action, reason, and context
        """
        failed_guards = failed_guards or []
        error_message = (error_message or "").lower()

        # ── Priority 0: Structured failure class (input resolution) ──
        # Direct routing — no string matching needed.
        if failure_class == "MISSING_DATA":
            return FailureVerdict(
                category=FailureCategory.MISSING_PARAMETER,
                action=RecoveryAction.ASK_USER,
                reason=f"Upstream data unavailable: {error_message}",
                node_id=node_id,
                skill_name=skill_name,
                context={"failure_class": failure_class, "error": error_message},
            )
        if failure_class in ("INVALID_REFERENCE", "TYPE_MISMATCH"):
            return FailureVerdict(
                category=FailureCategory.CAPABILITY_FAILURE,
                action=RecoveryAction.REPLAN,
                reason=f"Input resolution error ({failure_class}): {error_message}",
                node_id=node_id,
                skill_name=skill_name,
                context={"failure_class": failure_class, "error": error_message},
            )

        # ── Priority 1: Environment mismatch (guard failures) ──
        env_guards = [
            g for g in failed_guards
            if g.lower() in _ENVIRONMENT_GUARD_TYPES
        ]
        if env_guards:
            return FailureVerdict(
                category=FailureCategory.ENVIRONMENT_MISMATCH,
                action=(
                    RecoveryAction.REPLAN if retries_exhausted
                    else RecoveryAction.RETRY
                ),
                reason=f"Guard(s) failed: {', '.join(env_guards)}",
                node_id=node_id,
                skill_name=skill_name,
                context={"failed_guards": env_guards},
            )

        # ── Priority 2: Missing parameter ──
        if any(signal in error_message for signal in _MISSING_PARAM_SIGNALS):
            return FailureVerdict(
                category=FailureCategory.MISSING_PARAMETER,
                action=RecoveryAction.ASK_USER,
                reason=f"Missing parameter: {error_message}",
                node_id=node_id,
                skill_name=skill_name,
                context={"error": error_message},
            )

        # ── Priority 3: Capability failure (catchall) ──
        # After retries exhausted → replan
        # Before retries exhausted → retry
        action = (
            RecoveryAction.REPLAN if retries_exhausted
            else RecoveryAction.RETRY
        )
        return FailureVerdict(
            category=FailureCategory.CAPABILITY_FAILURE,
            action=action,
            reason=f"Skill execution failed: {error_message or 'unknown error'}",
            node_id=node_id,
            skill_name=skill_name,
            context={"error": error_message, "status": node_status},
        )

    def should_abort(self, verdicts: List[FailureVerdict]) -> bool:
        """Check if the mission should be aborted based on accumulated failures.

        Abort conditions:
        - Any verdict with ABORT action
        - Multiple REPLAN verdicts (plan is fundamentally broken)
        - Same node failed with REPLAN twice (loop detection)
        """
        if any(v.action == RecoveryAction.ABORT for v in verdicts):
            return True

        replan_count = sum(1 for v in verdicts if v.action == RecoveryAction.REPLAN)
        if replan_count >= 2:
            logger.warning(
                "[META] Multiple REPLAN verdicts — aborting mission"
            )
            return True

        # Loop detection: same node failed twice with REPLAN
        replan_nodes = [
            v.node_id for v in verdicts if v.action == RecoveryAction.REPLAN
        ]
        if len(replan_nodes) != len(set(replan_nodes)):
            logger.warning(
                "[META] Same node failed multiple times — aborting"
            )
            return True

        return False


# ─────────────────────────────────────────────────────────────
# OutcomeSeverity — node outcome classification
# ─────────────────────────────────────────────────────────────

class OutcomeSeverity(str, Enum):
    """Severity of a node outcome.

    BENIGN:       Intent already satisfied or idempotent no-op.
                  No replanning needed.
    SOFT_FAILURE: Skill ran but user intent is NOT met.
                  Replanning may recover.
    HARD_FAILURE: Skill crashed, timed out, or was denied.
                  Report failure.
    """
    BENIGN = "benign"
    SOFT_FAILURE = "soft_failure"
    HARD_FAILURE = "hard_failure"


# Reasons that mean the user's intent is ALREADY satisfied.
# "already_playing" = play was requested, media IS playing → success.
# These do NOT trigger replanning.
_IDEMPOTENT_REASONS = frozenset({
    "already_playing",
    "already_paused",
    "already_muted",
    "already_unmuted",
})


class OutcomeAnalyzer:
    """Deterministic outcome classifier. Zero domain logic.

    Classifies every node result into a severity level.
    Only SOFT_FAILURE triggers replanning via the existing
    coordinator → cortex pipeline.

    Classification rules:
      COMPLETED                    → BENIGN
      NO_OP + idempotent reason    → BENIGN
      NO_OP + any other reason     → SOFT_FAILURE
      FAILED / TIMED_OUT           → HARD_FAILURE
    """

    def classify(self, status: str, metadata: Optional[Dict[str, Any]] = None) -> OutcomeSeverity:
        """Classify a node's execution outcome.

        Args:
            status: NodeStatus string (completed, no_op, failed, timed_out)
            metadata: Skill execution metadata (contains 'reason' key)

        Returns:
            OutcomeSeverity
        """
        metadata = metadata or {}

        if status in ("completed",):
            return OutcomeSeverity.BENIGN

        if status in ("no_op",):
            reason = metadata.get("reason", "")
            if reason in _IDEMPOTENT_REASONS:
                return OutcomeSeverity.BENIGN
            return OutcomeSeverity.SOFT_FAILURE

        # Browser entity not found — recoverable via replanning
        # (e.g. entity_ref didn't match, index drifted, page changed)
        if status in ("failed",):
            reason = metadata.get("reason", "")
            error = metadata.get("error", "")
            combined = f"{reason} {error}".lower()
            if any(sig in combined for sig in (
                "no entity at index",
                "entity_index is required",
                "entity_ref",
            )):
                return OutcomeSeverity.SOFT_FAILURE

        # FAILED, TIMED_OUT, or unknown
        return OutcomeSeverity.HARD_FAILURE


# ─────────────────────────────────────────────────────────────
# DecisionEngine — bounded adaptive recovery (Stage 2: scored)
# ─────────────────────────────────────────────────────────────

# NOTE: The old _HEURISTIC_TABLE and _extract_filename were removed.
# Recovery is now fully contract-driven:
#   _check_requires → _find_creators → _find_revealers → _find_enablers
# This scales automatically as new skills declare produces/requires/effect_type.
# Dedup is handled in _find_creators/_find_revealers (skip already-attempted).


class DecisionEngine:
    """Bounded adaptive recovery for local failures.

    Stage 4 + Effect-driven recovery.

    7-step reasoning loop:
        1. Guard check (truth)       — evaluate contract.requires
        2. LLM diagnosis (hypothesis) — constrained to GuardType enum
        3. Normalize (bridge)         — _normalize_diagnosis → GuardType|None
        4. Validate (confirm)         — WorldState check, None → AmbiguityDecision
        5. Find producers (derived)   — scan skill.contract.produces
        6. Simulate transitions       — produced_guards reasoning
        7. Score + Decide             — ActionDecision | AmbiguityDecision | EscalationDecision

    Architecture position:
        MetaCognition.classify() → FailureVerdict        [EXISTING]
            ↓
        DecisionEngine.decide() → ActionDecision | AmbiguityDecision | EscalationDecision
            ↓
        ExecutionSupervisor enqueues / orchestrator asks user / escalates
    """

    _MIN_DECISIVENESS_GAP = 0.05

    def __init__(self, registry=None, llm_client=None):
        self._registry = registry
        self._llm = llm_client

    def decide(self, verdict, snapshot, original_plan=None):
        """Main entry point: classify complexity, then decide.

        Args:
            verdict: FailureVerdict from MetaCognition.classify()
            snapshot: DecisionSnapshot (frozen, immutable)
            original_plan: MissionPlan (for DAG dependency analysis)

        Returns:
            ActionDecision | AmbiguityDecision | EscalationDecision
        """
        from execution.cognitive_context import (
            ActionDecision, EscalationDecision, EscalationLevel,
            AmbiguityDecision, DecisionExplanation, Assumption,
        )

        # ── Budget check ──
        if not snapshot.within_budget:
            return EscalationDecision(
                level=EscalationLevel.GLOBAL,
                reason="Budget exhausted",
                verdict=verdict,
            )

        # ── 2-axis classification ──
        cause, scope = self.classify_complexity(verdict, snapshot, original_plan)

        # ── Route by scope ──
        if scope.value == "multi_step":
            return EscalationDecision(
                level=EscalationLevel.GLOBAL,
                reason=f"Multi-step failure: {cause.value}",
                verdict=verdict,
                context={"cause": cause.value, "scope": scope.value},
            )

        # ── Route by cause ──
        from execution.cognitive_context import FailureCause
        if cause == FailureCause.EXTERNAL_DEPENDENCY:
            return EscalationDecision(
                level=EscalationLevel.USER,
                reason=f"External dependency: {verdict.reason}",
                verdict=verdict,
            )

        # ── Effect-driven candidate generation (7-step loop) ──
        candidates, strategy = self._generate_candidates(verdict, snapshot)

        if candidates:
            # Score and rank candidates
            scored = []
            for skill_name, inputs, assumptions in candidates:
                score, components, lookahead_data = self._score_normalized(
                    skill_name, inputs, snapshot,
                )
                scored.append((score, skill_name, inputs, assumptions, components, lookahead_data))

            scored.sort(key=lambda x: x[0])  # lower = better
            best_score, best_skill, best_inputs, best_assumptions, best_comp, best_lookahead = scored[0]

            # Ambiguity check: top candidates too close
            if len(scored) >= 2:
                gap = abs(scored[1][0] - scored[0][0])
                if gap < self._MIN_DECISIVENESS_GAP:
                    choices = [
                        {"skill": sk, "inputs": inp, "score": round(s, 4)}
                        for s, sk, inp, _, _, _ in scored[:3]
                    ]
                    return AmbiguityDecision(
                        choices=choices,
                        question=(
                            f"Multiple options for '{verdict.reason}': "
                            + ", ".join(c["skill"] for c in choices)
                        ),
                        verdict=verdict,
                    )

            # Build rejection list for explanation (O1)
            rejected = []
            for s, sk, inp, _, comp, _ in scored[1:]:
                rejected.append({
                    "action": sk,
                    "score": round(s, 4),
                    "reason": self._rejection_reason(comp),
                })

            explanation = self._build_explanation(
                best_skill, best_score, best_comp, snapshot, rejected,
                best_lookahead,
            )

            return ActionDecision(
                skill=best_skill,
                inputs=best_inputs,
                assumptions=best_assumptions,
                score=best_score,
                explanation=explanation,
                strategy_source=strategy,
            )

        # ── No candidates → escalate ──
        if cause == FailureCause.MISSING_DATA:
            return EscalationDecision(
                level=EscalationLevel.USER,
                reason=f"Cannot resolve locally: {verdict.reason}",
                verdict=verdict,
            )

        return EscalationDecision(
            level=EscalationLevel.GLOBAL,
            reason=f"No recovery for {cause.value}: {verdict.reason}",
            verdict=verdict,
        )

    def classify_complexity(self, verdict, snapshot, original_plan=None):
        """2-axis failure classification (Invariant I4).

        Axis 1 (Cause): refined from existing MetaCognition categories.
        Axis 2 (Scope): inferred from DAG dependency structure.
        """
        from execution.cognitive_context import FailureCause, FailureScope

        # ── Axis 1: Cause ──
        # Primary: structured category from MetaCognition.classify()
        _CATEGORY_TO_CAUSE = {
            FailureCategory.ENVIRONMENT_MISMATCH: FailureCause.MISSING_STATE,
            FailureCategory.MISSING_PARAMETER: FailureCause.MISSING_DATA,
            FailureCategory.CAPABILITY_FAILURE: FailureCause.INVALID_ASSUMPTION,
        }

        if verdict.category in _CATEGORY_TO_CAUSE:
            cause = _CATEGORY_TO_CAUSE[verdict.category]
        else:
            # Fallback: conservative default for EXECUTION_ERROR etc.
            cause = FailureCause.INVALID_ASSUMPTION

        # ── Axis 2: Scope ──
        needs_prerequisites = (cause == FailureCause.MISSING_STATE)

        has_dependents = False
        if original_plan is not None:
            has_dependents = any(
                verdict.node_id in node.depends_on
                for node in original_plan.nodes
                if node.id != verdict.node_id
            )

        if has_dependents or needs_prerequisites:
            scope = FailureScope.MULTI_STEP
        else:
            scope = FailureScope.SINGLE_STEP

        logger.info(
            "[DECISION] classify_complexity: node=%s cause=%s scope=%s",
            verdict.node_id, cause.value, scope.value,
        )
        return cause, scope

    # NOTE: _try_heuristic and _already_satisfied were removed.
    # Recovery is now fully contract-driven via _check_requires →
    # _find_creators → _find_revealers → _find_enablers.
    # Dedup is already in _find_creators/_find_revealers
    # (skip already-attempted via snapshot.attempt_history).

    # ─────────────────────────────────────────────────────────
    # Effect-driven reasoning helpers (Steps 1-6)
    # ─────────────────────────────────────────────────────────

    def _check_requires(self, verdict, snapshot):
        """Step 1: Check declared requires. Returns first unmet GuardType or None."""
        from execution.supervisor import GuardType

        if not verdict.skill_name or not self._registry:
            return None
        try:
            contract = self._registry.get(verdict.skill_name).contract
        except (KeyError, AttributeError):
            return None

        for guard_val in contract.requires:
            try:
                guard_type = GuardType(guard_val)
            except ValueError:
                continue
            result = self._guard_is_unmet(guard_type, verdict, snapshot)
            if result is True:
                logger.info(
                    "[DECISION] Step 1: requires '%s' is UNMET for %s",
                    guard_val, verdict.skill_name,
                )
                return guard_type

        return None

    def _find_creators(self, guard_type, verdict, snapshot):
        """Step 5: Find skills that CREATE the missing state.

        Only returns skills with effect_type == 'create'.
        If no direct creator found, tries 1-level expansion:
        find skills whose produces enables a creator of the target.
        """
        from execution.cognitive_context import Assumption

        if not self._registry:
            return []

        guard_val = guard_type.value if hasattr(guard_type, 'value') else str(guard_type)
        failed_skill = verdict.skill_name or ""
        candidates = []

        # Direct creators: skills where guard_val in produces AND effect_type == "create"
        for name in self._registry.all_names():
            if name == failed_skill:
                continue
            try:
                contract = self._registry.get(name).contract
            except (KeyError, AttributeError):
                continue

            if guard_val not in contract.produces:
                continue
            if contract.effect_type != "create":
                continue

            # Skip already-attempted
            if any(h.get("skill") == name for h in snapshot.attempt_history):
                continue

            inputs = self._infer_repair_inputs(contract, verdict, snapshot)
            if contract.inputs and not inputs:
                continue  # Can't determine required inputs

            assumptions = [Assumption(
                type="effect_repair",
                params={"missing_guard": guard_val, "producer": name},
                guard_mapping=guard_val,
                invert=False,
            )]
            candidates.append((name, inputs, assumptions))

        # Fallback chain: revealers first (discovery), then enablers (indirect)
        if not candidates:
            candidates = self._find_revealers(guard_type, verdict, snapshot)
        if not candidates:
            candidates = self._find_enablers(guard_type, verdict, snapshot)

        return candidates

    def _find_revealers(self, guard_type, verdict, snapshot):
        """Find skills that REVEAL (discover) the missing state.

        Parallel to _find_creators but for effect_type == 'reveal'.
        Revealers reduce uncertainty about existing state rather than
        creating new state. Used when no direct creator is found.

        Example: file_reference has no 'create' producer, but search_file
        has effect_type='reveal' and produces=['file_reference'].
        """
        from execution.cognitive_context import Assumption

        if not self._registry:
            return []

        guard_val = guard_type.value if hasattr(guard_type, 'value') else str(guard_type)
        failed_skill = verdict.skill_name or ""
        candidates = []

        for name in self._registry.all_names():
            if name == failed_skill:
                continue
            try:
                contract = self._registry.get(name).contract
            except (KeyError, AttributeError):
                continue

            if guard_val not in contract.produces:
                continue
            if contract.effect_type != "reveal":
                continue

            # Skip already-attempted
            if any(h.get("skill") == name for h in snapshot.attempt_history):
                continue

            inputs = self._infer_repair_inputs(contract, verdict, snapshot)
            if contract.inputs and not inputs:
                continue

            assumptions = [Assumption(
                type="reveal_repair",
                params={"missing_guard": guard_val, "revealer": name},
                guard_mapping=guard_val,
                invert=False,
            )]
            candidates.append((name, inputs, assumptions))

        return candidates

    def _find_enablers(self, target_guard, verdict, snapshot):
        """1-level expansion: find skills that enable a creator of target_guard.

        Pattern: target_guard has no direct creator, but some skill
        produces a prerequisite that a creator of target_guard requires.

        Example: media_session_active has no direct 'create' skill,
        but open_app creates app_running, and an app_running state
        may enable media_session_active.

        Uses existing _generate_follow_ups pattern.
        """
        from execution.cognitive_context import Assumption

        if not self._registry:
            return []

        guard_val = target_guard.value if hasattr(target_guard, 'value') else str(target_guard)
        failed_skill = verdict.skill_name or ""
        candidates = []

        # Find which skills require guard_val (consumers of this state)
        # Then find creators of prerequisites those consumers need
        for name in self._registry.all_names():
            if name == failed_skill:
                continue
            try:
                contract = self._registry.get(name).contract
            except (KeyError, AttributeError):
                continue

            if contract.effect_type != "create":
                continue

            # Check if this creator's produces could transitively enable guard_val
            # by looking at domain relationships and type matches
            for produced in contract.produces:
                # If a creator produces something in the same domain
                # as the target guard, it's a potential enabler
                produced_domain = produced.split("_")[0] if "_" in produced else ""
                target_domain = guard_val.split("_")[0] if "_" in guard_val else ""

                if produced_domain and produced_domain == target_domain:
                    if any(h.get("skill") == name for h in snapshot.attempt_history):
                        continue

                    inputs = self._infer_repair_inputs(
                        contract, verdict, snapshot,
                    )
                    if contract.inputs and not inputs:
                        continue

                    assumptions = [Assumption(
                        type="enabler_repair",
                        params={
                            "target_guard": guard_val,
                            "enabler": name,
                            "produces": produced,
                        },
                        guard_mapping=guard_val,
                        invert=False,
                    )]
                    candidates.append((name, inputs, assumptions))
                    break  # One match per skill

        return candidates[:3]  # Bounded

    def _guard_is_unmet(self, guard_type, verdict, snapshot):
        """Step 4: Returns True (unmet), False (met), or None (unknown).

        Unknown → triggers AmbiguityDecision, not blind action.
        """
        from execution.supervisor import GuardType

        # Use timeline from snapshot if available for state evaluation
        if hasattr(snapshot, 'world') and snapshot.world:
            state = getattr(snapshot.world, 'state', None) if hasattr(snapshot.world, 'state') else None

            if guard_type == GuardType.MEDIA_SESSION_ACTIVE:
                if state:
                    media = getattr(state, 'media', None)
                    if media is not None:
                        has_session = bool(media.platform or media.title)
                        return not has_session
                return None  # Unknown

            if guard_type == GuardType.APP_RUNNING:
                if state:
                    system = getattr(state, 'system', None)
                    session = getattr(system, 'session', None) if system else None
                    if session:
                        tracked = getattr(session, 'tracked_apps', {})
                        entity = verdict.context.get("entity", "")
                        if entity and entity in tracked:
                            return not tracked[entity].running
                        open_apps = getattr(session, 'open_apps', [])
                        return len(open_apps) == 0
                return None

            if guard_type == GuardType.FILE_EXISTS:
                # Can't verify without execution — return None
                return None

        return None  # No state → unknown

    def _llm_diagnose(self, verdict, snapshot):
        """Step 2: LLM diagnostic constrained to known state types.

        Output MUST be a GuardType value from a closed set.
        Same pattern as mission_cortex: enumerate valid outputs, reject rest.
        Returns dict or None.
        """
        if self._llm is None:
            return None

        from execution.supervisor import GuardType

        valid_types = [gt.value for gt in GuardType
                       if gt != GuardType.REQUIRES_CONFIRMATION]
        world_facts = self._extract_world_facts(snapshot)

        prompt = f"""A system action failed. Diagnose.

ACTION: {verdict.skill_name}
FAILURE: {verdict.reason}
DETAILS: {verdict.context.get("error", "")}

WORLD STATE:
{world_facts}

Respond with JSON:
{{"state_type": "<{valid_types} or UNKNOWN>",
  "entity": "<specific thing, e.g. app name>",
  "cause": "<transient_error|missing_resource|permission_denied|already_satisfied|not_installed|unknown>"}}

RULES:
- state_type MUST be from the list or UNKNOWN
- cause determines next action (retry, repair, escalate, ask user)
- Do NOT invent values"""

        try:
            from cortex.json_extraction import extract_json_block
            raw = self._llm.complete(prompt, temperature=0.0)
            result = extract_json_block(raw)
            logger.info("[DECISION] LLM diagnosis: %s", result)
            return result
        except Exception as e:
            logger.warning("[DECISION] LLM diagnostic failed: %s", e)
            return None

    def _normalize_diagnosis(self, diagnosis):
        """Step 3: Map LLM output → GuardType. Returns GuardType or None.

        If 'UNKNOWN' or invalid → return None (triggers AmbiguityDecision).
        """
        from execution.supervisor import GuardType

        state_type = diagnosis.get("state_type", "")

        if state_type == "UNKNOWN" or not state_type:
            return None

        try:
            return GuardType(state_type)
        except ValueError:
            logger.info(
                "[DECISION] LLM output '%s' not valid GuardType — rejected",
                state_type,
            )
            return None

    def _simulate_and_rank(self, candidates, missing_guard, snapshot):
        """Step 6: Simulate → rank by state transition quality.

        Priority:
        1. DIRECT: candidate.produces contains missing_guard
        2. INDIRECT: simulation shows follow-up reaches goal
        3. DEAD-END: neither → discard
        """
        if not candidates:
            return candidates

        guard_val = (
            missing_guard.value
            if hasattr(missing_guard, 'value')
            else str(missing_guard)
        )
        ranked = []

        for skill, inputs, assumptions in candidates:
            sim = self._simulate(skill, snapshot)

            # DIRECT: this candidate produces the missing guard
            if guard_val and guard_val in sim.produced_guards:
                ranked.append((skill, inputs, assumptions, 2.0))
                continue

            # INDIRECT: follow-up can reach goal
            follow_ups = self._generate_follow_ups(sim, snapshot)
            if follow_ups:
                ranked.append((skill, inputs, assumptions, 1.0))
                continue

            # DEAD-END: no path → still include but penalise
            logger.info("[DECISION] %s: no direct path — weak candidate", skill)
            ranked.append((skill, inputs, assumptions, 0.5))

        ranked.sort(key=lambda x: -x[3])
        return [(s, i, a) for s, i, a, _ in ranked]

    def _infer_repair_inputs(self, contract, verdict, snapshot):
        """Infer inputs for a repair skill from context.

        Ranked inference:
        1. Last successful app for domain
        2. Currently running apps
        3. Tracked but stopped apps
        4. Original inputs from verdict
        """
        inputs = {}
        for key, stype in contract.inputs.items():
            if "app" in stype.lower() or "application" in stype.lower():
                app = self._infer_app_ranked(verdict, snapshot)
                if app:
                    inputs[key] = app
            elif key in verdict.context.get("original_inputs", {}):
                inputs[key] = verdict.context["original_inputs"][key]
        return inputs

    def _infer_app_ranked(self, verdict, snapshot):
        """Ranked app inference: domain history > running > tracked > None."""
        failed_domain = (verdict.skill_name or "").split(".")[0]

        # 1. Last successful app for domain
        for h in reversed(list(snapshot.attempt_history)):
            if (h.get("result") == "success"
                    and h.get("domain") == failed_domain
                    and h.get("app")):
                return h["app"]

        # 2-3. From world state
        if hasattr(snapshot, 'world') and snapshot.world:
            state = getattr(snapshot.world, 'state', None) if hasattr(snapshot.world, 'state') else None
            if state:
                system = getattr(state, 'system', None)
                session = getattr(system, 'session', None) if system else None
                if session:
                    # Running apps
                    open_apps = getattr(session, 'open_apps', [])
                    if open_apps:
                        return open_apps[0]
                    # Stopped but tracked
                    tracked = getattr(session, 'tracked_apps', {})
                    for app_id, app_state in tracked.items():
                        if not app_state.running:
                            return app_id

        return None

    def _extract_world_facts(self, snapshot):
        """Compress world state to decision-relevant facts for LLM prompt."""
        facts = []
        if hasattr(snapshot, 'world') and snapshot.world:
            state = getattr(snapshot.world, 'state', None) if hasattr(snapshot.world, 'state') else None
            if state:
                media = getattr(state, 'media', None)
                if media and (media.platform or media.title):
                    facts.append(
                        f"- Media: {media.platform}, "
                        f"playing={media.is_playing}"
                    )
                else:
                    facts.append("- No active media session")

                system = getattr(state, 'system', None)
                session = getattr(system, 'session', None) if system else None
                if session:
                    apps = getattr(session, 'open_apps', [])
                    if apps:
                        facts.append(f"- Running apps: {', '.join(apps)}")
                    tracked = getattr(session, 'tracked_apps', {})
                    stopped = [k for k, v in tracked.items()
                               if not v.running]
                    if stopped:
                        facts.append(
                            f"- Stopped (tracked): {', '.join(stopped)}"
                        )

        return "\n".join(facts) or "- No relevant state"

    # ─────────────────────────────────────────────────────────
    # Stage 2: Normalized scoring
    # ─────────────────────────────────────────────────────────

    def _generate_candidates(self, verdict, snapshot):
        """7-step effect-driven recovery. No static maps.

        Returns (candidates, strategy_source).
        candidates: list of (skill_name, inputs, assumptions)
        strategy_source: str identifying which tier produced candidates
        """
        # Recovery is fully contract-driven — no hardcoded fast path.
        # Flow: _check_requires → _find_creators → _find_revealers → _find_enablers

        # ── Step 1: Check requires (truth) ──
        unmet_guard = self._check_requires(verdict, snapshot)

        candidates = []
        if unmet_guard:
            # Guard CONFIRMED unmet → find creators (skip LLM)
            creators = self._find_creators(unmet_guard, verdict, snapshot)
            candidates.extend(creators)
            if candidates:
                candidates = self._simulate_and_rank(
                    candidates, unmet_guard, snapshot,
                )
                return candidates, "effect_requires"

        # ── Steps 2-4: LLM diagnosis ──
        # Runs when Step 1 didn't produce candidates:
        # either no guard was found, or guard found but no creators/revealers.
        # LLM may diagnose a DIFFERENT guard (e.g., file_reference vs file_exists).
        if not candidates and self._llm:
            diagnosis = self._llm_diagnose(verdict, snapshot)
            if diagnosis:
                guard_type = self._normalize_diagnosis(diagnosis)
                cause_str = diagnosis.get("cause", "unknown")

                # Cause-based routing
                if cause_str in ("not_installed", "permission_denied"):
                    return [], "llm_escalation"  # → EscalationDecision

                if guard_type:
                    confirmed = self._guard_is_unmet(
                        guard_type, verdict, snapshot,
                    )
                    if confirmed is True:
                        creators = self._find_creators(
                            guard_type, verdict, snapshot,
                        )
                        candidates.extend(creators)
                    elif confirmed is None:
                        pass  # Unknown → no candidates → Escalation
                    # False = LLM was wrong, skip

            if candidates:
                candidates = self._simulate_and_rank(
                    candidates, guard_type, snapshot,
                )
                return candidates, "effect_llm"

        return candidates, "none"

    @staticmethod
    def _normalize(value: float, lo: float, hi: float) -> float:
        """Normalize value to [-1.0, +1.0]. Stable when lo == hi."""
        if hi <= lo:
            return 0.0
        return max(-1.0, min(1.0, (value - lo) / (hi - lo) * 2 - 1))

    def _goal_distance(self, skill_name, snapshot) -> float:
        """How close does this action get us to the goal?

        Returns: -3.0 (direct outcome) to +3.0 (irrelevant).
        """
        from execution.cognitive_context import (
            ALWAYS_PROGRESSIVE, PREPARATORY_DOMAINS,
        )

        # Direct outcome match
        pending = set(snapshot.pending_outcomes)
        # Check if skill action matches any pending outcome
        # (e.g. skill "fs.read_file" action "read" matches outcome "read_file")
        skill_base = skill_name.split(".")[-1] if "." in skill_name else skill_name
        for outcome in pending:
            if skill_base in outcome or outcome in skill_base:
                return -3.0  # directly achieves outcome

        # Always-progressive infrastructure actions
        if skill_name in ALWAYS_PROGRESSIVE:
            return -1.0

        # Preparatory domain check
        skill_domain = skill_name.split(".")[0] if "." in skill_name else ""
        for target_domain, prep_domains in PREPARATORY_DOMAINS.items():
            if skill_domain in prep_domains:
                # This skill's domain is preparatory for a target domain
                # Check if any pending outcome is in the target domain
                for outcome in pending:
                    if target_domain in outcome:
                        return -1.5  # preparatory

        # Generic contribution check
        if self._contributes_to_outcome(skill_name, snapshot):
            return -1.0

        return 3.0  # irrelevant

    def _contributes_to_outcome(self, skill_name, snapshot) -> bool:
        """Does this action indirectly help achieve any pending outcome?"""
        from execution.cognitive_context import UNCERTAINTY_REDUCERS

        # Uncertainty reducers always contribute when uncertain
        if skill_name in UNCERTAINTY_REDUCERS:
            max_unc = max(snapshot.uncertainty.values()) if snapshot.uncertainty else 0
            if max_unc > 0.2:
                return True

        return False

    def _contradicts_commitment(self, skill_name, inputs, snapshot) -> bool:
        """Does this action contradict an active commitment?"""
        for key, commitment in snapshot.commitments.items():
            # If we committed to a file and this action searches for something else
            if key == "selected_file" and skill_name == "fs.search_file":
                committed_name = str(commitment.value).lower()
                search_name = str(inputs.get("name", "")).lower()
                if search_name and search_name not in committed_name:
                    return True
        return False

    def _score_normalized(self, skill_name, inputs, snapshot):
        """Normalized scoring (P1): 7 components, each ∈ [-1, +1].

        Returns (total_score, components_dict).
        Lower = better.
        """
        from execution.cognitive_context import (
            SCORING_WEIGHTS, COST_MAP, UNCERTAINTY_REDUCERS,
            EXPANSION_PROFILES,
        )

        # ── 1. Cost (from SkillContract.resource_cost if registry available) ──
        cost_raw = 1.0  # default = medium
        if self._registry is not None:
            try:
                skill = self._registry.get(skill_name)
                cost_raw = COST_MAP.get(skill.contract.resource_cost, 1.0)
            except Exception:
                pass
        cost_n = self._normalize(cost_raw, 0.2, 3.0)

        # ── 2. Distance ──
        dist_raw = self._goal_distance(skill_name, snapshot)
        dist_n = self._normalize(dist_raw, -3.0, 3.0)

        # ── 3. Uncertainty effect ──
        skill_domain = skill_name.split(".")[0] if "." in skill_name else "general"
        domain_unc = snapshot.uncertainty.get(skill_domain, 0.0)
        unc_raw = -domain_unc if skill_name in UNCERTAINTY_REDUCERS else 0.0
        unc_n = self._normalize(unc_raw, -1.0, 0.0)

        # ── 4. Exploration bonus ──
        expansion = EXPANSION_PROFILES.get(skill_name, 0.0)
        expl_raw = -expansion * domain_unc * 0.3
        expl_n = self._normalize(expl_raw, -1.5, 0.0)

        # ── 5. Attempt penalty ──
        prior = sum(
            1 for h in snapshot.attempt_history
            if h.get("skill") == skill_name
        )
        pen_raw = prior * 2.0
        pen_n = self._normalize(pen_raw, 0.0, 10.0)

        # ── 6. Commitment penalty ──
        commit_raw = 4.0 if self._contradicts_commitment(
            skill_name, inputs, snapshot
        ) else 0.0
        commit_n = self._normalize(commit_raw, 0.0, 4.0)

        # ── 7. Future: 2-step lookahead via contract type chains ──
        lookahead_data = self._score_with_lookahead(skill_name, snapshot)
        future_n = self._normalize(lookahead_data["expected_future"], -3.0, 3.0)

        components = {
            "cost": round(cost_n, 4),
            "distance": round(dist_n, 4),
            "uncertainty": round(unc_n, 4),
            "exploration": round(expl_n, 4),
            "penalty": round(pen_n, 4),
            "commitment": round(commit_n, 4),
            "future": round(future_n, 4),
        }

        total = sum(
            SCORING_WEIGHTS[k] * components[k] for k in components
        )

        return round(total, 4), components, lookahead_data

    def _build_explanation(
        self, skill_name, score, components, snapshot, rejected,
        lookahead_data=None,
    ):
        """Build DecisionExplanation with full component breakdown (O1)."""
        from execution.cognitive_context import (
            DecisionExplanation, SCORING_WEIGHTS,
        )

        explanation = DecisionExplanation(
            chosen_action=skill_name,
            final_score=score,
            components=components,
            weights=dict(SCORING_WEIGHTS),
            lookahead=lookahead_data or {},
            rejected=rejected,
            uncertainty_snapshot=dict(snapshot.uncertainty),
            active_commitments=list(snapshot.commitments.keys()),
        )

        # Log the decision for observability
        logger.info(
            "[DECISION] Chosen: %s [score: %.4f]", skill_name, score,
        )
        for comp_name, comp_val in components.items():
            weight = SCORING_WEIGHTS.get(comp_name, 0)
            logger.info(
                "  %s: %.4f × %.2f = %.4f",
                comp_name, comp_val, weight, comp_val * weight,
            )
        for rej in rejected:
            logger.info(
                "  Rejected: %s (score: %.4f, reason: %s)",
                rej["action"], rej["score"], rej["reason"],
            )

        return explanation

    @staticmethod
    def _rejection_reason(components) -> str:
        """Infer human-readable rejection reason from score components."""
        worst = max(components.items(), key=lambda x: x[1])
        reasons = {
            "cost": "higher cost",
            "distance": "less relevant to goal",
            "uncertainty": "doesn't reduce uncertainty",
            "exploration": "low exploration value",
            "penalty": "already attempted",
            "commitment": "contradicts commitment",
            "future": "poor follow-up potential",
        }
        return reasons.get(worst[0], f"high {worst[0]}")

    # ─────────────────────────────────────────────────────────
    # Stage 3: 2-step lookahead via SkillContract type chains
    # ─────────────────────────────────────────────────────────

    def _simulate(self, skill_name, snapshot):
        """Zero-cost state projection via SkillContract metadata.

        Returns SimulatedState representing the world after this
        skill succeeds. Uses contract outputs to infer produced
        semantic types. No actual execution.
        """
        from execution.cognitive_context import (
            SimulatedState, EXPANSION_PROFILES, UNCERTAINTY_REDUCERS,
        )

        produced_types = set()
        produced_guards = set()
        domain = skill_name.split(".")[0] if "." in skill_name else "general"

        # Derive produced types + guards from contract if registry available
        if self._registry is not None:
            try:
                skill = self._registry.get(skill_name)
                for out_key, out_type in skill.contract.outputs.items():
                    produced_types.add(out_type)
                for guard_val in skill.contract.produces:
                    produced_guards.add(guard_val)
            except Exception:
                pass

        # Fall back to skill-name inference
        if not produced_types:
            skill_base = skill_name.split(".")[-1] if "." in skill_name else skill_name
            produced_types.add(f"{domain}_{skill_base}_output")

        # Simulate uncertainty reduction
        new_uncertainty = dict(snapshot.uncertainty)
        if skill_name in UNCERTAINTY_REDUCERS:
            if domain in new_uncertainty:
                new_uncertainty[domain] = max(0.0, new_uncertainty[domain] - 0.3)

        # Simulate achieved outcomes
        achieved = set(snapshot.achieved_outcomes)
        for outcome in snapshot.pending_outcomes:
            skill_base = skill_name.split(".")[-1] if "." in skill_name else skill_name
            if skill_base in outcome or outcome in skill_base:
                achieved.add(outcome)

        pending = [o for o in snapshot.pending_outcomes if o not in achieved]

        expansion = EXPANSION_PROFILES.get(skill_name, 1.0)

        return SimulatedState(
            achieved_outcomes=achieved,
            produced_types=produced_types,
            pending_outcomes=pending,
            uncertainty=new_uncertainty,
            step_count=snapshot.step_count + 1,
            produced_guards=produced_guards,
            expansion_estimate=expansion,
            attempt_history=list(snapshot.attempt_history),
        )

    def _generate_follow_ups(self, sim_state, snapshot):
        """Find skills whose inputs match the produced output types.

        Registry-backed: checks SkillContract.inputs for type compatibility.
        Returns list of (skill_name, match_quality) sorted by quality.
        Limited to MAX_CANDIDATES.
        """
        from execution.cognitive_context import MAX_CANDIDATES

        if self._registry is None:
            return []

        follow_ups = []
        try:
            for skill_name, skill in self._registry.all():
                # Don't suggest ourselves as follow-up
                if not skill.contract.outputs:
                    continue

                # Check if any required input type matches a produced type
                match_count = 0
                total_inputs = len(skill.contract.inputs)
                if total_inputs == 0:
                    continue

                for in_key, in_type in skill.contract.inputs.items():
                    if in_type in sim_state.produced_types:
                        match_count += 1

                if match_count > 0:
                    quality = match_count / total_inputs
                    # Check if follow-up contributes to pending outcomes
                    skill_base = skill_name.split(".")[-1]
                    is_relevant = any(
                        skill_base in o or o in skill_base
                        for o in sim_state.pending_outcomes
                    )
                    if is_relevant:
                        quality += 1.0  # strong preference for relevant

                    follow_ups.append((skill_name, quality))
        except Exception:
            pass

        follow_ups.sort(key=lambda x: -x[1])
        return follow_ups[:MAX_CANDIDATES]

    def _estimate_success(self, skill_name, snapshot):
        """Derive success probability from current state.

        Factors:
        - Domain uncertainty (more uncertain = lower p)
        - Resource cost (higher cost = slightly lower p)
        - Prior failure count (more failures = lower p)

        Returns float in [0.1, 1.0].
        """
        from execution.cognitive_context import COST_MAP

        p = 1.0

        # Uncertainty penalty
        domain = skill_name.split(".")[0] if "." in skill_name else "general"
        domain_unc = snapshot.uncertainty.get(domain, 0.0)
        p -= domain_unc * 0.4  # high uncertainty → up to 40% reduction

        # Resource cost penalty
        cost_str = "low"
        if self._registry is not None:
            try:
                skill = self._registry.get(skill_name)
                cost_str = skill.contract.resource_cost
            except Exception:
                pass
        cost_val = COST_MAP.get(cost_str, 1.0)
        p -= (cost_val - 0.5) * 0.1  # high cost → up to 25% reduction

        # Prior failure penalty
        prior_fails = sum(
            1 for h in snapshot.attempt_history
            if h.get("skill") == skill_name and h.get("result") == "failed"
        )
        p -= prior_fails * 0.15

        return max(0.1, min(1.0, p))

    def _score_with_lookahead(self, skill_name, snapshot):
        """2-step expected-value scoring.

        1. Simulate: project state if skill succeeds
        2. Find follow-ups: skills whose inputs match projected outputs
        3. Score best follow-up
        4. Expected future = discount × p_success × score(best_follow_up)

        Returns dict with lookahead details for DecisionExplanation.
        """
        from execution.cognitive_context import LOOKAHEAD_DISCOUNT

        # Step 1: Simulate
        sim_state = self._simulate(skill_name, snapshot)

        # Step 2: Estimate success
        p_success = self._estimate_success(skill_name, snapshot)

        # Step 3: Find follow-ups
        follow_ups = self._generate_follow_ups(sim_state, snapshot)

        best_follow_up = None
        best_follow_score = 0.0

        if follow_ups:
            # Score the best follow-up using a lightweight snapshot
            best_fu_name, best_fu_quality = follow_ups[0]
            fu_dist = self._goal_distance(best_fu_name, snapshot)
            # Simplified score: just distance (full scoring would recurse)
            best_follow_score = -fu_dist  # negate: lower distance = better
            best_follow_up = best_fu_name

        # Step 4: Expected future value
        expected_future = (
            LOOKAHEAD_DISCOUNT * p_success * best_follow_score
        )

        return {
            "p_success": round(p_success, 4),
            "best_follow_up": best_follow_up,
            "expected_future": round(expected_future, 4),
            "candidates": len(follow_ups),
            "produced_types": sorted(sim_state.produced_types),
        }

    # ─────────────────────────────────────────────────────────
    # Stage 4: Intelligence layer — commitments, causal graph,
    #          goal versioning
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def create_commitment(
        execution_state,
        key: str,
        value,
        alternatives: list,
        confidence: float = 0.5,
        decision_id: str = "",
    ):
        """Create a commitment from a multi-option decision.

        Called by interpret_result() when 1-of-N selection is made.
        Tracks the alternatives for potential reconsideration.

        Args:
            execution_state: mutable ExecutionState
            key: commitment key (e.g. "selected_file")
            value: chosen value
            alternatives: rejected alternatives
            confidence: initial confidence [0,1]
            decision_id: the DecisionRecord.id that made this choice
        """
        from execution.cognitive_context import Commitment

        commitment = Commitment(
            key=key,
            value=value,
            alternatives=list(alternatives),
            confidence=confidence,
            source_decision_id=decision_id,
            created_at_step=execution_state.step_count,
        )
        execution_state.commitments[key] = commitment
        logger.info(
            "[COMMITMENT] Created: %s = %s (confidence: %.2f, "
            "alternatives: %d, decision: %s)",
            key, value, confidence, len(alternatives), decision_id,
        )

    @staticmethod
    def reconsider_commitment(execution_state, failed_decision_id: str):
        """Check if a downstream failure traces causally to a commitment.

        Uses the root_cache to find the root decision that caused the
        failure. If that root decision is a commitment's source, the
        commitment should be reconsidered.

        Returns:
            (commitment_key, commitment) if reconsideration needed,
            None otherwise.
        """
        root_id = execution_state.trace_root_cause(failed_decision_id)

        for key, commitment in execution_state.commitments.items():
            if commitment.source_decision_id == root_id:
                logger.info(
                    "[COMMITMENT] Reconsidering '%s': failure %s traces "
                    "to root %s which created this commitment",
                    key, failed_decision_id, root_id,
                )
                return key, commitment

        return None

    @staticmethod
    def record_decision_with_causal_link(
        execution_state,
        skill_name: str,
        inputs: dict,
        parent_decision_id: str = "",
        caused_by_node: str = "",
        strategy_source: str = "heuristic",
    ) -> str:
        """Record a decision in the causal graph.

        Returns the new decision ID for downstream linking.
        """
        from execution.cognitive_context import DecisionRecord
        import uuid

        decision_id = f"d_{uuid.uuid4().hex[:8]}"
        parent_ids = [parent_decision_id] if parent_decision_id else []

        record = DecisionRecord(
            id=decision_id,
            step=execution_state.step_count,
            action_skill=skill_name,
            action_inputs=inputs,
            strategy_source=strategy_source,
            parent_ids=parent_ids,
            caused_by=caused_by_node or None,
        )
        execution_state.record_decision(record)
        logger.info(
            "[CAUSAL] Recorded decision %s: %s (parents: %s, caused_by: %s)",
            decision_id, skill_name, parent_ids, caused_by_node,
        )
        return decision_id

    @staticmethod
    def invalidate_commitments_for_goal_change(
        execution_state,
        removed_outcomes: list,
    ) -> list:
        """Selectively invalidate commitments when goal changes.

        Only removes commitments whose keys reference outcomes that
        are no longer required. Preserves commitments for unchanged goals.

        Returns list of invalidated commitment keys.
        """
        invalidated = []
        for key in list(execution_state.commitments.keys()):
            commitment = execution_state.commitments[key]
            # Check if commitment's key or value references removed outcomes
            key_references_removed = any(
                outcome.lower() in key.lower()
                or outcome.lower() in str(commitment.value).lower()
                for outcome in removed_outcomes
            )
            if key_references_removed:
                del execution_state.commitments[key]
                invalidated.append(key)
                logger.info(
                    "[COMMITMENT] Invalidated '%s' due to goal change "
                    "(removed outcomes: %s)",
                    key, removed_outcomes,
                )

        return invalidated
