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
    ) -> FailureVerdict:
        """Classify a node failure into a recovery action.

        Args:
            node_id: ID of the failed node
            skill_name: Name of the skill that failed
            failed_guards: Guard type values that failed (if any)
            error_message: Error string from execution
            node_status: Status string from executor
            retries_exhausted: Whether max retries have been exceeded

        Returns:
            FailureVerdict with category, action, reason, and context
        """
        failed_guards = failed_guards or []
        error_message = (error_message or "").lower()

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
