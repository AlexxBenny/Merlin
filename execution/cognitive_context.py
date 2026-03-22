# execution/cognitive_context.py

"""
CognitiveContext — Unified shared truth for all decision-making.

This is a DATA module. No decision logic, no control flow, no imports
from other execution modules.  Every intelligent component reads from
the SAME CognitiveContext snapshot.

Contract 6 enforcement:
  - This file contains ONLY data models + serialization helpers.
  - Any method that performs reasoning, scoring, or classification
    belongs in metacognition.py or supervisor.py — NOT here.

Design rules:
  - CognitiveContext is the mutable container (built once per mission).
  - DecisionSnapshot is the per-step frozen view (immutable).
  - ExecutionState tracks runtime-mutable execution progress.
  - All other models (Assumption, Commitment, etc.) are data records.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Budget & Safety Constants
# ─────────────────────────────────────────────────────────────

MAX_RECOVERY_DEPTH = 3       # Max nested recovery attempts
MAX_TOTAL_STEPS = 30         # Hard ceiling on execution steps
MAX_DYNAMIC_QUEUE = 10       # Max pending recovery nodes
MAX_CANDIDATES = 5           # Max candidates per lookahead level
MAX_TRACE = 50               # Decision trace LRU cap
LOOKAHEAD_DISCOUNT = 0.5     # Future actions weighted less


# ─────────────────────────────────────────────────────────────
# Scoring Weights (normalized, sum to 1.0)
# ─────────────────────────────────────────────────────────────

SCORING_WEIGHTS = {
    "cost":        0.15,
    "distance":    0.30,  # most important
    "uncertainty": 0.15,
    "exploration": 0.10,
    "penalty":     0.10,
    "commitment":  0.05,
    "future":      0.15,
}

# SkillContract.resource_cost → numeric cost
COST_MAP = {"low": 0.5, "medium": 1.5, "high": 3.0}

# Skills that reduce uncertainty when executed
UNCERTAINTY_REDUCERS = frozenset({
    "fs.search_file", "fs.list_directory",
    "system.list_apps", "system.get_now_playing",
    "email.search_email", "email.read_inbox",
})

# Skills always considered progressive (infrastructure actions)
ALWAYS_PROGRESSIVE = frozenset({
    "system.focus_app", "system.open_app",
})

# Estimated state expansion per skill
EXPANSION_PROFILES = {
    "fs.list_directory": 5.0,
    "fs.search_file": 2.0,
    "email.read_inbox": 3.0,
    "email.search_email": 2.0,
    "system.list_apps": 2.0,
}

# Domain relationship: which domains can be preparatory for which
PREPARATORY_DOMAINS = {
    "email": {"fs"},        # email may need file search/read for attachments
    "browser": {"system"},  # browser may need system.focus_app
    "media": {"system"},    # media may need system.open_app
    "system": set(),
    "fs": set(),
    "memory": set(),
    "reasoning": set(),
}


# ─────────────────────────────────────────────────────────────
# Uncertainty Events
# ─────────────────────────────────────────────────────────────

UNCERTAINTY_INCREASE = {
    "multiple_matches": 0.4,
    "unknown_entity": 0.3,
    "partial_failure": 0.2,
    "file_not_found": 0.2,
    "ambiguous_input": 0.5,
}

UNCERTAINTY_DECREASE = {
    "entity_resolved": 0.3,
    "file_found": 0.3,
    "user_clarified": 0.5,
    "outcome_achieved": 0.2,
}


# ─────────────────────────────────────────────────────────────
# Failure Classification (2-axis)
# ─────────────────────────────────────────────────────────────

class FailureCause(str, Enum):
    """Axis 1: WHY did it fail?"""
    MISSING_DATA = "missing_data"
    MISSING_STATE = "missing_state"
    INVALID_ASSUMPTION = "invalid_assumption"
    EXTERNAL_DEPENDENCY = "external_dependency"


class FailureScope(str, Enum):
    """Axis 2: HOW BIG is the fix?"""
    SINGLE_STEP = "single_step"
    MULTI_STEP = "multi_step"


# ─────────────────────────────────────────────────────────────
# Assumption — data-driven pre-execution validity check
# ─────────────────────────────────────────────────────────────

class Assumption(BaseModel):
    """Pre-execution validity check for dynamic recovery nodes.

    Maps to existing GuardType for evaluation via _evaluate_guard().
    Tracked in ExecutionState.node_assumptions (NOT on MissionNode,
    which is frozen IR).

    Examples:
        Assumption(type="file_not_in_index",
                   params={"name": "report"},
                   guard_mapping="file_exists",
                   invert=True)
        → Checks FILE_EXISTS guard.  invert=True means the assumption
          holds when the guard FAILS (file does NOT exist yet).
    """
    model_config = ConfigDict(extra="forbid")

    type: str
    params: Dict[str, Any] = Field(default_factory=dict)
    guard_mapping: Optional[str] = None
    invert: bool = True


# ─────────────────────────────────────────────────────────────
# Commitment — decision under ambiguity
# ─────────────────────────────────────────────────────────────

class Commitment(BaseModel):
    """Records a decision point where one option was selected from many.

    Created when interpret_result() resolves ambiguity (1 of N).
    Reconsidered when downstream failure traces back causally.
    """
    model_config = ConfigDict(extra="forbid")

    key: str
    value: Any
    alternatives: List[Any] = Field(default_factory=list)
    confidence: float = 0.5
    source_decision_id: Optional[str] = None
    created_at_step: int = 0


# ─────────────────────────────────────────────────────────────
# DecisionRecord — causal trace entry
# ─────────────────────────────────────────────────────────────

class DecisionRecord(BaseModel):
    """One entry in the causal decision graph.

    parent_ids: which prior decisions led to this one.
    caused_by:  which node failure triggered this recovery decision.
    """
    model_config = ConfigDict(extra="forbid")

    id: str
    step: int
    action_skill: str
    action_inputs: Dict[str, Any] = Field(default_factory=dict)
    strategy_source: str = "heuristic"     # "heuristic", "llm", "memory", "user"
    parent_ids: List[str] = Field(default_factory=list)
    caused_by: Optional[str] = None
    reasoning: Dict[str, Any] = Field(default_factory=dict)
    outcome: Optional[str] = None          # filled after execution


# ─────────────────────────────────────────────────────────────
# DecisionExplanation — observability layer (O1)
# ─────────────────────────────────────────────────────────────

class DecisionExplanation(BaseModel):
    """Full reasoning trace for a single decision.

    Logged always. Contains normalized score breakdown, lookahead
    details, and rejection reasons for non-chosen candidates.
    """
    model_config = ConfigDict(extra="forbid")

    chosen_action: str
    final_score: float

    # Normalized score breakdown (each ∈ [-1, +1])
    components: Dict[str, float] = Field(default_factory=dict)
    weights: Dict[str, float] = Field(default_factory=dict)

    # Lookahead details
    lookahead: Dict[str, Any] = Field(default_factory=dict)

    # Why others were rejected
    rejected: List[Dict[str, Any]] = Field(default_factory=list)

    # State at decision time
    uncertainty_snapshot: Dict[str, float] = Field(default_factory=dict)
    active_commitments: List[str] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────
# EscalationDecision / ActionDecision
# ─────────────────────────────────────────────────────────────

class EscalationLevel(str, Enum):
    """Explicit escalation target. No implicit tier switching."""
    LOCAL = "local"    # Tier 1-2 (DecisionEngine handles)
    GLOBAL = "global"  # Tier 3 (cortex.compile recovery)
    USER = "user"      # ASK_USER


@dataclass(frozen=True)
class ActionDecision:
    """DecisionEngine chose a recovery action."""
    skill: str
    inputs: Dict[str, Any]
    assumptions: List[Assumption]
    score: float
    explanation: DecisionExplanation
    strategy_source: str = "heuristic"


@dataclass(frozen=True)
class EscalationDecision:
    """DecisionEngine cannot recover locally — escalate."""
    level: EscalationLevel
    reason: str
    verdict: Any = None   # FailureVerdict from MetaCognition
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AmbiguityDecision:
    """DecisionEngine found multiple viable options — ask user to choose.

    Triggered when:
    - Multiple candidates with close scores (low decisiveness gap)
    - LLM diagnosis returned UNKNOWN / unmappable GuardType
    - Guard evaluation returned None (unknown state)
    """
    choices: List[Dict[str, Any]]   # [{skill, inputs, score, reason}]
    question: str                    # User-facing question
    verdict: Any = None              # FailureVerdict


# ─────────────────────────────────────────────────────────────
# GoalState — tracks mission progress + versioning
# ─────────────────────────────────────────────────────────────

class GoalState(BaseModel):
    """Persistent goal tracking with versioning.

    Tracks required vs achieved outcomes, supports incremental
    refinement when the user clarifies mid-mission.
    """
    model_config = ConfigDict(extra="forbid")

    original_query: str = ""
    required_outcomes: List[str] = Field(default_factory=list)
    achieved_outcomes: List[str] = Field(default_factory=list)
    version: int = 1
    refinement_history: List[Dict[str, Any]] = Field(default_factory=list)

    def refine(self, new_query: str, new_outcomes: List[str]) -> List[str]:
        """Incremental goal refinement. Returns removed outcomes."""
        old_set = set(self.required_outcomes)
        new_set = set(new_outcomes)

        added = new_set - old_set
        removed = old_set - new_set
        unchanged = old_set & new_set

        self.refinement_history.append({
            "version": self.version,
            "old_query": self.original_query,
            "new_query": new_query,
            "added": sorted(added),
            "removed": sorted(removed),
        })

        self.version += 1
        self.original_query = new_query
        self.required_outcomes = sorted(new_set)
        self.achieved_outcomes = [
            o for o in self.achieved_outcomes if o in unchanged
        ]
        return sorted(removed)

    @property
    def is_complete(self) -> bool:
        return all(
            o in self.achieved_outcomes for o in self.required_outcomes
        )

    @property
    def pending_outcomes(self) -> List[str]:
        return [
            o for o in self.required_outcomes
            if o not in self.achieved_outcomes
        ]


# ─────────────────────────────────────────────────────────────
# ExecutionState — mutable runtime state
# ─────────────────────────────────────────────────────────────

class ExecutionState:
    """Mutable execution state. Lives outside the frozen IR.

    Tracks everything that changes during execution:
    - Step counter and recovery depth
    - Domain-keyed uncertainty
    - Node assumptions for dynamic recovery nodes
    - Commitments (decisions under ambiguity)
    - Decision trace with causal graph + compaction
    - Attempt history for deduplication
    """

    def __init__(self):
        self.step_count: int = 0
        self.recovery_depth: int = 0

        # Domain-keyed uncertainty (C2)
        self.uncertainty: Dict[str, float] = {
            "fs": 0.0,
            "email": 0.0,
            "system": 0.0,
            "browser": 0.0,
            "general": 0.0,
        }

        # Assumptions for dynamic recovery nodes (I2, outside frozen IR)
        self.node_assumptions: Dict[str, List[Assumption]] = {}

        # Commitments (R3)
        self.commitments: Dict[str, Commitment] = {}

        # Causal decision trace (C1) with compaction (P2)
        self.decision_trace: List[DecisionRecord] = []
        self.decision_summaries: List[Dict[str, Any]] = []
        self.root_cache: Dict[str, str] = {}

        # Attempt history for deduplication
        self.attempt_history: List[Dict[str, Any]] = []

        # Dynamic recovery queue
        self.dynamic_queue: List[Any] = []  # List[MissionNode]

        # FileRef registry — stores refs produced by search/list skills.
        # Scoped to mission duration (same as commitments, attempt_history).
        # Skills register refs here; DecisionEngine looks them up by ID.
        self.file_refs: Dict[str, Any] = {}  # ref_id → FileRef

        # Inline recovery tracking (per-node)
        self.inline_recovery_count: Dict[str, int] = {}   # node_id → attempt count
        self.inline_recovery_seen: Dict[str, Set[str]] = {}  # node_id → set of "skill:inputs_hash"

    def update_uncertainty(self, event_type: str, domain: str = "general"):
        """Domain-scoped uncertainty update."""
        if domain not in self.uncertainty:
            domain = "general"
        if event_type in UNCERTAINTY_INCREASE:
            delta = UNCERTAINTY_INCREASE[event_type]
        elif event_type in UNCERTAINTY_DECREASE:
            delta = -UNCERTAINTY_DECREASE[event_type]
        else:
            return
        self.uncertainty[domain] = max(0.0, min(1.0,
            self.uncertainty[domain] + delta))

    def record_decision(self, record: DecisionRecord):
        """Append decision with LRU eviction + root cache."""
        self.decision_trace.append(record)

        # O(1) root cache
        if record.parent_ids:
            parent_root = self.root_cache.get(record.parent_ids[0])
            self.root_cache[record.id] = parent_root or record.parent_ids[0]
        else:
            self.root_cache[record.id] = record.id

        # Compact when over limit (P2)
        if len(self.decision_trace) > MAX_TRACE:
            evicted = self.decision_trace[:10]
            self.decision_trace = self.decision_trace[10:]
            pattern = "→".join(
                r.action_skill.split(".")[-1] for r in evicted
            )
            successes = sum(1 for r in evicted if r.outcome == "success")
            self.decision_summaries.append({
                "pattern": pattern,
                "count": len(evicted),
                "successes": successes,
                "step_range": (evicted[0].step, evicted[-1].step),
            })

    def trace_root_cause(self, decision_id: str) -> str:
        """O(1) root cause lookup via cache."""
        return self.root_cache.get(decision_id, decision_id)

    def record_attempt(self, skill: str, inputs: Dict[str, Any],
                       result: str, error: str = ""):
        """Record an attempt for deduplication / penalty scoring."""
        self.attempt_history.append({
            "skill": skill,
            "inputs": inputs,
            "result": result,
            "error": error,
            "step": self.step_count,
        })

    @property
    def within_budget(self) -> bool:
        return (
            self.step_count < MAX_TOTAL_STEPS
            and self.recovery_depth < MAX_RECOVERY_DEPTH
            and len(self.dynamic_queue) < MAX_DYNAMIC_QUEUE
        )


# ─────────────────────────────────────────────────────────────
# SimulatedState — lightweight projected state for lookahead
# ─────────────────────────────────────────────────────────────

@dataclass
class SimulatedState:
    """Zero-cost projected state. No execution — contract metadata only."""
    achieved_outcomes: Set[str]
    produced_types: Set[str]
    pending_outcomes: List[str]
    uncertainty: Dict[str, float]
    step_count: int
    produced_guards: Set[str] = field(default_factory=set)  # GuardType values from contract.produces
    expansion_estimate: float = 0.0
    attempt_history: List[Dict[str, Any]] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────
# DecisionSnapshot — frozen per-step view (Contract 2)
# ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DecisionSnapshot:
    """Immutable state snapshot for one decision step.

    Created by CognitiveContext.snapshot_for_decision().
    All decision logic reads THIS, never the mutable context.
    """
    goal_original_query: str
    required_outcomes: Tuple[str, ...]
    achieved_outcomes: FrozenSet[str]
    pending_outcomes: Tuple[str, ...]

    step_count: int
    recovery_depth: int
    uncertainty: Dict[str, float]
    attempt_history: Tuple[Dict[str, Any], ...]
    commitments: Dict[str, Commitment]

    within_budget: bool


# ─────────────────────────────────────────────────────────────
# CognitiveContext — the unified shared truth object
# ─────────────────────────────────────────────────────────────

class CognitiveContext:
    """Single source of truth for all decision-making components.

    NOT a controller. Not intelligence. Just data.

    Built once per mission by the orchestrator. Read by:
    - DecisionEngine (via snapshot)
    - ContextProvider.build_from(ctx)
    - FilteredWSP.filter(ctx.world)
    - Tier 3 recovery (via to_failure_context())

    Contract 4: refresh_world() returns a NEW CognitiveContext.
    Contract 6: data + serialization only.
    """

    def __init__(
        self,
        goal: GoalState,
        execution: ExecutionState,
        world: Any = None,      # WorldSnapshot
        conversation: Any = None,  # ConversationFrame
    ):
        self.goal = goal
        self.execution = execution
        self.world = world
        self.conversation = conversation

    def snapshot_for_decision(self) -> DecisionSnapshot:
        """Freeze current state into an immutable snapshot (Contract 2)."""
        return DecisionSnapshot(
            goal_original_query=self.goal.original_query,
            required_outcomes=tuple(self.goal.required_outcomes),
            achieved_outcomes=frozenset(self.goal.achieved_outcomes),
            pending_outcomes=tuple(self.goal.pending_outcomes),
            step_count=self.execution.step_count,
            recovery_depth=self.execution.recovery_depth,
            uncertainty=dict(self.execution.uncertainty),
            attempt_history=tuple(self.execution.attempt_history),
            commitments=dict(self.execution.commitments),
            within_budget=self.execution.within_budget,
        )

    def refresh_world(self, new_world) -> "CognitiveContext":
        """Return a NEW CognitiveContext with updated world (Contract 4).

        Caller MUST rebind: ctx = ctx.refresh_world(new_snapshot)
        """
        return CognitiveContext(
            goal=self.goal,
            execution=self.execution,
            world=new_world,
            conversation=self.conversation,
        )

    def to_failure_context(self) -> Dict[str, Any]:
        """Serialize for Tier 3 recovery (Contract 5).

        Includes attempt history so cortex.compile() doesn't repeat
        failed strategies.
        """
        return {
            "goal": self.goal.original_query,
            "required_outcomes": self.goal.required_outcomes,
            "achieved_outcomes": self.goal.achieved_outcomes,
            "pending_outcomes": self.goal.pending_outcomes,
            "step_count": self.execution.step_count,
            "recovery_depth": self.execution.recovery_depth,
            "uncertainty": dict(self.execution.uncertainty),
            "attempt_history": list(self.execution.attempt_history),
            "commitments": {
                k: v.model_dump() for k, v in self.execution.commitments.items()
            },
            "decision_summaries": list(self.execution.decision_summaries),
        }
