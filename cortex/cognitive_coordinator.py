# cortex/cognitive_coordinator.py

"""
CognitiveCoordinator — Bounded reasoning engine.

Sits at the TOP of the MISSION pipeline, before tier classification.
Every non-reflex query passes through here first.

Responsibilities:
- Classify query mode (DIRECT_ANSWER | SKILL_PLAN | REASONED_PLAN | UNSUPPORTED)
- Produce direct answers for pure reasoning queries
- Compute intermediate variables for mixed reasoning+action
- Detect capability gaps before compilation is attempted

Phase 1: Single reasoning pass. No loops. No replanning.
Phase 2 seam: evaluate() for post-execution bounded replan.
Phase 3 seam: plan_iterative() for GoalGraph decomposition.

Architectural position: Cortex layer (§3.3a — Allowed: Reasoning).

Safety invariants:
- Cannot execute skills or mutate OS state.
- Maximum 1 LLM call per invocation.
- DIRECT_ANSWER is illegal if mutation intent detected.
- Ephemeral telemetry (battery, time, CPU, etc.) must use skills, not snapshot.
- UNSUPPORTED is capability-derived, not probabilistic.
- Fallback on any error → SKILL_PLAN (existing safe behavior).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from world.snapshot import WorldSnapshot
from models.base import LLMClient

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Coordinator output modes
# ─────────────────────────────────────────────────────────────

class CoordinatorMode(str, Enum):
    """Output mode from the CognitiveCoordinator.

    DIRECT_ANSWER:  Pure reasoning — final answer, no skills needed.
                    ILLEGAL if query contains mutation intent.
    SKILL_PLAN:     Pass-through — existing MissionCortex handles it.
    REASONED_PLAN:  Computed variables + refined query for compiler.
    UNSUPPORTED:    Required capabilities do not exist in skill manifest.
    PERSISTENT_JOB: Requires deferred/scheduled/recurring execution.
                    Phase 2 seam — not wired into merlin.py yet.
                    Coordinator does NOT create Task objects.
                    Task creation is SchedulerManager's responsibility (Phase 3).
    """
    DIRECT_ANSWER = "DIRECT_ANSWER"
    SKILL_PLAN = "SKILL_PLAN"
    REASONED_PLAN = "REASONED_PLAN"
    UNSUPPORTED = "UNSUPPORTED"
    PERSISTENT_JOB = "PERSISTENT_JOB"


@dataclass(frozen=True)
class CoordinatorResult:
    """Output of a CognitiveCoordinator invocation.

    Immutable. Auditable. Every field is explicit.

    Fields:
        mode:                 Which output path to take.
        answer:               Final user-facing text (DIRECT_ANSWER only).
        computed_vars:        Intermediate values for compiler (REASONED_PLAN).
        refined_query:        Rewritten query with resolved values (REASONED_PLAN).
        missing_capabilities: Skills/operations not found (UNSUPPORTED only).
        suggestion:           Helpful alternative suggestion (UNSUPPORTED only).
        reasoning_trace:      Human-readable trace of how the decision was made.
    """
    mode: CoordinatorMode
    answer: str = ""
    computed_vars: Dict[str, Any] = field(default_factory=dict)
    refined_query: str = ""
    missing_capabilities: List[str] = field(default_factory=list)
    suggestion: str = ""
    reasoning_trace: str = ""


# Deterministic fallback — used when coordinator fails or is unavailable
FALLBACK_RESULT = CoordinatorResult(
    mode=CoordinatorMode.SKILL_PLAN,
    reasoning_trace="Coordinator unavailable or failed — falling back to skill compilation",
)


# ─────────────────────────────────────────────────────────────
# Ephemeral domains — snapshot values are NOT authoritative
# ─────────────────────────────────────────────────────────────

EPHEMERAL_DOMAINS = frozenset({
    "battery", "brightness", "volume", "cpu", "ram", "disk",
    "memory", "media", "time", "clock", "network", "wifi",
    "bluetooth", "temperature",
})


# ─────────────────────────────────────────────────────────────
# Mutation verbs — if present, DIRECT_ANSWER is illegal
# ─────────────────────────────────────────────────────────────

MUTATION_VERBS = frozenset({
    "create", "delete", "remove", "rename", "move", "copy",
    "send", "open", "close", "set", "mute", "unmute",
    "play", "pause", "stop", "resume", "launch", "kill",
    "install", "uninstall", "download", "upload", "save",
    "write", "clear", "reset", "toggle", "enable", "disable",
    "increase", "decrease", "adjust", "switch",
})

# ─────────────────────────────────────────────────────────────
# Generation verbs — if present, DIRECT_ANSWER is illegal
# Content generation belongs in MISSION (as a skill), not DIRECT.
# ─────────────────────────────────────────────────────────────

GENERATION_VERBS = frozenset({
    "tell", "compose", "draft", "summarize", "describe",
    "explain", "generate", "narrate", "outline",
})


# ─────────────────────────────────────────────────────────────
# Protocol (seam for swappable implementations)
# ─────────────────────────────────────────────────────────────

@runtime_checkable
class CognitiveCoordinator(Protocol):
    """Protocol for cognitive coordination.

    Phase 1: process() only.
    Phase 2 seam: evaluate() for post-execution assessment.
    Phase 3 seam: plan_iterative() for goal-graph decomposition.


    Implementations must never execute skills or mutate OS state.
    """

    def process(
        self,
        query: str,
        snapshot: WorldSnapshot,
        skill_manifest: Dict[str, Any],
    ) -> CoordinatorResult:
        """Single-pass reasoning over query + world state.

        Must never raise. Returns FALLBACK_RESULT on any error.
        """
        ...

    # ── Phase 2 seam (not implemented in Phase 1) ──
    # def evaluate(
    #     self,
    #     query: str,
    #     plan: MissionPlan,
    #     execution_result: ExecutionResult,
    #     snapshot: WorldSnapshot,
    # ) -> EvaluationResult:
    #     """Post-execution assessment. Returns SUCCESS | REPLAN | FAIL."""
    #     ...


# ─────────────────────────────────────────────────────────────
# Phase 1 Implementation: LLM-backed coordinator
# ─────────────────────────────────────────────────────────────

class LLMCognitiveCoordinator:
    """LLM-backed cognitive coordinator. Phase 1: no loops.

    Single LLM call per invocation. Bounded, auditable, safe.
    """

    def __init__(self, llm: LLMClient):
        self._llm = llm

    def process(
        self,
        query: str,
        snapshot: WorldSnapshot,
        skill_manifest: Dict[str, Any],
    ) -> CoordinatorResult:
        """Single-pass reasoning. Never raises. Never loops."""
        try:
            prompt = self._build_prompt(query, snapshot, skill_manifest)
            raw = self._llm.complete(prompt)
            result = self._parse_response(raw, query)

            # ── Imperative guard: block DIRECT_ANSWER for mutation queries ──
            # Only applies to imperative statements, NOT interrogative queries.
            # "open the app" = imperative → override to SKILL_PLAN (safe)
            # "why can't you open it?" = interrogative → trust DIRECT_ANSWER
            if result.mode == CoordinatorMode.DIRECT_ANSWER:
                tokens = set(query.lower().split())
                first_word = query.lower().split()[0] if query.strip() else ""
                _QUESTION_WORDS = frozenset({
                    "why", "how", "what", "when", "where", "who", "which",
                    "can", "could", "would", "should", "do", "does", "did",
                    "is", "are", "was", "were", "will", "don't", "doesn't",
                    "isn't", "aren't", "won't", "wouldn't",
                })
                is_interrogative = first_word in _QUESTION_WORDS or "?" in query
                if tokens & MUTATION_VERBS and not is_interrogative:
                    logger.warning(
                        "[COORDINATOR] DIRECT_ANSWER blocked — mutation verb "
                        "detected in '%s'. Upgrading to SKILL_PLAN.",
                        query[:60],
                    )
                    return CoordinatorResult(
                        mode=CoordinatorMode.SKILL_PLAN,
                        reasoning_trace=(
                            f"Mutation intent detected ({tokens & MUTATION_VERBS}). "
                            "DIRECT_ANSWER overridden to SKILL_PLAN for safety."
                        ),
                    )

                # ── Generation guard: content generation → SKILL_PLAN ──
                # "tell me a story" / "compose a poem" must use
                # reasoning.generate_text skill, not DIRECT chat.
                # Rule: if a registered skill can handle it → MISSION.
                if result.mode == CoordinatorMode.DIRECT_ANSWER:
                    if tokens & GENERATION_VERBS and not is_interrogative:
                        logger.info(
                            "[COORDINATOR] DIRECT_ANSWER blocked — generation "
                            "verb detected in '%s'. Upgrading to SKILL_PLAN.",
                            query[:60],
                        )
                        return CoordinatorResult(
                            mode=CoordinatorMode.SKILL_PLAN,
                            reasoning_trace=(
                                f"Generation intent detected "
                                f"({tokens & GENERATION_VERBS}). "
                                "DIRECT_ANSWER overridden to SKILL_PLAN."
                            ),
                        )

            return result

        except Exception as e:
            logger.error(
                "CognitiveCoordinator failed for '%s': %s",
                query[:80], e, exc_info=True,
            )
            return FALLBACK_RESULT

    def _build_prompt(
        self,
        query: str,
        snapshot: WorldSnapshot,
        skill_manifest: Dict[str, Any],
    ) -> str:
        """Build constrained reasoning prompt with structured capability derivation."""

        world_facts = self._extract_world_facts(snapshot)

        skill_lines = "\n".join(
            f"  - {name}: {info.get('description', '?')}"
            for name, info in skill_manifest.items()
        )

        ephemeral_list = ", ".join(sorted(EPHEMERAL_DOMAINS))

        return f"""You are MERLIN's cognitive coordinator.

Your task: analyze the user's query and decide how to handle it.

═══════════════════════════════════════════════════
USER QUERY: "{query}"
═══════════════════════════════════════════════════

CURRENT WORLD STATE (snapshot — may be stale for ephemeral data):
{world_facts}

AVAILABLE SKILLS (these are the ONLY OS actions MERLIN can perform):
{skill_lines}

═══════════════════════════════════════════════════
DECISION PROCEDURE (follow these steps exactly):
═══════════════════════════════════════════════════

Step 1: List every operation the user's query requires.
Step 2: For each operation, check if there is a matching skill in the list above.
Step 3: Determine if reasoning/computation is needed (date math, arithmetic, logic, knowledge).
Step 4: Choose the correct mode based on this matrix:

  ┌─────────────────────────┬──────────────────┬───────────────────────┐
  │                         │ Skills sufficient │ Skills missing        │
  ├─────────────────────────┼──────────────────┼───────────────────────┤
  │ No reasoning needed     │ SKILL_PLAN       │ UNSUPPORTED           │
  │ Reasoning needed        │ REASONED_PLAN    │ UNSUPPORTED           │
  │ No skills needed at all │ DIRECT_ANSWER    │ DIRECT_ANSWER         │
  └─────────────────────────┴──────────────────┴───────────────────────┘

═══════════════════════════════════════════════════
HARD RULES (these override everything):
═══════════════════════════════════════════════════

1. MUTATION GUARD: If the query expresses ANY intent to change OS state
   (create, delete, open, close, set, mute, play, etc.) — even conditionally —
   you MUST choose SKILL_PLAN or REASONED_PLAN. NEVER DIRECT_ANSWER.

2. EPHEMERAL DATA: The following are ephemeral and snapshot values are NOT
   authoritative: {ephemeral_list}.
   If the user asks about these AND computation is needed, choose REASONED_PLAN
   and include a note that the skill must be called for fresh data.
   If the user just asks the current value (e.g. "what is the time"), choose SKILL_PLAN.

3. UNSUPPORTED must be CAPABILITY-DERIVED: If any required operation has NO
   corresponding skill in the list above, you MUST choose UNSUPPORTED.
   Do not guess. Do not hallucinate skills.

4. When UNCERTAIN, default to SKILL_PLAN. This is always safe.

═══════════════════════════════════════════════════
OUTPUT FORMAT (strict JSON, no markdown fences):
═══════════════════════════════════════════════════

For DIRECT_ANSWER (pure reasoning, knowledge, explanation — NO OS action):
{{"mode": "DIRECT_ANSWER", "answer": "concise answer", "reasoning": "how you derived it"}}

For SKILL_PLAN (skills exist, no reasoning needed):
{{"mode": "SKILL_PLAN", "reasoning": "skills cover this, no computation needed"}}

For REASONED_PLAN (reasoning first, then skills):
{{"mode": "REASONED_PLAN", "computed": {{"var_name": "value"}}, "refined_query": "rewritten query with literal values", "reasoning": "what you computed and why"}}

For UNSUPPORTED (required capabilities do not exist):
{{"mode": "UNSUPPORTED", "missing": ["capability1", "capability2"], "suggestion": "what the user could do instead", "reasoning": "which operations have no skill"}}
"""

    @staticmethod
    def _extract_world_facts(snapshot: WorldSnapshot) -> str:
        """Extract world facts for LLM reasoning — schema-driven, zero hardcoded fields.

        Uses Pydantic model_dump(exclude_none=True) so field names are always
        in sync with the actual WorldState model. If a field is added or renamed
        in state.py, this method automatically picks it up.
        """
        state_dict = snapshot.state.model_dump(exclude_none=True)
        if not state_dict:
            return "  (no state available)"

        lines = []
        LLMCognitiveCoordinator._flatten_dict(state_dict, lines, prefix="")
        return "\n".join(lines) if lines else "  (no state available)"

    @staticmethod
    def _flatten_dict(
        d: dict,
        lines: list,
        prefix: str,
        max_list_items: int = 10,
    ) -> None:
        """Recursively flatten a dict into human-readable '  key: value' lines.

        - Nested dicts are flattened with dot-separated keys.
        - Lists >max_list_items are summarized by count only (prevents prompt bloat).
        - Empty dicts/lists are skipped.
        """
        for key, value in d.items():
            full_key = f"{prefix}{key}" if prefix else key

            if isinstance(value, dict):
                if value:  # skip empty dicts
                    LLMCognitiveCoordinator._flatten_dict(
                        value, lines, prefix=f"{full_key}.", max_list_items=max_list_items,
                    )
            elif isinstance(value, list):
                if len(value) > max_list_items:
                    lines.append(f"  {full_key}: [{len(value)} items]")
                elif value:
                    lines.append(f"  {full_key}: {value}")
            else:
                lines.append(f"  {full_key}: {value}")

    def _parse_response(self, raw: str, query: str) -> CoordinatorResult:
        """Parse LLM JSON response into CoordinatorResult."""
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning(
                "[COORDINATOR] Invalid JSON for '%s': %s",
                query[:50], text[:200],
            )
            return FALLBACK_RESULT

        mode_str = data.get("mode", "")
        reasoning = data.get("reasoning", "")

        if mode_str == "DIRECT_ANSWER":
            answer = data.get("answer", "")
            if not answer:
                logger.warning("[COORDINATOR] DIRECT_ANSWER with empty answer")
                return FALLBACK_RESULT
            return CoordinatorResult(
                mode=CoordinatorMode.DIRECT_ANSWER,
                answer=answer,
                reasoning_trace=reasoning,
            )

        if mode_str == "SKILL_PLAN":
            return CoordinatorResult(
                mode=CoordinatorMode.SKILL_PLAN,
                reasoning_trace=reasoning,
            )

        if mode_str == "REASONED_PLAN":
            computed = data.get("computed", {})
            refined = data.get("refined_query", "")
            if not refined:
                logger.warning(
                    "[COORDINATOR] REASONED_PLAN without refined_query"
                )
                return FALLBACK_RESULT
            return CoordinatorResult(
                mode=CoordinatorMode.REASONED_PLAN,
                computed_vars=computed,
                refined_query=refined,
                reasoning_trace=reasoning,
            )

        if mode_str == "UNSUPPORTED":
            missing = data.get("missing", [])
            suggestion = data.get("suggestion", "")
            return CoordinatorResult(
                mode=CoordinatorMode.UNSUPPORTED,
                missing_capabilities=missing if isinstance(missing, list) else [str(missing)],
                suggestion=suggestion,
                reasoning_trace=reasoning,
            )

        logger.warning("[COORDINATOR] Unknown mode '%s'", mode_str)
        return FALLBACK_RESULT
