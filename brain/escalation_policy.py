# brain/escalation_policy.py

"""
EscalationPolicy — The Triage Nurse.

This module runs ONLY after BrainCore routes to MISSION.
It decides WHAT KIND of cognition is needed, not WHETHER cognition
is needed (that's BrainCore's job).

Design rules:
- Deterministic — no LLM, no NLP, no semantic parsing
- Context-aware — checks ConversationFrame and WorldSnapshot
- Config-driven — referential markers from config/routing.yaml
- Never reasons about intent or selects skills

Decisions:
- MISSION  → full cortex compilation
- CLARIFY  → ask user for more context
- IGNORE   → silence (rare for user input)

Cognitive Tier Classification (Phase 5A):
- SIMPLE         → Tier 1: single LLM compile
- MULTI_INTENT   → Tier 2: decompose + compile + coverage verify
- HIERARCHICAL   → Tier 3: future (GoalGraph)
"""

import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Set, runtime_checkable

from world.snapshot import WorldSnapshot
from conversation.frame import ConversationFrame
from world.timeline import WorldEvent


logger = logging.getLogger(__name__)


class EscalationDecision(str, Enum):
    IGNORE = "ignore"
    REFLEX = "reflex"
    MISSION = "mission"
    CLARIFY = "clarify"


# ───────────────────────────────────────────────────────────────
# Cognitive Tier — how deep the compilation pipeline runs
# ───────────────────────────────────────────────────────────────

class CognitiveTier(str, Enum):
    """Cognitive depth tier for mission compilation.

    SIMPLE:       Tier 1 — single LLM compile (existing behavior).
    MULTI_INTENT: Tier 2 — decompose → compile with checklist → coverage verify.
    HIERARCHICAL: Tier 3 — future (GoalGraph decomposition). Gate only in 5A.
    """
    SIMPLE = "simple"
    MULTI_INTENT = "multi"
    HIERARCHICAL = "hierarchical"


# ───────────────────────────────────────────────────────────────
# TierClassifier — replaceable seam for tier classification
# ───────────────────────────────────────────────────────────────

@runtime_checkable
class TierClassifier(Protocol):
    """Protocol for cognitive tier classification.

    Replaceable seam: swap with LLM-based, embedding-based,
    config-driven, or any other classification strategy.
    """
    def classify(self, user_text: str) -> CognitiveTier: ...


# Conjunction words that signal multi-action queries
_CONJUNCTIONS: Set[str] = {
    "and", "then", "also", "while", "plus", "after", "before",
    "next", "additionally", "moreover",
}


class HeuristicTierClassifier:
    """Default TierClassifier — deterministic, O(query_length) dictionary lookups.

    Builds a keyword→domain index from SkillRegistry contracts at init time:
    - Skill name tokens (e.g., "create", "folder" from "fs.create_folder")
    - Input semantic type tokens (e.g., "brightness" from "brightness_percentage")
    - Description tokens (e.g., "playback" from "Resume or start media playback")

    Classification logic:
    1. Map query tokens to domains via keyword index
    2. If distinct_domains >= 2 → MULTI_INTENT (cross-domain)
    3. If verb_count >= 2 AND conjunctions >= 1 → MULTI_INTENT (within-domain)
    4. If conjunctions >= 4 OR verb_count >= 4 → HIERARCHICAL
    5. Else → SIMPLE

    Scales to 500+ skills: index built once, classification is pure dict lookups.
    """

    def __init__(self, registry: Any = None):
        """Build keyword→domain index from SkillRegistry.

        Args:
            registry: SkillRegistry instance. If None, classifier always
                      returns SIMPLE (safe degradation for testing).
        """
        self._keyword_index: Dict[str, str] = {}
        self._verb_set: Set[str] = set()

        if registry is not None:
            self._build_index(registry)

    def _build_index(self, registry: Any) -> None:
        """Build keyword→domain and verb set from contract metadata."""
        for name in registry.all_names():
            skill = registry.get(name)
            c = skill.contract
            domain = c.domain or name.split(".")[0]

            # Skill name tokens: "fs.create_folder" → {"create", "folder"}
            parts = name.split(".")
            if len(parts) >= 2:
                action_tokens = parts[1].split("_")
                for token in action_tokens:
                    if len(token) > 2:
                        self._keyword_index[token] = domain
                        self._verb_set.add(token)

            # Input semantic type tokens: "folder_name" → {"folder", "name"}
            all_inputs = {**c.inputs, **c.optional_inputs}
            for _key, stype in all_inputs.items():
                for token in stype.split("_"):
                    if len(token) > 2:
                        self._keyword_index[token] = domain

            # Input key names themselves: "app_name" → {"app"}
            for key in all_inputs:
                for token in key.split("_"):
                    if len(token) > 2:
                        self._keyword_index[token] = domain

            # Description tokens (>3 chars to skip articles)
            for token in c.description.lower().split():
                # Strip punctuation
                clean = token.strip("(),.:;!?")
                if len(clean) > 3:
                    self._keyword_index[clean] = domain

        logger.debug(
            "TierClassifier: built keyword index with %d entries, %d verbs, "
            "from %d skills",
            len(self._keyword_index), len(self._verb_set),
            len(registry.all_names()),
        )

    def classify(self, user_text: str) -> CognitiveTier:
        """Classify query cognitive depth. Deterministic, O(n) lookups."""
        if not self._keyword_index:
            # No registry → safe fallback
            return CognitiveTier.SIMPLE

        tokens = user_text.lower().split()

        # Map query tokens to domains
        matched_domains: Set[str] = set()
        verb_count = 0
        conjunction_count = 0

        for token in tokens:
            # Strip basic punctuation for matching
            clean = token.strip("(),.:;!?\"'")

            if clean in _CONJUNCTIONS:
                conjunction_count += 1

            if clean in self._keyword_index:
                matched_domains.add(self._keyword_index[clean])

            if clean in self._verb_set:
                verb_count += 1

        # Tier 3 gate: very complex queries
        if conjunction_count >= 4 or verb_count >= 4:
            return CognitiveTier.HIERARCHICAL

        # Tier 2: cross-domain dispersion
        if len(matched_domains) >= 2:
            return CognitiveTier.MULTI_INTENT

        # Tier 2: within-domain multi-intent
        if verb_count >= 2 and conjunction_count >= 1:
            return CognitiveTier.MULTI_INTENT

        return CognitiveTier.SIMPLE


class EscalationPolicy:
    """
    Deterministic escalation policy.

    This module decides WHAT KIND of cognition is needed,
    not WHETHER cognition is needed.

    Only called when BrainCore has already decided: MISSION.
    """

    def __init__(
        self,
        referential_markers: List[str] | None = None,
    ):
        self._referential_markers = [
            m.lower() for m in (referential_markers or [])
        ]

    # ─────────────────────────────────────────────────────────
    # Event-driven escalation
    # ─────────────────────────────────────────────────────────

    def decide_for_event(
        self,
        snapshot: WorldSnapshot,
        event: WorldEvent,
    ) -> EscalationDecision:
        """
        Escalation decision for world-driven events.

        Events that reach here have already passed BrainCore.
        """

        # Ads should never reach cortex
        if event.type == "ad_detected":
            return EscalationDecision.REFLEX

        # Critical system events → deterministic response (no LLM)
        if event.type in {"battery_low", "battery_critical", "memory_pressure"}:
            return EscalationDecision.REFLEX

        # Background informational events → log only
        if event.type in {
            "download_progress", "network_fluctuation",
            "cpu_high", "cpu_normal", "memory_normal",
            "foreground_window_changed", "idle_detected", "idle_ended",
            "time_tick", "hour_changed", "date_changed",
            "media_started", "media_stopped", "media_track_changed",
            "media_source_changed", "battery_charging",
            "app_launched", "app_closed", "app_focused",
            # Hardware actuation events (logged in execution.yaml, not escalated)
            "brightness_changed", "volume_changed",
            "mute_toggled", "nightlight_toggled",
        }:
            return EscalationDecision.IGNORE

        # Anything ambiguous escalates
        return EscalationDecision.MISSION

    # ─────────────────────────────────────────────────────────
    # User input escalation
    # ─────────────────────────────────────────────────────────

    def decide_for_user_input(
        self,
        user_text: str,
        snapshot: WorldSnapshot,
        frame: ConversationFrame,
    ) -> EscalationDecision:
        """
        Escalation decision for user input.

        This is ONLY called after BrainCore routes to MISSION.
        No keyword checks here — BrainCore already handled that.

        Focus: context-dependent decisions only.
        """

        text = user_text.lower()

        # ── Referential language: does context exist to resolve it? ──
        if self._has_referential_language(text):
            if self._can_resolve_reference(snapshot, frame):
                return EscalationDecision.MISSION   # Context exists, proceed
            return EscalationDecision.CLARIFY        # No context, ask user

        # ── Default: MISSION (safe bias) ──
        return EscalationDecision.MISSION

    # ─────────────────────────────────────────────────────────
    # Context helpers (deterministic, no LLM)
    # ─────────────────────────────────────────────────────────

    def _has_referential_language(self, text: str) -> bool:
        """
        Does the text contain words that refer to prior context?

        Examples: "the second video", "that one", "it"
        """
        return any(marker in text for marker in self._referential_markers)

    def _can_resolve_reference(
        self,
        snapshot: WorldSnapshot,
        frame: ConversationFrame,
    ) -> bool:
        """
        Is there enough context to resolve the reference?

        Checks:
        1. Are there visible lists in WorldState? (e.g., search results)
        2. Are there context frames in ConversationFrame? (e.g., prior domain)

        If either exists, assume context is sufficient.
        """

        # Check world state for visible lists
        if snapshot.state.visible_lists:
            return True

        # Check conversation frames for prior domain context
        if frame.context_frames:
            return True

        return False
