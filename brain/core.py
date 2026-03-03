# brain/core.py

"""
BrainCore — The Circuit Breaker.

This is NOT an intelligence module.
It answers ONE binary question:
    "Is this safe to handle reflexively, or do we need full cognition?"

Architecture: Three-Stage Hybrid Routing

    Stage 0 — Deterministic Fast Path (~50ms, no LLM)
        Reflex template match with no disqualifier tokens → REFLEX
        Covers: mute, play, volume 50, open chrome

    Stage 1 — Structural Feature Classifier (phi3:mini, ~200ms)
        Returns boolean feature vector (NOT single label):
        {temporal, history, computation, condition, context}
        Only invoked when Stage 0 can't decide.

    Stage 2 — Capability Gate
        All flags false + skill match → REFLEX
        Any flag true → MISSION

    Fallback — Deterministic Heuristics (if LLM unavailable)
        Relational gate + IntentMatcher (v1 behavior)

Routing invariant:
    False positives (escalating to MISSION unnecessarily) = SAFE
    False negatives (handling complex task as REFLEX) = CATASTROPHIC
    Therefore: default bias → MISSION
"""

import logging
from dataclasses import dataclass
from typing import Any, FrozenSet, List, Optional, Set, Tuple

from perception.normalize import normalize_for_matching


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Percept:
    modality: str       # "text", "speech", "vision"
    payload: str        # raw content
    confidence: float   # 1.0 for text, lower for speech/vision
    timestamp: float


class CognitiveRoute:
    REFLEX = "reflex"
    MULTI_REFLEX = "multi_reflex"
    MISSION = "mission"
    REFUSE = "refuse"


class BrainCore:
    """
    FINAL AUTHORITY ON ROUTING.

    Three-stage hybrid routing:
    - Stage 0: deterministic fast path (reflex match, no disqualifiers)
    - Stage 1: deterministic structural analyzer (~5ms, no LLM)
    - Stage 2: capability gate (feature flags → reflex or mission)
    - Fallback: heuristic routing (if analyzer not injected)
    """

    def __init__(
        self,
        mission_indicators: List[str] | None = None,
        refuse_indicators: List[str] | None = None,
        relational_indicators: List[str] | None = None,
        reflex_engine: Any | None = None,
        analyzer: Any | None = None,
    ):
        self._mission_indicators = [
            k.lower() for k in (mission_indicators or [])
        ]
        self._refuse_indicators = [
            k.lower() for k in (refuse_indicators or [])
        ]
        self._reflex_engine = reflex_engine
        self._analyzer = analyzer

        # ── Relational gate: tokenized matching (fallback only) ──
        self._relational_unigrams: FrozenSet[str] = frozenset()
        self._relational_bigrams: FrozenSet[Tuple[str, str]] = frozenset()
        if relational_indicators:
            unigrams: Set[str] = set()
            bigrams: Set[Tuple[str, str]] = set()
            for indicator in relational_indicators:
                parts = indicator.lower().split()
                if len(parts) == 1:
                    unigrams.add(parts[0])
                elif len(parts) == 2:
                    bigrams.add((parts[0], parts[1]))
                else:
                    bigrams.add((parts[0], parts[1]))
            self._relational_unigrams = frozenset(unigrams)
            self._relational_bigrams = frozenset(bigrams)

    # Last computed structural features — exposed for downstream layers
    # (e.g., EscalationPolicy) so they don't re-run linguistic heuristics.
    last_features: Any = None

    def _has_relational_dependency(self, text: str) -> bool:
        """Detect inter-clause semantic dependency via tokenized matching.

        Uses proper token-level matching (not substring).
        Used as FALLBACK when classifier is unavailable.
        """
        if not self._relational_unigrams and not self._relational_bigrams:
            return False

        tokens = text.lower().split()

        for token in tokens:
            if token in self._relational_unigrams:
                logger.debug(
                    "BrainCore: relational token '%s' detected → MISSION",
                    token,
                )
                return True

        if self._relational_bigrams and len(tokens) >= 2:
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                if pair in self._relational_bigrams:
                    logger.debug(
                        "BrainCore: relational bigram '%s %s' detected → MISSION",
                        pair[0], pair[1],
                    )
                    return True

        return False

    def route(self, percept: Percept) -> str:
        """
        Route a percept to the correct cognitive path.

        Three-stage hybrid routing:

        Stage 0 — Deterministic Fast Path:
            1. REFUSE — safety gate (always runs, cheap)
            2. Reflex match + no disqualifiers → REFLEX (skip analysis)

        Stage 1 — Structural Feature Analysis (deterministic, ~5ms):
            3. Analyze into feature vector (temporal, history, etc.)
            4. Capability gate: any flag true → MISSION

        Default → MISSION (safe bias)
        """
        text = normalize_for_matching(percept.payload)

        # ── Stage 0: Safety gate (always runs) ──
        if any(k in text for k in self._refuse_indicators):
            return CognitiveRoute.REFUSE

        # ── Stage 0: Deterministic fast path ──
        # If reflex match exists AND no disqualifier tokens → skip analysis.
        # Covers: mute, play, pause, volume 50, open chrome, etc.
        # These commands have no ambiguity — no need for feature analysis.
        if self._reflex_engine:
            from brain.structural_classifier import StructuralAnalyzer
            if not StructuralAnalyzer.has_disqualifier_tokens(text):
                # No question words, no temporal tokens, no pronouns
                # → safe for deterministic reflex
                if self._reflex_engine.try_match(text):
                    return CognitiveRoute.REFLEX

        # ── Stage 1: Structural Feature Analysis ──
        # Deterministic token/bigram analysis. ~5ms. No LLM.
        if self._analyzer:
            features = self._analyzer.analyze(text)
            self.last_features = features

            # Stage 2: Capability gate
            if features.reflex_eligible:
                # All flags false → safe for reflex
                # Try multi-reflex first (clause-aware split)
                if self._reflex_engine and self._reflex_engine.try_match_multi(text):
                    return CognitiveRoute.MULTI_REFLEX

                # If conjunctions present but multi-reflex failed:
                # User clearly intended multiple actions ("X and Y").
                # Single reflex would greedily match ONE skill and
                # silently drop the rest → lossy, incorrect.
                # Escalate to MISSION where LLM can decompose properly.
                from brain.structural_classifier import _CONJUNCTION_TOKENS
                if frozenset(text.split()) & _CONJUNCTION_TOKENS:
                    logger.info(
                        "BrainCore: conjunction detected but multi-reflex failed "
                        "for '%s' → MISSION",
                        text[:50],
                    )
                    return CognitiveRoute.MISSION

                # No conjunctions, single intent → single reflex
                if self._reflex_engine and self._reflex_engine.try_match(text):
                    return CognitiveRoute.REFLEX

            # Either a feature flag blocked reflex, or no unambiguous
            # IntentMatcher match was found despite reflex eligibility.
            logger.info(
                "BrainCore: no unambiguous reflex match for '%s' (features=%s)",
                text[:50], features,
            )
            return CognitiveRoute.MISSION

        # ── Fallback: heuristic routing (no analyzer) ──
        # Only reached in testing or misconfigured environments.
        if self._has_relational_dependency(text):
            return CognitiveRoute.MISSION

        if self._reflex_engine and self._reflex_engine.try_match_multi(text):
            return CognitiveRoute.MULTI_REFLEX

        if self._reflex_engine and self._reflex_engine.try_match(text):
            return CognitiveRoute.REFLEX

        if any(k in text for k in self._mission_indicators):
            return CognitiveRoute.MISSION

        return CognitiveRoute.MISSION


