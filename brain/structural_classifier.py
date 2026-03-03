# brain/structural_classifier.py

"""
Structural Feature Analyzer — Deterministic Semantic Gating.

Detects structural linguistic features that indicate whether a query
can be handled by deterministic reflex skills or requires LLM reasoning.

This is NOT an LLM classifier. It is a deterministic pattern analyzer.
Latency: ~5ms per query. No model. No drift. CI-safe.

Design rules:
- Features are boolean flags, NOT single-label categories.
- Each feature dimension has its own token/pattern set.
- Context detection uses bigram context to avoid false positives
  (e.g., "is it" = grammar, "delete it" = coreference).
- clause_count is computed deterministically from conjunctions.
- All detection is O(n) over tokens. No LLM, no models.

Feature dimensions:
- requires_temporal: date/time arithmetic, projection, timezone
- requires_history: past state, trends, timeline data
- requires_computation: math, aggregation, comparison, semantic judgment
- requires_condition: if/when/trigger logic
- requires_context: references prior conversation or entities

Reflex eligibility rule:
    ALL flags False → reflex allowed
    ANY flag True → MISSION
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Set

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Query Features
# ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class QueryFeatures:
    """Structural feature vector for a user query.

    Each boolean flag represents a semantic dimension that
    reflex skills CANNOT handle. If ANY flag is True, the
    query must be escalated to MISSION.
    """
    requires_temporal: bool = False
    requires_history: bool = False
    requires_computation: bool = False
    requires_condition: bool = False
    requires_context: bool = False
    requires_scheduling: bool = False
    is_multi_clause: bool = False

    @property
    def reflex_eligible(self) -> bool:
        """Can this query be handled by a single reflex skill?"""
        return (
            not self.requires_temporal
            and not self.requires_history
            and not self.requires_computation
            and not self.requires_condition
            and not self.requires_context
            and not self.requires_scheduling
        )

    @property
    def intra_query_coreference(self) -> bool:
        """Does this query contain pronouns that refer to entities
        introduced in the SAME query, not in prior conversation?

        True when BOTH:
        - requires_context is True (coreferential pronoun detected)
        - is_multi_clause is True (query has multiple clauses)

        When multi-clause, pronouns like 'it' likely refer to an
        entity introduced in an earlier clause of the same query:
          'Create folder alex. Inside IT create man.'
        The compiler can resolve this without prior context.

        Single-clause with pronoun = conversational reference:
          'Delete it.' → refers to prior conversation
        """
        return self.requires_context and self.is_multi_clause

    def __repr__(self) -> str:
        flags = []
        if self.requires_temporal:
            flags.append("temporal")
        if self.requires_history:
            flags.append("history")
        if self.requires_computation:
            flags.append("computation")
        if self.requires_condition:
            flags.append("condition")
        if self.requires_context:
            flags.append("context")
        if self.requires_scheduling:
            flags.append("scheduling")
        if not flags:
            return "QueryFeatures(reflex_eligible)"
        label = f"QueryFeatures(requires=[{', '.join(flags)}])"
        if self.intra_query_coreference:
            label = label[:-1] + ", intra_query_coref)"
        return label


# ─────────────────────────────────────────────────────────────
# Feature Detection Token Sets
# ─────────────────────────────────────────────────────────────

# Temporal: date/time arithmetic, projection, timezone
# Detects queries that need more than reading current time.
_TEMPORAL_TOKENS: frozenset = frozenset({
    "tomorrow", "yesterday", "ago", "until", "since",
    "earlier", "later",
    # Duration
    "duration", "elapsed",
})

# Temporal bigrams: two-token sequences that signal temporal reasoning.
# These avoid false positives from single tokens like "last" or "next"
# which have non-temporal meanings (e.g., "last song", "next track").
_TEMPORAL_BIGRAMS: frozenset = frozenset({
    ("last", "hour"), ("last", "week"), ("last", "month"),
    ("last", "year"), ("last", "day"), ("last", "night"),
    ("next", "week"), ("next", "month"), ("next", "year"),
    ("next", "day"),
    ("how", "long"), ("how", "many"), ("how", "much"),
    ("time", "in"),     # "time in Tokyo"
    ("time", "zone"),
    ("in", "hours"), ("in", "minutes"), ("in", "seconds"),
    ("in", "days"), ("in", "weeks"), ("in", "months"),
})


# History: past state, trends, timeline data.
# Detects queries about what HAPPENED, not what IS.
_HISTORY_TOKENS: frozenset = frozenset({
    "was", "were", "been", "had",
    "dropped", "increased", "decreased", "changed",
    "rose", "fell", "spiked", "dipped",
    "trend", "trending", "history", "historical",
    "before",
})

_HISTORY_BIGRAMS: frozenset = frozenset({
    ("has", "been"), ("had", "been"), ("did", "it"),
    ("used", "to"),
})


# Computation: math, aggregation, comparison, semantic judgment.
# Detects queries needing reasoning beyond raw state reads.
_COMPUTATION_TOKENS: frozenset = frozenset({
    "average", "total", "count", "sum",
    "compare", "comparison", "versus", "vs",
    "than", "better", "worse", "more", "less", "fewer",
    "enough", "sufficient", "excessive",
    "fast", "slow", "high", "low",
    "stable", "unstable", "okay", "ok",
    "normal", "abnormal", "unusual",
    "percent", "percentage", "ratio",
    "maximum", "minimum", "max", "min",
})

_COMPUTATION_BIGRAMS: frozenset = frozenset({
    ("is", "it"),     # NOT context — "is it okay?" = seeking judgment
    # Captured here because "is it X?" is asking for evaluation,
    # which reflex cannot do. Pure state reads don't use "is it".
    # "what time is it" is handled by excluding "is it" when
    # preceded by a known query noun (time/day/date/battery).
    ("how", "much"), ("how", "many"), ("how", "far"),
    ("more", "than"), ("less", "than"),
    ("greater", "than"), ("fewer", "than"),
    ("compared", "to"),
})


# Condition: if/when/trigger/conditional logic.
# Detects commands that define dependent execution.
_CONDITION_TOKENS: frozenset = frozenset({
    "if", "unless", "whether",
})

_CONDITION_BIGRAMS: frozenset = frozenset({
    ("notify", "when"), ("tell", "when"), ("alert", "when"),
    ("notify", "if"), ("tell", "if"), ("alert", "if"),
    ("only", "if"), ("only", "when"),
    ("in", "case"),
})


# Scheduling: deferred, timed, or recurring execution.
# Distinct from temporal (date arithmetic) — this is ACTION TIMING.
# "after 10 seconds" = scheduling. "what day is tomorrow" = temporal.
#
# Safety bias: FALSE POSITIVES > FALSE NEGATIVES.
# If uncertain → force MISSION → coordinator classifies.
# Silent truncation (reflex dropping "after 10 seconds") is the
# worst failure mode. An unnecessary MISSION route is acceptable.
_SCHEDULING_TOKENS: frozenset = frozenset({
    "after",                              # "pause after 10 seconds"
    "wait", "delay", "delayed",            # "wait 5 seconds then pause"
    "schedule", "scheduled",              # "schedule a mute at 3pm"
    "timer", "timeout",                   # "set a timer for 10 minutes"
    "remind", "reminder",                 # "remind me to pause in 5 min"
    "recurring", "repeat", "every",       # "check battery every hour"
    "once",                               # "once it finishes, pause"
})

_SCHEDULING_BIGRAMS: frozenset = frozenset({
    # "in N seconds/minutes/hours" — deferred execution
    ("in", "seconds"), ("in", "minutes"), ("in", "hours"),
    # "at TIME" — scheduled execution
    ("at", "pm"), ("at", "am"),
    # "after N units" — delay
    ("after", "seconds"), ("after", "minutes"), ("after", "hours"),
    # "every N units" — recurring execution
    ("every", "minute"), ("every", "hour"), ("every", "second"),
    ("every", "day"), ("every", "week"), ("every", "month"),
    # "N seconds/minutes later" — deferred
    ("seconds", "later"), ("minutes", "later"), ("hours", "later"),
})

# Temporal pattern regex — CLASS detection, not token enumeration.
# Catches digit-adjacent time units (fused or separated), vague temporal
# phrases, and "at + digit" scheduling patterns.
#
# Purpose: REFLEX SAFETY GATING ONLY.
#   False positives → acceptable (mission path handles correctly).
#   False negatives → dangerous (silent semantic truncation).
#
# This covers gaps that bigram detection cannot:
#   "mute at 3pm"      → fused "3pm" not caught by ("at","pm") bigram
#   "mute at 3 pm"     → number between "at" and "pm" breaks adjacency
#   "pause music in a bit" → vague temporal not in any token set
_TEMPORAL_PATTERN: re.Pattern = re.compile(
    r'\d+\s*(?:seconds?|secs?|minutes?|mins?|hours?|hrs?|'
    r'pm|am|oclock)\b'
    r'|(?:in\s+(?:a\s+)?(?:bit|moment|while|sec|min))'
    r'|(?:at\s+\d)',
    re.IGNORECASE,
)


# Context: references to prior conversation or unspecified entities.
# The HARD problem: "it" in "what time is it" = grammar,
# but "it" in "delete it" = coreference.
#
# Solution: INVERTED pronoun detection.
# Instead of whitelisting action+pronoun pairs (infinite, unscalable),
# we BLACKLIST grammatical patterns (finite, closed set).
# ANY pronoun NOT in a grammatical pattern = coreference.
#
# Grammatical patterns are auxiliary/copula verbs + pronoun:
#   "is it", "was it", "does it", "did it", "will it"
# These are FINITE. Everything else is coreference:
#   "delete it", "inside it", "from it", "after it", etc.

# Pronouns that can be coreferential:
_PRONOUNS: frozenset = frozenset({
    "it", "this", "that", "them", "those", "these",
})

# These NEVER indicate context — grammatical dummy pronouns:
_GRAMMATICAL_PRONOUN_PATTERNS: frozenset = frozenset({
    # Copula/auxiliary + pronoun (questions like "what time is it")
    ("is", "it"), ("was", "it"), ("will", "it"),
    ("does", "it"), ("did", "it"), ("do", "it"),
    ("has", "it"), ("had", "it"), ("can", "it"),
    ("could", "it"), ("would", "it"), ("should", "it"),
    # "is this/that" in questions like "what is this"
    ("is", "this"), ("was", "this"), ("is", "that"),
    # "are these/those" in questions
    ("are", "these"), ("are", "those"),
    ("were", "these"), ("were", "those"),
})

# Non-pronoun context tokens — always signal context reference:
_CONTEXT_TOKENS: frozenset = frozenset({
    "again",
})

_CONTEXT_BIGRAMS: frozenset = frozenset({
    ("the", "previous"), ("the", "last"), ("the", "other"),
    ("the", "same"), ("the", "first"), ("the", "second"),
    ("that", "one"), ("this", "one"),
    ("do", "again"), ("try", "again"),
})


# Conjunction tokens: signal multi-clause structure.
# MUST disqualify from Stage 0 fast path because Stage 0 only
# tries try_match() (single skill). Conjunctions need Stage 1
# where try_match_multi() is attempted first.
_CONJUNCTION_TOKENS: frozenset = frozenset({
    "and", "then", "also", "plus", "additionally",
})

# Disqualifier tokens: if ANY present, Stage 0 fast path is skipped
# and the structural analyzer runs. These are broad — the analyzer
# then does precise classification per feature dimension.
_QUESTION_WORDS: frozenset = frozenset({
    "what", "how", "why", "when", "which", "where",
    "is", "are", "was", "did", "does", "can", "will",
    "do", "could", "should", "would",
})


# ─────────────────────────────────────────────────────────────
# Structural Analyzer
# ─────────────────────────────────────────────────────────────

class StructuralAnalyzer:
    """Deterministic structural feature analyzer.

    Detects whether a query contains structural elements that
    require reasoning beyond simple state reads.

    Architecture:
    - Each feature dimension has its own token/bigram sets.
    - Context detection uses bigram context to distinguish
      grammatical pronouns from coreferential pronouns.
    - All detection is O(n) over tokens. ~5ms per query.
    - No LLM. No model. No drift. Fully deterministic.
    - CI-testable. Explainable. Auditable.
    """

    def analyze(self, text: str) -> QueryFeatures:
        """Analyze a query and return structural features.

        Args:
            text: Normalized input text (lowercase, stripped).

        Returns:
            QueryFeatures with boolean flags for each dimension.
        """
        tokens = text.lower().split()
        token_set = frozenset(tokens)
        bigrams = self._extract_bigrams(tokens)

        features = QueryFeatures(
            requires_temporal=self._detect_temporal(token_set, bigrams),
            requires_history=self._detect_history(token_set, bigrams),
            requires_computation=self._detect_computation(
                token_set, bigrams, tokens,
            ),
            requires_condition=self._detect_condition(token_set, bigrams),
            requires_context=self._detect_context(token_set, bigrams, tokens),
            requires_scheduling=(
                self._detect_scheduling(token_set, bigrams)
                or self._detect_scheduling_pattern(text)
            ),
            is_multi_clause=self._detect_multi_clause(text, token_set),
        )

        logger.info(
            "StructuralAnalyzer: '%s' → %s",
            text[:50], features,
        )

        return features

    @staticmethod
    def has_disqualifier_tokens(text: str) -> bool:
        """Check if text contains tokens requiring structural analysis.

        Used by BrainCore Stage 0 to skip analyzer for trivial commands.
        If no disqualifiers → safe for deterministic single reflex.
        If any disqualifier → needs structural analysis (may be multi-reflex).

        CRITICAL: conjunctions MUST disqualify because Stage 0 only
        checks try_match() (single skill). Multi-command queries need
        Stage 1 where try_match_multi() is attempted first.
        """
        tokens = frozenset(text.lower().split())
        # Conjunctions — multi-clause structure needs multi-reflex path
        if tokens & _CONJUNCTION_TOKENS:
            return True
        # Question words, conditional tokens, or context tokens
        # indicate the query needs structural analysis
        if tokens & _QUESTION_WORDS:
            return True
        if tokens & _TEMPORAL_TOKENS:
            return True
        if tokens & _COMPUTATION_TOKENS:
            return True
        if tokens & _CONDITION_TOKENS:
            return True
        if tokens & _HISTORY_TOKENS:
            return True
        if tokens & _CONTEXT_TOKENS:
            return True
        if tokens & _SCHEDULING_TOKENS:
            return True
        # Temporal pattern regex — catches fused time units and
        # vague temporal phrases missed by token/bigram detection
        if _TEMPORAL_PATTERN.search(text):
            return True
        # Check for pronouns that MIGHT be coreferential
        if tokens & {"it", "this", "that", "them", "those", "these"}:
            return True
        return False

    # ── Internal feature detectors ──

    @staticmethod
    def _extract_bigrams(tokens: List[str]) -> Set[tuple]:
        """Extract all consecutive token pairs."""
        return {
            (tokens[i], tokens[i + 1])
            for i in range(len(tokens) - 1)
        } if len(tokens) >= 2 else set()

    @staticmethod
    def _detect_temporal(
        token_set: frozenset, bigrams: Set[tuple],
    ) -> bool:
        """Detect temporal shift / date arithmetic requirements."""
        if token_set & _TEMPORAL_TOKENS:
            return True
        if bigrams & _TEMPORAL_BIGRAMS:
            return True
        return False

    @staticmethod
    def _detect_history(
        token_set: frozenset, bigrams: Set[tuple],
    ) -> bool:
        """Detect past-state / trend / timeline requirements."""
        if token_set & _HISTORY_TOKENS:
            return True
        if bigrams & _HISTORY_BIGRAMS:
            return True
        return False

    @staticmethod
    def _detect_computation(
        token_set: frozenset, bigrams: Set[tuple],
        tokens: List[str],
    ) -> bool:
        """Detect aggregation / comparison / judgment requirements.

        Special handling for "is it X?" pattern:
        - "is it okay?" → computation (seeking judgment) → True
        - "what time is it" → grammar (not computation) → False

        Disambiguation rule: if "is it" is preceded by a known
        query noun (time, day, date, battery, cpu, etc.), it's
        grammatical. Otherwise, it's a judgment question.
        """
        if token_set & _COMPUTATION_TOKENS:
            return True
        if bigrams & _COMPUTATION_BIGRAMS:
            # Special case: "is it" may be grammatical
            if ("is", "it") in bigrams:
                return not _is_grammatical_is_it(tokens)
            return True
        return False

    @staticmethod
    def _detect_condition(
        token_set: frozenset, bigrams: Set[tuple],
    ) -> bool:
        """Detect conditional / trigger logic."""
        if token_set & _CONDITION_TOKENS:
            return True
        if bigrams & _CONDITION_BIGRAMS:
            return True
        return False

    @staticmethod
    def _detect_scheduling(
        token_set: frozenset, bigrams: Set[tuple],
    ) -> bool:
        """Detect deferred, scheduled, or recurring execution qualifiers.

        Distinct from temporal detection (date/time arithmetic).
        This catches action TIMING modifiers that reflex cannot handle.

        Safety bias: false positives > false negatives.
        Silent truncation (reflex ignoring scheduling qualifier)
        is worse than an unnecessary MISSION escalation.
        """
        if token_set & _SCHEDULING_TOKENS:
            return True
        if bigrams & _SCHEDULING_BIGRAMS:
            return True
        return False

    @staticmethod
    def _detect_scheduling_pattern(text: str) -> bool:
        """Regex-based temporal pattern detection.

        Complements token/bigram detection for cases where
        time expressions are fused ('3pm') or have intervening
        tokens ('at 3 pm').

        Called by analyze() to augment _detect_scheduling.
        """
        return bool(_TEMPORAL_PATTERN.search(text))

    @staticmethod
    def _detect_multi_clause(text: str, token_set: frozenset) -> bool:
        """Detect multi-clause structure in the query.

        Multi-clause queries contain multiple actions or clauses
        joined by conjunctions or sentence boundaries.

        Used to derive intra_query_coreference: if a query has
        coreferential pronouns AND multiple clauses, the pronoun
        likely refers to an entity in an earlier clause, not
        to prior conversation.

        Reuses _CONJUNCTION_TOKENS already defined for BrainCore
        Stage 1 disqualification. No new token sets needed.
        """
        # Conjunctions: "and", "then", "also", "plus", etc.
        if token_set & _CONJUNCTION_TOKENS:
            return True
        # Sentence-internal periods: "Create folder alex. Inside it..."
        # Exclude trailing period (not a clause boundary).
        stripped = text.rstrip(". ")
        if "." in stripped:
            return True
        # Comma-separated clauses: "create folder alex, inside it..."
        if "," in text:
            return True
        return False

    @staticmethod
    def _detect_context(
        token_set: frozenset, bigrams: Set[tuple],
        tokens: List[str],
    ) -> bool:
        """Detect context references (coreference to prior entities).

        Uses INVERTED pronoun detection:
        - Blacklist grammatical patterns (finite: "is it", "was it")
        - Everything else = coreference ("delete it", "inside it")

        This is scalable because grammatical patterns are a closed
        set (aux/copula verbs + pronoun), while coreferential
        patterns are infinite (any verb/preposition + pronoun).

        Examples:
        - "what time is it"  → ("is", "it") grammatical → False
        - "delete it"        → ("delete", "it") NOT grammatical → True
        - "inside it create" → ("inside", "it") NOT grammatical → True
        - "is it okay"       → ("is", "it") grammatical → False
        """
        # Non-pronoun context tokens
        if token_set & _CONTEXT_TOKENS:
            return True
        # Context bigrams (discourse markers)
        if bigrams & _CONTEXT_BIGRAMS:
            return True

        # Pronoun coreference detection (inverted logic)
        # If any pronoun appears in a non-grammatical context → coreference
        present_pronouns = token_set & _PRONOUNS
        if not present_pronouns:
            return False

        for i, token in enumerate(tokens):
            if token not in present_pronouns:
                continue
            if i == 0:
                # Pronoun as first word (e.g., "it broke") → coreference
                return True
            # Check if (preceding_word, pronoun) is grammatical
            prev = tokens[i - 1]
            if (prev, token) not in _GRAMMATICAL_PRONOUN_PATTERNS:
                # Not a known grammatical pattern → coreference
                return True

        return False


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

# Query nouns that precede "is it" in grammatical questions.
# "what time is it" → "time" before "is it" → grammatical.
_QUERY_NOUNS = frozenset({
    "time", "day", "date", "year", "month", "week",
    "hour", "minute", "second",
    "battery", "cpu", "ram", "disk", "volume", "brightness",
    "status", "weather", "temperature",
    "playing", "song", "track", "music",
})


def _is_grammatical_is_it(tokens: List[str]) -> bool:
    """Check if 'is it' in the token list is grammatical (not judgment).

    Rule: if a known query noun appears anywhere before 'is it',
    the 'is it' is part of a grammatical question like
    'what time is it' or 'what day is it'.

    If no query noun precedes it, 'is it' is likely a judgment
    question like 'is it okay?' or 'is it safe?'.
    """
    try:
        idx = tokens.index("is")
        if idx + 1 < len(tokens) and tokens[idx + 1] == "it":
            # Check if any query noun appears before "is it"
            preceding = frozenset(tokens[:idx])
            if preceding & _QUERY_NOUNS:
                return True
    except ValueError:
        pass
    return False
