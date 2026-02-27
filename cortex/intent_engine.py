# cortex/intent_engine.py

"""
IntentEngine — Scored semantic intent matching. No regex.

Replaces regex-based reflex matching with:
1. Inverted verb/keyword indexes (built once at startup)
2. Synonym expansion (global dictionary)
3. Scored clause matching (verb + keyword + target_type)
4. Type-driven parameter extraction (via SemanticType aliases)

Scaling properties:
- O(types), not O(skills) — shared types share coercion
- O(K × M) per query — K clauses × M candidate skills per clause
- Inverted index reduces M to small candidate sets
- New skills auto-index via contract metadata — zero regex to maintain

Design rules:
- High-confidence matches only → low confidence escalates to LLM
- Top match must exceed THRESHOLD and have MARGIN over runner-up
- Intent matching is deterministic, fast, and testable
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

from cortex.synonyms import expand_token, VERB_SYNONYMS, NOUN_SYNONYMS
from cortex.semantic_types import SEMANTIC_TYPES
from perception.normalize import normalize_for_matching

if TYPE_CHECKING:
    from execution.registry import SkillRegistry

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Scoring constants
# ─────────────────────────────────────────────────────────────

VERB_WEIGHT = 3          # Verb match is strongest signal
KEYWORD_WEIGHT = 2       # Each keyword match
TARGET_WEIGHT = 2        # target_type match
PARAM_WEIGHT = 1         # Parameter-presence bonus (numeric or alias detected)
MATCH_THRESHOLD = 3      # Minimum score to accept a match
                         # Verb(3) alone is sufficient for self-identifying
                         # commands (mute, play, pause). Margin rule (1.5×)
                         # protects against ambiguity.
MARGIN_RATIO = 1.5       # Top score must be >= this × runner-up

# Verbs that are semantically generic — they describe an action class,
# not a specific operation. Require keyword disambiguation.
# "create X" could be folder/file/project/etc — needs keyword to resolve.
# "open chrome" is self-identifying — no keyword needed.
# Long-term: move to per-skill contract.verb_specificity metadata.
GENERIC_VERBS = frozenset({
    "create", "make", "new",           # Creation verbs
    "set", "adjust", "change",         # Configuration verbs
    "show", "get", "check",            # Query verbs
    "start",                           # Ambiguous (app? music? timer?)
})


# ─────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────

@dataclass
class IntentMatch:
    """Result of a successful intent match."""
    skill_name: str              # e.g. "system.set_volume"
    params: Dict[str, Any]       # extracted parameters
    score: float                 # match confidence score
    verb_matched: str = ""       # which verb matched
    keywords_matched: List[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────
# IntentIndex — inverted index built at startup
# ─────────────────────────────────────────────────────────────

class IntentIndex:
    """Inverted index from verbs/keywords to skill names.

    Built once from SkillRegistry at startup.
    Near-constant-time candidate lookup.
    """

    def __init__(self):
        # verb → set of skill names
        self.verb_index: Dict[str, Set[str]] = {}
        # keyword → set of skill names
        self.keyword_index: Dict[str, Set[str]] = {}
        # skill_name → contract (for scoring)
        self.contracts: Dict[str, Any] = {}

    def build(self, registry: "SkillRegistry") -> None:
        """Build inverted indexes from all registered skills."""
        for name in registry.all_names():
            skill = registry.get(name)
            c = skill.contract
            self.contracts[name] = c

            # Index declared intent verbs
            for verb in c.intent_verbs:
                v = verb.lower()
                self.verb_index.setdefault(v, set()).add(name)

            # Index declared intent keywords
            for kw in c.intent_keywords:
                k = kw.lower()
                self.keyword_index.setdefault(k, set()).add(name)

            # Index action tokens by semantic role:
            # First part is the verb (set, open, mute) → verb_index
            # Remaining parts are nouns (brightness, volume) → keyword_index
            action_parts = c.action.split("_")
            if action_parts and len(action_parts[0]) > 2:
                self.verb_index.setdefault(action_parts[0], set()).add(name)
            for token in action_parts[1:]:
                if len(token) > 2:
                    self.keyword_index.setdefault(token, set()).add(name)

            # Also index target_type
            if c.target_type:
                self.keyword_index.setdefault(c.target_type.lower(), set()).add(name)

        logger.info(
            "IntentIndex: %d verb entries, %d keyword entries, %d skills",
            len(self.verb_index), len(self.keyword_index), len(self.contracts),
        )


# ─────────────────────────────────────────────────────────────
# IntentMatcher — scored clause matching
# ─────────────────────────────────────────────────────────────

class IntentMatcher:
    """Score-based intent matching. No regex.

    Process per clause:
    1. Tokenize + normalize
    2. Expand synonyms
    3. Collect candidate skills from index
    4. Score each candidate
    5. Apply threshold + margin rules
    6. Extract parameters via type system
    """

    def __init__(self, index: IntentIndex, registry: "SkillRegistry"):
        self._index = index
        self._registry = registry

    def match_clause(self, clause: str) -> Optional[IntentMatch]:
        """Match a single clause to a skill. Returns None if no confident match."""
        # 1. Tokenize
        tokens = clause.lower().split()

        # 2. Expand synonyms
        expanded = [expand_token(t) for t in tokens]

        # 3. Collect candidates from index
        candidates: Set[str] = set()
        for token in expanded:
            if token in self._index.verb_index:
                candidates.update(self._index.verb_index[token])
            if token in self._index.keyword_index:
                candidates.update(self._index.keyword_index[token])

        if not candidates:
            logger.debug("IntentMatcher: no candidates for '%s'", clause)
            return None

        # 4. Score each candidate
        scores: List[tuple[str, float, str, List[str]]] = []
        for skill_name in candidates:
            contract = self._index.contracts[skill_name]
            score, verb_matched, kw_matched = self._score(
                expanded, tokens, contract,
            )
            if score > 0:
                scores.append((skill_name, score, verb_matched, kw_matched))

        if not scores:
            return None

        # Sort by score desc, then by intent_priority desc
        scores.sort(
            key=lambda x: (x[1], self._index.contracts[x[0]].intent_priority),
            reverse=True,
        )

        top_name, top_score, top_verb, top_kw = scores[0]

        # 5. Threshold check
        if top_score < MATCH_THRESHOLD:
            logger.debug(
                "IntentMatcher: '%s' → top score %.1f < threshold %d",
                clause, top_score, MATCH_THRESHOLD,
            )
            return None

        # 6. Margin check (if runner-up exists)
        if len(scores) > 1:
            runner_score = scores[1][1]
            if runner_score > 0 and top_score < runner_score * MARGIN_RATIO:
                logger.debug(
                    "IntentMatcher: '%s' → ambiguous: %s(%.1f) vs %s(%.1f)",
                    clause, top_name, top_score, scores[1][0], runner_score,
                )
                return None

        # 7. Generic verb gate (symmetric — applies to ALL skills)
        #    Generic verbs describe action classes, not specific operations.
        #    Without keyword disambiguation, intent is ambiguous.
        #    "create X" → folder? file? project? → need keyword.
        #    "open chrome" → self-identifying verb → no keyword needed.
        if top_verb in GENERIC_VERBS and not top_kw:
            logger.debug(
                "IntentMatcher: '%s' → generic verb '%s' without keyword → reject",
                clause, top_verb,
            )
            return None

        # 8. Extract parameters
        params = self._extract_params(top_name, tokens, expanded)

        # Extraction failure → reject match entirely.
        # This happens when escalation prepositions are detected
        # for skills with optional inputs — the intent is too
        # complex for deterministic extraction → escalate to MISSION.
        if params is None:
            logger.debug(
                "IntentMatcher: '%s' → %s extraction failed → reject",
                clause, top_name,
            )
            return None

        logger.info(
            "IntentMatcher: '%s' → %s (score=%.1f, verb=%s, kw=%s, params=%r)",
            clause, top_name, top_score, top_verb, top_kw, params,
        )

        return IntentMatch(
            skill_name=top_name,
            params=params,
            score=top_score,
            verb_matched=top_verb,
            keywords_matched=top_kw,
        )

    def _score(
        self,
        expanded_tokens: List[str],
        raw_tokens: List[str],
        contract: Any,
    ) -> tuple[float, str, List[str]]:
        """Score a single skill against clause tokens."""
        score = 0.0
        verb_matched = ""
        kw_matched: List[str] = []

        # Verb match — ONLY declared intent_verbs, not action tokens
        contract_verbs = {v.lower() for v in contract.intent_verbs}

        for token in expanded_tokens:
            if token in contract_verbs:
                score += VERB_WEIGHT
                verb_matched = token
                break  # Only count one verb match

        # Keyword match
        contract_keywords = {k.lower() for k in contract.intent_keywords}
        if contract.target_type:
            contract_keywords.add(contract.target_type.lower())

        for token in expanded_tokens:
            if token in contract_keywords:
                score += KEYWORD_WEIGHT
                kw_matched.append(token)

        # Deduplicate keyword scoring
        kw_matched = list(set(kw_matched))

        # Parameter-presence bonus: if skill has required inputs
        # and clause contains a numeric token or semantic alias,
        # give a small boost. This ensures "brightness 10" (kw+param=3)
        # passes threshold without allowing pure-noun matches.
        if contract.inputs and score > 0:
            for token in raw_tokens:
                # Numeric token
                try:
                    float(token)
                    score += PARAM_WEIGHT
                    break
                except ValueError:
                    pass
                # Semantic alias token (e.g. "full", "half", "dim")
                for sem_type_name in contract.inputs.values():
                    sem_type = SEMANTIC_TYPES.get(sem_type_name)
                    if sem_type and sem_type.aliases and token.lower() in sem_type.aliases:
                        score += PARAM_WEIGHT
                        break
                else:
                    continue
                break

        return score, verb_matched, kw_matched

    # ─────────────────────────────────────────────────────────────
    # Reflex extraction boundary constants
    # ─────────────────────────────────────────────────────────────

    # Structural words that introduce a parameter but aren't part of it
    _STRUCTURAL_FILLERS = frozenset({"named", "called", "titled"})

    # Prepositions that signal complex intent (location, direction, condition)
    # Only trigger escalation for skills WITH optional inputs
    _ESCALATION_PREPOSITIONS = frozenset({
        "on", "in", "inside", "under", "at", "into", "from", "for",
    })

    # Conversational noise — always stripped from string extraction
    # NOTE: 'my'/'your' intentionally excluded — they can be part of names
    _CONVERSATIONAL_NOISE = frozenset({
        "the", "a", "an", "to", "please",
        "can", "you", "could", "would", "i", "want",
    })

    def _extract_params(
        self,
        skill_name: str,
        raw_tokens: List[str],
        expanded_tokens: List[str],
    ) -> Dict[str, Any]:
        """Extract parameters using deterministic slot-filling.

        Reflex extraction boundary rules:
        1. ONLY required inputs are extracted — optional inputs use defaults
        2. Numeric extraction restricted to numeric semantic types
        3. Skills with 2+ required string params → return what we have (skip strings)
        4. Escalation prepositions + optional inputs → skip string extraction
        5. Single string param → positional (remaining tokens after noise)
        """
        contract = self._index.contracts[skill_name]
        params: Dict[str, Any] = {}

        # ── Step 1: Extract numeric + alias params (required only) ──
        for key, sem_type_name in contract.inputs.items():
            sem_type = SEMANTIC_TYPES.get(sem_type_name)
            if not sem_type:
                continue

            # Alias check (e.g. "full" → 100, "max" → 100)
            if sem_type.aliases:
                for token in raw_tokens:
                    if token.lower() in sem_type.aliases:
                        params[key] = token
                        break

            # Numeric check — ONLY for numeric semantic types
            if key not in params and sem_type.python_type in (int, float):
                for token in raw_tokens:
                    try:
                        float(token)
                        params[key] = token
                        break
                    except ValueError:
                        pass

        # ── Step 2: String param extraction ──
        string_params = [
            (k, v) for k, v in contract.inputs.items()
            if k not in params  # not already extracted
            and SEMANTIC_TYPES.get(v) is not None
            and SEMANTIC_TYPES[v].python_type is str
        ]

        # 2a: Multiple required string params → cannot assign deterministically
        #     Return what we have (numeric/alias), skip strings → forces mission
        if len(string_params) >= 2:
            return params

        # 2b: Exactly one required string param → positional extraction
        if len(string_params) == 1:
            key, sem_type_name = string_params[0]

            # Escalation guard: if skill has optional inputs AND
            # a preposition is detected → extraction FAILS entirely.
            # Return None to signal that this match should be rejected.
            # The query is too complex for deterministic extraction.
            if contract.optional_inputs:
                for token in raw_tokens:
                    if token.lower() in self._ESCALATION_PREPOSITIONS:
                        logger.debug(
                            "IntentMatcher: preposition '%s' detected for "
                            "skill with optional inputs → extraction failed",
                            token,
                        )
                        return None

            # Build comprehensive noise set
            noise: set = set()
            for v in contract.intent_verbs:
                noise.add(v.lower())
            for k in contract.intent_keywords:
                noise.add(k.lower())
            noise.update(self._STRUCTURAL_FILLERS)
            noise.update(self._CONVERSATIONAL_NOISE)
            # Synonym canonical forms
            for v in VERB_SYNONYMS.values():
                noise.add(v)
            for n in NOUN_SYNONYMS.values():
                noise.add(n)
            # Exclude numeric tokens (already extracted or not relevant)
            numeric_tokens = set()
            for t in raw_tokens:
                try:
                    float(t)
                    numeric_tokens.add(t)
                except ValueError:
                    pass

            remaining = [
                t for t in raw_tokens
                if t.lower() not in noise and t not in numeric_tokens
            ]
            if remaining:
                params[key] = " ".join(remaining)

        return params

