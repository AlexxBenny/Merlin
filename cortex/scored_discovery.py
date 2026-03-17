# cortex/scored_discovery.py

"""
DomainScoredDiscovery — Two-phase skill selection for scalable LLM prompts.

Phase 1: Score all skills by intent_verb + intent_keyword match against query.
          Return top-K scored skills.

Phase 2: Domain expansion — for each matched skill's domain, include up to
          MAX_PER_DOMAIN sibling skills (sorted by their own scores).

Safety rails:
    - Domain expansion cap: max N siblings per domain (prevents domain explosion)
    - Hard global cap: MAX_MANIFEST total skills in output (absolute ceiling)
    - Deterministic ordering: (score desc, name asc) for stable prompts
    - Fallback: if total registered skills <= fallback_threshold, return ALL
      (no filtering needed — our current state at 15 skills)

Future: EmbeddingDiscovery replaces Phase 1 scoring with vector search.
        Same interface, same Phase 2 expansion, zero cortex changes.
"""

import re
import logging
from typing import Any, Dict, List, Set, Tuple

from cortex.skill_discovery import SkillDiscovery, AllSkillsDiscovery
from execution.registry import SkillRegistry

logger = logging.getLogger(__name__)


class DomainScoredDiscovery(SkillDiscovery):
    """
    Two-phase skill selection:
      Phase 1: intent_verb + intent_keyword scoring (deterministic, zero-dependency)
      Phase 2: bounded domain expansion for chaining safety

    Config:
        top_k:              Phase 1 cutoff (default 20)
        max_per_domain:     Max sibling skills expanded per domain (default 12)
        max_manifest:       Hard global cap on total skills returned (default 60)
        fallback_threshold: Return all skills if total < this (default 30)
    """

    def __init__(
        self,
        top_k: int = 20,
        max_per_domain: int = 12,
        max_manifest: int = 60,
        fallback_threshold: int = 30,
    ):
        self.top_k = top_k
        self.max_per_domain = max_per_domain
        self.max_manifest = max_manifest
        self.fallback_threshold = fallback_threshold
        self._all_skills = AllSkillsDiscovery()

    def find_candidates(
        self,
        query: str,
        registry: SkillRegistry,
    ) -> Dict[str, Any]:
        all_names = registry.all_names()

        # Below threshold: return everything (no filtering benefit)
        if len(all_names) <= self.fallback_threshold:
            return self._all_skills.find_candidates(query, registry)

        # Phase 1: score all skills
        scores = self._score_skills(query, registry)

        # Phase 1 top-K (deterministic sort: score desc, name asc)
        sorted_skills = sorted(
            scores.items(),
            key=lambda x: (-x[1], x[0]),
        )
        top_skills = [name for name, score in sorted_skills[:self.top_k]
                      if score > 0]

        # Safety: if Phase 1 found nothing, fall back to all skills.
        # Empty manifest would crash the compiler — never allow it.
        if not top_skills:
            logger.warning(
                "[DISCOVERY] Phase 1 returned 0 matches for query=%r, "
                "falling back to AllSkillsDiscovery",
                query[:80],
            )
            return self._all_skills.find_candidates(query, registry)

        # Phase 2: bounded domain expansion
        matched_domains: Set[str] = set()
        for name in top_skills:
            skill = registry.get(name)
            matched_domains.add(skill.contract.domain)

        expanded = self._expand_domains(
            matched_domains, top_skills, scores, registry,
        )

        # Merge: top skills + expanded siblings (no duplicates)
        final_names = list(dict.fromkeys(top_skills + expanded))

        # Hard global cap
        if len(final_names) > self.max_manifest:
            final_names = final_names[:self.max_manifest]

        logger.info(
            "[DISCOVERY] Query=%r → Phase1=%d skills, Phase2=%d expanded, "
            "Final=%d (from %d total)",
            query[:80], len(top_skills), len(expanded),
            len(final_names), len(all_names),
        )

        return self._build_manifest(final_names, registry)

    def _score_skills(
        self,
        query: str,
        registry: SkillRegistry,
    ) -> Dict[str, float]:
        """Score each skill by intent_verb + intent_keyword overlap with query.

        Scoring:
            - Each verb match: +2.0 (verbs are high-signal)
            - Each keyword match: +1.0
            - Scores are not normalized — absolute ranking only

        Uses word tokenization of the query (lowercased).
        """
        query_words = set(re.findall(r'\b[a-z]+\b', query.lower()))
        # Simple plural normalization: 'emails' → also match 'email'
        # Prevents singular/plural mismatch across all skills.
        depluralized = set()
        for w in query_words:
            if len(w) > 3 and w.endswith('s') and not w.endswith('ss'):
                depluralized.add(w[:-1])
        query_words |= depluralized
        scores: Dict[str, float] = {}

        for name in registry.all_names():
            skill = registry.get(name)
            contract = skill.contract

            score = 0.0

            # Verb match (high weight — verbs carry intent)
            for verb in contract.intent_verbs:
                if verb.lower() in query_words:
                    score += 2.0

            # Keyword match
            for kw in contract.intent_keywords:
                if kw.lower() in query_words:
                    score += 1.0

            scores[name] = score

        return scores

    def _expand_domains(
        self,
        matched_domains: Set[str],
        already_selected: List[str],
        scores: Dict[str, float],
        registry: SkillRegistry,
    ) -> List[str]:
        """Expand matched domains with bounded sibling inclusion.

        For each matched domain, include up to max_per_domain sibling skills
        that aren't already in the selected set. Siblings are sorted by score
        (desc) then name (asc) for deterministic ordering.
        """
        selected_set = set(already_selected)
        expanded: List[str] = []

        for domain in sorted(matched_domains):  # Deterministic iteration
            # Find all skills in this domain not yet selected
            siblings = []
            for name in registry.all_names():
                if name in selected_set:
                    continue
                skill = registry.get(name)
                if skill.contract.domain == domain:
                    siblings.append(name)

            # Sort by score (desc), then name (asc)
            siblings.sort(key=lambda n: (-scores.get(n, 0.0), n))

            # Cap per domain
            for sib in siblings[:self.max_per_domain]:
                expanded.append(sib)
                selected_set.add(sib)

        return expanded

    def _build_manifest(
        self,
        skill_names: List[str],
        registry: SkillRegistry,
    ) -> Dict[str, Any]:
        """Build manifest dict in the same format as AllSkillsDiscovery."""
        manifest: Dict[str, Any] = {}
        for name in skill_names:
            skill = registry.get(name)
            manifest[skill.contract.name] = {
                "description": skill.contract.description,
                "action": skill.contract.action,
                "target_type": skill.contract.target_type,
                "inputs": {
                    k: v for k, v in skill.contract.inputs.items()
                },
                "outputs": {
                    k: v for k, v in skill.contract.outputs.items()
                },
                "allowed_modes": sorted(
                    m.value for m in skill.contract.allowed_modes
                ),
            }
        return manifest
