# cortex/entity_resolver.py

"""
EntityResolver — Post-compilation entity normalization.

Sits in the MissionOrchestrator transform chain as Phase 9C:

    Compiler → ParameterResolver (9A) → PreferenceResolver (9B) → EntityResolver (9C) → Executor

Responsibilities:
- Scan plan nodes for skills with entity_params in their contract
- Resolve each entity parameter to a canonical app_id via ApplicationRegistry
- Add app_id alongside original app_name (never overwrite)
- Return structured outcomes (resolved/ambiguous/not_found)
- Raise EntityResolutionError for ambiguous or not_found (user clarification)

Design:
- Follows same pattern as ParameterResolver (operates on MissionPlan)
- Never mutates the original plan — produces new MissionPlan
- Uses SkillContract.entity_params — NOT hardcoded skill names
- Candidate set is extensible (future: favorites, recent, GUI apps)
- Skips IRReference values (runtime-resolved pipes)
"""

import copy
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from infrastructure.application_registry import (
    ApplicationEntity,
    ApplicationRegistry,
)
from ir.mission import MissionPlan, MissionNode, IRReference

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Resolution result types
# ─────────────────────────────────────────────────────────────

class ResolutionType(str, Enum):
    """Outcome of a single entity resolution attempt."""
    RESOLVED = "resolved"       # Exact match → single app_id
    AMBIGUOUS = "ambiguous"     # Multiple candidates → user must choose
    NOT_FOUND = "not_found"     # No match at all


@dataclass(frozen=True)
class ResolutionResult:
    """Structured outcome of resolving one application term.

    Fields:
        type:       Resolution outcome (resolved/ambiguous/not_found)
        term:       Original user term that was resolved
        app_id:     Canonical app_id (only set when type=RESOLVED)
        entity:     Full ApplicationEntity (only set when type=RESOLVED)
        candidates: List of candidate app_ids (only set when type=AMBIGUOUS)
        score:      Match confidence 0.0–1.0 (for diagnostics)
    """
    type: ResolutionType
    term: str
    app_id: Optional[str] = None
    entity: Optional[ApplicationEntity] = None
    candidates: List[str] = field(default_factory=list)
    score: float = 0.0


# ─────────────────────────────────────────────────────────────
# Structured error (never a raw ValueError)
# ─────────────────────────────────────────────────────────────

@dataclass
class EntityViolation:
    """A single entity resolution failure."""
    node_id: str
    skill: str
    param_key: str
    raw_value: str
    resolution_type: str   # "ambiguous" or "not_found"
    candidates: List[str] = field(default_factory=list)


@dataclass
class EntityResolutionError(Exception):
    """Structured error for entity resolution failures.

    Contains ALL violations found in the plan (not just the first).
    Provides user-facing clarification messages.
    """
    violations: List[EntityViolation] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [f"{len(self.violations)} entity resolution error(s):"]
        for v in self.violations:
            if v.resolution_type == "ambiguous":
                lines.append(
                    f"  • {v.node_id} ({v.skill}): "
                    f"'{v.raw_value}' is ambiguous — candidates: {v.candidates}"
                )
            else:
                lines.append(
                    f"  • {v.node_id} ({v.skill}): "
                    f"'{v.raw_value}' not found"
                )
        return "\n".join(lines)

    def user_message(self) -> str:
        """Clean user-facing clarification message."""
        parts = []
        for v in self.violations:
            if v.resolution_type == "ambiguous":
                options = ", ".join(v.candidates[:5])
                parts.append(
                    f"Which did you mean by \"{v.raw_value}\"? "
                    f"Options: {options}"
                )
            else:
                if v.candidates:
                    suggestion = v.candidates[0]
                    parts.append(
                        f"I couldn't find an application named "
                        f"\"{v.raw_value}\". Did you mean {suggestion}?"
                    )
                else:
                    parts.append(
                        f"I couldn't find an application named "
                        f"\"{v.raw_value}\"."
                    )
        return " ".join(parts)


# ─────────────────────────────────────────────────────────────
# EntityResolver — Post-compilation transform
# ─────────────────────────────────────────────────────────────

class EntityResolver:
    """Resolve entity parameters in a compiled MissionPlan.

    Post-compilation transform (Phase 9C). Same pattern as
    ParameterResolver and PreferenceResolver.

    Usage:
        resolver = EntityResolver(registry, skill_registry, aliases)
        resolved_plan = resolver.resolve_plan(plan)

    Resolution pipeline per entity param:
    1. Direct registry lookup (by app_id → exact)
    2. Name index lookup (display names → exact)
    3. Alias table ("browser" → "chrome")
    4. Fuzzy match (substring in display names)
    5. Ambiguous or NOT_FOUND → raise EntityResolutionError
    """

    def __init__(
        self,
        registry: ApplicationRegistry,
        skill_registry: "SkillRegistry",
        alias_map: Optional[Dict[str, str]] = None,
    ):
        self._registry = registry
        self._skill_registry = skill_registry
        self._aliases: Dict[str, str] = {}

        if alias_map:
            for key, value in alias_map.items():
                self._aliases[key.lower().strip()] = value.lower().strip()

    def resolve_plan(self, plan: MissionPlan) -> MissionPlan:
        """Produce a new MissionPlan with entity params resolved.

        Never mutates the original plan.
        Raises EntityResolutionError if any entity cannot be resolved.
        """
        violations: List[EntityViolation] = []
        resolved_nodes: List[MissionNode] = []

        for node in plan.nodes:
            resolved_inputs, node_violations = self._resolve_node(node)
            violations.extend(node_violations)

            new_node = MissionNode(
                id=node.id,
                skill=node.skill,
                inputs=resolved_inputs,
                outputs=node.outputs,
                depends_on=list(node.depends_on),
                mode=node.mode,
            )
            resolved_nodes.append(new_node)

        if violations:
            raise EntityResolutionError(violations=violations)

        return MissionPlan(
            id=plan.id,
            nodes=resolved_nodes,
            metadata=dict(plan.metadata) if plan.metadata else {},
        )

    def _resolve_node(
        self, node: MissionNode,
    ) -> tuple:
        """Resolve entity params in a single node. Returns (inputs, violations)."""
        violations: List[EntityViolation] = []
        resolved: Dict[str, Any] = dict(node.inputs)

        # Look up skill contract to find entity_params
        try:
            skill = self._skill_registry.get(node.skill)
        except KeyError:
            # Unknown skill — pass inputs through unchanged
            return resolved, []

        entity_params = getattr(skill.contract, 'entity_params', [])
        if not entity_params:
            return resolved, []

        for param_key in entity_params:
            if param_key not in resolved:
                continue

            raw_value = resolved[param_key]

            # Skip IRReference values — runtime-resolved pipes
            if isinstance(raw_value, IRReference):
                continue

            # Skip non-string values
            if not isinstance(raw_value, str):
                continue

            # Resolve the entity
            result = self.resolve(raw_value)

            if result.type == ResolutionType.RESOLVED:
                # Add app_id alongside original param (never overwrite)
                resolved["app_id"] = result.app_id
                logger.info(
                    "[ENTITY] %s.%s: '%s' → app_id='%s' (score=%.2f)",
                    node.id, param_key, raw_value, result.app_id, result.score,
                )
            elif result.type == ResolutionType.AMBIGUOUS:
                violations.append(EntityViolation(
                    node_id=node.id,
                    skill=node.skill,
                    param_key=param_key,
                    raw_value=raw_value,
                    resolution_type="ambiguous",
                    candidates=result.candidates,
                ))
            else:  # NOT_FOUND
                # Find closest match for suggestion
                closest = self._find_closest(raw_value)
                violations.append(EntityViolation(
                    node_id=node.id,
                    skill=node.skill,
                    param_key=param_key,
                    raw_value=raw_value,
                    resolution_type="not_found",
                    candidates=[closest] if closest else [],
                ))

        return resolved, violations

    def resolve(self, term: str) -> ResolutionResult:
        """Resolve a single application term to a canonical app_id.

        Never raises. Returns NOT_FOUND on error.
        """
        try:
            return self._resolve_inner(term)
        except Exception as e:
            logger.warning("EntityResolver error for '%s': %s", term, e)
            return ResolutionResult(
                type=ResolutionType.NOT_FOUND,
                term=term,
            )

    def _resolve_inner(self, term: str) -> ResolutionResult:
        """Core resolution pipeline."""
        normalized = term.lower().strip()

        if not normalized:
            return ResolutionResult(type=ResolutionType.NOT_FOUND, term=term)

        # ── Step 1: Direct registry lookup (app_id match) ──
        entity = self._registry.get(normalized)
        if entity:
            return ResolutionResult(
                type=ResolutionType.RESOLVED, term=term,
                app_id=entity.app_id, entity=entity, score=1.0,
            )

        # ── Step 2: Name index lookup (display name match) ──
        entity = self._registry.lookup_by_name(normalized)
        if entity:
            return ResolutionResult(
                type=ResolutionType.RESOLVED, term=term,
                app_id=entity.app_id, entity=entity, score=1.0,
            )

        # ── Step 3: Alias table ──
        if normalized in self._aliases:
            alias_target = self._aliases[normalized]
            entity = self._registry.get(alias_target)
            if entity:
                logger.debug("EntityResolver: alias '%s' → '%s'", normalized, alias_target)
                return ResolutionResult(
                    type=ResolutionType.RESOLVED, term=term,
                    app_id=entity.app_id, entity=entity, score=0.9,
                )

        # ── Step 4: Fuzzy match (substring in candidate names) ──
        candidates = self._candidate_set()
        matches = self._fuzzy_match(normalized, candidates)

        if len(matches) == 1:
            app_id, score = matches[0]
            entity = self._registry.get(app_id)
            return ResolutionResult(
                type=ResolutionType.RESOLVED, term=term,
                app_id=app_id, entity=entity, score=score,
            )

        if len(matches) > 1:
            candidate_ids = [m[0] for m in matches]
            logger.info("EntityResolver: ambiguous '%s' → %s", term, candidate_ids)
            return ResolutionResult(
                type=ResolutionType.AMBIGUOUS, term=term,
                candidates=candidate_ids,
            )

        # ── Step 5: NOT_FOUND ──
        return ResolutionResult(type=ResolutionType.NOT_FOUND, term=term)

    def _candidate_set(self) -> List[ApplicationEntity]:
        """Return the current candidate set for matching.

        Currently returns all entities. Future extension point:
        - Filter by GUI apps only
        - Prioritize frequently used
        - Include favorites
        """
        return [
            self._registry.get(app_id)
            for app_id in self._registry.all_ids()
            if self._registry.get(app_id) is not None
        ]

    def _fuzzy_match(
        self, term: str, candidates: List[ApplicationEntity],
    ) -> List[tuple]:
        """Find candidates where term is a substring of any name.

        Returns list of (app_id, score), sorted by score desc.
        Only returns candidates above threshold.
        """
        SCORE_THRESHOLD = 0.4
        results = []

        for entity in candidates:
            best_score = 0.0

            if term == entity.app_id:
                best_score = 1.0
            elif term in entity.app_id:
                best_score = max(best_score, len(term) / len(entity.app_id))

            for name in entity.display_names:
                name_lower = name.lower()
                if term == name_lower:
                    best_score = 1.0
                    break
                if term in name_lower:
                    coverage = len(term) / len(name_lower)
                    best_score = max(best_score, coverage)

            for pn in entity.canonical_process_names:
                pn_lower = pn.lower().replace(".exe", "")
                if term == pn_lower:
                    best_score = max(best_score, 0.95)

            if best_score >= SCORE_THRESHOLD:
                results.append((entity.app_id, best_score))

        results.sort(key=lambda x: -x[1])

        if len(results) >= 2:
            top_score = results[0][1]
            results = [
                (app_id, score)
                for app_id, score in results
                if score >= top_score * 0.85
            ]

        return results[:5]

    def _find_closest(self, term: str) -> Optional[str]:
        """Find the single closest match for a not-found term (for suggestions)."""
        candidates = self._candidate_set()
        matches = self._fuzzy_match(term.lower().strip(), candidates)
        # Use a lower threshold for suggestions
        if matches:
            return matches[0][0]
        return None

    def resolve_terms(self, terms: List[str]) -> List[ResolutionResult]:
        """Resolve multiple terms (batch convenience method)."""
        return [self.resolve(term) for term in terms]
