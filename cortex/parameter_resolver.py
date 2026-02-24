# cortex/parameter_resolver.py

"""
ParameterResolver — Typed parameter normalization between compilation and execution.

This is the boundary enforcement layer between:
  LLM output (untyped JSON) → Executor (typed skill inputs)

Design rules:
- NEVER mutates the original plan — always produces a new one
- Raises structured ParameterError (not ValueError/RuntimeError)
- Resolution is deterministic — no LLM calls
- Output-only types are skipped (no user input to coerce)
- Unknown semantic types pass through unchanged (forward-compatible)

Scaling property:
  Coercion is O(types), not O(skills).
  40 skills using volume_percentage all share one resolve() path.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from cortex.semantic_types import SEMANTIC_TYPES
from ir.mission import MissionPlan, MissionNode

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Structured error (never a raw ValueError)
# ─────────────────────────────────────────────────────────────

@dataclass
class ParameterViolation:
    """A single parameter resolution failure."""
    node_id: str
    skill: str
    param_key: str
    raw_value: Any
    semantic_type: str
    reason: str


@dataclass
class ParameterError(Exception):
    """Structured error for parameter resolution failures.

    Contains ALL violations found in the plan (not just the first).
    This allows batch reporting to the user.
    """
    violations: List[ParameterViolation] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [f"{len(self.violations)} parameter error(s):"]
        for v in self.violations:
            lines.append(
                f"  • {v.node_id} ({v.skill}): "
                f"'{v.param_key}' = {v.raw_value!r} — {v.reason}"
            )
        return "\n".join(lines)

    def user_message(self) -> str:
        """Clean user-facing error message (no stack trace)."""
        parts = []
        for v in self.violations:
            parts.append(f"'{v.param_key}' ({v.raw_value!r}): {v.reason}")
        return "I couldn't resolve some parameters: " + "; ".join(parts)


# ─────────────────────────────────────────────────────────────
# Resolver
# ─────────────────────────────────────────────────────────────

class ParameterResolver:
    """Resolve raw plan inputs through SemanticType contracts.

    Usage:
        resolver = ParameterResolver(registry)
        resolved_plan = resolver.resolve_plan(plan)
    """

    def __init__(self, registry: "SkillRegistry"):
        self._registry = registry

    def resolve_plan(self, plan: MissionPlan) -> MissionPlan:
        """Produce a new MissionPlan with all inputs type-resolved.

        Never mutates the original plan.
        Raises ParameterError if any input cannot be resolved.
        """
        violations: List[ParameterViolation] = []
        resolved_nodes: List[MissionNode] = []

        for node in plan.nodes:
            resolved_inputs, node_violations = self._resolve_node(node)
            violations.extend(node_violations)

            # Build new node with resolved inputs (shallow copy)
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
            raise ParameterError(violations=violations)

        # Build new plan (never mutate original)
        return MissionPlan(
            id=plan.id,
            nodes=resolved_nodes,
            metadata=dict(plan.metadata) if plan.metadata else {},
        )

    def _resolve_node(
        self, node: MissionNode,
    ) -> tuple[Dict[str, Any], List[ParameterViolation]]:
        """Resolve a single node's inputs. Returns (resolved_inputs, violations)."""
        violations: List[ParameterViolation] = []
        resolved: Dict[str, Any] = {}

        # Look up skill contract
        try:
            skill = self._registry.get(node.skill)
        except KeyError:
            # Unknown skill — pass inputs through unchanged
            logger.debug(
                "ParameterResolver: unknown skill '%s', passthrough", node.skill,
            )
            return dict(node.inputs), []

        contract = skill.contract
        all_inputs = {**contract.inputs, **contract.optional_inputs}

        for key, raw_value in node.inputs.items():
            semantic_type_name = all_inputs.get(key)

            if not semantic_type_name or semantic_type_name not in SEMANTIC_TYPES:
                # No type mapping → passthrough
                resolved[key] = raw_value
                continue

            sem_type = SEMANTIC_TYPES[semantic_type_name]

            # Output-only types should never appear in inputs
            # (already caught at registration, but belt-and-suspenders)
            if sem_type.direction == "output":
                resolved[key] = raw_value
                continue

            # Skip resolution for types with no coercion logic
            if not sem_type.is_resolvable:
                resolved[key] = raw_value
                continue

            try:
                resolved[key] = sem_type.resolve(raw_value)
                if resolved[key] != raw_value:
                    logger.info(
                        "[RESOLVE] %s.%s: %r → %r (type=%s)",
                        node.id, key, raw_value, resolved[key],
                        semantic_type_name,
                    )
            except ValueError as e:
                violations.append(ParameterViolation(
                    node_id=node.id,
                    skill=node.skill,
                    param_key=key,
                    raw_value=raw_value,
                    semantic_type=semantic_type_name,
                    reason=str(e),
                ))
                resolved[key] = raw_value  # Keep raw for error reporting

        return resolved, violations
