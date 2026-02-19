import logging
from typing import Any, Dict, List, Optional, Protocol, Set, Tuple, runtime_checkable

from ir.mission import (
    MissionPlan,
    MissionNode,
    ExecutionMode,
    OutputReference,
    ConditionExpr,
)


class MissionValidationError(Exception):
    pass


def validate_mission_plan(
    plan: MissionPlan,
    available_skills: Set[str],
    registry=None,
) -> None:
    """
    Hard validation gate for MissionPlan.

    This is the ONLY place where structural integrity of a compiled
    mission is verified. No silent fixes. No best-effort.

    Validates:
    1. Skill existence
    2. Dependency existence
    3. Cycle detection
    4. Background dependency prohibition
    5. Condition expression validity
    6. OutputReference target validity
    7. At least one root node
    8. Required input coverage (compile-time, needs registry)
    9. Unexpected input rejection (compile-time, needs registry)

    Ordering invariant:
    - Checks 1-7 are structural (only need available_skills set)
    - Checks 8-9 are contract-aware (need registry for SkillContract access)
    - Checks 8-9 run AFTER normalization and ID resolution
    """

    # Build node index for O(1) lookup from List[MissionNode]
    node_index: Dict[str, MissionNode] = {}
    for node in plan.nodes:
        if node.id in node_index:
            raise MissionValidationError(
                f"Duplicate node id '{node.id}'"
            )
        node_index[node.id] = node

    # 1. Skill existence
    for node in plan.nodes:
        if node.skill not in available_skills:
            raise MissionValidationError(
                f"Unknown skill '{node.skill}' in node '{node.id}'"
            )

    # 2. Dependency existence
    for node in plan.nodes:
        for dep in node.depends_on:
            if dep not in node_index:
                raise MissionValidationError(
                    f"Node '{node.id}' depends on missing node '{dep}'"
                )

    # 3. Cycle detection (DFS)
    visited: set = set()
    stack: set = set()

    def visit(nid: str) -> None:
        if nid in stack:
            raise MissionValidationError("Cycle detected in mission DAG")
        if nid in visited:
            return
        stack.add(nid)
        for d in node_index[nid].depends_on:
            visit(d)
        stack.remove(nid)
        visited.add(nid)

    for nid in node_index:
        visit(nid)

    # 4. Background nodes cannot be depended upon
    background_nodes = {
        n.id for n in plan.nodes
        if n.mode == ExecutionMode.background
    }
    for node in plan.nodes:
        for dep in node.depends_on:
            if dep in background_nodes:
                raise MissionValidationError(
                    f"Node '{node.id}' depends on background node '{dep}'"
                )

    # 5. Conditional nodes — validate ConditionExpr
    for node in plan.nodes:
        if node.condition_on is not None:
            cond: ConditionExpr = node.condition_on

            # If source references a node id, that node must exist
            # and must have outputs
            if not cond.source.startswith("world."):
                if cond.source not in node_index:
                    raise MissionValidationError(
                        f"Node '{node.id}' condition_on references "
                        f"missing node '{cond.source}'"
                    )
                if not node_index[cond.source].outputs:
                    raise MissionValidationError(
                        f"condition_on node '{cond.source}' has no outputs"
                    )

    # 6. OutputReference target validity
    for node in plan.nodes:
        for key, value in node.inputs.items():
            if isinstance(value, OutputReference):
                # Referenced node must exist
                if value.node not in node_index:
                    raise MissionValidationError(
                        f"Node '{node.id}' input '{key}' references "
                        f"missing node '{value.node}'"
                    )
                # Referenced output must be declared
                ref_node = node_index[value.node]
                if value.output not in ref_node.outputs:
                    raise MissionValidationError(
                        f"Node '{node.id}' input '{key}' references "
                        f"output '{value.output}' not declared by "
                        f"node '{value.node}'"
                    )
                # Referenced node must be a dependency (direct or transitive)
                if value.node not in node.depends_on:
                    raise MissionValidationError(
                        f"Node '{node.id}' references output from "
                        f"node '{value.node}' but does not depend on it"
                    )

    # 7. At least one root
    if not any(len(n.depends_on) == 0 for n in plan.nodes):
        raise MissionValidationError(
            "Mission must have at least one root node"
        )

    # ── Contract-aware checks (only if registry provided) ──
    # These run AFTER structural checks (correct ordering).
    # Normalizer has already coerced values at this point.

    if registry is not None:
        for node in plan.nodes:
            skill = registry.get(node.skill)
            if skill is None:
                continue  # Check #1 already caught unknown skills

            contract = skill.contract
            required = set(contract.inputs.keys())
            optional = set(contract.optional_inputs.keys())
            allowed = required | optional
            provided = set(node.inputs.keys())

            # 8. Missing required inputs
            missing = required - provided
            if missing:
                raise MissionValidationError(
                    f"Node '{node.id}': skill '{node.skill}' requires "
                    f"inputs {sorted(missing)} but they were not provided"
                )

            # 9. Unexpected inputs (provided ⊈ required ∪ optional)
            unexpected = provided - allowed
            if unexpected:
                raise MissionValidationError(
                    f"Node '{node.id}': skill '{node.skill}' received "
                    f"unexpected inputs {sorted(unexpected)}. "
                    f"Allowed: {sorted(allowed)}"
                )


# ──────────────────────────────────────────────────────────────
# Intent Coverage Verification (Phase 5A)
#
# Separate from structural validation above.
# Structural validation = "is this DAG internally consistent?"
# Coverage verification = "does this DAG cover the user's intents?"
# ──────────────────────────────────────────────────────────────

_coverage_logger = logging.getLogger(__name__ + ".coverage")


@runtime_checkable
class IntentMatcher(Protocol):
    """Replaceable seam for intent-to-node matching strategy.

    Swap with embedding-based, LLM-based, or config-driven matcher.
    """
    def match(
        self,
        intent: Dict[str, str],
        candidate_nodes: List[MissionNode],
        registry: Any,
    ) -> Optional[str]:
        """Return node_id of the best match, or None if no match."""
        ...


class HeuristicIntentMatcher:
    """Default intent matcher — deterministic, argument-level.

    Three-stage matching:
    1. Domain-biased candidate filtering (soft — fallback to all)
    2. Skill name/description verb matching
    3. Argument-level value alignment
    """

    def match(
        self,
        intent: Dict[str, str],
        candidate_nodes: List[MissionNode],
        registry: Any,
    ) -> Optional[str]:
        """Find the best matching node for an intent unit.

        Returns node.id if matched, None otherwise.
        """
        if not candidate_nodes:
            return None

        verb = intent.get("verb", "").lower()
        obj = intent.get("object", "").lower()
        domain_hint = intent.get("domain_hint", "").lower()

        # Stage 1: Domain-biased filtering (SOFT)
        if domain_hint:
            domain_candidates = [
                n for n in candidate_nodes
                if self._node_domain(n) == domain_hint
            ]
            # Soft fallback: if no domain match, use ALL candidates
            if domain_candidates:
                candidates = domain_candidates
            else:
                candidates = candidate_nodes
        else:
            candidates = candidate_nodes

        # Stage 2 + 3: Verb match + argument alignment
        for node in candidates:
            skill_name = node.skill.lower()
            skill_tokens = set(skill_name.replace(".", "_").split("_"))

            # Stage 2: Verb match — verb appears in skill name tokens
            verb_match = verb in skill_tokens if verb else False

            # Also check if verb is a substring of any skill token
            if not verb_match and verb:
                verb_match = any(verb in t for t in skill_tokens)

            if not verb_match:
                continue

            # Stage 3: Argument alignment
            if not obj:
                # No object to match — verb match alone is sufficient
                return node.id

            # Check intent.object against node.inputs values
            node_values = [
                str(v).lower()
                for v in node.inputs.values()
                if not hasattr(v, 'node')  # Skip OutputReference
            ]

            # Case-insensitive substring match
            obj_match = any(
                obj in v or v in obj
                for v in node_values
            )

            if obj_match:
                return node.id

            # Relaxed: if node has no literal input values (e.g., media_play
            # with empty inputs), verb match alone is sufficient
            if not node_values:
                return node.id

        return None

    @staticmethod
    def _node_domain(node: MissionNode) -> str:
        """Extract domain from node's skill name (e.g., 'system' from 'system.open_app')."""
        return node.skill.split(".")[0].lower() if "." in node.skill else ""


def verify_intent_coverage(
    plan: MissionPlan,
    intent_units: List[Dict[str, str]],
    registry: Any,
    matcher: Optional[IntentMatcher] = None,
) -> Tuple[bool, List[Dict[str, str]]]:
    """Verify that a compiled MissionPlan covers all declared intent units.

    This is a SEMANTIC coverage check, separate from structural validation.
    Called only for Tier 2+ queries (after decomposition).

    Args:
        plan: The compiled MissionPlan to verify.
        intent_units: The INJECTED intent set (≤8, post-cap).
            Verification and injection operate on the SAME set.
        registry: SkillRegistry for contract access.
        matcher: IntentMatcher implementation. Defaults to HeuristicIntentMatcher.

    Returns:
        (all_covered, uncovered_intents):
            all_covered: True if every intent maps to a DAG node.
            uncovered_intents: List of intent dicts that have no matching node.

    Design rules:
    - Each DAG node covers at most ONE intent (no double-counting).
    - domain_hint is a soft bias, not a hard gate.
    - Matching is argument-level: intent.object must appear in node inputs.
    """
    if matcher is None:
        matcher = HeuristicIntentMatcher()

    # Available nodes pool — consumed nodes are removed
    available_nodes: List[MissionNode] = list(plan.nodes)
    uncovered: List[Dict[str, str]] = []

    for intent in intent_units:
        matched_id = matcher.match(intent, available_nodes, registry)

        if matched_id is not None:
            # Consume the matched node — cannot cover another intent
            available_nodes = [n for n in available_nodes if n.id != matched_id]
            _coverage_logger.debug(
                "Intent [%s %s] → covered by node '%s'",
                intent.get("verb", ""), intent.get("object", ""),
                matched_id,
            )
        else:
            uncovered.append(intent)
            _coverage_logger.info(
                "Intent [%s %s] → UNCOVERED (no matching node)",
                intent.get("verb", ""), intent.get("object", ""),
            )

    all_covered = len(uncovered) == 0

    _coverage_logger.info(
        "Coverage result: %d/%d intents covered, %d uncovered",
        len(intent_units) - len(uncovered),
        len(intent_units),
        len(uncovered),
    )

    return all_covered, uncovered
