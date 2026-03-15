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


logger = logging.getLogger(__name__)


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

                # 6b. Heuristic type guard for index/field (compile-time)
                # Uses semantic type name from referenced node's output declaration.
                # Not runtime-safe — just catches obvious LLM misuse early.
                if registry is not None and (value.index is not None or value.field is not None):
                    ref_skill = registry.get(ref_node.skill)
                    if ref_skill:
                        out_semantic_type = ref_skill.contract.outputs.get(
                            value.output, ""
                        ).lower()
                        if value.index is not None and "list" not in out_semantic_type:
                            logger.warning(
                                "OutputReference index=%d on '%s.%s' but "
                                "semantic type '%s' does not contain 'list'. "
                                "Possible LLM hallucination — may fail at runtime.",
                                value.index, value.node, value.output,
                                out_semantic_type,
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

            # 10. Input group constraints (at least one from each group)
            for group in getattr(contract, 'input_groups', None) or []:
                if not (provided & group):
                    raise MissionValidationError(
                        f"Node '{node.id}': skill '{node.skill}' requires "
                        f"at least one of {sorted(group)} but none were "
                        f"provided"
                    )


# ──────────────────────────────────────────────────────────────
# Intent Coverage Verification (Phase 5A)
#
# Separate from structural validation above.
# Structural validation = "is this DAG internally consistent?"
# Coverage verification = "does this DAG cover the user's intents?"
# ──────────────────────────────────────────────────────────────

_coverage_logger = logging.getLogger(__name__ + ".coverage")

# Subsumption table: broad action → set of specific actions it covers.
# One-directional: "autonomous_task" covers everything, but "search"
# cannot subsume "autonomous_task". Used as safety net when decomposer
# and compiler disagree on action granularity.
_ACTION_SUBSUMPTION: Dict[str, set] = {
    "autonomous_task": {
        "search", "select_result", "navigate", "click", "fill",
        "scroll", "keypress", "go_back", "go_forward",
    },
}


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


class CapabilityIntentMatcher:
    """Capability-based intent matcher — deterministic, O(intents × nodes).

    Matching rule:
        intent.action == node_action (derived from node.skill name)

    When multiple nodes share the same action (e.g., two create_folder nodes),
    falls back to parameter value matching for disambiguation.

    No token parsing. No substring matching. No heuristics.
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

        action = intent.get("action", "").lower()
        if not action:
            return None

        # Find all nodes whose skill action matches the intent action
        action_matches = []
        for node in candidate_nodes:
            # Extract action from skill name: "system.set_volume" → "set_volume"
            node_action = (
                node.skill.split(".", 1)[1] if "." in node.skill else node.skill
            )
            if node_action.lower() == action:
                action_matches.append(node)

        # Subsumption fallback: check both directions.
        #
        # Forward: intent is broad, plan is specific.
        #   Example: decomposer says "autonomous_task", compiler produced "search"
        #   → intent "autonomous_task" subsumes node "search" ✓
        #
        # Reverse: intent is specific, plan is broad.
        #   Example: decomposer says "search", compiler produced "autonomous_task"
        #   → node "autonomous_task" subsumes intent "search" ✓
        if not action_matches:
            # Forward: intent action subsumes node action
            subsumable = _ACTION_SUBSUMPTION.get(action, set())
            if subsumable:
                for node in candidate_nodes:
                    node_action = (
                        node.skill.split(".", 1)[1]
                        if "." in node.skill else node.skill
                    )
                    if node_action.lower() in subsumable:
                        action_matches.append(node)
                if action_matches:
                    _coverage_logger.debug(
                        "Intent [%s] matched via forward subsumption → node '%s'",
                        action, action_matches[0].id,
                    )

            # Reverse: node action subsumes intent action
            if not action_matches:
                for node in candidate_nodes:
                    node_action = (
                        node.skill.split(".", 1)[1]
                        if "." in node.skill else node.skill
                    ).lower()
                    node_subsumable = _ACTION_SUBSUMPTION.get(node_action, set())
                    if action in node_subsumable:
                        action_matches.append(node)
                if action_matches:
                    _coverage_logger.debug(
                        "Intent [%s] matched via reverse subsumption → node '%s'",
                        action, action_matches[0].id,
                    )

        if not action_matches:
            return None

        # Single match — done
        if len(action_matches) == 1:
            return action_matches[0].id

        # Multiple matches (e.g., two create_folder nodes) — disambiguate by parameters
        params = intent.get("parameters", {})
        if not params:
            # No parameters to disambiguate — return first match
            return action_matches[0].id

        # Score each candidate by parameter value overlap
        param_values = {str(v).lower() for v in params.values() if v != ""}
        best_node = action_matches[0]
        best_overlap = 0

        for node in action_matches:
            node_values = {
                str(v).lower()
                for v in node.inputs.values()
                if not hasattr(v, 'node')  # Skip OutputReference
            }
            overlap = len(param_values & node_values)
            if overlap > best_overlap:
                best_overlap = overlap
                best_node = node

        return best_node.id

    @staticmethod
    def _node_domain(node: MissionNode) -> str:
        """Extract domain from node's skill name."""
        return node.skill.split(".")[0].lower() if "." in node.skill else ""


# Keep backward-compatible alias
HeuristicIntentMatcher = CapabilityIntentMatcher


def verify_intent_coverage(
    plan: MissionPlan,
    intent_units: List[Dict[str, str]],
    registry: Any,
    matcher: Optional[IntentMatcher] = None,
) -> Tuple[bool, List[Dict[str, str]]]:
    """Verify that a compiled MissionPlan covers all declared intent units.

    This is a CAPABILITY coverage check, separate from structural validation.
    Called only for Tier 2+ queries (after decomposition).

    Args:
        plan: The compiled MissionPlan to verify.
        intent_units: The INJECTED intent set (≤8, post-cap).
            Verification and injection operate on the SAME set.
        registry: SkillRegistry for contract access.
        matcher: IntentMatcher implementation. Defaults to CapabilityIntentMatcher.

    Returns:
        (all_covered, uncovered_intents):
            all_covered: True if every intent maps to a DAG node.
            uncovered_intents: List of intent dicts that have no matching node.

    Design rules:
    - Each DAG node covers at most ONE intent (no double-counting).
    - Matching is capability-based: intent.action == node skill action.
    """
    if matcher is None:
        matcher = CapabilityIntentMatcher()

    # Available nodes pool — consumed nodes are removed
    available_nodes: List[MissionNode] = list(plan.nodes)
    uncovered: List[Dict[str, str]] = []

    for intent in intent_units:
        matched_id = matcher.match(intent, available_nodes, registry)

        if matched_id is not None:
            # Consume the matched node — cannot cover another intent
            available_nodes = [n for n in available_nodes if n.id != matched_id]
            _coverage_logger.debug(
                "Intent [%s] → covered by node '%s'",
                intent.get("action", ""), matched_id,
            )
        else:
            uncovered.append(intent)
            _coverage_logger.info(
                "Intent [%s] → UNCOVERED (no matching node)",
                intent.get("action", ""),
            )

    all_covered = len(uncovered) == 0

    _coverage_logger.info(
        "Coverage result: %d/%d intents covered, %d uncovered",
        len(intent_units) - len(uncovered),
        len(intent_units),
        len(uncovered),
    )

    return all_covered, uncovered
