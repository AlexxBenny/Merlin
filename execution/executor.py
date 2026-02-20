# execution/executor.py

"""
MissionExecutor — execute MissionPlan DAGs with IR v1 semantics.

Phase 3B: Uses DAGScheduler for topological planning,
ThreadPoolExecutor for parallel compute within layers.

Enforces:
- IR version gate
- Dependency order (via scheduler layers)
- Condition evaluation (skip semantics)
- Skill contract: allowed_modes, failure_policy, emits_events, mutates_world
- OutputReference-only input resolution
- $-string defense-in-depth rejection
"""

import logging
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_EXCEPTION
from typing import Any, Callable, Dict, Optional, Set, Tuple

from ir.mission import (
    IR_VERSION,
    ExecutionMode,
    MissionPlan,
    MissionNode,
    OutputReference,
    ConditionExpr,
)
from skills.contract import FailurePolicy
from skills.skill_result import SkillResult
from world.timeline import WorldTimeline
from world.snapshot import WorldSnapshot

# Optional: COM threading support (Windows)
_HAS_COMTYPES = False
try:
    import comtypes
    _HAS_COMTYPES = True
except ImportError:
    pass
from .registry import SkillRegistry
from .scheduler import DAGScheduler


logger = logging.getLogger(__name__)


# ---------------------------
# Node execution status
# ---------------------------
# Centralized enum — every layer references this, never raw strings.
# To add a new status (e.g. partial_success, blocked, policy_denied):
#   1. Add the value here
#   2. Update ExecutionResult.record() routing
#   3. Update ReportBuilder._STATUS_LABELS mapping
# No other file should hardcode status strings.

class NodeStatus(str, Enum):
    """Semantic execution outcome for a single node.

    COMPLETED:  skill ran and produced useful outputs
    NO_OP:      skill ran but made no change (e.g. already_playing,
                no_media_session).  Still 'completed' from DAG
                perspective — downstream depends_on is not blocked.
    SKIPPED:    node was never executed (condition_on / dependency)
    FAILED:     skill raised an exception
    TIMED_OUT:  skill exceeded its timeout

    Future values may include: partial_success, blocked,
    policy_denied, retry_scheduled.

    NOTE: NO_OP is currently *inferred* from outputs.changed=False +
    metadata.reason.  Long-term, skills should signal semantic_status
    explicitly via SkillResult.  See skills/skill_result.py.
    """
    COMPLETED = "completed"
    NO_OP = "no_op"
    SKIPPED = "skipped"
    FAILED = "failed"
    TIMED_OUT = "timed_out"


class ExecutionResult:
    """
    Typed result of executor.run().

    Carries:
    - results: node_id → outputs (only completed nodes)
    - node_statuses: node_id → NodeStatus string
    - completed/skipped/failed sets for efficient lookup

    This replaces the bare Dict[str, Any] return.
    Consumers that only need outputs can use .results directly.
    MissionOutcome builders use .node_statuses for typed failure states.
    """

    __slots__ = ("results", "metadata", "node_statuses", "completed", "skipped", "failed")

    def __init__(self):
        self.results: Dict[str, Dict[str, Any]] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}  # node_id → side-channel metadata
        self.node_statuses: Dict[str, str] = {}
        self.completed: Set[str] = set()
        self.skipped: Set[str] = set()
        self.failed: Set[str] = set()  # includes TIMED_OUT

    def record(
        self,
        node_id: str,
        status: NodeStatus,
        outputs: Dict[str, Any],
        metadata: Dict[str, Any] | None = None,
    ) -> None:
        """Record a single node's execution result."""
        self.node_statuses[node_id] = status
        if status in (NodeStatus.COMPLETED, NodeStatus.NO_OP):
            # NO_OP is still "completed" from the DAG scheduler's
            # perspective — downstream depends_on is not blocked.
            # The distinction is purely semantic, for reporting.
            self.results[node_id] = outputs
            self.completed.add(node_id)
            if metadata:
                self.metadata[node_id] = metadata
        elif status == NodeStatus.SKIPPED:
            self.skipped.add(node_id)
        elif status in (NodeStatus.FAILED, NodeStatus.TIMED_OUT):
            self.failed.add(node_id)


class MissionExecutor:
    def __init__(
        self,
        registry: SkillRegistry,
        timeline: WorldTimeline,
        max_workers: int = 4,
        node_timeout: Optional[float] = None,
    ):
        self.registry = registry
        self.timeline = timeline
        self.max_workers = max_workers
        self.node_timeout = node_timeout  # seconds, None = no timeout

    def _resolve_input(
        self,
        raw_value: Any,
        results: Dict[str, Dict[str, Any]],
    ) -> Tuple[bool, Any, str]:
        """
        Resolve an individual input value.

        Only two legal forms:
        - OutputReference -> resolved from prior node results
        - Literal value -> passed through unchanged

        $-string rejection: enforcement point 2 of 2 (defense-in-depth).
        """

        # Defense-in-depth: reject $-prefixed strings at runtime
        # even if they somehow passed IR model validation
        if isinstance(raw_value, str) and raw_value.startswith("$"):
            return (
                False,
                None,
                f"Banned $-prefixed string '{raw_value}' "
                f"reached executor. Use OutputReference.",
            )

        # OutputReference resolution
        if isinstance(raw_value, OutputReference):
            node_id = raw_value.node
            out_key = raw_value.output

            if node_id not in results:
                return (
                    False,
                    None,
                    f"Referenced node '{node_id}' has not "
                    f"produced outputs yet",
                )

            node_outputs = results[node_id]
            if out_key not in node_outputs:
                return (
                    False,
                    None,
                    f"Referenced output '{out_key}' missing "
                    f"on node '{node_id}'",
                )

            return True, node_outputs[out_key], ""

        # Literal value — pass through
        return True, raw_value, ""

    def _evaluate_condition(
        self,
        condition: ConditionExpr,
        results: Dict[str, Dict[str, Any]],
        world_snapshot: Optional[WorldSnapshot] = None,
    ) -> bool:
        """
        Evaluate a ConditionExpr. Returns True if condition is met.

        Evaluated once at node scheduling time.
        """
        source = condition.source

        if source.startswith("world."):
            # Resolve from world snapshot
            if world_snapshot is None:
                return False
            # Convert to dict once for dotted-path navigation
            # This is the ONLY place dict access is used — read-only
            state_dict = world_snapshot.state.model_dump()
            keys = source.split(".")[1:]  # strip "world." prefix
            current = state_dict
            for key in keys:
                if isinstance(current, dict) and key in current:
                    current = current[key]
                else:
                    return False
            return current == condition.equals

        # Resolve from node results
        if source not in results:
            return False

        node_outputs = results[source]
        # Check if any output matches
        return any(v == condition.equals for v in node_outputs.values())

    def _execute_node(
        self,
        node: MissionNode,
        exec_result: 'ExecutionResult',
        world_snapshot: Optional[WorldSnapshot],
    ) -> Tuple[str, str, Dict[str, Any], Dict[str, Any] | None]:
        """
        Execute a single node. Returns (node_id, status, outputs, metadata).

        This method is called from within ThreadPoolExecutor.
        It must NOT modify shared state except through the return value.
        World mutations still happen inline (serialized via WorldTimeline._lock).
        """
        # Check if dependency was skipped/failed
        skipped_or_failed_deps = [
            d for d in node.depends_on
            if d in exec_result.skipped or d in exec_result.failed
        ]
        if skipped_or_failed_deps:
            logger.info(
                "Node '%s' skipped: dependency %s was skipped/failed",
                node.id, skipped_or_failed_deps,
            )
            return node.id, NodeStatus.SKIPPED, {}, None

        # Evaluate condition (if present)
        if node.condition_on is not None:
            if not self._evaluate_condition(
                node.condition_on, exec_result.results, world_snapshot
            ):
                logger.info(
                    "Node '%s' skipped: condition not met "
                    "(source='%s', expected=%r)",
                    node.id, node.condition_on.source,
                    node.condition_on.equals,
                )
                return node.id, NodeStatus.SKIPPED, {}, None

        # Resolve inputs
        inputs_for_skill: Dict[str, Any] = {}
        for key, raw_val in node.inputs.items():
            ok, resolved, err = self._resolve_input(raw_val, exec_result.results)
            if not ok:
                raise RuntimeError(
                    f"Node '{node.id}' input resolution failed: {key}: {err}"
                )
            inputs_for_skill[key] = resolved

        logger.info(
            "[TRACE] Node '%s' (skill=%s) resolved inputs: %r",
            node.id, node.skill, inputs_for_skill,
        )

        # Fetch skill
        try:
            skill = self.registry.get(node.skill)
        except KeyError as e:
            raise RuntimeError(
                f"Missing skill '{node.skill}' for node '{node.id}'"
            ) from e

        # -------------------------------------------------------
        # SKILL CONTRACT ENFORCEMENT
        # -------------------------------------------------------

        # 1. Allowed mode check
        if node.mode not in skill.contract.allowed_modes:
            raise RuntimeError(
                f"Node '{node.id}' uses mode '{node.mode.value}' "
                f"but skill '{skill.name}' only allows: "
                f"{sorted(m.value for m in skill.contract.allowed_modes)}"
            )

        # 2. Input key validation — from contract
        declared_inputs = skill.input_keys
        extra_inputs = set(inputs_for_skill.keys()) - declared_inputs
        if extra_inputs:
            raise RuntimeError(
                f"Node '{node.id}' provides inputs not "
                f"declared by skill '{skill.name}': {extra_inputs}"
            )

        # Snapshot timeline event count before execution
        # (for emits_events / mutates_world enforcement)
        # Uses atomic event_count() for thread safety under parallel compute
        events_before = self.timeline.event_count()

        # 3. Execute skill with contract-aware failure semantics
        try:
            raw_result = skill.execute(inputs_for_skill, self.timeline, world_snapshot)
            if not isinstance(raw_result, SkillResult):
                raise RuntimeError(
                    f"Skill '{skill.name}' returned {type(raw_result).__name__} "
                    f"instead of SkillResult for node '{node.id}'"
                )
            outputs = raw_result.outputs
            metadata = raw_result.metadata
        except Exception as e:
            # Read failure policy from contract
            policy = skill.contract.failure_policy.get(
                node.mode, FailurePolicy.FAIL
            )

            if policy == FailurePolicy.FAIL:
                raise RuntimeError(
                    f"Node '{node.id}' (mode={node.mode.value}) "
                    f"failed with policy=FAIL: {e}"
                ) from e
            elif policy == FailurePolicy.CONTINUE:
                logger.warning(
                    "Node '%s' (mode=%s) failed with policy=CONTINUE: %s",
                    node.id, node.mode.value, e,
                )
                return node.id, NodeStatus.FAILED, {}, None
            else:  # IGNORE
                logger.debug(
                    "Node '%s' (mode=%s) failed with policy=IGNORE: %s",
                    node.id, node.mode.value, e,
                )
                return node.id, NodeStatus.FAILED, {}, None

        # 4. Validate outputs against contract
        declared_outputs = skill.output_keys
        if not set(outputs.keys()).issubset(declared_outputs):
            undeclared = set(outputs.keys()) - declared_outputs
            raise RuntimeError(
                f"Skill '{skill.name}' returned undeclared "
                f"outputs for node '{node.id}': {undeclared}"
            )

        # 5. Event emission enforcement (thread-safe via events_since_index)
        #
        # CRITICAL: Filter to skill-sourced events only.
        # Background sources (time, system, media) emit concurrently
        # to the same timeline on the RuntimeEventLoop daemon thread.
        # Without filtering, those events cause false contract violations.
        #
        # Convention: skills emit with source=contract.name (e.g. "system.set_brightness").
        # Background sources use bare names ("time", "system", "media") — no dot,
        # so namespaces never collide.
        #
        # We do NOT drop or suppress background events from the timeline.
        # They remain for WorldState building, proactive logic, and goal evaluation.
        new_events = self.timeline.events_since_index(events_before)
        skill_source = skill.contract.name
        skill_events = [e for e in new_events if e.source == skill_source]

        if not skill.contract.mutates_world and skill_events:
            raise RuntimeError(
                f"Skill '{skill.name}' (mutates_world=False) "
                f"emitted {len(skill_events)} event(s) during "
                f"node '{node.id}'"
            )

        if skill.contract.emits_events:
            declared_event_types = set(skill.contract.emits_events)
            for ev in skill_events:
                if ev.type not in declared_event_types:
                    raise RuntimeError(
                        f"Skill '{skill.name}' emitted undeclared "
                        f"event type '{ev.type}' during node "
                        f"'{node.id}'. Declared: "
                        f"{declared_event_types}"
                    )
        elif skill_events:
            # No events declared but skill-sourced events were emitted
            raise RuntimeError(
                f"Skill '{skill.name}' emitted events but "
                f"declares emits_events=[] for node '{node.id}'"
            )

        # ── Derive semantic status ──────────────────────────
        # If the skill reports no change AND provides a reason,
        # this is a semantic no-op (e.g. already_playing,
        # no_media_session).  Classified here — the single
        # place where execution semantics are determined.
        #
        # Future: skills may signal semantic_status explicitly
        # via SkillResult, replacing this inference.
        if outputs.get("changed") is False and metadata.get("reason"):
            return node.id, NodeStatus.NO_OP, outputs, metadata

        return node.id, NodeStatus.COMPLETED, outputs, metadata

    def run(
        self,
        plan: MissionPlan,
        world_snapshot: Optional[WorldSnapshot] = None,
        on_layer_start: Optional[Callable] = None,
        on_layer_complete: Optional[Callable] = None,
    ) -> 'ExecutionResult':
        """
        Execute a mission DAG with IR v1 semantics + Skill Contract v1 enforcement.

        Uses DAGScheduler for topological planning.
        Within each layer, nodes execute concurrently via ThreadPoolExecutor.
        Between layers, all nodes must complete before proceeding.

        Args:
            plan: Immutable mission plan to execute.
            world_snapshot: Typed WorldSnapshot — the ONLY accepted type.
                            Never pass dict, model_dump(), or partial state.
                            Condition evaluation converts internally.
            on_layer_start: Optional callback(layer_ids, node_index, layer_idx, total_layers).
                            Fires BEFORE each layer executes. Fire-and-forget.
            on_layer_complete: Optional callback(layer_ids, layer_idx).
                            Fires AFTER each layer completes. Fire-and-forget.

        Returns ExecutionResult with typed per-node status.
        """

        # -------------------------------------------------------
        # RUNTIME TYPE GUARD — Non-negotiable
        # -------------------------------------------------------
        if world_snapshot is not None and not isinstance(world_snapshot, WorldSnapshot):
            raise TypeError(
                f"Executor.run() requires WorldSnapshot or None, "
                f"got {type(world_snapshot).__name__}. "
                f"Never pass dict or model_dump() to execution logic."
            )

        # -------------------------------------------------------
        # IR VERSION GATE — Non-negotiable
        # -------------------------------------------------------
        if plan.ir_version != IR_VERSION:
            raise RuntimeError(
                f"Unsupported IR version: '{plan.ir_version}'. "
                f"Expected '{IR_VERSION}'."
            )

        # Plan execution layers
        layers = DAGScheduler.plan(plan)
        node_index = DAGScheduler.get_node_index(plan)

        exec_result = ExecutionResult()

        for layer_idx, layer in enumerate(layers):
            # Layer-start callback (narration hook — fire-and-forget)
            if on_layer_start:
                try:
                    on_layer_start(layer, node_index, layer_idx, len(layers))
                except Exception:
                    pass  # Narration failure never blocks execution

            self._execute_layer(
                layer, node_index, exec_result, world_snapshot,
            )

            # Layer-complete callback (fire-and-forget)
            if on_layer_complete:
                try:
                    on_layer_complete(layer, layer_idx)
                except Exception:
                    pass

        return exec_result

    def _execute_layer(
        self,
        layer: list,
        node_index: Dict[str, MissionNode],
        exec_result: 'ExecutionResult',
        world_snapshot: Optional[WorldSnapshot],
    ) -> None:
        """
        Execute one dependency layer.

        Nodes in the same layer are independent and run concurrently,
        EXCEPT:
        - Nodes with requires_focus=True are serialized (focus safety)
        - Nodes listed in each other's conflicts_with run sequentially

        After timeout, the pool is force-shutdown (wait=False, cancel_futures=True)
        so the executor never blocks on hung threads.
        """
        # ── Split by focus/conflict constraints ──
        focus_nodes = []
        parallel_nodes = []

        for nid in layer:
            node = node_index[nid]
            if self._needs_focus(node) or self._has_conflicts(node, layer, node_index):
                focus_nodes.append(nid)
            else:
                parallel_nodes.append(nid)

        # ── Phase 1: Execute non-focus nodes in parallel ──
        if parallel_nodes:
            self._execute_parallel(parallel_nodes, node_index, exec_result, world_snapshot)

        # ── Phase 2: Execute focus/conflicting nodes sequentially ──
        for nid in focus_nodes:
            node = node_index[nid]
            try:
                node_id, status, outputs, meta = self._com_safe_execute(
                    node, exec_result, world_snapshot,
                )
                exec_result.record(node_id, status, outputs, meta)
            except Exception:
                raise

    def _execute_parallel(
        self,
        node_ids: list,
        node_index: Dict[str, MissionNode],
        exec_result: 'ExecutionResult',
        world_snapshot: Optional[WorldSnapshot],
    ) -> None:
        """Execute a batch of nodes concurrently via ThreadPoolExecutor."""
        workers = min(self.max_workers, len(node_ids))
        pool = ThreadPoolExecutor(max_workers=workers)

        try:
            futures = {}
            for nid in node_ids:
                node = node_index[nid]
                future = pool.submit(
                    self._com_safe_execute,
                    node, exec_result, world_snapshot,
                )
                futures[future] = nid

            for future in as_completed(futures, timeout=self.node_timeout):
                nid = futures[future]
                try:
                    node_id, status, outputs, meta = future.result(timeout=0)
                except Exception as e:
                    # Unexpected exception from _execute_node propagates
                    raise
                exec_result.record(node_id, status, outputs, meta)

        except TimeoutError:
            # Some futures didn't complete within timeout
            for future, nid in futures.items():
                if not future.done():
                    logger.warning(
                        "Node '%s' timed out after %s seconds",
                        nid, self.node_timeout,
                    )
                    future.cancel()
                    exec_result.record(nid, NodeStatus.TIMED_OUT, {})

        finally:
            # Force-shutdown: don't wait for hung threads
            pool.shutdown(wait=False, cancel_futures=True)

    def _com_safe_execute(
        self,
        node: MissionNode,
        exec_result: 'ExecutionResult',
        world_snapshot: Optional[WorldSnapshot],
    ) -> Tuple[str, str, Dict[str, Any], Dict[str, Any] | None]:
        """
        COM-aware wrapper around _execute_node.

        Worker threads need CoInitialize() for COM objects (pycaw, etc.).
        This keeps executor domain-agnostic — no skill knows about threading.
        """
        if _HAS_COMTYPES:
            comtypes.CoInitialize()
        try:
            return self._execute_node(node, exec_result, world_snapshot)
        finally:
            if _HAS_COMTYPES:
                comtypes.CoUninitialize()

    def _needs_focus(self, node: MissionNode) -> bool:
        """Check if a node's skill requires foreground window control."""
        skill = self.registry.get(node.skill)
        if skill and hasattr(skill, 'contract'):
            return skill.contract.requires_focus
        return False

    def _has_conflicts(
        self,
        node: MissionNode,
        layer: list,
        node_index: Dict[str, MissionNode],
    ) -> bool:
        """Check if a node conflicts with any other node in the same layer."""
        skill = self.registry.get(node.skill)
        if not skill or not hasattr(skill, 'contract'):
            return False

        conflicts = set(skill.contract.conflicts_with)
        if not conflicts:
            return False

        for other_nid in layer:
            if other_nid == node.id:
                continue
            other_node = node_index[other_nid]
            if other_node.skill in conflicts:
                return True

        return False
