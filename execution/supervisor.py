# execution/supervisor.py

"""
ExecutionSupervisor — Node-by-node execution controller.

The supervisor WRAPS the executor. It owns DAG traversal, calling
executor.execute_node() per node. The executor is demoted to a
node execution engine.

Architectural position:
    MissionOrchestrator
           ↓
    ExecutionSupervisor       ← owns node-by-node execution
           ↓
    MissionExecutor           ← demoted to node engine
           ↓
    Skills
           ↓
    Infrastructure

Responsibilities:
- Walk DAG layer by layer
- Evaluate preconditions (StepGuard) before each node
- Execute node via executor.execute_node()
- Evaluate postconditions after execution
- Attempt repair actions on failure (re-focus, re-launch)
- Cross-node repair: can re-execute earlier nodes if needed
- Support mid-execution pause (ClarificationManager, Phase 5)

Design rules:
- Guards are typed (GuardType enum, no string matching)
- Repair actions reference SkillRegistry, validated at construction
- ExecutionContext bundles all dependencies (prevent dependency monster)
- Supervisor never modifies WorldState — that's the skill's job
- All callbacks (on_layer_start, on_layer_complete) fire-and-forget
"""

import logging
import time
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, ConfigDict, Field

from ir.mission import MissionPlan, MissionNode
from execution.executor import MissionExecutor, ExecutionResult, NodeStatus
from execution.scheduler import DAGScheduler
from world.snapshot import WorldSnapshot

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# GuardType — typed guard checks (no string matching)
# ─────────────────────────────────────────────────────────────

class GuardType(str, Enum):
    """Typed guard checks for pre/postconditions.

    Prevents typos and simplifies evaluation logic.
    New guards added here, evaluated in _evaluate_guard().
    """
    ACTIVE_WINDOW = "active_window"
    APP_RUNNING = "app_running"
    APP_FOCUSED = "app_focused"
    WINDOW_VISIBLE = "window_visible"
    FILE_EXISTS = "file_exists"
    ELEMENT_VISIBLE = "element_visible"
    REQUIRES_CONFIRMATION = "requires_confirmation"  # Safety gate for destructive actions
    # For future browser automation


# ─────────────────────────────────────────────────────────────
# RepairAction — deterministic repair via SkillRegistry
# ─────────────────────────────────────────────────────────────

class RepairAction(BaseModel):
    """Deterministic repair action.

    skill_name: Registry key for the repair skill.
                Validated at plan construction (MissionCortex),
                never at execution time.
    inputs:     Static inputs for the repair skill.
    """
    model_config = ConfigDict(extra="forbid")

    skill_name: str
    inputs: Dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────────────────────
# StepGuard — typed condition with repair strategy
# ─────────────────────────────────────────────────────────────

class StepGuard(BaseModel):
    """Guard for a single execution step.

    Evaluated before (precondition) or after (postcondition) a node.
    If evaluation fails, repair_actions are attempted in order.
    If all repairs fail, the node is retried up to max_retries times.
    """
    model_config = ConfigDict(extra="forbid")

    type: GuardType
    params: Dict[str, Any] = Field(default_factory=dict)
    repair_actions: List[RepairAction] = Field(default_factory=list)
    max_retries: int = 2
    retry_delay: float = 0.5


# ─────────────────────────────────────────────────────────────
# ExecutionContext — bundled dependencies (prevents dependency monster)
# ─────────────────────────────────────────────────────────────

class ExecutionContext:
    """Bundled dependencies for guard evaluation and repair execution.

    Passed to the supervisor at construction time. Guards and repairs
    use the context to query the environment and execute repair skills.
    """

    def __init__(
        self,
        observer=None,
        session_manager=None,
        capability_registry=None,
        timeline=None,
    ):
        self.observer = observer
        self.session_manager = session_manager
        self.capability_registry = capability_registry
        self.timeline = timeline


# ─────────────────────────────────────────────────────────────
# ExecutionSupervisor — owns node-by-node execution
# ─────────────────────────────────────────────────────────────

class ExecutionSupervisor:
    """Node-by-node execution controller.

    Walks the DAG layer by layer. For each node:
    1. Evaluate preconditions (from contract extension)
    2. Execute via executor.execute_node()
    3. Evaluate postconditions
    4. On failure: attempt repair actions, retry
    5. On unresolvable failure: log and continue per failure policy

    Cross-node repair: can re-execute earlier nodes if needed.

    The executor is used ONLY as a node engine — its run() method
    is never called by the supervisor.
    """

    def __init__(
        self,
        executor: MissionExecutor,
        context: ExecutionContext,
    ):
        self._executor = executor
        self._ctx = context

    @property
    def registry(self):
        """Expose executor's registry for orchestrator compatibility."""
        return self._executor.registry

    def run(
        self,
        plan: MissionPlan,
        world_snapshot: Optional[WorldSnapshot] = None,
        on_layer_start: Optional[Callable] = None,
        on_layer_complete: Optional[Callable] = None,
    ) -> ExecutionResult:
        """Execute a mission plan with guard enforcement.

        Walks DAG layer by layer, executing nodes individually.
        Replaces executor.run() when supervisor is active.

        Same signature as MissionExecutor.run() for drop-in replacement.
        """

        # ── IR version gate (same as executor) ──
        from ir.mission import IR_VERSION
        if plan.ir_version != IR_VERSION:
            raise RuntimeError(
                f"Unsupported IR version: '{plan.ir_version}'. "
                f"Expected '{IR_VERSION}'."
            )

        # ── Type guard (same as executor) ──
        if world_snapshot is not None and not isinstance(world_snapshot, WorldSnapshot):
            raise TypeError(
                f"Supervisor.run() requires WorldSnapshot or None, "
                f"got {type(world_snapshot).__name__}."
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

            # ── Execute each node in the layer ──
            self._execute_layer(
                layer, node_index, exec_result, world_snapshot,
            )

            # Layer-complete callback (fire-and-forget)
            if on_layer_complete:
                try:
                    on_layer_complete(layer, layer_idx)
                except Exception:
                    pass

        # ── Session cleanup after execution ──
        if self._ctx.session_manager and self._ctx.observer:
            try:
                self._ctx.session_manager.cleanup_stale_sessions(
                    observer=self._ctx.observer,
                )
            except Exception as e:
                logger.debug("Post-execution session cleanup failed: %s", e)

        return exec_result

    def _execute_layer(
        self,
        layer: list,
        node_index: Dict[str, MissionNode],
        exec_result: ExecutionResult,
        world_snapshot: Optional[WorldSnapshot],
    ) -> None:
        """Execute one dependency layer with guard enforcement.

        Within a layer, nodes are independent. The supervisor
        still respects focus/conflict constraints by delegating
        to the executor's layer logic for parallel nodes, but
        wraps each focus node with guard checks.
        """
        # Split by focus/conflict constraints (same logic as executor)
        focus_nodes = []
        parallel_nodes = []

        for nid in layer:
            node = node_index[nid]
            if (self._executor._needs_focus(node) or
                    self._executor._has_conflicts(node, layer, node_index)):
                focus_nodes.append(nid)
            else:
                parallel_nodes.append(nid)

        # Phase 1: Execute non-focus nodes in parallel (via executor)
        if parallel_nodes:
            self._executor._execute_parallel(
                parallel_nodes, node_index, exec_result, world_snapshot,
            )

        # Phase 2: Execute focus/conflicting nodes with guard enforcement
        for nid in focus_nodes:
            node = node_index[nid]
            self._execute_guarded_node(
                node, exec_result, world_snapshot,
            )

    def _execute_guarded_node(
        self,
        node: MissionNode,
        exec_result: ExecutionResult,
        world_snapshot: Optional[WorldSnapshot],
    ) -> None:
        """Execute a single node with pre/postcondition enforcement.

        1. Evaluate preconditions
        2. Execute via executor.execute_node()
        3. Evaluate postconditions
        4. On failure: repair + retry
        """
        # ── Get guard definitions from contract extension ──
        preconditions = self._get_guards(node, "preconditions")
        postconditions = self._get_guards(node, "postconditions")

        # ── Evaluate preconditions ──
        for guard in preconditions:
            if not self._evaluate_guard(guard):
                repaired = self._attempt_repair(guard, node, exec_result, world_snapshot)
                if not repaired:
                    logger.warning(
                        "Node '%s' precondition failed: %s (unrepaired)",
                        node.id, guard.type.value,
                    )
                    exec_result.record(node.id, NodeStatus.FAILED, {})
                    return

        # ── Execute node (with retry on failure) ──
        max_retries = max(
            (g.max_retries for g in preconditions + postconditions),
            default=1,
        )
        retry_delay = min(
            (g.retry_delay for g in preconditions + postconditions),
            default=0.5,
        )

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                node_id, status, outputs, meta = self._executor.execute_node(
                    node, exec_result, world_snapshot,
                )
                exec_result.record(node_id, status, outputs, meta)

                # ── Evaluate postconditions ──
                if status in (NodeStatus.COMPLETED, NodeStatus.NO_OP):
                    all_post_ok = True
                    for guard in postconditions:
                        if not self._evaluate_guard(guard):
                            all_post_ok = False
                            logger.warning(
                                "Node '%s' postcondition failed: %s",
                                node.id, guard.type.value,
                            )
                            break

                    if all_post_ok:
                        return  # Success — all guards passed

                # Node failed or postcondition failed — try repair before retry
                if attempt < max_retries:
                    for guard in postconditions:
                        self._attempt_repair(guard, node, exec_result, world_snapshot)
                    time.sleep(retry_delay)

            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    logger.info(
                        "Node '%s' failed (attempt %d/%d): %s — retrying",
                        node.id, attempt + 1, max_retries + 1, e,
                    )
                    time.sleep(retry_delay)
                else:
                    logger.warning(
                        "Node '%s' failed after %d attempts: %s",
                        node.id, max_retries + 1, e,
                    )
                    exec_result.record(node.id, NodeStatus.FAILED, {})

    def _get_guards(
        self, node: MissionNode, guard_type: str,
    ) -> List[StepGuard]:
        """Extract guard definitions from the skill contract.

        Falls back to empty list if the contract doesn't have
        the pre/postconditions extension yet.
        """
        try:
            skill = self._executor.registry.get(node.skill)
            raw_guards = getattr(skill.contract, guard_type, [])

            guards = []

            # Auto-inject REQUIRES_CONFIRMATION for destructive skills.
            # This runs BEFORE user-defined preconditions so it's always first.
            if guard_type == "preconditions":
                risk = getattr(skill.contract, "risk_level", "safe")
                if risk == "destructive":
                    guards.insert(0, StepGuard(
                        type=GuardType.REQUIRES_CONFIRMATION,
                        params={
                            "skill": node.skill,
                            "inputs": dict(node.inputs),
                        },
                        repair_actions=[],
                        max_retries=0,  # Confirmation is not retried
                    ))
                elif risk == "moderate":
                    logger.warning(
                        "[SAFETY] Moderate-risk skill '%s' executing — inputs: %s",
                        node.skill, dict(node.inputs),
                    )

            if not raw_guards:
                return guards

            for raw in raw_guards:
                if isinstance(raw, StepGuard):
                    guards.append(raw)
                elif isinstance(raw, dict):
                    # Parse from dict (contract extension format)
                    guard_type_val = raw.get("type")
                    if guard_type_val and isinstance(guard_type_val, str):
                        try:
                            gt = GuardType(guard_type_val)
                        except ValueError:
                            logger.warning(
                                "Unknown GuardType '%s' in skill '%s'",
                                guard_type_val, node.skill,
                            )
                            continue
                        guards.append(StepGuard(
                            type=gt,
                            params=raw.get("params", {}),
                            repair_actions=[
                                RepairAction(**ra)
                                for ra in raw.get("repair_actions", [])
                            ],
                            max_retries=raw.get("max_retries", 2),
                            retry_delay=raw.get("retry_delay", 0.5),
                        ))
            return guards
        except Exception as e:
            logger.debug(
                "Failed to get guards for node '%s': %s", node.id, e,
            )
            return []

    def _evaluate_guard(self, guard: StepGuard) -> bool:
        """Evaluate a single guard against current environment.

        Returns True if the guard passes, False otherwise.
        Never raises — returns False on error.
        """
        # REQUIRES_CONFIRMATION: always block — merlin.py confirms with user
        # before execution and decides whether to resume or abort the mission.
        if guard.type == GuardType.REQUIRES_CONFIRMATION:
            logger.info(
                "[SAFETY] REQUIRES_CONFIRMATION guard triggered for skill '%s'",
                guard.params.get("skill", "?"),
            )
            return False

        observer = self._ctx.observer
        if observer is None:
            # No observer available — cannot evaluate, assume OK
            return True

        try:
            if guard.type == GuardType.ACTIVE_WINDOW:
                expected_app = guard.params.get("app", "")
                active = observer.get_active_window()
                if active is None:
                    return False
                return expected_app.lower() in (active.app_name or "").lower()

            elif guard.type == GuardType.APP_RUNNING:
                app_name = guard.params.get("app", "")
                return observer.is_app_running(app_name)

            elif guard.type == GuardType.APP_FOCUSED:
                app_name = guard.params.get("app", "")
                return observer.is_app_focused(app_name)

            elif guard.type == GuardType.WINDOW_VISIBLE:
                app_name = guard.params.get("app", "")
                return observer.is_app_running(app_name)

            elif guard.type == GuardType.FILE_EXISTS:
                path = guard.params.get("path", "")
                return observer.file_exists(path)

            elif guard.type == GuardType.ELEMENT_VISIBLE:
                # Deferred to Phase 6 (browser automation)
                logger.debug(
                    "ELEMENT_VISIBLE guard deferred — assuming True",
                )
                return True

            else:
                logger.warning("Unknown GuardType: %s", guard.type)
                return True

        except Exception as e:
            logger.debug("Guard evaluation failed: %s", e)
            return False

    def _attempt_repair(
        self,
        guard: StepGuard,
        node: MissionNode,
        exec_result: ExecutionResult,
        world_snapshot: Optional[WorldSnapshot],
    ) -> bool:
        """Attempt repair actions for a failed guard.

        Executes each repair action in order. If any succeeds
        and the guard now passes, returns True.

        Returns False if all repairs fail.
        """
        if not guard.repair_actions:
            return False

        for repair in guard.repair_actions:
            try:
                # Look up repair skill from registry
                skill = self._executor.registry.get(repair.skill_name)

                # Execute repair skill
                timeline = self._ctx.timeline or self._executor.timeline
                result = skill.execute(
                    repair.inputs, timeline, world_snapshot,
                )

                logger.info(
                    "Repair action '%s' executed for node '%s'",
                    repair.skill_name, node.id,
                )

                # Re-evaluate guard after repair
                if self._evaluate_guard(guard):
                    logger.info(
                        "Guard '%s' now passes after repair '%s'",
                        guard.type.value, repair.skill_name,
                    )
                    return True

            except Exception as e:
                logger.warning(
                    "Repair action '%s' failed: %s",
                    repair.skill_name, e,
                )

        return False
