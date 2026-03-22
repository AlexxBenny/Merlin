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

from ir.mission import MissionPlan, MissionNode, ExecutionMode
from execution.executor import MissionExecutor, ExecutionResult, NodeStatus
from execution.scheduler import DAGScheduler
from execution.metacognition import (
    MetaCognitionEngine, FailureVerdict, RecoveryAction,
    OutcomeAnalyzer, OutcomeSeverity,
)
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
    FILE_REFERENCE = "file_reference"      # File identity known (path resolved via search/list)
    ELEMENT_VISIBLE = "element_visible"
    REQUIRES_CONFIRMATION = "requires_confirmation"  # Safety gate for destructive actions
    MEDIA_SESSION_ACTIVE = "media_session_active"     # Active media session exists
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
        self._metacognition = MetaCognitionEngine()
        self._outcome_analyzer = OutcomeAnalyzer()
        self._verdicts: List[FailureVerdict] = []
        self._cognitive_ctx = None       # Set per-run via run()
        self._decision_engine = None     # Set per-run via run()

    @property
    def verdicts(self) -> List[FailureVerdict]:
        """Accumulated failure verdicts from this execution run."""
        return list(self._verdicts)

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
        cognitive_ctx=None,
        decision_engine=None,
        override_outputs=None,
    ) -> ExecutionResult:
        """Execute a mission plan with guard enforcement.

        Walks DAG layer by layer, executing nodes individually.
        Replaces executor.run() when supervisor is active.

        Args:
            cognitive_ctx: Optional CognitiveContext for assumption
                gating and uncertainty tracking. When None, all
                existing behavior is preserved unchanged.
            override_outputs: Optional dict of {node_id: outputs} for
                pre-completed nodes. Used for clarification resume:
                the user selected a match, so we inject it as the
                node's output and skip re-execution.  Downstream
                nodes resolve OutputReferences normally.
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

        # Store cognitive context for this run (None = backward-compat)
        self._cognitive_ctx = cognitive_ctx
        # DecisionEngine for inline recovery (None = disabled)
        self._decision_engine = decision_engine

        # Plan execution layers
        layers = DAGScheduler.plan(plan)
        node_index = DAGScheduler.get_node_index(plan)

        exec_result = ExecutionResult()
        self._verdicts = []  # Reset for new run

        # ── Pre-seed override outputs (clarification resume) ──
        # Nodes whose outputs are already known (e.g. user selected
        # a match from an ambiguous list) are recorded as COMPLETED
        # so downstream OutputReference resolution works normally.
        if override_outputs:
            for node_id, outputs in override_outputs.items():
                exec_result.record(
                    node_id, NodeStatus.COMPLETED, outputs,
                    metadata={"source": "clarification_resume"},
                )
                logger.info(
                    "[RESUME] Pre-seeded node '%s' with override outputs",
                    node_id,
                )

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

            # ── Meta-cognition: check if plan is fundamentally broken ──
            if self._metacognition.should_abort(self._verdicts):
                logger.warning(
                    "[META] Aborting execution after layer %d/%d — "
                    "plan is fundamentally broken (%d verdicts)",
                    layer_idx + 1, len(layers), len(self._verdicts),
                )
                break

        # Store verdicts on exec_result for upstream consumption
        exec_result.meta_verdicts = list(self._verdicts)

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
            # Skip nodes already completed via override_outputs
            if nid in exec_result.completed:
                logger.info(
                    "[RESUME] Skipping pre-completed node '%s'", nid,
                )
                continue
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
            # Classify outcomes for parallel nodes (executor skips guards)
            for nid in parallel_nodes:
                status = exec_result.node_statuses.get(nid)
                meta = exec_result.metadata.get(nid, {})
                if status is not None:
                    severity = self._outcome_analyzer.classify(status, meta)
                    if severity != OutcomeSeverity.BENIGN:
                        node = node_index[nid]
                        exec_result.outcome_verdicts.append({
                            "node_id": nid,
                            "skill": node.skill,
                            "status": str(status),
                            "reason": meta.get("reason", ""),
                            "severity": severity.value,
                        })
                        logger.info(
                            "[OUTCOME] Node '%s' (%s): %s → %s",
                            nid, node.skill, status, severity.value,
                        )

                        # ── Ambiguity: block dependent nodes ──
                        # When a node signals ambiguous_input, its outputs
                        # are NOT safe for downstream consumption.  Mark it
                        # as failed so the executor's cascade-skip (line 356)
                        # prevents dependents from executing with bad data.
                        # The orchestrator's recovery loop will convert the
                        # SOFT_FAILURE → EscalationDecision(USER) → ask-back.
                        if meta.get("reason") == "ambiguous_input":
                            exec_result.failed.add(nid)
                            logger.info(
                                "[OUTCOME] Node '%s' blocked dependents "
                                "(ambiguous_input)", nid,
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

        Order: Guard → Repair → Assumption → Execute → Postcondition

        1. Evaluate preconditions (existing StepGuard flow)
        2. Assumption gate (if CognitiveContext active)
        3. Execute via executor.execute_node()
        4. Evaluate postconditions
        5. On failure: repair + retry
        6. Uncertainty update (if CognitiveContext active)
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
                    # Meta-cognition: classify precondition failure
                    verdict = self._metacognition.classify(
                        node_id=node.id,
                        skill_name=node.skill,
                        failed_guards=[guard.type.value],
                        retries_exhausted=True,
                    )
                    # ── Inline recovery: try before giving up ──
                    if self._attempt_inline_recovery(
                        node, verdict, exec_result, world_snapshot,
                    ):
                        return self._execute_guarded_node(
                            node, exec_result, world_snapshot,
                        )
                    self._verdicts.append(verdict)
                    logger.info(
                        "[META] Node '%s' classified: %s → %s",
                        node.id, verdict.category.value, verdict.action.value,
                    )
                    return

        # ── Assumption gate (after preconditions, before execution) ──
        if self._cognitive_ctx:
            exec_state = self._cognitive_ctx.execution
            assumptions = exec_state.node_assumptions.get(node.id, [])
            if assumptions and not self._should_still_execute(node, assumptions):
                logger.info(
                    "[ASSUMPTION] Node '%s' skipped — assumption invalid",
                    node.id,
                )
                exec_result.record(node.id, NodeStatus.SKIPPED, {})
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
        failed_post_guards: List[str] = []
        for attempt in range(max_retries + 1):
            try:
                node_id, status, outputs, meta = self._executor.execute_node(
                    node, exec_result, world_snapshot,
                )
                exec_result.record(node_id, status, outputs, meta)

                # ── Input resolution failure: skip retries, route to recovery ──
                # Retrying is pointless — upstream output won't change.
                if (status == NodeStatus.FAILED
                        and meta and "failure_class" in meta):
                    failure_class = meta["failure_class"]
                    logger.info(
                        "[INPUT] Node '%s' input resolution failed: %s",
                        node.id, failure_class,
                    )
                    verdict = self._metacognition.classify(
                        node_id=node.id,
                        skill_name=node.skill,
                        error_message=meta.get("reason", ""),
                        failure_class=failure_class,
                        retries_exhausted=True,
                    )
                    if self._attempt_inline_recovery(
                        node, verdict, exec_result, world_snapshot,
                    ):
                        return self._execute_guarded_node(
                            node, exec_result, world_snapshot,
                        )
                    self._verdicts.append(verdict)
                    logger.info(
                        "[META] Node '%s' input resolution: %s → %s (%s)",
                        node.id, verdict.category.value,
                        verdict.action.value, verdict.reason,
                    )
                    return

                # ── Outcome classification (every node) ──
                severity = self._outcome_analyzer.classify(
                    status, meta,
                )
                if severity != OutcomeSeverity.BENIGN:
                    exec_result.outcome_verdicts.append({
                        "node_id": node_id,
                        "skill": node.skill,
                        "status": str(status),
                        "reason": (meta or {}).get("reason", ""),
                        "severity": severity.value,
                    })
                    logger.info(
                        "[OUTCOME] Node '%s' (%s): %s → %s",
                        node_id, node.skill, status, severity.value,
                    )

                    # ── Ambiguity: block dependent nodes ──
                    if (meta or {}).get("reason") == "ambiguous_input":
                        exec_result.failed.add(node_id)
                        logger.info(
                            "[OUTCOME] Node '%s' blocked dependents "
                            "(ambiguous_input)", node_id,
                        )

                # ── Uncertainty update (after every node) ──
                if self._cognitive_ctx:
                    domain = self._get_skill_domain(node.skill)
                    if status == NodeStatus.COMPLETED:
                        self._cognitive_ctx.execution.update_uncertainty(
                            "outcome_achieved", domain,
                        )
                    elif status == NodeStatus.FAILED:
                        error_str = (meta or {}).get("reason", "")
                        if "not found" in error_str.lower():
                            self._cognitive_ctx.execution.update_uncertainty(
                                "file_not_found", domain,
                            )
                        elif "multiple" in error_str.lower():
                            self._cognitive_ctx.execution.update_uncertainty(
                                "multiple_matches", domain,
                            )

                # ── Evaluate postconditions ──
                if status in (NodeStatus.COMPLETED, NodeStatus.NO_OP):
                    all_post_ok = True
                    failed_post_guards = []
                    for guard in postconditions:
                        if not self._evaluate_guard(guard):
                            all_post_ok = False
                            failed_post_guards.append(guard.type.value)
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
                err_str = str(e)

                # Input resolution failures are deterministic — retrying won't help
                # because the upstream output hasn't changed.
                if "input_resolution" in err_str.lower():
                    logger.info(
                        "[INPUT] Node '%s' input resolution error: %s",
                        node.id, err_str,
                    )
                    exec_result.record(node.id, NodeStatus.FAILED, {})
                    # Extract failure_class from error context
                    fc = None
                    for cls in ("MISSING_DATA", "INVALID_REFERENCE", "TYPE_MISMATCH"):
                        if cls in err_str:
                            fc = cls
                            break
                    verdict = self._metacognition.classify(
                        node_id=node.id,
                        skill_name=node.skill,
                        error_message=err_str,
                        failure_class=fc,
                        retries_exhausted=True,
                    )
                    if self._attempt_inline_recovery(
                        node, verdict, exec_result, world_snapshot,
                    ):
                        return self._execute_guarded_node(
                            node, exec_result, world_snapshot,
                        )
                    self._verdicts.append(verdict)
                    logger.info(
                        "[META] Node '%s' input resolution: %s → %s (%s)",
                        node.id, verdict.category.value,
                        verdict.action.value, verdict.reason,
                    )
                    return

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

        # ── Meta-cognition: classify exhausted failure ──
        verdict = self._metacognition.classify(
            node_id=node.id,
            skill_name=node.skill,
            failed_guards=failed_post_guards,
            error_message=str(last_error) if last_error else None,
            retries_exhausted=True,
        )
        # ── Inline recovery: try before giving up ──
        if self._attempt_inline_recovery(
            node, verdict, exec_result, world_snapshot,
        ):
            return self._execute_guarded_node(
                node, exec_result, world_snapshot,
            )
        self._verdicts.append(verdict)
        logger.info(
            "[META] Node '%s' exhausted retries — classified: %s → %s (%s)",
            node.id, verdict.category.value, verdict.action.value,
            verdict.reason,
        )

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

        Guard evaluation follows a tiered model:
        - APP_RUNNING: WorldState primary → observer fallback
          (authoritative check — WorldState tracks all MERLIN-launched apps)
        - APP_FOCUSED: Observer primary (safety-critical — must be real-time
          before typing/clicking, 500ms stale snapshot is dangerous)
        - WINDOW_VISIBLE: WorldState `visible` flag → observer confirmation
        - Others: observer-based

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
        timeline = self._ctx.timeline

        try:
            if guard.type == GuardType.APP_RUNNING:
                app_name = guard.params.get("app", "")
                # WorldState primary: check tracked_apps
                if timeline is not None:
                    from world.state import WorldState
                    ws = WorldState.from_events(timeline.all_events())
                    key = app_name.lower()
                    tracked = ws.system.session.tracked_apps.get(key)
                    if tracked is not None:
                        return tracked.running
                # Fallback: observer for apps not tracked in WorldState
                if observer is not None:
                    return observer.is_app_running(app_name)
                return True  # No source → assume OK

            elif guard.type == GuardType.APP_FOCUSED:
                # Safety-critical: observer-primary (real-time)
                # Must never use stale snapshot before typing/clicking
                app_name = guard.params.get("app", "")
                if observer is not None:
                    return observer.is_app_focused(app_name)
                # No observer: fall back to WorldState (better than nothing)
                if timeline is not None:
                    from world.state import WorldState
                    ws = WorldState.from_events(timeline.all_events())
                    key = app_name.lower()
                    tracked = ws.system.session.tracked_apps.get(key)
                    if tracked is not None:
                        return tracked.focused
                return True

            elif guard.type == GuardType.WINDOW_VISIBLE:
                app_name = guard.params.get("app", "")
                # WorldState check first (visible flag)
                if timeline is not None:
                    from world.state import WorldState
                    ws = WorldState.from_events(timeline.all_events())
                    key = app_name.lower()
                    tracked = ws.system.session.tracked_apps.get(key)
                    if tracked is not None and not tracked.visible:
                        return False  # Definitely not visible
                # Observer confirmation for positive case
                if observer is not None:
                    return observer.is_app_running(app_name)
                return True

            elif guard.type == GuardType.ACTIVE_WINDOW:
                expected_app = guard.params.get("app", "")
                if observer is None:
                    return True
                active = observer.get_active_window()
                if active is None:
                    return False
                return expected_app.lower() in (active.app_name or "").lower()

            elif guard.type == GuardType.FILE_EXISTS:
                path = guard.params.get("path", "")
                if observer is None:
                    return True
                return observer.file_exists(path)

            elif guard.type == GuardType.ELEMENT_VISIBLE:
                # Deferred to Phase 6 (browser automation)
                logger.debug(
                    "ELEMENT_VISIBLE guard deferred — assuming True",
                )
                return True

            elif guard.type == GuardType.MEDIA_SESSION_ACTIVE:
                if timeline is not None:
                    from world.state import WorldState
                    ws = WorldState.from_events(timeline.all_events())
                    media = ws.media
                    return bool(
                        media and (media.platform or media.title)
                    )
                return False

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

    # ─────────────────────────────────────────────────────────
    # Inline recovery via DecisionEngine
    # ─────────────────────────────────────────────────────────

    MAX_INLINE_RECOVERY = 2
    _SAFE_INLINE_EFFECTS = frozenset({"reveal", "create", "maintain"})

    def _attempt_inline_recovery(
        self,
        node: MissionNode,
        verdict: FailureVerdict,
        exec_result: ExecutionResult,
        world_snapshot: Optional[WorldSnapshot],
    ) -> bool:
        """Attempt inline recovery at point of failure.

        Builds a MissionNode from the ActionDecision and routes it
        through executor.execute_node() — the SAME pipeline as normal
        nodes (contract enforcement, timeline tracking, context injection).

        Returns True if recovery succeeded AND caller should retry
        the original node (via _execute_guarded_node re-entry, which
        re-evaluates ALL preconditions before execution).

        Constraints:
        - Bounded: MAX_INLINE_RECOVERY per node
        - Deduped: same (skill, inputs) never tried twice for same node
        - Safety: only effect_type in _SAFE_INLINE_EFFECTS
        - Causal graph: recorded via record_decision_with_causal_link()
        - No DAG mutation: recovery node is ephemeral (not in plan)
        """
        if not self._decision_engine or not self._cognitive_ctx:
            return False

        # Per-node budget check
        exec_state = self._cognitive_ctx.execution
        count = exec_state.inline_recovery_count.get(node.id, 0)
        if count >= self.MAX_INLINE_RECOVERY:
            logger.info(
                "[INLINE] Max recovery attempts (%d) for node '%s'",
                self.MAX_INLINE_RECOVERY, node.id,
            )
            return False

        # Fresh snapshot (re-snapshot every iteration)
        snapshot = self._cognitive_ctx.snapshot_for_decision()

        # Lazy import to avoid circular
        from execution.cognitive_context import ActionDecision

        decision = self._decision_engine.decide(verdict, snapshot)

        if not isinstance(decision, ActionDecision):
            return False

        # Dedup: have we already tried this exact (skill, inputs) for this node?
        import hashlib, json
        try:
            inputs_str = json.dumps(decision.inputs, sort_keys=True, default=str)
        except (TypeError, ValueError):
            inputs_str = str(sorted(decision.inputs.items()))
        dedup_key = f"{decision.skill}:{hashlib.md5(inputs_str.encode()).hexdigest()}"

        seen = exec_state.inline_recovery_seen.setdefault(node.id, set())
        if dedup_key in seen:
            logger.info(
                "[INLINE] Skipping duplicate recovery '%s' for node '%s'",
                decision.skill, node.id,
            )
            return False

        # Safety gate: check effect_type
        try:
            skill = self._executor.registry.get(decision.skill)
            effect = getattr(skill.contract, 'effect_type', 'maintain')
            if effect not in self._SAFE_INLINE_EFFECTS:
                logger.info(
                    "[INLINE] Skipping unsafe recovery: %s (effect=%s)",
                    decision.skill, effect,
                )
                return False
        except KeyError:
            return False

        # Record in causal graph
        from execution.metacognition import DecisionEngine
        DecisionEngine.record_decision_with_causal_link(
            exec_state,
            skill_name=decision.skill,
            inputs=decision.inputs,
            caused_by_node=node.id,
            strategy_source=decision.strategy_source,
        )

        # Build ephemeral recovery node + route through executor
        recovery_node = MissionNode(
            id=f"recovery_{node.id}_{count}",
            skill=decision.skill,
            inputs=decision.inputs,
            mode=ExecutionMode.foreground,
        )

        logger.info(
            "[INLINE] Executing recovery '%s' for node '%s' (attempt %d)",
            decision.skill, node.id, count + 1,
        )

        try:
            nid, status, outputs, meta = self._executor.execute_node(
                recovery_node, exec_result, world_snapshot,
            )
            exec_result.record(nid, status, outputs, meta)

            exec_state.record_attempt(
                skill=decision.skill,
                inputs=decision.inputs,
                result=str(status),
            )
            exec_state.inline_recovery_count[node.id] = count + 1
            seen.add(dedup_key)

            if status in (NodeStatus.COMPLETED, NodeStatus.NO_OP):
                logger.info(
                    "[INLINE] Recovery '%s' succeeded for node '%s'",
                    decision.skill, node.id,
                )
                return True  # caller retries original node

            logger.warning(
                "[INLINE] Recovery '%s' failed for node '%s': status=%s",
                decision.skill, node.id, status,
            )
            return False

        except Exception as e:
            exec_state.record_attempt(
                skill=decision.skill,
                inputs=decision.inputs,
                result="failed",
                error=str(e),
            )
            exec_state.inline_recovery_count[node.id] = count + 1
            seen.add(dedup_key)
            logger.warning(
                "[INLINE] Recovery '%s' threw for node '%s': %s",
                decision.skill, node.id, e,
            )
            return False

    # ─────────────────────────────────────────────────────────
    # Assumption validation + uncertainty helpers
    # ─────────────────────────────────────────────────────────

    def _should_still_execute(
        self, node: MissionNode, assumptions: list,
    ) -> bool:
        """Check if assumptions for a dynamic recovery node still hold.

        Maps each Assumption to an existing GuardType, evaluates via
        _evaluate_guard(), and respects the 'invert' flag.

        No new evaluation logic — all checks route through guards.
        """
        for assumption in assumptions:
            if not assumption.guard_mapping:
                continue  # No guard mapping → assumption not checkable
            try:
                guard_type = GuardType(assumption.guard_mapping)
            except ValueError:
                logger.debug(
                    "Unknown guard mapping '%s' for assumption on node '%s'",
                    assumption.guard_mapping, node.id,
                )
                continue
            guard = StepGuard(type=guard_type, params=assumption.params)
            result = self._evaluate_guard(guard)
            if assumption.invert:
                result = not result
            if not result:
                logger.info(
                    "[ASSUMPTION] Assumption '%s' (guard: %s, invert: %s) "
                    "failed for node '%s'",
                    assumption.type, assumption.guard_mapping,
                    assumption.invert, node.id,
                )
                return False
        return True

    def _get_skill_domain(self, skill_name: str) -> str:
        """Get authoritative domain from SkillContract via registry."""
        try:
            return self._executor.registry.get(skill_name).contract.domain
        except Exception:
            return "general"
