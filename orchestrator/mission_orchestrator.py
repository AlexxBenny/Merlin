from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from ir.mission import MissionPlan, ExecutionMode
from execution.executor import MissionExecutor, ExecutionResult, NodeStatus
from errors import FailureIR
from cortex.parameter_resolver import ParameterResolver, ParameterError
from cortex.mission_cortex import MissionCortex
from cortex.validators import verify_intent_coverage
from brain.escalation_policy import CognitiveTier
from world.timeline import WorldTimeline
from world.state import WorldState
from world.snapshot import WorldSnapshot
from conversation.frame import ConversationFrame, ContextFrame
from conversation.outcome import MissionOutcome
from reporting.report_builder import ReportBuilder
from reporting.output import OutputChannel
from memory.store import MemoryStore

if TYPE_CHECKING:
    from runtime.attention import AttentionManager, MissionState
    from reporting.narration import NarrationPolicy


logger = logging.getLogger(__name__)


class MissionOrchestrator:
    """
    Owns the full mission lifecycle: compile → execute → report → deliver.

    Three responsibilities (not one):
    1. Run missions (execution)
    2. Decide whether to report (via ReportBuilder)
    3. Deliver the report (via OutputChannel)

    The orchestrator coordinates. It does not reason.
    """

    def __init__(
        self,
        cortex: MissionCortex,
        executor: MissionExecutor,
        timeline: WorldTimeline,
        report_builder: ReportBuilder,
        output_channel: OutputChannel,
        max_workers: int = 4,
        memory: Optional[MemoryStore] = None,
        attention_manager: Optional["AttentionManager"] = None,
        narration_policy: Optional["NarrationPolicy"] = None,
    ):
        self.cortex = cortex
        self.executor = executor
        self.timeline = timeline
        self.report_builder = report_builder
        self.output_channel = output_channel
        self.pool = ThreadPoolExecutor(max_workers=max_workers)
        self.memory = memory
        self._attention = attention_manager
        self._narration = narration_policy
        self._resolver = ParameterResolver(executor.registry)

    # ─────────────────────────────────────────────────────────
    # PUBLIC API: Full mission lifecycle
    # ─────────────────────────────────────────────────────────

    def handle_user_input(
        self,
        user_text: str,
        conversation: ConversationFrame,
        world_state_schema: Dict[str, Any],
        cognitive_tier: Optional[CognitiveTier] = None,
        intent_units: Optional[List[Dict[str, str]]] = None,
    ) -> Optional[str]:
        """
        The correct public API for user-initiated missions.

        1. Compile user text → MissionPlan DAG (LLM sees world_state_schema)
        2. Coverage verification (Tier 2+ only)
        3. Build world snapshot at execution time
        4. Execute the DAG
        5. Build report from results
        6. Deliver report via output channel

        Args:
            cognitive_tier: Optional tier classification for this query.
            intent_units: Optional decomposed intent list (Tier 2+ only).
                Passed into compile as checklist and used for coverage gate.

        Returns the report text (or None if silent).
        """

        # ── Bootstrap gate: fail hard if world not initialized ──
        if not self.timeline.bootstrapped:
            raise RuntimeError(
                "MissionOrchestrator: world not bootstrapped. "
                "Cannot execute missions on uninitialized state."
            )

        # 1. Compile — LLM sees world_state_schema
        result = self.cortex.compile(
            user_query=user_text,
            world_state_schema=world_state_schema,
            conversation=conversation,
            intent_checklist=intent_units,  # None for Tier 1
        )

        # 1.5 FailureIR path — skip execution, report error
        if isinstance(result, FailureIR):
            report = self._build_failure_report(result)
            if report:
                self.output_channel.send(report)
            else:
                self.output_channel.send_silent()
            return report

        plan = result

        # ── Coverage gate (Tier 2+ only, after structural validation) ──
        if intent_units:
            all_covered, uncovered = verify_intent_coverage(
                plan, intent_units, self.cortex.registry,
            )
            if not all_covered:
                uncovered_desc = "; ".join(
                    u.get("action", "unknown") for u in uncovered
                )
                failure = FailureIR(
                    error_type="incomplete_coverage",
                    error_message=(
                        f"Plan covers {len(intent_units) - len(uncovered)}/"
                        f"{len(intent_units)} intents. "
                        f"Uncovered: {uncovered_desc}"
                    ),
                    user_query=user_text,
                )
                report = self._build_failure_report(failure)
                if report:
                    self.output_channel.send(report)
                return report

            # ── Compiler alignment check (detect cross-LLM drift) ──
            # Verify that the compiler chose skills whose actions match
            # the canonical actions the decomposer intended.
            decomposed_actions = {
                u.get("action", "") for u in intent_units if u.get("action")
            }
            compiled_actions = set()
            for node in plan.nodes:
                parts = node.skill.split(".", 1)
                node_action = parts[1] if len(parts) >= 2 else parts[0]
                compiled_actions.add(node_action)

            # Actions in decomposition that don't appear in the compiled DAG
            drift = decomposed_actions - compiled_actions
            if drift:
                logger.warning(
                    "[DRIFT] Decomposer intended actions %s but compiler "
                    "produced nodes with actions %s — "
                    "potential cross-LLM semantic drift",
                    sorted(drift),
                    sorted(compiled_actions),
                )

        # [TRACE] Log the compiled plan
        logger.info(
            "[TRACE] Compiled MissionPlan: id=%s, nodes=%d",
            plan.id, len(plan.nodes),
        )
        for node in plan.nodes:
            logger.info(
                "[TRACE]   Node '%s': skill=%s, inputs=%r, depends_on=%r, mode=%s",
                node.id, node.skill, node.inputs, node.depends_on, node.mode.value,
            )

        # ── Phase 9A: Typed parameter resolution (deterministic) ──
        # Runs BEFORE narration so resolved values feed narration phrases.
        # Runs BEFORE EXECUTING transition so errors stay in COMPILING.
        try:
            plan = self._resolver.resolve_plan(plan)
        except ParameterError as pe:
            logger.warning(
                "Parameter resolution failed: %s", pe,
            )
            # Structured error → clean user message (no stack trace)
            return pe.user_message()

        # 2. Snapshot world at execution time
        events = self.timeline.all_events()
        state = WorldState.from_events(events)
        snapshot = WorldSnapshot.build(state, events[-10:] if events else [])

        # ── Phase 8: Pre-narration (deterministic, no LLM) ──
        pre_narration = None
        pre_narration_fired = False
        if self._narration:
            from execution.scheduler import DAGScheduler
            node_index = DAGScheduler.get_node_index(plan)
            pre_narration = self._narration.narrate_pre_execution(
                plan, node_index, self.executor.registry,
            )
            if pre_narration:
                pre_narration_fired = True
                self.output_channel.send(pre_narration)

        # ── Transition: COMPILING → EXECUTING ──
        if self._attention:
            from runtime.attention import MissionState
            self._attention.set_mission_state(MissionState.EXECUTING)

        # 3. Execute — get typed execution result (with layer callbacks)
        exec_start = time.monotonic()

        def _on_layer_start(layer_ids, node_index, layer_idx, total_layers):
            """Layer-start narration hook (fire-and-forget)."""
            if self._narration:
                text = self._narration.narrate_layer_start(
                    layer_ids, node_index,
                    self.executor.registry,
                    layer_idx, total_layers,
                    pre_narration_fired,
                )
                if text:
                    self.output_channel.send(text)
            # Heartbeat check
            if self._narration:
                elapsed = time.monotonic() - exec_start
                heartbeat = self._narration.narrate_heartbeat(elapsed)
                if heartbeat:
                    self.output_channel.send(heartbeat)

        exec_result = self.run(
            plan, snapshot,
            on_layer_start=_on_layer_start,
        )

        # ── Transition: EXECUTING → REPORTING ──
        if self._attention:
            from runtime.attention import MissionState
            self._attention.set_mission_state(MissionState.REPORTING)

        # 3.5. Build and persist MissionOutcome
        outcome = self._build_outcome(plan, exec_result)
        conversation.outcomes.append(outcome)

        # Enforce outcome cap — prevent unbounded memory growth
        OUTCOME_CAP = 10
        if len(conversation.outcomes) > OUTCOME_CAP:
            conversation.outcomes[:] = conversation.outcomes[-OUTCOME_CAP:]

        # Store episode in memory for future retrieval
        if self.memory is not None:
            try:
                self.memory.store_episode(
                    mission_id=plan.id,
                    query=user_text,
                    outcome_summary=f"executed {len(outcome.nodes_executed)} nodes, "
                                    f"failed {len(outcome.nodes_failed)}",
                    metadata={
                        "domain": outcome.active_domain,
                        "entity": outcome.active_entity,
                    },
                )
            except Exception as e:
                logger.warning("Failed to store memory episode: %s", e)

        # ── Drain queued proactive insights BEFORE building report ──
        # CRITICAL: drain during REPORTING state, before IDLE transition.
        # This prevents auto-flush race and enables contextual merging.
        queued_insights: Optional[List[str]] = None
        if self._attention:
            queued = self._attention.drain_queue()
            if queued:
                queued_insights = [q.text for q in queued]
                logger.debug(
                    "Orchestrator: drained %d insight(s) for report",
                    len(queued_insights),
                )

        # 4. Build report (with tonal continuity + insights)
        report = self.report_builder.build(
            mission=plan,
            execution_result=exec_result,
            timeline=self.timeline,
            snapshot=snapshot,
            conversation=conversation,
            queued_insights=queued_insights,
            pre_narration=pre_narration,
        )

        # 5. Deliver
        if report:
            self.output_channel.send(report)
        else:
            self.output_channel.send_silent()

        # ── Transition: REPORTING → IDLE ──
        # Queue is already drained — IDLE auto-flush is a no-op.
        if self._attention:
            from runtime.attention import MissionState
            self._attention.set_mission_state(MissionState.IDLE)

        # 6. Update conversation frame
        conversation.last_mission_id = plan.id
        if outcome.active_domain:
            conversation.active_domain = outcome.active_domain
            # Persist domain-scoped context frame
            conversation.context_frames[outcome.active_domain] = ContextFrame(
                domain=outcome.active_domain,
                data={
                    "entity": outcome.active_entity,
                    "artifacts": outcome.artifacts,
                    "mission_id": plan.id,
                },
                produced_by=plan.id,
            )
        if outcome.active_entity:
            conversation.active_entity = outcome.active_entity

        # 7. Structured conversation state updates
        self._update_conversation_state(
            conversation=conversation,
            outcome=outcome,
            exec_result=exec_result,
            user_text=user_text,
            mission_id=plan.id,
        )

        return report


    def handle_prebuilt_plan(
        self,
        plan: MissionPlan,
        user_text: str,
        conversation: ConversationFrame,
    ) -> Optional[str]:
        """Execute a pre-built plan (e.g. from multi-reflex).

        Same lifecycle as handle_user_input() but skips:
        - LLM compilation (plan already built)
        - Tier classification
        - Coverage verification
        - Decomposition

        This is the <200ms fast path for deterministic plans.
        """
        # ── Phase 9A: Typed parameter resolution ──
        try:
            plan = self._resolver.resolve_plan(plan)
        except ParameterError as pe:
            logger.warning("Parameter resolution failed: %s", pe)
            return pe.user_message()

        # Snapshot world
        events = self.timeline.all_events()
        state = WorldState.from_events(events)
        snapshot = WorldSnapshot.build(state, events[-10:] if events else [])

        # ── Pre-narration ──
        pre_narration = None
        pre_narration_fired = False
        if self._narration:
            from execution.scheduler import DAGScheduler
            node_index = DAGScheduler.get_node_index(plan)
            pre_narration = self._narration.narrate_pre_execution(
                plan, node_index, self.executor.registry,
            )
            if pre_narration:
                pre_narration_fired = True
                self.output_channel.send(pre_narration)

        # ── Transition: → EXECUTING ──
        if self._attention:
            from runtime.attention import MissionState
            self._attention.set_mission_state(MissionState.EXECUTING)

        # Execute with layer callbacks
        exec_start = time.monotonic()

        def _on_layer_start(layer_ids, node_index, layer_idx, total_layers):
            if self._narration:
                text = self._narration.narrate_layer_start(
                    layer_ids, node_index,
                    self.executor.registry,
                    layer_idx, total_layers,
                    pre_narration_fired,
                )
                if text:
                    self.output_channel.send(text)
            if self._narration:
                elapsed = time.monotonic() - exec_start
                heartbeat = self._narration.narrate_heartbeat(elapsed)
                if heartbeat:
                    self.output_channel.send(heartbeat)

        exec_result = self.run(
            plan, snapshot,
            on_layer_start=_on_layer_start,
        )

        # ── Transition: EXECUTING → REPORTING ──
        if self._attention:
            from runtime.attention import MissionState
            self._attention.set_mission_state(MissionState.REPORTING)

        # Build outcome
        outcome = self._build_outcome(plan, exec_result)
        conversation.outcomes.append(outcome)
        OUTCOME_CAP = 10
        if len(conversation.outcomes) > OUTCOME_CAP:
            conversation.outcomes[:] = conversation.outcomes[-OUTCOME_CAP:]

        # Drain insights
        queued_insights: Optional[List[str]] = None
        if self._attention:
            queued = self._attention.drain_queue()
            if queued:
                queued_insights = [q.text for q in queued]

        # Build report
        report = self.report_builder.build(
            mission=plan,
            execution_result=exec_result,
            timeline=self.timeline,
            snapshot=snapshot,
            conversation=conversation,
            queued_insights=queued_insights,
            pre_narration=pre_narration,
        )

        # Deliver
        if report:
            self.output_channel.send(report)
        else:
            self.output_channel.send_silent()

        # ── Transition: REPORTING → IDLE ──
        if self._attention:
            from runtime.attention import MissionState
            self._attention.set_mission_state(MissionState.IDLE)

        # Update conversation
        conversation.last_mission_id = plan.id
        if outcome.active_domain:
            conversation.active_domain = outcome.active_domain
            conversation.context_frames[outcome.active_domain] = ContextFrame(
                domain=outcome.active_domain,
                data={
                    "entity": outcome.active_entity,
                    "artifacts": outcome.artifacts,
                    "mission_id": plan.id,
                },
                produced_by=plan.id,
            )
        if outcome.active_entity:
            conversation.active_entity = outcome.active_entity

        self._update_conversation_state(
            conversation=conversation,
            outcome=outcome,
            exec_result=exec_result,
            user_text=user_text,
            mission_id=plan.id,
        )

        return report


    # ─────────────────────────────────────────────────────────
    # Conversation state updates (v2)
    # ─────────────────────────────────────────────────────────

    def _update_conversation_state(
        self,
        conversation: ConversationFrame,
        outcome: MissionOutcome,
        exec_result: ExecutionResult,
        user_text: str,
        mission_id: str,
    ) -> None:
        """Populate structured conversation state after mission execution.

        Updates:
        - entity_registry: typed EntityRecords from artifacts + visible_lists
        - last_results: full node→output structure (preserves provenance)
        - recent_intents: user text (truncated)

        Never raises — failures logged and swallowed.
        """
        try:
            # 7a. Entity registry — register visible_lists as typed entities
            for key, items in outcome.visible_lists.items():
                clean_key = key.split(".", 1)[-1] if "." in key else key
                conversation.register_entity(
                    key=clean_key,
                    value=items,
                    entity_type="list",
                    source_mission=mission_id,
                )

            # 7b. Entity registry — register artifacts as typed entities
            for key, val in outcome.artifacts.items():
                clean_key = key.split(".", 1)[-1] if "." in key else key
                entity_type = "path" if isinstance(val, str) and ("/" in val or "\\" in val) else "scalar"
                conversation.register_entity(
                    key=clean_key,
                    value=val,
                    entity_type=entity_type,
                    source_mission=mission_id,
                )

            # 7c. Last results — preserve full structure (no flattening)
            conversation.store_results(exec_result.results.copy())

            # 7d. Push intent — truncated user text
            conversation.push_intent(user_text[:100])

        except Exception as e:
            logger.warning("Failed to update conversation state: %s", e)

    # ─────────────────────────────────────────────────────────
    # LOW-LEVEL: Execution only (no reporting)
    # ─────────────────────────────────────────────────────────

    def run(
        self,
        mission: MissionPlan,
        world_snapshot: Optional[WorldSnapshot] = None,
        on_layer_start: Optional[Callable] = None,
        on_layer_complete: Optional[Callable] = None,
    ) -> ExecutionResult:
        """
        Execute a mission DAG. Returns typed ExecutionResult.

        This is the low-level execution path — no reporting.
        Used by handle_user_input() and available for testing.
        """

        # Submit full DAG execution (with narration callbacks)
        future: Future = self.pool.submit(
            self.executor.run, mission, world_snapshot,
            on_layer_start, on_layer_complete,
        )

        # Wait for full DAG execution to finish
        # (executor enforces dependencies)
        exec_result = future.result()

        return exec_result

    # ─────────────────────────────────────────────────────────
    # Outcome construction
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _build_outcome(
        plan: MissionPlan,
        exec_result: ExecutionResult,
    ) -> MissionOutcome:
        """
        Build a MissionOutcome from executed plan + typed execution result.

        Created ONCE per mission, AFTER execution completes.
        Uses exec_result.node_statuses for typed failure classification.
        """
        executed = sorted(exec_result.completed)
        skipped = sorted(exec_result.skipped)
        failed = sorted(
            nid for nid, st in exec_result.node_statuses.items()
            if st == NodeStatus.FAILED
        )
        timed_out = sorted(
            nid for nid, st in exec_result.node_statuses.items()
            if st == NodeStatus.TIMED_OUT
        )

        # Extract artifacts and visible_lists from completed node outputs
        artifacts: Dict[str, Any] = {}
        visible_lists: Dict[str, list] = {}
        active_domain: str | None = None
        active_entity: str | None = None

        for node in plan.nodes:
            if node.id not in exec_result.results:
                continue

            output = exec_result.results[node.id]

            for key, value in output.items():
                if isinstance(value, list):
                    visible_lists[f"{node.id}.{key}"] = value
                else:
                    artifacts[f"{node.id}.{key}"] = value

            # Read entity/domain from metadata channel (not outputs)
            meta = exec_result.metadata.get(node.id, {})
            if "domain" in meta:
                active_domain = meta["domain"]
            if "entity" in meta:
                active_entity = meta["entity"]

        # Infer domain from skill name if not explicitly set
        if not active_domain and executed:
            last_node = next(
                (n for n in plan.nodes if n.id == executed[-1]), None
            )
            if last_node:
                active_domain = last_node.skill.split(".")[0]

        # Infer entity from last completed node output if not explicitly set
        if not active_entity and executed:
            last_node = next(
                (n for n in plan.nodes if n.id == executed[-1]), None
            )
            if last_node and last_node.id in exec_result.results:
                last_output = exec_result.results[last_node.id]
                for key, val in last_output.items():
                    if isinstance(val, str):
                        active_entity = val
                        break

        return MissionOutcome(
            mission_id=plan.id,
            nodes_executed=executed,
            nodes_skipped=skipped,
            nodes_failed=failed,
            nodes_timed_out=timed_out,
            artifacts=artifacts,
            visible_lists=visible_lists,
            active_domain=active_domain,
            active_entity=active_entity,
        )

    # ─────────────────────────────────────────────────────────
    # Failure reporting
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _build_failure_report(failure: FailureIR) -> str:
        """Build a human-readable error report from FailureIR.

        Flows through the SAME output channel as success reports.
        No UI inconsistency — user sees a structured message.
        """
        if failure.error_type == "llm_unavailable":
            return (
                "I'm unable to process your request right now — "
                "the language model is not available.\n\n"
                f"Error: {failure.error_message}"
            )

        if failure.error_type == "parse_error":
            return (
                "I wasn't able to understand the model's response. "
                "Please try again.\n\n"
                f"Details: {failure.error_message}"
            )

        # malformed_plan or incomplete_coverage
        if failure.error_type == "incomplete_coverage":
            return (
                "I built a plan, but it doesn't cover all of your requests.\n\n"
                f"Details: {failure.error_message}\n\n"
                "Please try rephrasing or simplifying your request."
            )

        return (
            "I couldn't build a valid plan for your request.\n\n"
            f"Reason: {failure.error_message}"
        )
