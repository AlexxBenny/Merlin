from concurrent.futures import ThreadPoolExecutor, Future
from typing import Any, Dict, List, Optional
import logging

from ir.mission import MissionPlan, ExecutionMode
from execution.executor import MissionExecutor, ExecutionResult, NodeStatus
from errors import FailureIR
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
    ):
        self.cortex = cortex
        self.executor = executor
        self.timeline = timeline
        self.report_builder = report_builder
        self.output_channel = output_channel
        self.pool = ThreadPoolExecutor(max_workers=max_workers)
        self.memory = memory

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
                    f"{u.get('verb', '')} {u.get('object', '')}" for u in uncovered
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

        # 2. Snapshot world at execution time
        events = self.timeline.all_events()
        state = WorldState.from_events(events)
        snapshot = WorldSnapshot.build(state, events[-10:] if events else [])

        # 3. Execute — get typed execution result
        exec_result = self.run(plan, snapshot)

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

        # 4. Build report
        report = self.report_builder.build(
            mission=plan,
            execution_result=exec_result,
            timeline=self.timeline,
            snapshot=snapshot,
            conversation=conversation,
        )

        # 5. Deliver
        if report:
            self.output_channel.send(report)
        else:
            self.output_channel.send_silent()

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
    ) -> ExecutionResult:
        """
        Execute a mission DAG. Returns typed ExecutionResult.

        This is the low-level execution path — no reporting.
        Used by handle_user_input() and available for testing.
        """

        # Submit full DAG execution
        future: Future = self.pool.submit(
            self.executor.run, mission, world_snapshot
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
