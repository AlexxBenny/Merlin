from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from ir.mission import MissionPlan, ExecutionMode
from execution.executor import MissionExecutor, ExecutionResult, NodeStatus
from errors import FailureIR
from cortex.parameter_resolver import ParameterResolver, ParameterError
from cortex.preference_resolver import PreferenceResolver
from cortex.entity_resolver import EntityResolver, EntityResolutionError
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
        supervisor=None,
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
        self._supervisor = supervisor
        self._resolver = ParameterResolver(executor.registry)
        self._pref_resolver: Optional[PreferenceResolver] = None
        self._entity_resolver: Optional[EntityResolver] = None

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
        unsupported_intents: Optional[List[Dict[str, str]]] = None,
        original_query: Optional[str] = None,
        computed_variables: Optional[Dict[str, Any]] = None,
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
            original_query: Pre-coordinator query text (for metadata/debugging).
            computed_variables: Values computed by coordinator (REASONED_PLAN).

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

        # ── Inject metadata for explainability ──
        plan.metadata["unsupported_intents"] = unsupported_intents or []
        if original_query and original_query != user_text:
            plan.metadata["original_query"] = original_query
        if computed_variables:
            plan.metadata["computed_variables"] = computed_variables

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

        # ── Phase 9B: Preference resolution (semantic memory) ──
        if self._pref_resolver is not None:
            plan = self._pref_resolver.resolve_plan(plan)

        # ── Phase 9C/9D: Entity resolution (app + browser entities) ──
        # Build WorldState BEFORE entity resolution so browser entity
        # resolver can access WorldState.browser.top_entities.
        events = self.timeline.all_events()
        state = WorldState.from_events(events)
        snapshot = WorldSnapshot.build(state, events[-10:] if events else [])

        if self._entity_resolver is not None:
            try:
                plan = self._entity_resolver.resolve_plan(
                    plan, world_snapshot=state,
                )
            except EntityResolutionError as ere:
                # Separate browser NOT_FOUND from app ambiguous/not_found
                browser_not_found = [
                    v for v in ere.violations
                    if v.resolution_type == "not_found_browser"
                ]
                other_violations = [
                    v for v in ere.violations
                    if v.resolution_type != "not_found_browser"
                ]

                # App entity violations → ask user (existing flow)
                if other_violations:
                    logger.info("Entity resolution clarification: %s", ere)
                    return ere.user_message()

                # Browser NOT_FOUND → recovery recompile
                # (entity not on page — compiler can plan search instead)
                if browser_not_found:
                    logger.info(
                        "[RECOVERY] Browser entity not found — "
                        "triggering recompile for: %s",
                        [v.raw_value for v in browser_not_found],
                    )
                    recovery_failures = [{
                        "skill": v.skill,
                        "status": "entity_not_found",
                        "reason": (
                            f"Entity '{v.raw_value}' not present on page. "
                            f"Available: {', '.join(v.candidates[:3])}"
                            if v.candidates else
                            f"Entity '{v.raw_value}' not present on page."
                        ),
                        "severity": "soft_failure",
                    } for v in browser_not_found]
                    try:
                        recovery_ws = state.model_dump()
                        recovery_plan = self.cortex.compile(
                            user_query=user_text,
                            world_state_schema=recovery_ws,
                            conversation=conversation,
                            execution_failures=recovery_failures,
                        )
                        if isinstance(recovery_plan, MissionPlan):
                            plan = recovery_plan
                            logger.info(
                                "[RECOVERY] Recompiled plan: %d nodes",
                                len(plan.nodes),
                            )
                        else:
                            # Recompile also failed → ask user
                            return ere.user_message()
                    except Exception as e:
                        logger.warning(
                            "[RECOVERY] Recompile failed: %s", e,
                        )
                        return ere.user_message()

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

        # ── CognitiveContext: single source of truth for this mission ──
        from execution.cognitive_context import (
            CognitiveContext,
            GoalState as CognitiveGoalState,
            ExecutionState,
            ActionDecision,
            EscalationDecision,
            EscalationLevel,
            AmbiguityDecision,
        )
        from execution.metacognition import DecisionEngine

        # DecisionEngine: instantiate ONCE per mission
        # Get optional LLM client for recovery reasoning (graceful degradation)
        llm_client = None
        try:
            from models.router import ModelRouter
            import yaml
            from pathlib import Path
            models_path = Path(__file__).resolve().parent.parent / "config" / "models.yaml"
            if models_path.exists():
                with open(models_path) as f:
                    models_cfg = yaml.safe_load(f)
                if "recovery_reasoner" in models_cfg:
                    router = ModelRouter(models_cfg)
                    llm_client = router.get_client("recovery_reasoner")
        except Exception as e:
            logger.debug("[DECISION] No LLM client for recovery: %s", e)

        decision_engine = DecisionEngine(
            registry=self.executor.registry,
            llm_client=llm_client,
        )

        cognitive_goal = CognitiveGoalState(
            original_query=user_text,
            required_outcomes=[
                n.skill.split(".")[-1] for n in plan.nodes
            ],
        )
        cognitive_ctx = CognitiveContext(
            goal=cognitive_goal,
            execution=ExecutionState(),
            world=snapshot,
            conversation=conversation,
        )

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
            cognitive_ctx=cognitive_ctx,
            decision_engine=decision_engine,
        )

        # ── Recovery via DecisionEngine (deterministic, no LLM) ──
        if exec_result.meta_verdicts:
            for verdict in exec_result.meta_verdicts:
                snapshot_for_decision = cognitive_ctx.snapshot_for_decision()
                decision = decision_engine.decide(
                    verdict, snapshot_for_decision, plan,
                )

                if isinstance(decision, ActionDecision):
                    # Record in causal graph
                    from execution.metacognition import DecisionEngine as DE
                    DE.record_decision_with_causal_link(
                        cognitive_ctx.execution,
                        skill_name=decision.skill,
                        inputs=decision.inputs,
                        caused_by_node=verdict.node_id,
                        strategy_source=decision.strategy_source,
                    )
                    # Store assumptions for the recovery node
                    recovery_node_id = f"recovery_{verdict.node_id}"
                    if decision.assumptions:
                        cognitive_ctx.execution.node_assumptions[
                            recovery_node_id
                        ] = decision.assumptions
                    # Record attempt
                    cognitive_ctx.execution.record_attempt(
                        skill=decision.skill,
                        inputs=decision.inputs,
                        result="pending",
                    )
                    logger.info(
                        "[DECISION] ActionDecision for node '%s': %s(%s) "
                        "[score: %.3f, strategy: %s]",
                        verdict.node_id, decision.skill,
                        decision.inputs, decision.score,
                        decision.strategy_source,
                    )
                    exec_result.recovery_explanation = (
                        f"DecisionEngine: {decision.skill} "
                        f"(strategy: {decision.strategy_source})"
                    )

                elif isinstance(decision, AmbiguityDecision):
                    logger.info(
                        "[DECISION] AmbiguityDecision (meta): %s",
                        decision.question,
                    )
                    exec_result.clarification_needed = {
                        "question": decision.question,
                        "options": decision.choices,
                        "context": {
                            "source": "decision_engine",
                            "verdict": str(decision.verdict),
                        },
                    }
                    break  # Pause for user input

                elif isinstance(decision, EscalationDecision):
                    if decision.level == EscalationLevel.GLOBAL:
                        # Tier 3: refresh world + recompile
                        logger.info(
                            "[ESCALATION] GLOBAL: %s — triggering Tier 3 replan",
                            decision.reason,
                        )
                        try:
                            recovery_events = self.timeline.all_events()
                            recovery_state = WorldState.from_events(
                                recovery_events,
                            )
                            recovery_snapshot = WorldSnapshot.build(
                                recovery_state,
                                recovery_events[-10:]
                                if recovery_events else [],
                            )
                            cognitive_ctx = cognitive_ctx.refresh_world(
                                recovery_snapshot,
                            )
                            recovery_ws = recovery_state.model_dump()
                            recovery_plan = self.cortex.compile(
                                user_query=user_text,
                                world_state_schema=recovery_ws,
                                conversation=conversation,
                                execution_failures=[
                                    v for v in
                                    exec_result.outcome_verdicts
                                ],
                            )
                            if isinstance(recovery_plan, MissionPlan):
                                if not self._plans_equivalent(
                                    plan, recovery_plan,
                                ):
                                    logger.info(
                                        "[RECOVERY] Executing Tier 3 plan: "
                                        "%d nodes",
                                        len(recovery_plan.nodes),
                                    )
                                    recovery_result = self.run(
                                        recovery_plan,
                                        recovery_snapshot,
                                        cognitive_ctx=cognitive_ctx,
                                    )
                                    for nid in recovery_result.completed:
                                        exec_result.completed.add(nid)
                                        if nid in recovery_result.results:
                                            exec_result.results[nid] = (
                                                recovery_result.results[nid]
                                            )
                                    for nid in recovery_result.failed:
                                        exec_result.failed.add(nid)
                                    exec_result.recovery_explanation = (
                                        f"Tier 3 recovery: "
                                        f"{len(recovery_plan.nodes)} node(s)"
                                    )
                                    plan = recovery_plan
                                else:
                                    logger.info(
                                        "[RECOVERY] Tier 3 plan identical "
                                        "— skipping",
                                    )
                            else:
                                logger.warning(
                                    "[RECOVERY] Tier 3 compile failed: %s",
                                    recovery_plan,
                                )
                        except Exception as e:
                            logger.warning(
                                "[RECOVERY] Tier 3 replan failed: %s", e,
                            )

                    elif decision.level == EscalationLevel.USER:
                        logger.info(
                            "[ESCALATION] USER: %s",
                            decision.reason,
                        )
                        exec_result.clarification_needed = {
                            "question": decision.reason,
                            "options": [],
                            "context": {
                                "source": "escalation",
                                "level": "user",
                            },
                        }
                        break  # Pause for user input

        # ── Outcome verdicts → DecisionEngine (unified routing) ──
        if exec_result.outcome_verdicts:
            soft_failures = [
                v for v in exec_result.outcome_verdicts
                if v.get("severity") == "soft_failure"
            ]
            if soft_failures:
                from execution.metacognition import (
                    FailureVerdict, FailureCategory, RecoveryAction,
                )
                logger.info(
                    "[RECOVERY] %d soft failure(s) → DecisionEngine",
                    len(soft_failures),
                )
                for sf in soft_failures:
                    # ── Ambiguity: short-circuit to ask-back ──
                    # Ambiguity is NOT a failure requiring recovery.
                    # It's a user interaction signal.  The skill already
                    # stored a user-facing question + options in metadata.
                    # Route directly to clarification, bypassing the
                    # DecisionEngine (which would GLOBAL-escalate and
                    # trigger a futile Tier 3 replan because the node
                    # has dependents → multi_step scope).
                    if sf.get("reason") == "ambiguous_input":
                        node_meta = exec_result.metadata.get(
                            sf.get("node_id", ""), {},
                        )
                        question = node_meta.get(
                            "message",
                            f"Ambiguity detected for "
                            f"{sf.get('skill', 'unknown')}. "
                            "Could you clarify?",
                        )
                        exec_result.clarification_needed = {
                            "question": question,
                            "options": node_meta.get("options", []),
                            "context": {
                                "source": "ambiguous_input",
                                "node_id": sf.get("node_id"),
                                "skill": sf.get("skill"),
                                "plan": plan,
                                "node_results": {
                                    nid: {
                                        **exec_result.results.get(nid, {}),
                                        **exec_result.metadata.get(nid, {}),
                                    }
                                    for nid in exec_result.results
                                },
                            },
                        }
                        logger.info(
                            "[RECOVERY] Ambiguity → ask-back: %s",
                            question[:100],
                        )
                        break  # Pause for user input

                    # Convert outcome dict → FailureVerdict
                    verdict = FailureVerdict(
                        category=FailureCategory.CAPABILITY_FAILURE,
                        action=RecoveryAction.RETRY,
                        reason=sf.get("reason", "unknown"),
                        node_id=sf.get("node_id", "unknown"),
                        skill_name=sf.get("skill"),
                        context={
                            "error": sf.get("reason", ""),
                            "original_inputs": sf.get("original_inputs", {}),
                            "domain": sf.get("domain", ""),
                        },
                    )
                    snapshot_for_decision = cognitive_ctx.snapshot_for_decision()
                    decision = decision_engine.decide(
                        verdict, snapshot_for_decision, plan,
                    )

                    if isinstance(decision, ActionDecision):
                        from execution.metacognition import DecisionEngine as DE
                        DE.record_decision_with_causal_link(
                            cognitive_ctx.execution,
                            skill_name=decision.skill,
                            inputs=decision.inputs,
                            caused_by_node=verdict.node_id,
                            strategy_source=decision.strategy_source,
                        )
                        recovery_node_id = f"recovery_{verdict.node_id}"
                        if decision.assumptions:
                            cognitive_ctx.execution.node_assumptions[
                                recovery_node_id
                            ] = decision.assumptions
                        cognitive_ctx.execution.record_attempt(
                            skill=decision.skill,
                            inputs=decision.inputs,
                            result="pending",
                        )
                        logger.info(
                            "[DECISION] ActionDecision for outcome '%s': "
                            "%s(%s) [score: %.3f, strategy: %s]",
                            verdict.node_id, decision.skill,
                            decision.inputs, decision.score,
                            decision.strategy_source,
                        )
                        exec_result.recovery_explanation = (
                            f"DecisionEngine: {decision.skill} "
                            f"(strategy: {decision.strategy_source})"
                        )

                    elif isinstance(decision, AmbiguityDecision):
                        logger.info(
                            "[DECISION] AmbiguityDecision: %s",
                            decision.question,
                        )
                        exec_result.clarification_needed = {
                            "question": decision.question,
                            "options": decision.choices,
                            "context": {
                                "source": "decision_engine_outcome",
                                "verdict": str(decision.verdict),
                            },
                        }
                        break  # Pause for user input

                    elif isinstance(decision, EscalationDecision):
                        if decision.level == EscalationLevel.GLOBAL:
                            logger.info(
                                "[ESCALATION] GLOBAL (outcome): %s",
                                decision.reason,
                            )
                            try:
                                recovery_events = self.timeline.all_events()
                                recovery_state = WorldState.from_events(
                                    recovery_events,
                                )
                                recovery_snapshot = WorldSnapshot.build(
                                    recovery_state,
                                    recovery_events[-10:]
                                    if recovery_events else [],
                                )
                                cognitive_ctx = cognitive_ctx.refresh_world(
                                    recovery_snapshot,
                                )
                                recovery_ws = recovery_state.model_dump()
                                recovery_plan = self.cortex.compile(
                                    user_query=user_text,
                                    world_state_schema=recovery_ws,
                                    conversation=conversation,
                                    execution_failures=soft_failures,
                                )
                                if isinstance(recovery_plan, MissionPlan):
                                    if not self._plans_equivalent(
                                        plan, recovery_plan,
                                    ):
                                        logger.info(
                                            "[RECOVERY] Tier 3 recovery: "
                                            "%d nodes",
                                            len(recovery_plan.nodes),
                                        )
                                        recovery_result = self.run(
                                            recovery_plan,
                                            recovery_snapshot,
                                            cognitive_ctx=cognitive_ctx,
                                        )
                                        for nid in recovery_result.completed:
                                            exec_result.completed.add(nid)
                                            if nid in recovery_result.results:
                                                exec_result.results[nid] = (
                                                    recovery_result.results[nid]
                                                )
                                        for nid in recovery_result.failed:
                                            exec_result.failed.add(nid)
                                        exec_result.recovery_explanation = (
                                            f"Tier 3 recovery: "
                                            f"{len(recovery_plan.nodes)} node(s)"
                                        )
                                        plan = recovery_plan
                            except Exception as e:
                                logger.warning(
                                    "[RECOVERY] Tier 3 replan failed: %s", e,
                                )
                        elif decision.level == EscalationLevel.USER:
                            logger.info(
                                "[ESCALATION] USER (outcome): %s",
                                decision.reason,
                            )
                            exec_result.clarification_needed = {
                                "question": decision.reason,
                                "options": [],
                                "context": {
                                    "source": "escalation_outcome",
                                    "level": "user",
                                },
                            }
                            break  # Pause for user input

        # ── Clarification needed → return signal to merlin ──
        if exec_result.clarification_needed:
            logger.info(
                "[CLARIFICATION] Pausing for user: %s",
                exec_result.clarification_needed["question"][:100],
            )
            return exec_result.clarification_needed

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
            plan=plan,
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

        # ── Phase 9B: Preference resolution ──
        if self._pref_resolver is not None:
            plan = self._pref_resolver.resolve_plan(plan)

        # ── Phase 9C/9D: Entity resolution ──
        events = self.timeline.all_events()
        state = WorldState.from_events(events)
        snapshot = WorldSnapshot.build(state, events[-10:] if events else [])

        if self._entity_resolver is not None:
            try:
                plan = self._entity_resolver.resolve_plan(
                    plan, world_snapshot=state,
                )
            except EntityResolutionError as ere:
                # Browser NOT_FOUND → recovery recompile
                browser_not_found = [
                    v for v in ere.violations
                    if v.resolution_type == "not_found_browser"
                ]
                other_violations = [
                    v for v in ere.violations
                    if v.resolution_type != "not_found_browser"
                ]
                if other_violations:
                    logger.info("Entity resolution clarification: %s", ere)
                    return ere.user_message()
                if browser_not_found:
                    logger.info(
                        "[RECOVERY] Browser entity not found — "
                        "triggering recompile",
                    )
                    recovery_failures = [{
                        "skill": v.skill,
                        "status": "entity_not_found",
                        "reason": (
                            f"Entity '{v.raw_value}' not present on page. "
                            f"Available: {', '.join(v.candidates[:3])}"
                            if v.candidates else
                            f"Entity '{v.raw_value}' not present on page."
                        ),
                        "severity": "soft_failure",
                    } for v in browser_not_found]
                    try:
                        recovery_ws = state.model_dump()
                        recovery_plan = self.cortex.compile(
                            user_query=user_text,
                            world_state_schema=recovery_ws,
                            conversation=conversation,
                            execution_failures=recovery_failures,
                        )
                        if isinstance(recovery_plan, MissionPlan):
                            plan = recovery_plan
                        else:
                            return ere.user_message()
                    except Exception:
                        return ere.user_message()

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
            plan=plan,
        )

        return report

    def handle_resumed_plan(
        self,
        plan: MissionPlan,
        user_text: str,
        conversation,
        override_outputs: Dict[str, Dict[str, Any]],
    ) -> Optional[str]:
        """Execute a previously-compiled plan with pre-seeded outputs.

        Used for clarification resume: the user selected a match
        from an ambiguous list, so we inject it as the override output
        for the ambiguous node and resume execution from downstream.

        Skips: LLM compilation, parameter resolution, entity resolution.
        These were already done on the first run.

        Does: execution (with override_outputs) → outcome → report →
              conversation state update.

        Args:
            plan: The SAME MissionPlan from the paused first run.
            user_text: Original user query (for reporting).
            conversation: ConversationFrame for state update.
            override_outputs: {node_id: outputs} for pre-completed nodes.
        """
        # Rebuild snapshot for execution context
        events = self.timeline.all_events()
        state = WorldState.from_events(events)
        snapshot = WorldSnapshot.build(state, events[-10:] if events else [])

        # ── Transition: → EXECUTING ──
        if self._attention:
            from runtime.attention import MissionState
            self._attention.set_mission_state(MissionState.EXECUTING)

        logger.info(
            "[RESUME] Executing resumed plan '%s' with %d override(s)",
            plan.id, len(override_outputs),
        )

        # Execute with override outputs — pre-completed nodes are skipped
        exec_result = self.run(
            plan, snapshot,
            override_outputs=override_outputs,
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

        # Update conversation state
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
            plan=plan,
        )

        return report


    # ─────────────────────────────────────────────────────────
    # PUBLIC API: Unified skill result rendering (Phase 14)
    # ─────────────────────────────────────────────────────────

    # Reason-based responses for terse formatting (state-aware)
    _REASON_RESPONSES = {
        "already_playing": "Already playing.",
        "already_paused": "Already paused.",
        "already_muted": "Already muted.",
        "already_unmuted": "Already unmuted.",
        "no_media_session": "No media session detected.",
    }

    def render_skill_result(
        self,
        skill_name: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        metadata: Optional[Dict[str, Any]],
        output_style: str,
        user_query: str,
        snapshot: WorldSnapshot,
        conversation: ConversationFrame,
    ) -> str:
        """Unified rendering for reflex skill results.

        Single entrypoint that handles all three output styles.
        Keeps merlin.py decoupled from ReportBuilder internals.

        Args:
            skill_name:   e.g. "system.list_jobs"
            inputs:       resolved inputs dict
            outputs:      skill result outputs dict
            metadata:     skill result metadata dict
            output_style: "terse" | "templated" | "rich"
            user_query:   original user text
            snapshot:     current world snapshot
            conversation: conversation frame
        """
        if output_style == "rich":
            return self.report_builder.build_from_skill_result(
                skill_name=skill_name,
                inputs=inputs,
                outputs=outputs,
                user_query=user_query,
                snapshot=snapshot,
                conversation=conversation,
            )

        if output_style == "templated":
            template = metadata.get("response_template") if metadata else None
            if template:
                try:
                    return template.format(**outputs)
                except (KeyError, IndexError):
                    pass  # fall through to terse

        # ── Terse (default) ──
        return self._format_terse_response(skill_name, outputs, metadata)

    def _format_terse_response(
        self,
        skill_name: str,
        outputs: Dict[str, Any],
        metadata: Optional[Dict[str, Any]],
    ) -> str:
        """Deterministic formatting for terse reflex responses.

        Priority:
        1. reason-based (state-aware: already_playing, no_media_session)
        2. changed-flag (play → Playing., pause → Paused.)
        3. generic output dump (query skills without template)
        4. "Done." fallback
        """
        reason = metadata.get("reason") if metadata else None

        # Reason-based responses take priority
        if reason and reason in self._REASON_RESPONSES:
            return self._REASON_RESPONSES[reason]

        # Changed flag
        changed = outputs.get("changed")
        if changed is True:
            if "play" in skill_name:
                return "Playing."
            elif "pause" in skill_name:
                return "Paused."
            elif "mute" in skill_name:
                return "Muted." if "unmute" not in skill_name else "Unmuted."
            elif "next" in skill_name:
                return "Next track."
            elif "previous" in skill_name:
                return "Previous track."
        elif changed is False and reason:
            return self._REASON_RESPONSES.get(reason, f"Done. ({skill_name})")

        # Generic output dump
        if outputs:
            data_outputs = {
                k: v for k, v in outputs.items()
                if v is not None and v != "unknown" and k != "changed"
            }
            if data_outputs:
                parts = [f"{k}: {v}" for k, v in data_outputs.items()]
                return ", ".join(parts)

        return f"Done. ({skill_name})"


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
        plan: Optional[MissionPlan] = None,
    ) -> None:
        """Populate structured conversation state after mission execution.

        Updates:
        - entity_registry: typed EntityRecords from artifacts + visible_lists
        - last_results: full node→output structure (preserves provenance)
        - recent_intents: user text (truncated)

        When plan is provided, derives semantic output types from skill
        contracts for richer entity_type tagging (e.g., 'file_ref_list'
        instead of generic 'list').

        Never raises — failures logged and swallowed.
        """
        try:
            # Build node_id.output_key → semantic_type map from plan
            output_type_map: Dict[str, str] = {}
            if plan is not None:
                for node in plan.nodes:
                    skill = self.executor.registry.get(node.skill)
                    if skill and hasattr(skill, 'contract'):
                        for out_name, sem_type in skill.contract.outputs.items():
                            map_key = f"{node.id}.{out_name}"
                            output_type_map[map_key] = sem_type

            # 7a. Entity registry — register visible_lists as typed entities
            for key, items in outcome.visible_lists.items():
                clean_key = key.split(".", 1)[-1] if "." in key else key
                # Use semantic output type from contract if available
                entity_type = output_type_map.get(key, "list")
                conversation.register_entity(
                    key=clean_key,
                    value=items,
                    entity_type=entity_type,
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
        cognitive_ctx=None,
        decision_engine=None,
        override_outputs=None,
    ) -> ExecutionResult:
        """
        Execute a mission DAG. Returns typed ExecutionResult.

        This is the low-level execution path — no reporting.
        Used by handle_user_input() and available for testing.

        Args:
            cognitive_ctx: Optional CognitiveContext for assumption gating
                and uncertainty tracking. Forwarded to supervisor.
            decision_engine: Optional DecisionEngine for inline recovery.
                Forwarded to supervisor.
            override_outputs: Optional dict of {node_id: outputs} for
                pre-completed nodes. Forwarded to supervisor for
                clarification resume.
        """

        # ── Build fresh SkillContext per mission (not per-startup) ──
        # Keeps time accurate and user profile up-to-date.
        try:
            from execution.skill_context import SkillContext, UserProfile
            from datetime import datetime

            user_knowledge = getattr(self, '_user_knowledge', None)
            if user_knowledge is not None:
                profile = UserProfile.from_profile_dict(
                    user_knowledge.get_user_profile()
                )
            else:
                profile = UserProfile()

            context = SkillContext(user=profile, time=datetime.now())
            self.executor.set_context(context)
        except Exception as e:
            logger.debug("SkillContext build failed (non-fatal): %s", e)

        # Route through supervisor (guard enforcement) or executor (direct)
        if self._supervisor is not None:
            future: Future = self.pool.submit(
                self._supervisor.run, mission, world_snapshot,
                on_layer_start, on_layer_complete,
                cognitive_ctx, decision_engine,
                override_outputs,
            )
        else:
            future: Future = self.pool.submit(
                self.executor.run, mission, world_snapshot,
                on_layer_start, on_layer_complete,
            )

        # Wait for full DAG execution to finish
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
            recovery_attempted=exec_result.recovery_explanation is not None,
        )

    @staticmethod
    def _plans_equivalent(plan_a: MissionPlan, plan_b: MissionPlan) -> bool:
        """Check if two plans are structurally equivalent.

        Compares skill names, input keys, and order.
        NOT graph isomorphism — just a practical dedup check.
        """
        if len(plan_a.nodes) != len(plan_b.nodes):
            return False
        for na, nb in zip(plan_a.nodes, plan_b.nodes):
            if na.skill != nb.skill:
                return False
            if set(na.inputs.keys()) != set(nb.inputs.keys()):
                return False
        return True

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
            logger.warning(
                "[ORCHESTRATOR] Incomplete coverage: %s",
                failure.error_message,
            )
            return (
                "I couldn't fully handle everything you asked for. "
                "Could you try rephrasing or simplifying your request?"
            )

        logger.warning(
            "[ORCHESTRATOR] Plan failure (%s): %s",
            failure.error_type, failure.error_message,
        )
        return (
            "Something went wrong while preparing that task. "
            "Would you like me to try again?"
        )
