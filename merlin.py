# merlin.py

"""
MERLIN — The Conductor.

This is the single authority that:
- Owns all component instances
- Routes percepts to the correct cognitive path
- Manages conversation lifecycle
- Starts and stops the runtime event loop

Design rules:
- merlin.py coordinates. It does not reason.
- All decisions are delegated to deterministic components.
- LLM is NEVER called from this file.
- There is exactly ONE Merlin instance per process.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional
import logging

from brain.core import BrainCore, CognitiveRoute, Percept
from brain.escalation_policy import (
    EscalationPolicy, EscalationDecision,
    CognitiveTier, HeuristicTierClassifier,
)
from cortex.mission_cortex import MissionCortex, DecompositionResult
from execution.executor import MissionExecutor
from execution.registry import SkillRegistry
from world.timeline import WorldTimeline
from world.state import WorldState
from world.snapshot import WorldSnapshot
from world.resolver import WorldResolver
from conversation.frame import ConversationFrame
from orchestrator.mission_orchestrator import MissionOrchestrator
from runtime.event_loop import RuntimeEventLoop
from runtime.reflex_engine import ReflexEngine, ReflexResult
from runtime.sources.base import EventSource
from runtime.attention import AttentionManager, MissionState
from reporting.report_builder import ReportBuilder
from reporting.output import OutputChannel
from reporting.notification_policy import NotificationPolicy
from cortex.cognitive_coordinator import (
    CognitiveCoordinator, CoordinatorMode, FALLBACK_RESULT,
)
from brain.structural_classifier import SpeechActType
from models.base import LLMClient
from cortex.world_state_provider import WorldStateProvider, SimpleWorldStateProvider
from memory.store import MemoryStore


logger = logging.getLogger(__name__)


@dataclass
class PendingMission:
    """Suspended mission context awaiting user response.

    Immutable after creation. One-shot consumption.
    Consumed or discarded on the next user percept.

    Kinds:
        clarification: EscalationPolicy → CLARIFY.
            Stores original percept + snapshot.
            User's response is merged and mission re-enters pipeline.
        partial: Decomposer found unsupported intents.
            Stores valid + unsupported intents + snapshot.
            User confirms → resume with valid only.
    """
    kind: Literal["clarification", "partial"]
    original_percept: Percept
    snapshot: WorldSnapshot
    question: str
    tier: Optional[CognitiveTier] = None
    valid_intents: Optional[List[Dict[str, Any]]] = None
    unsupported_intents: Optional[List[Dict[str, Any]]] = None
    created_at: float = field(default_factory=time.time)


class Merlin:
    """
    The conductor. Owns everything. Reasons about nothing.
    """

    def __init__(
        self,
        brain: BrainCore,
        escalation_policy: EscalationPolicy,
        cortex: MissionCortex,
        registry: SkillRegistry,
        timeline: WorldTimeline,
        reflex_engine: ReflexEngine,
        report_builder: ReportBuilder,
        output_channel: OutputChannel,
        notification_policy: NotificationPolicy,
        event_sources: List[EventSource],
        max_workers: int = 4,
        node_timeout: float | None = None,
        clarifier_llm: Optional[LLMClient] = None,
        world_state_provider: Optional[WorldStateProvider] = None,
        memory: Optional[MemoryStore] = None,
        attention_manager: Optional[AttentionManager] = None,
        narration_policy=None,  # Optional[NarrationPolicy]
        coordinator: Optional[CognitiveCoordinator] = None,
        scheduler: Optional["TickSchedulerManager"] = None,
        completion_queue: Optional["CompletionQueue"] = None,
        supervisor=None,
    ):
        # ── Cognitive components (frozen) ──
        self.brain = brain
        self.escalation_policy = escalation_policy
        self.clarifier_llm = clarifier_llm
        self.world_state_provider = world_state_provider or SimpleWorldStateProvider()

        # ── Attention arbitration ──
        self.attention_manager = attention_manager

        # ── Reflex engine (inject executor for contract enforcement) ──
        self.reflex_engine = reflex_engine

        # ── World state (shared mutable) ──
        self.timeline = timeline

        # ── Conversation (working context) ──
        self.conversation = ConversationFrame()

        # ── Shared executor (used by orchestrator, reflex engine, and scheduler) ──
        executor = MissionExecutor(
            registry, timeline,
            max_workers=max_workers,
            node_timeout=node_timeout,
        )
        self.executor = executor  # exposed for job_executor callback
        self.reflex_engine.executor = executor

        # ── Execution ──
        self.orchestrator = MissionOrchestrator(
            cortex=cortex,
            executor=executor,
            timeline=timeline,
            report_builder=report_builder,
            output_channel=output_channel,
            max_workers=max_workers,
            memory=memory,
            attention_manager=attention_manager,
            narration_policy=narration_policy,
            supervisor=supervisor,
        )

        # ── Tier classification (Phase 5A: deterministic, init-time) ──
        self.tier_classifier = HeuristicTierClassifier(registry)

        # ── Cognitive Coordinator (Phase 1: bounded reasoning pre-phase) ──
        # Optional — if None, REASONING tier degrades to SIMPLE.
        self.coordinator = coordinator

        # ── Job Scheduling Subsystem ──
        self.scheduler = scheduler
        self.completion_queue = completion_queue

        # ── Output ──
        self.output_channel = output_channel
        self.report_builder = report_builder

        # ── Pending mission state (suspend/resume) ──
        # Stores mission context when waiting for user response.
        # Consumed on next percept, then cleared. Never recomputed.
        # Covers: clarification, partial capability confirmation.
        self._pending_mission: Optional[PendingMission] = None

        # ── Deterministic confirmation tokens ──
        self._CONFIRM_TOKENS = frozenset({
            "yes", "proceed", "continue", "do it", "go ahead",
            "sure", "ok", "okay", "yep", "yeah", "y",
        })
        self._DECLINE_TOKENS = frozenset({
            "no", "cancel", "stop", "abort", "nevermind",
            "never mind", "don't", "nope", "n",
        })

        # ── Runtime event loop (background thread) ──
        # job_executor is raw executor.run — the scheduler thread must NEVER
        # touch orchestrator, ConversationFrame, or AttentionManager state.
        # Output delivery happens via CompletionQueue → _drain_completions
        # → AttentionManager (on the event loop thread).
        self.event_loop = RuntimeEventLoop(
            timeline=timeline,
            reflex_engine=reflex_engine,
            sources=event_sources,
            notification_policy=notification_policy,
            report_builder=report_builder,
            output_channel=output_channel,
            get_conversation=lambda: self.conversation,
            attention_manager=attention_manager,
            scheduler=scheduler,
            completion_queue=completion_queue,
            job_executor=executor.run if scheduler else None,
        )

    # ─────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background event loop."""
        logger.info("MERLIN starting...")
        self.event_loop.start()
        logger.info("MERLIN runtime event loop started.")

    def stop(self) -> None:
        """Stop the background event loop."""
        logger.info("MERLIN shutting down...")
        self.event_loop.stop()

    # ─────────────────────────────────────────────────────────
    # Percept handling (the main entry point)
    # ─────────────────────────────────────────────────────────

    def handle_percept(self, percept: Percept) -> Optional[str]:
        """
        Route a percept through the cognitive pipeline.

        History writes at system boundaries:
        - User turn: immediately here (entry point)
        - Assistant turn: after each handler returns

        Two-gate routing:
        Gate 1: BrainCore (circuit breaker)
            → REFUSE: reject immediately
            → REFLEX: try template match, fallback to MISSION
            → MISSION: proceed to Gate 2

        Gate 2: EscalationPolicy (triage, only for MISSION)
            → MISSION: full cortex compilation
            → CLARIFY: ask user for more context
            → IGNORE: silence
        """

        # ── Record user turn (system boundary) ──
        self.conversation.append_turn("user", percept.payload)

        # ── Gate 0: Pending mission response (clarification or partial) ──
        if self._pending_mission is not None:
            return self._handle_pending_response(percept)

        # ── Gate 1: BrainCore circuit breaker ──
        route = self.brain.route(percept)

        logger.info(
            "BrainCore route for '%s': %s",
            percept.payload[:50],
            route,
        )

        if route == CognitiveRoute.REFUSE:
            return self._handle_refuse(percept)

        # ── Build world snapshot ONCE, UPSTREAM ──
        # Invariant: both REFLEX and MISSION paths see the same world.
        events = self.timeline.all_events()
        state = WorldState.from_events(events)
        snapshot = WorldSnapshot.build(
            state, events[-10:] if events else []
        )

        if route == CognitiveRoute.REFLEX:
            return self._handle_reflex(percept, snapshot)

        if route == CognitiveRoute.MULTI_REFLEX:
            return self._handle_multi_reflex(percept, snapshot)

        # ── Speech-act interception: PREFERENCE / DECLARATION ──
        # Catches knowledge declarations BEFORE escalation policy.
        # Only for single-clause queries \u2014 multi-clause queries go through
        # the full decomposition pipeline where memory skills (memory.set_preference,
        # memory.add_policy) handle PREFERENCE/MEMORY_WRITE clauses as DAG nodes.
        features = self.brain.last_features
        is_single_clause = not (
            features.is_multi_clause if features and hasattr(features, "is_multi_clause") else False
        )
        if features and is_single_clause and features.speech_act in (
            SpeechActType.PREFERENCE,
            SpeechActType.DECLARATION,
        ):
            return self._handle_knowledge_declaration(
                percept, features.speech_act,
            )

        # ── Gate 2: EscalationPolicy (only reached for MISSION) ──
        decision = self.escalation_policy.decide_for_user_input(
            user_text=percept.payload,
            snapshot=snapshot,
            frame=self.conversation,
            features=self.brain.last_features,
        )

        logger.info(
            "Escalation decision for '%s': %s",
            percept.payload[:50],
            decision.value,
        )

        if decision == EscalationDecision.IGNORE:
            self.output_channel.send_silent()
            return None

        if decision == EscalationDecision.CLARIFY:
            return self._handle_clarify(percept, snapshot)

        return self._handle_mission(percept, snapshot)

    # ─────────────────────────────────────────────────────────
    # Route handlers
    # ─────────────────────────────────────────────────────────

    def _handle_knowledge_declaration(
        self, percept: Percept, speech_act: SpeechActType,
    ) -> Optional[str]:
        """Handle PREFERENCE or DECLARATION speech acts.

        Extracts key-value from the user's statement and stores
        it in UserKnowledgeStore. No LLM needed.
        """
        import re
        text = percept.payload
        user_knowledge = getattr(self, '_user_knowledge', None)

        if user_knowledge is None:
            logger.warning(
                "[KNOWLEDGE] No UserKnowledgeStore — falling back to mission"
            )
            return self._handle_mission(percept, self._build_snapshot())

        # Extract key-value from common patterns
        # "my preferred volume is 80" → ("volume", "80")
        # "my name is Alex" → ("name", "Alex")
        m = re.search(
            r"\bmy\s+(?:preferred|favorite|favourite|usual|default)?\s*"
            r"(\w+)\s+is\s+(.+)",
            text, re.IGNORECASE,
        )
        if not m:
            # "i prefer volume at 80" → ("volume", "80")
            m = re.search(
                r"\bi\s+(?:prefer|like)\s+(?:my\s+)?(\w+)\s+(?:at|to\s+be)\s+(.+)",
                text, re.IGNORECASE,
            )

        if not m:
            logger.info(
                "[KNOWLEDGE] Could not extract key-value from '%s' — "
                "falling back to mission",
                text[:50],
            )
            return self._handle_mission(percept, self._build_snapshot())

        key = m.group(1).strip()
        raw_value = m.group(2).strip().rstrip(".,!")

        # Try numeric coercion
        value: object = raw_value
        try:
            value = int(raw_value)
        except ValueError:
            try:
                value = float(raw_value)
            except ValueError:
                pass

        # Store based on speech act type
        try:
            if speech_act == SpeechActType.PREFERENCE:
                user_knowledge.set_preference(key, value)
                response = f"Got it. I'll remember your preferred {key} is {value}."
            else:  # DECLARATION
                user_knowledge.set_fact(key, value)
                response = f"Got it. I'll remember your {key} is {value}."

            logger.info(
                "[KNOWLEDGE] Stored %s: %s = %r from '%s'",
                speech_act.value, key, value, text[:50],
            )
        except ValueError as e:
            response = f"I couldn't store that: {e}"
            logger.warning("[KNOWLEDGE] Validation failed: %s", e)

        self.conversation.append_turn("assistant", response)
        self.output_channel.send(response)
        return response

    def _build_snapshot(self) -> "WorldSnapshot":
        """Build a fresh WorldSnapshot. Helper for fallback paths."""
        events = self.timeline.all_events()
        state = WorldState.from_events(events)
        return WorldSnapshot.build(
            state, events[-10:] if events else []
        )

    def _check_destructive_nodes(self, plan: "MissionPlan") -> list:
        """Scan a compiled MissionPlan for destructive nodes.

        Returns list of MissionNode objects whose skill contract has
        risk_level='destructive'. Used pre-execution to surface
        confirmation prompts via PendingMission flow.

        Never raises — returns empty list on error so mission proceeds.
        """
        try:
            from ir.mission import MissionPlan  # noqa: F401 (avoids circular)
            destructive = []
            for node in plan.nodes:
                skill = self.executor.registry.get(node.skill)
                if skill is None:
                    continue
                risk = getattr(skill.contract, "risk_level", "safe")
                if risk == "destructive":
                    destructive.append(node)
            return destructive
        except Exception as e:
            logger.debug("[SAFETY] _check_destructive_nodes failed: %s", e)
            return []

    def _schedule_decomposed_clause(
        self,
        sched_clause: dict,
        percept: "Percept",
        snapshot: "WorldSnapshot",
    ) -> bool:
        """Schedule a SCHEDULED clause via TickSchedulerManager.

        Reuses the existing TemporalResolver + compile + submit pipeline.

        Handles both explicit actions ("mute in 10 min") and implicit
        actions ("remind me every hour" → default notify/reminder).

        Never raises — scheduling failures are logged and silently dropped
        so that the rest of the mission still executes.

        Returns True if the task was successfully submitted, False otherwise.
        """
        try:
            if not self.scheduler:
                logger.warning(
                    "[SCHEDULED] No scheduler available for clause: %s",
                    sched_clause.get("action", "?"),
                )
                return False

            from runtime.temporal_resolver import TemporalResolver
            from runtime.task_store import Task, TaskSchedule, TaskStatus, TaskType
            from ir.mission import IR_VERSION
            from errors import FailureIR
            import uuid
            import time as _time

            trigger_expr = sched_clause.get("trigger", "")
            action = sched_clause.get("action", "")
            parameters = sched_clause.get("parameters", {})

            # ── Implicit action default ──
            # Queries like "remind me every hour" have no explicit skill action.
            # Default to a notification/reminder action so the scheduler has
            # something to execute.
            if not action or action.lower() in (
                "remind", "reminder", "notify", "notification",
                "alert", "ping",
            ):
                # Use the original query text as the notification message
                original_text = percept.payload if percept else ""
                notify_msg = parameters.get("message", original_text)
                action = "notify"
                parameters = {"message": notify_msg}

            # Build a natural-language deferred query from clause fields
            param_desc = " ".join(
                f"{k}={v}" for k, v in parameters.items()
            )
            deferred_query = f"{action} {param_desc}".strip()

            # Instantiate resolver for trigger time computation
            resolver = TemporalResolver()

            # ── Determine task type from trigger expression ──
            trigger_lower = trigger_expr.lower()
            if any(kw in trigger_lower for kw in ("every", "recurring", "repeat")):
                task_type = TaskType.RECURRING
            elif any(kw in trigger_lower for kw in ("at ", "pm", "am", ":")):
                task_type = TaskType.SCHEDULED
            else:
                task_type = TaskType.DELAYED

            # ── Resolve trigger time based on detected type ──
            now = _time.time()
            repeat_interval = None
            max_repeats = 1  # default: single execution

            if task_type == TaskType.RECURRING:
                recurring = resolver.resolve_recurring(trigger_expr)
                if recurring:
                    repeat_interval = recurring.interval_seconds
                    max_repeats = recurring.max_repeats
                    next_run = now + recurring.interval_seconds
                else:
                    # Fallback: parse as delay if recurring parsing fails
                    next_run = resolver.resolve(
                        {"expression": trigger_expr, "kind": "delay"}
                    )
            elif task_type == TaskType.SCHEDULED:
                next_run = resolver.resolve(
                    {"expression": trigger_expr, "kind": "absolute_time"}
                )
            else:
                next_run = resolver.resolve(
                    {"expression": trigger_expr, "kind": "delay"}
                )

            if next_run is None:
                logger.warning(
                    "[SCHEDULED] Could not resolve trigger '%s' for clause '%s'",
                    trigger_expr, action,
                )
                response = (
                    f"I understood '{action}' but couldn't parse the timing "
                    f"'{trigger_expr}'. Could you rephrase?"
                )
                self.output_channel.send(response)
                return False

            # Compute delay for TaskSchedule (used by DELAYED type)
            delay_secs = max(0, int(next_run - now))


            # Compile the deferred action NOW (at schedule time, not dispatch time)
            compiled = self.orchestrator.cortex.compile(
                user_query=deferred_query,
                world_state_schema=self.world_state_provider.build_schema(
                    snapshot, query=deferred_query,
                ),
                conversation=self.conversation,
            )

            if isinstance(compiled, FailureIR):
                logger.warning(
                    "[SCHEDULED] Could not compile '%s': %s",
                    deferred_query, compiled.error_message,
                )
                return False

            task = Task(
                id=str(uuid.uuid4()),
                type=task_type,
                query=deferred_query,
                next_run=next_run,
                status=TaskStatus.PENDING,
                schedule=TaskSchedule(
                    delay_seconds=delay_secs if task_type == TaskType.DELAYED else None,
                    schedule_at=next_run if task_type == TaskType.SCHEDULED else None,
                    repeat_interval=repeat_interval,
                    max_repeats=max_repeats,
                    time_expression=trigger_expr,
                ),
                metadata=(
                    {"total_repeats": 0}
                    if task_type == TaskType.RECURRING else {}
                ),
                mission_data={
                    "compiled_plan": compiled.model_dump(),
                    "deferred_query": deferred_query,
                    "ir_version": IR_VERSION,
                },
            )
            self.scheduler.submit(task)
            logger.info(
                "[SCHEDULED] Clause '%s' scheduled for epoch=%.1f (type=%s)",
                action, next_run, task_type.value,
            )
            return True
        except Exception as e:
            logger.warning(
                "[SCHEDULED] Failed to schedule clause '%s': %s",
                sched_clause.get("action", "?"), e, exc_info=True,
            )
            return False

    def _handle_refuse(self, percept: Percept) -> str:
        """Reject a dangerous or prohibited command."""
        response = "I'm not able to do that."
        self.output_channel.send(response)
        self.conversation.append_turn("assistant", response)
        return response

    def _handle_reflex(
        self, percept: Percept, snapshot: WorldSnapshot,
    ) -> Optional[str]:
        """
        Handle reflex-level commands.

        Flow:
        1. Try template match (pattern → parameters → skill)
        2. Resolve parameters (alias coercion: "full"→100, "half"→50)
        3. Execute directly (no LLM narration, no LLM report)
        4. Format response from skill metadata (state-aware)
        5. If no match → fallback to MISSION (safe bias)

        Why not orchestrator? Reflex must be fast (<200ms).
        Orchestrator adds LLM narration + LLM report generation.
        ParameterResolver is injected directly here instead.
        """

        # Try template matching
        reflex_match = self.reflex_engine.try_match(percept.payload)

        if reflex_match:
            logger.info(
                "Reflex template matched: skill=%s params=%s",
                reflex_match.skill,
                reflex_match.params,
            )

            # ── Phase 9A: Resolve parameters (alias coercion) ──
            # Build a temporary plan so resolver can do typed coercion
            import time as _time
            from ir.mission import MissionPlan, MissionNode, IR_VERSION
            from cortex.parameter_resolver import ParameterResolver, ParameterError

            temp_plan = MissionPlan(
                id=f"reflex_{int(_time.time())}",
                nodes=[
                    MissionNode(
                        id="reflex_0",
                        skill=reflex_match.skill,
                        inputs=reflex_match.params,
                    )
                ],
                metadata={"ir_version": IR_VERSION},
            )

            try:
                resolver = ParameterResolver(self.reflex_engine.registry)
                resolved_plan = resolver.resolve_plan(temp_plan)
                # Update match params with resolved values
                resolved_inputs = resolved_plan.nodes[0].inputs
                from runtime.reflex_engine import ReflexMatch
                reflex_match = ReflexMatch(
                    skill=reflex_match.skill,
                    params=resolved_inputs,
                )
            except ParameterError as pe:
                response = pe.user_message()
                self.output_channel.send(response)
                self.conversation.append_turn("assistant", response)
                return response

            # Execute directly — no narration, no LLM report
            result: ReflexResult = self.reflex_engine.execute_reflex(
                reflex_match, snapshot=snapshot,
            )

            if result.success:
                # Route through unified rendering pipeline (Phase 14)
                contract = self.reflex_engine.registry.get(
                    reflex_match.skill,
                ).contract
                response = self.orchestrator.render_skill_result(
                    skill_name=reflex_match.skill,
                    inputs=reflex_match.params,
                    outputs=result.outputs,
                    metadata=result.metadata,
                    output_style=contract.output_style,
                    user_query=percept.payload,
                    snapshot=snapshot,
                    conversation=self.conversation,
                )
                self.output_channel.send(response)
                self.conversation.append_turn("assistant", response)
                return response
            else:
                # Reflex is authoritative — report failure, NEVER escalate
                response = f"Couldn't do that: {result.error}"
                self.output_channel.send(response)
                self.conversation.append_turn("assistant", response)
                return response

        # No template match → escalate to MISSION (snapshot already built)
        logger.info(
            "No reflex template matched for '%s', escalating to MISSION",
            percept.payload[:50],
        )
        return self._handle_mission(percept, snapshot)


    def _handle_multi_reflex(
        self, percept: Percept, snapshot: WorldSnapshot,
    ) -> Optional[str]:
        """
        Handle conjunction commands deterministically. Zero LLM.

        Flow:
        1. Split on conjunctions, match each clause
        2. Build multi-node plan from matches
        3. Route through orchestrator (gets resolve + narrate + report for free)
        4. Full lifecycle management (EXECUTING → REPORTING → IDLE)
        """
        try:
            # Use cached matches from route() — avoid duplicate work
            matches = self.reflex_engine._last_multi_matches
            if not matches:
                # Safety fallback — re-match if cache was somehow cleared
                matches = self.reflex_engine.try_match_multi(percept.payload)
            # Clear cache after consumption
            self.reflex_engine._last_multi_matches = None

            if not matches:
                # Safety fallback — should not happen since route already matched
                return self._handle_mission(percept, snapshot)

            logger.info(
                "Multi-reflex: %d clauses → %s",
                len(matches),
                [m.skill for m in matches],
            )

            plan = self.reflex_engine.execute_multi_reflex(matches)

            result = self.orchestrator.handle_prebuilt_plan(
                plan=plan,
                user_text=percept.payload,
                conversation=self.conversation,
            )

            if result:
                self.conversation.append_turn(
                    "assistant", result,
                    mission_id=self.conversation.last_mission_id,
                )
            return result

        except Exception as e:
            # Safety net: fall back to mission if multi-reflex fails
            if self.attention_manager:
                from runtime.attention import MissionState
                self.attention_manager.set_mission_state(MissionState.IDLE)

            logger.error(
                "Multi-reflex failed for '%s': %s",
                percept.payload[:80], e, exc_info=True,
            )
            response = f"I couldn't complete that request. Error: {e}"
            self.output_channel.send(response)
            return response

    # ─────────────────────────────────────────────────────────
    # Partial capability gate (deterministic capability boundary)
    # ─────────────────────────────────────────────────────────

    def _handle_partial_capability(
        self,
        percept: Percept,
        decomp: DecompositionResult,
        snapshot: WorldSnapshot,
        tier: CognitiveTier,
    ) -> str:
        """Gate: unsupported intents detected. Ask user to confirm partial execution.

        If zero valid intents → deterministic rejection, no dialogue.
        If some valid intents → store PendingMission, ask confirmation.
        State is consumed on next percept via _handle_pending_response.
        """
        unsupported_desc = "; ".join(
            u["description"] for u in decomp.unsupported_intents
            if u.get("description")
        ) or "some parts of your request"

        if not decomp.valid_intents:
            # All-unsupported: deterministic rejection, no pending
            logger.info(
                "[CAPABILITY] All intents unsupported — rejecting: %s",
                unsupported_desc[:200],
            )
            response = (
                f"I'm not able to perform any part of this request. "
                f"Unsupported: {unsupported_desc}."
            )
            # Reset mission state since nothing will execute
            if self.attention_manager:
                self.attention_manager.set_mission_state(MissionState.IDLE)
            self.output_channel.send(response)
            self.conversation.append_turn("assistant", response)
            return response

        # Some valid, some unsupported: store state and ask
        logger.info(
            "[CAPABILITY] Partial capability — valid: %d, unsupported: %d",
            len(decomp.valid_intents), len(decomp.unsupported_intents),
        )

        response = (
            f"I can handle part of your request, "
            f"but I'm not able to: {unsupported_desc}. "
            f"Should I proceed with what I can do?"
        )

        # Store immutable PendingMission — no recomputation on confirmation
        self._pending_mission = PendingMission(
            kind="partial",
            original_percept=percept,
            snapshot=snapshot,
            question=response,
            tier=tier,
            valid_intents=decomp.valid_intents,
            unsupported_intents=decomp.unsupported_intents,
        )

        # Keep mission state as COMPILING — we're waiting for confirmation
        self.output_channel.send(response)
        self.conversation.append_turn("assistant", response)
        return response

    # ─────────────────────────────────────────────────────────
    # Unified pending response handler (suspend/resume)
    # ─────────────────────────────────────────────────────────

    def _handle_pending_response(self, percept: Percept) -> Optional[str]:
        """Handle user response to a suspended mission (clarification or partial).

        Deterministic routing by pending.kind:
        - partial: confirm/decline/new-query (existing behavior)
        - clarification: merge answer into original query and resume

        Non-confirmation input (new query) always clears pending
        and re-routes through the normal pipeline.
        """
        pending = self._pending_mission
        self._pending_mission = None  # Always clear — one-shot consumption

        user_text = percept.payload.strip().lower()

        # ── Decline tokens: cancel regardless of kind ──
        if user_text in self._DECLINE_TOKENS:
            logger.info(
                "[PENDING] User declined (%s)", pending.kind,
            )
            if self.attention_manager:
                self.attention_manager.set_mission_state(MissionState.IDLE)
            response = "Understood. Request cancelled."
            self.output_channel.send(response)
            self.conversation.append_turn("assistant", response)
            return response

        # ── Kind-specific handling ──
        if pending.kind == "partial":
            if user_text in self._CONFIRM_TOKENS:
                logger.info("[PENDING] User confirmed partial execution")
                return self._resume_partial_mission(pending)
            # Not a confirmation — fall through to new-query routing
        elif pending.kind == "clarification":
            # Any non-decline response is treated as the clarification answer
            logger.info(
                "[PENDING] Clarification answer received: '%s'",
                percept.payload[:80],
            )
            return self._resume_from_clarification(pending, percept.payload)

        # ── New query: clear pending, route normally ──
        logger.info(
            "[PENDING] New query during pending (%s) — routing normally",
            pending.kind,
        )
        if self.attention_manager:
            self.attention_manager.set_mission_state(MissionState.IDLE)

        # Re-route through normal pipeline (Gate 1 onward)
        route = self.brain.route(percept)
        events = self.timeline.all_events()
        state = WorldState.from_events(events)
        snapshot = WorldSnapshot.build(
            state, events[-10:] if events else []
        )

        if route == CognitiveRoute.REFUSE:
            return self._handle_refuse(percept)
        if route == CognitiveRoute.REFLEX:
            return self._handle_reflex(percept, snapshot)
        if route == CognitiveRoute.MULTI_REFLEX:
            return self._handle_multi_reflex(percept, snapshot)

        decision = self.escalation_policy.decide_for_user_input(
            user_text=percept.payload,
            snapshot=snapshot,
            frame=self.conversation,
        )
        if decision == EscalationDecision.IGNORE:
            self.output_channel.send_silent()
            return None
        if decision == EscalationDecision.CLARIFY:
            return self._handle_clarify(percept, snapshot)
        return self._handle_mission(percept, snapshot)

    def _resume_partial_mission(
        self, pending: PendingMission,
    ) -> Optional[str]:
        """Resume mission compilation with stored valid intents.

        Uses the ORIGINAL snapshot — does not rebuild world state.
        Deterministic decision, non-deterministic environment is acceptable.
        """
        percept = pending.original_percept
        snapshot = pending.snapshot
        tier = pending.tier
        valid_intents = pending.valid_intents
        unsupported_intents = pending.unsupported_intents

        try:
            result = self.orchestrator.handle_user_input(
                user_text=percept.payload,
                conversation=self.conversation,
                world_state_schema=self.world_state_provider.build_schema(
                    snapshot, query=percept.payload,
                ),
                cognitive_tier=tier,
                intent_units=valid_intents,
                unsupported_intents=unsupported_intents,
            )

            if result:
                self.conversation.append_turn(
                    "assistant", result,
                    mission_id=self.conversation.last_mission_id,
                )
            return result

        except Exception as e:
            if self.attention_manager:
                self.attention_manager.set_mission_state(MissionState.IDLE)
            logger.error(
                "Partial mission failed for '%s': %s",
                percept.payload[:80], e, exc_info=True,
            )
            response = f"I couldn't complete that request. Error: {e}"
            self.output_channel.send(response)
            self.conversation.append_turn("assistant", response)
            return response

    def _resume_from_clarification(
        self, pending: PendingMission, user_answer: str,
    ) -> Optional[str]:
        """Resume mission after user answers a clarification question.

        Merges the original query with the user's clarification answer
        and re-enters the mission pipeline with the ORIGINAL snapshot.

        The conversation history already contains the clarification exchange
        (question + answer), so the compiler LLM sees full context.
        """
        # Merge original query + clarification answer
        merged_query = f"{pending.original_percept.payload} ({user_answer})"

        merged_percept = Percept(
            modality=pending.original_percept.modality,
            payload=merged_query,
            confidence=pending.original_percept.confidence,
            timestamp=pending.original_percept.timestamp,
        )

        logger.info(
            "[PENDING] Resuming from clarification: '%s' → '%s'",
            pending.original_percept.payload[:50],
            merged_query[:80],
        )

        # Re-enter mission pipeline with original snapshot
        return self._handle_mission(merged_percept, pending.snapshot)

    # ─────────────────────────────────────────────────────────
    # Clarification handler (now stores context for resume)
    # ─────────────────────────────────────────────────────────

    def _handle_clarify(
        self, percept: Percept, snapshot: WorldSnapshot,
    ) -> str:
        """Ask for clarification when context is insufficient.

        Stores mission context as PendingMission so the user's
        response resumes the mission instead of starting fresh.

        Uses clarifier LLM if available for context-aware questions.
        Falls back to deterministic response otherwise.
        """
        if self.clarifier_llm:
            try:
                prompt = self._build_clarify_prompt(percept)
                logger.info(
                    "[CLARIFY] Sending to LLM (%d char prompt)...",
                    len(prompt),
                )
                response = self.clarifier_llm.complete(prompt)
                # Sanitize — strip any LLM framing
                response = response.strip()
                if not response:
                    response = self._default_clarify_response()
            except Exception:
                response = self._default_clarify_response()
        else:
            response = self._default_clarify_response()

        # Store context for resume
        self._pending_mission = PendingMission(
            kind="clarification",
            original_percept=percept,
            snapshot=snapshot,
            question=response,
        )

        self.output_channel.send(response)
        self.conversation.append_turn("assistant", response)
        return response

    @staticmethod
    def _default_clarify_response() -> str:
        return (
            "I need a bit more context. "
            "Could you clarify what you're referring to?"
        )

    def _build_clarify_prompt(self, percept: Percept) -> str:
        """
        Build a prompt for the clarifier LLM. Includes conversation
        context so the clarifier can ask SPECIFIC questions.
        """
        context_parts = []
        if self.conversation.active_domain:
            context_parts.append(f"Active domain: {self.conversation.active_domain}")
        if self.conversation.active_entity:
            context_parts.append(f"Active entity: {self.conversation.active_entity}")

        recent = self.conversation.history[-3:]
        if recent:
            turn_lines = []
            for t in recent:
                prefix = "User" if t.role == "user" else "Assistant"
                turn_lines.append(f"{prefix}: {t.text[:150]}")
            context_parts.append("Recent turns:\n" + "\n".join(turn_lines))

        context = "\n".join(context_parts) if context_parts else "No prior context."

        return (
            f"The user said: \"{percept.payload}\"\n\n"
            f"Context:\n{context}\n\n"
            "The request is ambiguous. Ask ONE specific, helpful "
            "clarification question. Be concise. Do not explain yourself."
        )

    def _handle_mission(
        self, percept: Percept, snapshot: WorldSnapshot,
        _coordinator_done: bool = False,
    ) -> Optional[str]:
        """
        Compile and execute a full mission.

        Flow:
        1. Coordinator pre-phase (all non-reflex queries go here first)
           - DIRECT_ANSWER → respond immediately (no skills)
           - UNSUPPORTED → explain missing capabilities
           - REASONED_PLAN → mutate percept with refined query → compile
           - SKILL_PLAN → continue to tier classification + compile
        2. Tier classification (deterministic, decomposition complexity only)
        3. Conditional decomposition (Tier 2+ only)
        4. Compile + execute via MissionOrchestrator

        Args:
            percept: The user percept (may be refined by coordinator).
            snapshot: Pre-built WorldSnapshot.
            _coordinator_done: Internal flag to prevent re-entry.
        """
        try:
            # ── Reference resolution (BEFORE cortex) ──
            self._resolve_references(percept.payload)

            # ── Signal mission start to attention manager ──
            if self.attention_manager:
                self.attention_manager.set_mission_state(MissionState.COMPILING)

            # ─────────────────────────────────────────────────
            # Coordinator pre-phase (runs ONCE, before tier)
            # ─────────────────────────────────────────────────
            original_query = percept.payload
            computed_vars = {}

            if self.coordinator and not _coordinator_done:
                try:
                    skill_manifest = (
                        self.orchestrator.cortex._build_skill_manifest()
                    )
                    result = self.coordinator.process(
                        query=percept.payload,
                        snapshot=snapshot,
                        skill_manifest=skill_manifest,
                        user_knowledge=getattr(self, '_user_knowledge', None),
                        speech_act=(
                            self.brain.last_features.speech_act
                            if self.brain.last_features else None
                        ),
                    )

                    logger.info(
                        "[COORDINATOR] '%s' → mode=%s, trace=%s",
                        percept.payload[:50], result.mode.value,
                        result.reasoning_trace[:100]
                        if result.reasoning_trace else "",
                    )

                    # ── DIRECT_ANSWER: respond immediately ──
                    if result.mode == CoordinatorMode.DIRECT_ANSWER:
                        if self.attention_manager:
                            self.attention_manager.set_mission_state(
                                MissionState.IDLE
                            )
                        response = result.answer
                        self.output_channel.send(response)
                        self.conversation.append_turn(
                            "assistant", response
                        )
                        return response

                    # ── REASONED_PLAN: refine percept ──
                    if result.mode == CoordinatorMode.REASONED_PLAN:
                        logger.info(
                            "[COORDINATOR] computed=%s, refined='%s'",
                            result.computed_vars,
                            result.refined_query[:80],
                        )
                        computed_vars = result.computed_vars
                        percept = Percept(
                            modality=percept.modality,
                            payload=result.refined_query,
                            confidence=percept.confidence,
                            timestamp=percept.timestamp,
                        )

                    # SKILL_PLAN: fall through to tier classification

                except Exception as coord_err:
                    logger.warning(
                        "[COORDINATOR] Failed, continuing without: %s",
                        coord_err,
                    )
                    # Continue to compile — safe degradation

            # ─────────────────────────────────────────────────
            # Tier classification (decomposition complexity)
            # ─────────────────────────────────────────────────
            tier = self.tier_classifier.classify(percept.payload)
            logger.info(
                "[TIER] Query '%s' → %s",
                percept.payload[:80], tier.value,
            )

            # ── Deterministic tier upgrade for scheduling queries ──
            # Single-clause scheduling queries ("remind me to X in Y")
            # are classified as SIMPLE by the tier classifier (single verb,
            # no conjunction). But they MUST reach the decomposer so the
            # SCHEDULED clause type is recognized and routed to the scheduler.
            # Without this upgrade, they'd go straight to the compiler which
            # would fail (no "remind" skill exists).
            scheduling_required = (
                self.brain.last_features
                and self.brain.last_features.requires_scheduling
            )
            if scheduling_required and tier == CognitiveTier.SIMPLE:
                tier = CognitiveTier.MULTI_INTENT
                logger.info(
                    "[TIER] Upgraded SIMPLE → MULTI_INTENT "
                    "(scheduling detected by StructuralAnalyzer)"
                )

            # ── Conditional decomposition (Tier 2+ only) ──
            intent_units = None
            unsupported_intents = []
            if tier != CognitiveTier.SIMPLE:
                decomp = self.orchestrator.cortex.decompose_intents(
                    percept.payload,
                )
                if decomp is None:
                    # Decomposition failed → graceful degradation to Tier 1
                    logger.info(
                        "[TIER] Decomposition failed, degrading to Tier 1"
                    )
                    tier = CognitiveTier.SIMPLE
                elif decomp.unsupported_intents and not decomp.executable_intents:
                    # ── CAPABILITY GATE: do NOT compile if ALL unsupported ──
                    return self._handle_partial_capability(
                        percept, decomp, snapshot, tier,
                    )
                else:
                    # ── Typed clause dispatch ──
                    # 1. VAGUE clauses → ask for clarification, skip execution
                    if decomp.vague_intents and not decomp.executable_intents:
                        vague_descs = "; ".join(
                            f"{v.get('text', v.get('action', '?'))} "
                            f"(unclear: {v.get('missing', 'details')})"
                            for v in decomp.vague_intents
                        )
                        response = (
                            f"Could you clarify what you mean? "
                            f"I wasn't sure about: {vague_descs}"
                        )
                        self.output_channel.send(response)
                        self.conversation.append_turn("assistant", response)
                        return response

                    # 2. SCHEDULED clauses → route to TickSchedulerManager
                    sched_ok = []
                    sched_fail = []
                    if decomp.scheduled_intents:
                        for sched_clause in decomp.scheduled_intents:
                            if self._schedule_decomposed_clause(
                                sched_clause, percept, snapshot,
                            ):
                                sched_ok.append(sched_clause)
                            else:
                                sched_fail.append(sched_clause)

                    # 3. INFORMATIONAL clauses → collect for appending to response
                    info_acknowledgements = []
                    for info in decomp.informational_intents:
                        text = info.get("text", "")
                        if text:
                            info_acknowledgements.append(text)

                    # 4. Executable intents → compile + safety check + run
                    if decomp.executable_intents:
                        intent_units = decomp.valid_intents
                        # Log scheduling failures for mixed queries
                        # (user will still see executable results, but
                        #  should know if scheduling part failed)
                        if sched_fail:
                            fail_actions = [
                                s.get("action", "?") for s in sched_fail
                            ]
                            logger.warning(
                                "[SCHEDULED] %d scheduled clause(s) failed "
                                "in mixed query: %s",
                                len(sched_fail), fail_actions,
                            )
                    elif decomp.scheduled_intents:
                        # SCHEDULED-only query (e.g., "remind me to X in Y").
                        # Scheduling is already dispatched above. Return
                        # honest acknowledgment based on actual success/failure.
                        if self.attention_manager:
                            self.attention_manager.set_mission_state(
                                MissionState.IDLE
                            )
                        if sched_ok and not sched_fail:
                            # All succeeded
                            sched_descs = []
                            for s in sched_ok:
                                action = s.get("action", "your request")
                                trigger = s.get("trigger", "")
                                if trigger:
                                    sched_descs.append(
                                        f"{action} ({trigger})"
                                    )
                                else:
                                    sched_descs.append(action)
                            response = (
                                "Got it — scheduled: "
                                + "; ".join(sched_descs)
                                + "."
                            )
                        elif sched_fail and not sched_ok:
                            # All failed
                            response = (
                                "Scheduling failed. "
                                "I couldn't create the reminder."
                            )
                        else:
                            # Partial success
                            ok_descs = [
                                s.get("action", "?") for s in sched_ok
                            ]
                            fail_descs = [
                                s.get("action", "?") for s in sched_fail
                            ]
                            response = (
                                f"Partially scheduled: "
                                f"{'; '.join(ok_descs)}. "
                                f"Failed: {'; '.join(fail_descs)}."
                            )
                        self.output_channel.send(response)
                        self.conversation.append_turn(
                            "assistant", response
                        )
                        return response
                    else:
                        # Nothing to execute — pure INFORMATIONAL/VAGUE
                        info_text = " ".join(
                            i.get("text", "") for i in decomp.informational_intents
                        ).strip()
                        response = (
                            f"Understood. {info_text}" if info_text
                            else "Got it — nothing to execute."
                        )
                        self.output_channel.send(response)
                        self.conversation.append_turn("assistant", response)
                        return response

                    logger.info(
                        "[TIER] Decomposed %d executable, %d scheduled, "
                        "%d informational, %d vague intent units",
                        len(decomp.executable_intents), len(decomp.scheduled_intents),
                        len(decomp.informational_intents), len(decomp.vague_intents),
                    )

            result = self.orchestrator.handle_user_input(
                user_text=percept.payload,
                conversation=self.conversation,
                world_state_schema=self.world_state_provider.build_schema(
                    snapshot, query=percept.payload,
                ),
                cognitive_tier=tier,
                intent_units=intent_units,
                unsupported_intents=unsupported_intents,
                original_query=original_query,
                computed_variables=computed_vars,
            )

            # Orchestrator handles EXECUTING → REPORTING → IDLE lifecycle.
            # No IDLE set here — the orchestrator owns it now.

            if result:
                self.conversation.append_turn(
                    "assistant", result,
                    mission_id=self.conversation.last_mission_id,
                )
            return result
        except Exception as e:
            # ── Ensure mission state returns to IDLE on error ──
            if self.attention_manager:
                self.attention_manager.set_mission_state(MissionState.IDLE)

            logger.error(
                "Mission failed for '%s': %s",
                percept.payload[:80],
                e,
                exc_info=True,
            )
            response = f"I couldn't complete that request. Error: {e}"
            self.output_channel.send(response)
            self.conversation.append_turn("assistant", response)
            return response

    # ─────────────────────────────────────────────────────────
    # Job scheduling handlers
    # ─────────────────────────────────────────────────────────

    def _handle_persistent_job(
        self, percept: Percept, snapshot: WorldSnapshot,
        coord_result: "CoordinatorResult",
    ) -> Optional[str]:
        """Handle PERSISTENT_JOB: schedule a deferred action.

        Flow:
            1. Execute immediate_actions synchronously (if any)
            2. If immediate fails → abort scheduling (preserve logical integrity)
            3. Resolve trigger via TemporalResolver
            4. Compile deferred action into Task
            5. Submit to scheduler
            6. Respond deterministically (no LLM)
        """
        import uuid
        from runtime.temporal_resolver import TemporalResolver
        from runtime.task_store import Task, TaskSchedule, TaskStatus, TaskType

        try:
            if self.attention_manager:
                self.attention_manager.set_mission_state(MissionState.IDLE)

            # ── 1. Execute immediate actions ──
            if coord_result.immediate_actions:
                for action_spec in coord_result.immediate_actions:
                    action_desc = action_spec.get("action", "")
                    if action_desc:
                        logger.info(
                            "[PERSISTENT_JOB] Executing immediate action: %s",
                            action_desc,
                        )
                        try:
                            imm_result = self.orchestrator.handle_user_input(
                                user_text=action_desc,
                                conversation=self.conversation,
                                world_state_schema=self.world_state_provider.build_schema(
                                    snapshot, query=action_desc,
                                ),
                            )
                            if imm_result is None:
                                # Immediate action failed → abort scheduling
                                response = (
                                    f"I couldn't complete the immediate action "
                                    f"'{action_desc}', so I won't schedule "
                                    f"the deferred action."
                                )
                                self.output_channel.send(response)
                                self.conversation.append_turn(
                                    "assistant", response,
                                )
                                return response
                        except Exception as imm_err:
                            response = (
                                f"The immediate action '{action_desc}' failed: "
                                f"{imm_err}. Scheduling aborted."
                            )
                            self.output_channel.send(response)
                            self.conversation.append_turn(
                                "assistant", response,
                            )
                            return response

            # ── 2. Check scheduler availability ──
            if not self.scheduler:
                response = (
                    "Scheduling is not currently available. "
                    "I can only execute actions immediately."
                )
                self.output_channel.send(response)
                self.conversation.append_turn("assistant", response)
                return response

            # ── 3. Resolve trigger time ──
            resolver = TemporalResolver()
            trigger_spec = coord_result.trigger_spec
            next_run = resolver.resolve(trigger_spec)

            if next_run is None:
                response = (
                    f"I couldn't understand the timing: "
                    f"'{trigger_spec.get('expression', '?')}'. "
                    f"Could you rephrase?"
                )
                self.output_channel.send(response)
                self.conversation.append_turn("assistant", response)
                return response

            # ── 4. Compile deferred action into MissionPlan NOW ──
            # Critical: compile once at scheduling time.
            # At dispatch time, execute the compiled plan directly
            # via MissionExecutor.run() — no re-interpretation.
            from ir.mission import MissionPlan, IR_VERSION
            from errors import FailureIR

            deferred_query = coord_result.deferred_action_query
            compiled = self.orchestrator.cortex.compile(
                user_query=deferred_query,
                world_state_schema=self.world_state_provider.build_schema(
                    snapshot, query=deferred_query,
                ),
                conversation=self.conversation,
            )

            if isinstance(compiled, FailureIR):
                response = (
                    f"I understood the timing, but couldn't compile "
                    f"the action '{deferred_query}': {compiled.error_message}"
                )
                self.output_channel.send(response)
                self.conversation.append_turn("assistant", response)
                return response

            # ── 5. Determine task type ──
            kind = trigger_spec.get("kind", "delay")
            if kind == "delay":
                task_type = TaskType.DELAYED
            elif kind == "absolute_time":
                task_type = TaskType.SCHEDULED
            else:
                task_type = TaskType.DELAYED

            # ── 6. Build Task with serialized compiled plan ──
            task = Task(
                id=str(uuid.uuid4()),
                type=task_type,
                query=percept.payload,
                mission_data={
                    "compiled_plan": compiled.model_dump(mode="json"),
                    "ir_version": IR_VERSION,
                    "schema_version": 1,
                    "deferred_query": deferred_query,
                },
                schedule=TaskSchedule(
                    delay_seconds=(
                        resolver.resolve_delay_seconds(
                            trigger_spec.get("expression", ""),
                        )
                        if kind == "delay" else None
                    ),
                    schedule_at=next_run if kind == "absolute_time" else None,
                    time_expression=trigger_spec.get("expression", ""),
                    timezone=resolver.get_local_timezone(),
                ),
                next_run=next_run,
            )

            # ── 7. Submit to scheduler ──
            task_id = self.scheduler.submit(task)
            stored = self.scheduler._store.get(task_id)
            short_id = stored.short_id if stored else "?"

            # ── 8. Build deterministic response ──
            expression = trigger_spec.get("expression", "?")
            plan_nodes = len(compiled.nodes)

            if kind == "delay":
                response = (
                    f"Got it. I'll {deferred_query} in {expression}. "
                    f"(Job {short_id}, {plan_nodes} step{'s' if plan_nodes > 1 else ''} compiled)"
                )
            else:
                response = (
                    f"Scheduled: {deferred_query} at {expression}. "
                    f"(Job {short_id}, {plan_nodes} step{'s' if plan_nodes > 1 else ''} compiled)"
                )

            logger.info(
                "[PERSISTENT_JOB] Submitted %s: '%s' → next_run=%.1f, plan_nodes=%d",
                short_id, deferred_query, next_run, plan_nodes,
            )

            self.output_channel.send(response)
            self.conversation.append_turn("assistant", response)
            return response

        except Exception as e:
            if self.attention_manager:
                self.attention_manager.set_mission_state(MissionState.IDLE)

            logger.error(
                "[PERSISTENT_JOB] Failed: %s", e, exc_info=True,
            )
            response = f"I couldn't schedule that: {e}"
            self.output_channel.send(response)
            self.conversation.append_turn("assistant", response)
            return response



    def _resolve_references(self, user_text: str) -> None:
        """
        Run WorldResolver BEFORE cortex.compile().

        Annotates conversation frame with resolved references.
        Never mutates MissionPlan (no plan exists yet).
        Never crashes — resolution failure is silent.
        """
        # Get visible lists from last outcome (if any)
        visible_lists: dict = {}
        active_entity = self.conversation.active_entity

        if self.conversation.outcomes:
            last_outcome = self.conversation.outcomes[-1]
            if hasattr(last_outcome, 'visible_lists'):
                visible_lists = last_outcome.visible_lists

        query_ctx = WorldResolver.resolve(
            user_text=user_text,
            visible_lists=visible_lists,
            active_entity=active_entity,
            entity_registry=self.conversation.entity_registry,
        )

        # Store resolved context for cortex to consume
        if query_ctx.is_resolved:
            self.conversation.unresolved_references = {
                "resolved": [
                    {
                        "ordinal": ref.ordinal,
                        "index": ref.index,
                        "list_key": ref.list_key,
                        "entity_hint": ref.entity_hint,
                        "value": ref.resolved_value,
                    }
                    for ref in query_ctx.resolved_references
                ]
            }
            logger.info(
                "Resolved %d reference(s) for '%s'",
                len(query_ctx.resolved_references),
                user_text[:50],
            )
        elif query_ctx.has_referential_language:
            # Referential language detected but couldn't resolve
            self.conversation.unresolved_references = {
                "unresolved": True,
                "text": user_text[:100],
            }
            logger.info(
                "Referential language detected but unresolved: '%s'",
                user_text[:50],
            )
        else:
            # No referential language— clear any stale references
            self.conversation.unresolved_references = {}
