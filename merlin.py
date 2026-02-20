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

from typing import Any, Dict, List, Optional
import logging

from brain.core import BrainCore, CognitiveRoute, Percept
from brain.escalation_policy import (
    EscalationPolicy, EscalationDecision,
    CognitiveTier, HeuristicTierClassifier,
)
from cortex.mission_cortex import MissionCortex
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
from models.base import LLMClient
from cortex.world_state_provider import WorldStateProvider, SimpleWorldStateProvider
from memory.store import MemoryStore


logger = logging.getLogger(__name__)


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

        # ── Shared executor (used by both orchestrator and reflex engine) ──
        executor = MissionExecutor(
            registry, timeline,
            max_workers=max_workers,
            node_timeout=node_timeout,
        )
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
        )

        # ── Tier classification (Phase 5A: deterministic, init-time) ──
        self.tier_classifier = HeuristicTierClassifier(registry)

        # ── Output ──
        self.output_channel = output_channel
        self.report_builder = report_builder

        # ── Runtime event loop (background thread) ──
        self.event_loop = RuntimeEventLoop(
            timeline=timeline,
            reflex_engine=reflex_engine,
            sources=event_sources,
            notification_policy=notification_policy,
            report_builder=report_builder,
            output_channel=output_channel,
            get_conversation=lambda: self.conversation,
            attention_manager=attention_manager,
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

        # ── Gate 2: EscalationPolicy (only reached for MISSION) ──
        decision = self.escalation_policy.decide_for_user_input(
            user_text=percept.payload,
            snapshot=snapshot,
            frame=self.conversation,
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
            return self._handle_clarify(percept)

        return self._handle_mission(percept, snapshot)

    # ─────────────────────────────────────────────────────────
    # Route handlers
    # ─────────────────────────────────────────────────────────

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
        2. If template matches → execute, return structured result
        3. If execution fails → report failure (NEVER escalate)
        4. If no match → fallback to MISSION (safe bias)
        """

        # Try template matching
        reflex_match = self.reflex_engine.try_match(percept.payload)

        if reflex_match:
            logger.info(
                "Reflex template matched: skill=%s params=%s",
                reflex_match.skill,
                reflex_match.params,
            )
            result: ReflexResult = self.reflex_engine.execute_reflex(
                reflex_match, snapshot=snapshot,
            )

            if result.success:
                # Format intelligent response from skill metadata
                response = self._format_reflex_response(reflex_match.skill, result)
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

    # ─────────────────────────────────────────────────────────
    # Reflex response formatting
    # ─────────────────────────────────────────────────────────

    # Maps skill metadata 'reason' → natural-language response.
    # Skills that return reason in metadata get intelligent replies.
    # Skills without reason metadata get the generic fallback.
    _REASON_RESPONSES = {
        "already_playing": "Already playing.",
        "already_paused": "Already paused.",
        "already_muted": "Already muted.",
        "already_unmuted": "Already unmuted.",
        "no_media_session": "No media session detected.",
    }

    def _format_reflex_response(
        self, skill_name: str, result: ReflexResult,
    ) -> str:
        """
        Format an intelligent reflex response from skill outputs/metadata.

        State-aware skills return metadata with 'reason' keys.
        This method maps those reasons to natural-language responses.
        Falls back to generic "Done." for skills without metadata.
        """
        # Read reason from skill metadata channel (SkillResult.metadata)
        reason = result.metadata.get("reason") if result.metadata else None

        # Reason-based responses take priority
        if reason and reason in self._REASON_RESPONSES:
            return self._REASON_RESPONSES[reason]

        # Check if skill returned a 'changed' flag in outputs
        changed = result.outputs.get("changed")
        if changed is True:
            # Derive action verb from skill name
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

        return f"Done. ({skill_name})"

    def _handle_clarify(self, percept: Percept) -> str:
        """
        Ask for clarification when context is insufficient.

        Uses clarifier LLM if available for context-aware questions.
        Falls back to deterministic response otherwise.
        """
        if self.clarifier_llm:
            try:
                prompt = self._build_clarify_prompt(percept)
                response = self.clarifier_llm.complete(prompt)
                # Sanitize — strip any LLM framing
                response = response.strip()
                if not response:
                    response = self._default_clarify_response()
            except Exception:
                response = self._default_clarify_response()
        else:
            response = self._default_clarify_response()
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
    ) -> Optional[str]:
        """
        Compile and execute a full mission.

        Tiered cognitive flow (Phase 5A):
        - Tier 1 (SIMPLE):       compile directly (existing behavior, zero overhead)
        - Tier 2 (MULTI_INTENT): decompose → compile with checklist → coverage verify
        - Tier 3 (HIERARCHICAL): gate only — treated as Tier 2 in 5A

        Delegates to MissionOrchestrator.handle_user_input().
        Catches all errors to prevent LLM/compilation failures from
        crashing the interactive loop.

        Args:
            snapshot: Pre-built WorldSnapshot (built once in handle_percept,
                      shared with EscalationPolicy — never rebuilt here).
        """
        try:
            # ── Reference resolution (BEFORE cortex) ──
            self._resolve_references(percept.payload)

            # ── Signal mission start to attention manager ──
            if self.attention_manager:
                self.attention_manager.set_mission_state(MissionState.COMPILING)

            # ── Tier classification (deterministic, no LLM) ──
            tier = self.tier_classifier.classify(percept.payload)
            logger.info(
                "[TIER] Query '%s' → %s",
                percept.payload[:80], tier.value,
            )

            # ── Conditional decomposition (Tier 2+ only) ──
            intent_units = None
            if tier != CognitiveTier.SIMPLE:
                intent_units = self.orchestrator.cortex.decompose_intents(
                    percept.payload,
                )
                if intent_units is None:
                    # Decomposition failed → graceful degradation to Tier 1
                    logger.info(
                        "[TIER] Decomposition failed, degrading to Tier 1"
                    )
                    tier = CognitiveTier.SIMPLE
                else:
                    logger.info(
                        "[TIER] Decomposed %d intent units",
                        len(intent_units),
                    )

            result = self.orchestrator.handle_user_input(
                user_text=percept.payload,
                conversation=self.conversation,
                world_state_schema=self.world_state_provider.build_schema(
                    snapshot, query=percept.payload,
                ),
                cognitive_tier=tier,
                intent_units=intent_units,
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
