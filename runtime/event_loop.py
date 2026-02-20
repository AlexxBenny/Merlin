# runtime/event_loop.py

from typing import Callable, List, Optional
import time
import logging
import threading

from world.timeline import WorldTimeline
from world.state import WorldState
from world.snapshot import WorldSnapshot
from conversation.frame import ConversationFrame
from runtime.reflex_engine import ReflexEngine
from runtime.sources.base import EventSource
from reporting.notification_policy import NotificationPolicy
from reporting.report_builder import ReportBuilder
from reporting.output import OutputChannel
from runtime.attention import AttentionManager, AttentionDecision


logger = logging.getLogger(__name__)


class RuntimeEventLoop:
    """
    Always-on runtime loop.

    - Polls event sources
    - Emits events to WorldTimeline
    - Triggers reflexes (deterministic, zero-LLM)
    - Proactively notifies user when justified (via NotificationPolicy)
    - Never reasons
    """

    def __init__(
        self,
        timeline: WorldTimeline,
        reflex_engine: ReflexEngine,
        sources: List[EventSource],
        notification_policy: NotificationPolicy,
        report_builder: ReportBuilder,
        output_channel: OutputChannel,
        get_conversation: Callable[[], ConversationFrame],
        attention_manager: Optional[AttentionManager] = None,
        tick_interval: float = 0.1,
    ):
        self.timeline = timeline
        self.reflex_engine = reflex_engine
        self.sources = sources
        self.notification_policy = notification_policy
        self.report_builder = report_builder
        self.output_channel = output_channel
        self.get_conversation = get_conversation
        self.attention_manager = attention_manager
        self.tick_interval = tick_interval

        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        if self._running:
            return
        self._bootstrap()  # Authoritative init — blocks until complete
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def _bootstrap(self):
        """Transactional bootstrap: collect all events, batch-commit, mark ready.

        Invariants:
        - Idempotent: skips if already bootstrapped
        - Transactional: all events committed atomically via emit_batch
        - Blocking: completes before poll loop starts
        - No partial world is ever visible to any reader
        """
        if self.timeline.bootstrapped:
            logger.info("RuntimeEventLoop: already bootstrapped, skipping")
            return

        logger.info("RuntimeEventLoop: bootstrapping world state...")
        all_events = []
        for source in self.sources:
            try:
                events = source.bootstrap()
                all_events.extend(events)
                if events:
                    logger.info(
                        "  %s: %d bootstrap events",
                        type(source).__name__, len(events),
                    )
            except Exception as e:
                logger.warning(
                    "  %s: bootstrap failed: %s",
                    type(source).__name__, e,
                )

        # Single atomic commit — no partial world
        self.timeline.emit_batch(all_events)
        self.timeline.mark_bootstrapped()
        logger.info(
            "World authoritative snapshot established. "
            "%d events committed. World ready.",
            len(all_events),
        )

    def _run(self):
        while self._running:
            for source in self.sources:
                try:
                    events = source.poll()
                except Exception:
                    # Runtime must NEVER crash
                    continue

                for event in events:
                    # 1. Append to world timeline
                    self.timeline.emit(
                        source=event.source,
                        event_type=event.type,
                        payload=event.payload,
                    )

                    # 2. Trigger reflex engine (deterministic, immediate)
                    self.reflex_engine.on_event(event)

                    # 3. Proactive reporting (the JARVIS behavior)
                    self._maybe_notify_user(event)

            time.sleep(self.tick_interval)

    def _maybe_notify_user(self, event) -> None:
        """
        Proactive notification with attention arbitration.

        Flow:
        1. NotificationPolicy decides if event is worth reporting
        2. AttentionManager decides timing (INTERRUPT / QUEUE / SUPPRESS)
        3. ReportBuilder formats the message (LLM for language only)
        4. OutputChannel delivers (or defers)

        If any step returns None/fails, silence is chosen.
        Runtime must NEVER crash from reporting.
        """
        try:
            # Build snapshot for policy check
            all_events = self.timeline.all_events()
            state = WorldState.from_events(all_events)
            snapshot = WorldSnapshot.build(
                state, all_events[-10:] if all_events else []
            )

            # Policy gate: should the user hear about this?
            if not self.notification_policy.should_notify(event, snapshot):
                return

            # Determine event priority for attention arbitration
            priority = event.payload.get("severity", "info")

            # Attention gate: should we deliver NOW?
            if self.attention_manager:
                decision = self.attention_manager.decide(priority, event.type)

                if decision == AttentionDecision.SUPPRESS:
                    return

                # Build report text (needed for both INTERRUPT and QUEUE)
                conversation = self.get_conversation()
                report = self.report_builder.build_from_event(
                    event, snapshot, conversation
                )
                if not report:
                    return

                if decision == AttentionDecision.QUEUE:
                    self.attention_manager.enqueue(report, priority, event.type)
                    return

                # INTERRUPT — deliver immediately
                self.attention_manager.deliver(report)
            else:
                # No attention manager — legacy behavior (always deliver)
                conversation = self.get_conversation()
                report = self.report_builder.build_from_event(
                    event, snapshot, conversation
                )
                if report:
                    self.output_channel.send(report)

        except Exception:
            # Proactive reporting must NEVER break runtime
            logger.debug(
                "Proactive reporting failed for event '%s', "
                "continuing silently",
                event.type,
                exc_info=True,
            )

