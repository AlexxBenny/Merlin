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
    - Dispatches scheduled jobs (via TickSchedulerManager)
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
        scheduler: Optional["TickSchedulerManager"] = None,
        completion_queue: Optional["CompletionQueue"] = None,
        job_executor: Optional[Callable] = None,
        user_knowledge=None,  # Optional[UserKnowledgeStore] — for proactive policy eval
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
        self._user_knowledge = user_knowledge  # proactive policy eval

        # Job scheduling infrastructure (optional)
        self._scheduler = scheduler
        self._completion_queue = completion_queue
        self._job_executor = job_executor  # Callable: (Task, WorldSnapshot) -> (bool, str|None)

        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        if self._running:
            return
        self._bootstrap()  # Authoritative init — blocks until complete

        # Scheduler recovery BEFORE tick loop starts.
        # Critical: prevents dispatching stale/misaligned jobs.
        if self._scheduler:
            self._scheduler.recover()

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

            # 4. Scheduler tick — dispatch due jobs
            self._tick_scheduler()

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
                # Even if notification is suppressed, check user policies
                # for proactive action suggestions from UserKnowledgeStore.
                self._maybe_apply_user_policy(event, snapshot)
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

    def _maybe_apply_user_policy(self, event, snapshot) -> None:
        """Evaluate UserKnowledgeStore policies against the triggering event.

        Called when a world event fires, regardless of notification policy.
        If policies match (e.g., 'when media_started → set_volume=90'),
        the suggested action is surfaced as a proactive insight.

        Design invariants:
        - NEVER executes skills directly (read-only; merlin owns execution)
        - NEVER raises (runtime must not crash)
        - Insights are enqueued and merged with next user report by ReportBuilder
        - No LLM call here — pure deterministic policy matching
        """
        if not self._user_knowledge:
            return

        try:
            # Build context dict from event for policy matching
            context: dict = {
                "event_type": str(event.type),
                **{k: v for k, v in (event.payload or {}).items()
                   if isinstance(v, (str, int, float, bool))},
            }

            matching_policies = self._user_knowledge.get_matching_policies(context)
            if not matching_policies:
                return

            for policy in matching_policies:
                action = policy.action or {}
                label = getattr(policy, "label", "") or str(action)

                # Build insight text for proactive notification
                action_desc = "; ".join(f"{k}={v}" for k, v in action.items())
                insight = (
                    f"Your preference suggests: {action_desc}"
                    if action_desc else f"User policy triggered: {label}"
                )

                logger.info(
                    "[PROACTIVE] Policy matched (event=%s): %s → %s",
                    event.type, context, action,
                )

                if self.attention_manager:
                    self.attention_manager.enqueue(insight, "info", "policy_trigger")
                else:
                    # Fallback: deliver directly
                    self.output_channel.send(insight)

        except Exception:
            logger.debug(
                "_maybe_apply_user_policy failed for event '%s', continuing silently",
                event.type,
                exc_info=True,
            )

    def _tick_scheduler(self) -> None:
        """Check scheduler for due jobs and dispatch them.

        Called on every tick cycle. Non-blocking.
        Jobs are executed in background threads to avoid
        blocking the event loop.

        Also drains CompletionQueue and delivers notifications.
        """
        if self._scheduler is None:
            return

        # ── Drain completion queue → proactive notification ──
        self._drain_completions()

        try:
            due_tasks = self._scheduler.tick()
        except Exception:
            logger.debug(
                "Scheduler tick failed, continuing",
                exc_info=True,
            )
            return

        for task in due_tasks:
            # Execute each job in a background thread
            thread = threading.Thread(
                target=self._execute_scheduled_job,
                args=(task,),
                daemon=True,
                name=f"job-{task.short_id}",
            )
            thread.start()

    def _drain_completions(self) -> None:
        """Drain CompletionQueue and route through AttentionManager.

        Called on every tick (event loop thread). Non-blocking.

        Output delivery strategy:
            IDLE      → INTERRUPT — speak immediately
            EXECUTING → QUEUE    — merged into user report later
            REPORTING → QUEUE    — merged into same report

        The orchestrator drains AttentionManager's queue before building
        the user report, so queued job completions are naturally merged
        by ReportBuilder via queued_insights.
        """
        if not self._completion_queue:
            return

        events = self._completion_queue.drain()
        if not events:
            return

        for event in events:
            try:
                # Use natural output text, not robot status
                text = event.output or f"Completed: {event.query[:50]}"
                if event.status != "completed" and event.error:
                    text = (
                        f"A scheduled job failed: {event.query[:50]} "
                        f"— {event.error}"
                    )

                # Route through AttentionManager
                if self.attention_manager:
                    priority = "warning" if event.status != "completed" else "info"
                    decision = self.attention_manager.decide(
                        priority, f"job_{event.status}",
                    )
                    if decision == AttentionDecision.SUPPRESS:
                        continue
                    if decision == AttentionDecision.QUEUE:
                        # Will be drained into user report via queued_insights
                        self.attention_manager.enqueue(
                            text, priority, f"job_{event.status}",
                        )
                        logger.info(
                            "[EVENT_LOOP] Job %s queued for merge: %s",
                            event.short_id, event.status,
                        )
                        continue
                    # INTERRUPT — speak immediately (user is idle)
                    self.attention_manager.deliver(text)
                    logger.info(
                        "[EVENT_LOOP] Job %s delivered: %s",
                        event.short_id, event.status,
                    )
                else:
                    self.output_channel.send(text)
            except Exception:
                logger.debug(
                    "Completion notification failed for %s",
                    event.short_id, exc_info=True,
                )

    def _execute_scheduled_job(self, task) -> None:
        """Execute a scheduled job in isolation.

        Runs in a background thread. Never raises.

        Critical invariants:
            - COMPILED MissionPlan from mission_data (no NL re-interpretation)
            - Raw executor.run() — no orchestrator, no ConversationFrame,
              no AttentionManager state transitions
            - Output delivered via CompletionQueue → event loop thread →
              AttentionManager (respects QUEUE/INTERRUPT/SUPPRESS)
            - source="scheduler" in timeline events for auditing

        Thread safety:
            Only touches: executor (thread-pool), timeline (thread-safe),
            completion_queue (locked), scheduler (locked).
            NEVER touches: orchestrator, ConversationFrame, AttentionManager.
        """
        from runtime.completion_event import CompletionEvent, ExecutionContext

        context = ExecutionContext(
            source="scheduler",
            job_id=task.short_id,
            priority=getattr(task, 'priority', "normal") or "normal",
        )

        logger.info(
            "[SCHEDULER] Executing job %s (source=%s, priority=%s): %s",
            task.short_id, context.source, context.priority,
            task.query[:60],
        )

        success = False
        error = None
        output_text = None

        try:
            mission_data = task.mission_data or {}
            compiled_plan_data = mission_data.get("compiled_plan")

            if not compiled_plan_data:
                success = False
                error = "No compiled plan in mission_data"
            elif not self._job_executor:
                success = False
                error = "No job executor configured"
            else:
                from ir.mission import MissionPlan
                plan = MissionPlan.model_validate(compiled_plan_data)

                # Build fresh snapshot (current world state)
                all_events = self.timeline.all_events()
                from world.state import WorldState
                state = WorldState.from_events(all_events)
                snapshot = WorldSnapshot.build(
                    state, all_events[-10:] if all_events else []
                )

                # Execute via raw MissionExecutor.run()
                # NO orchestrator. NO ConversationFrame. NO AttentionManager.
                exec_result = self._job_executor(plan, snapshot)

                if hasattr(exec_result, 'failed') and exec_result.failed:
                    success = False
                    error = f"Failed nodes: {', '.join(exec_result.failed)}"
                else:
                    success = True

                # Build natural output text from execution results
                deferred_query = mission_data.get(
                    "deferred_query", task.query,
                )
                output_text = self._build_job_summary(
                    deferred_query, exec_result,
                )

        except Exception as e:
            success = False
            error = str(e)
            logger.warning(
                "[SCHEDULER] Job %s execution failed: %s",
                task.short_id, e,
            )

        # Report to scheduler (updates task status in store)
        if self._scheduler:
            self._scheduler.on_completion(task.id, success, error)

        # Push to completion queue (drained on next tick by event loop thread)
        if self._completion_queue:
            import time as _time
            event = CompletionEvent(
                task_id=task.id,
                short_id=task.short_id,
                query=task.query,
                status="completed" if success else "failed",
                error=error,
                output=output_text,
                completed_at=_time.time(),
            )
            self._completion_queue.push(event)

        # Emit to WorldTimeline for state tracking
        event_type = "job_completed" if success else "job_failed"
        self.timeline.emit(
            source="scheduler",
            event_type=event_type,
            payload={
                "task_id": task.id,
                "short_id": task.short_id,
                "query": task.query,
                "error": error,
                "execution_context": {
                    "source": context.source,
                    "job_id": context.job_id,
                    "priority": context.priority,
                },
            },
        )

        logger.info(
            "[SCHEDULER] Job %s %s",
            task.short_id,
            "completed" if success else f"failed: {error}",
        )

    @staticmethod
    def _build_job_summary(deferred_query: str, exec_result) -> str:
        """Build natural completion text from execution result.

        Priority order:
            1. Generated text from reasoning nodes (reminder content, etc.)
            2. Deferred query as-is (brief confirmation)

        Does NOT try to be clever — just extracts the most user-facing
        output from the ExecutionResult.
        """
        if not hasattr(exec_result, 'results'):
            return f"Completed: {deferred_query[:60]}"

        # Look for reasoning/generated text outputs
        # These are the user-facing content (reminder text, summaries, etc.)
        for node_id, outputs in exec_result.results.items():
            if not isinstance(outputs, dict):
                continue
            # Prefer 'text' key (reasoning.generate_text output)
            if 'text' in outputs:
                val = outputs['text']
                if isinstance(val, str) and len(val) > 5:
                    return val

        # Fallback: check for any substantial string output
        for node_id, outputs in exec_result.results.items():
            if not isinstance(outputs, dict):
                continue
            for key, val in outputs.items():
                if isinstance(val, str) and len(val) > 20:
                    return val

        # No meaningful output — use the query as confirmation
        return f"Completed: {deferred_query[:60]}"
