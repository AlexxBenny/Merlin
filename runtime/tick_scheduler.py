# runtime/tick_scheduler.py

"""
TickSchedulerManager — Concrete scheduler for persistent jobs.

Implements the SchedulerManager ABC with cooperative tick-based scheduling.
Called by RuntimeEventLoop._run() on each cycle (~100ms).

This is INFRASTRUCTURE, not cognition.
No LLM. No cortex. No IR imports.

Design invariants:
    - tick() is non-blocking, deterministic, O(n) over pending tasks
    - Concurrency cap: at most max_concurrent_jobs executing at once
    - Concurrency cap uses SLICING, not blocking (due[:available_slots])
    - Retry backoff: failed tasks re-enter PENDING with exponential delay
    - Recurring: completed recurring tasks respawn with aligned next_run
    - Boot recovery: fast-forward recurring jobs, apply missed_policy
    - Failure terminal: after max_retries, mark FAILED, emit ONE notification

Ownership:
    SchedulerManager owns ALL status transitions on Task objects.
    No other component may mutate task lifecycle.
"""

import logging
import time as _time
from typing import List, Optional

from runtime.scheduler import SchedulerManager
from runtime.task_store import Task, TaskSchedule, TaskStatus, TaskStore, TaskType

logger = logging.getLogger(__name__)

# Retry backoff base (seconds). Doubles each attempt: 2, 4, 8, 16...
_RETRY_BACKOFF_BASE = 2.0


class TickSchedulerManager(SchedulerManager):
    """Cooperative tick-based scheduler.

    Architecture:
        submit()        → compute next_run, store task
        tick()          → find due tasks, respect concurrency cap,
                          transition PENDING→RUNNING, return dispatch list
        on_completion() → RUNNING→COMPLETED/FAILED, handle recurring/retry
        cancel()        → PENDING→CANCELLED
        recover()       → boot-time recovery: fast-forward, missed_policy

    Thread safety: delegates to TaskStore's thread safety.
    This class itself is single-threaded (called from event loop thread).
    """

    def __init__(
        self,
        store: TaskStore,
        max_concurrent_jobs: int = 2,
        missed_job_staleness_seconds: float = 300.0,
        default_max_retries: int = 3,
    ) -> None:
        self._store = store
        self._max_concurrent = max_concurrent_jobs
        self._staleness_threshold = missed_job_staleness_seconds
        self._default_max_retries = default_max_retries

    # ─────────────────────────────────────────────────────────
    # SchedulerManager ABC implementation
    # ─────────────────────────────────────────────────────────

    def submit(self, task: Task) -> str:
        """Accept a task for scheduling.

        Computes next_run from task.schedule if not already set.
        Sets max_retries from config if not overridden.
        Stores task and returns task.id.
        """
        # Compute next_run if not already set
        if task.next_run is None:
            task.next_run = self._compute_next_run(task)

        # Apply default max_retries if not overridden
        if task.max_retries <= 1:
            task.max_retries = self._default_max_retries

        task_id = self._store.create(task)

        logger.info(
            "[SCHEDULER] Submitted job %s (%s): '%s' → next_run=%.1f",
            task.short_id, task.type.value, task.query[:50],
            task.next_run or 0,
        )
        return task_id

    def cancel(self, task_id: str) -> bool:
        """Cancel a pending task."""
        result = self._store.cancel(task_id)
        if result:
            logger.info("[SCHEDULER] Cancelled task %s", task_id)
        return result

    def tick(self) -> List[Task]:
        """Check for due tasks, respecting concurrency cap.

        Returns tasks transitioned to RUNNING, ready for dispatch.

        Concurrency cap uses SLICING (due[:available_slots]):
            - Never blocks ALL due tasks just because some are running
            - Excess due tasks deferred to next tick
            - Prevents boot-time explosion (50 overdue → dispatch 2)

        Must be non-blocking. Must be deterministic.
        """
        now = _time.time()

        # Count currently running jobs
        running = self._store.list_by_status(TaskStatus.RUNNING)
        running_count = len(running)

        # Check capacity
        available_slots = self._max_concurrent - running_count
        if available_slots <= 0:
            return []

        # Get due tasks (PENDING with next_run <= now)
        due = self._store.get_due(now)
        if not due:
            return []

        # Sort by priority (high first), then by next_run (earliest first)
        priority_order = {"high": 0, "normal": 1, "low": 2}
        due.sort(key=lambda t: (
            priority_order.get(t.priority, 1),
            t.next_run or 0,
        ))

        # Slice to available capacity — NEVER block all due tasks
        dispatch = due[:available_slots]

        # Transition each to RUNNING
        for task in dispatch:
            task.status = TaskStatus.RUNNING
            task.attempts += 1
            self._store.update_task(task)
            logger.info(
                "[SCHEDULER] Dispatching %s (%s): '%s' (attempt %d/%d)",
                task.short_id, task.id, task.query[:40],
                task.attempts, task.max_retries,
            )

        return dispatch

    def pending_count(self) -> int:
        """Number of tasks in PENDING status."""
        return len(self._store.list_by_status(TaskStatus.PENDING))

    # ─────────────────────────────────────────────────────────
    # Completion callback
    # ─────────────────────────────────────────────────────────

    def on_completion(
        self,
        task_id: str,
        success: bool,
        error: Optional[str] = None,
    ) -> None:
        """Handle execution completion.

        On success:
            - RUNNING → COMPLETED
            - If recurring: respawn with aligned next_run

        On failure:
            - If retries remaining: RUNNING → PENDING with backoff delay
            - If retries exhausted: RUNNING → FAILED (terminal)
              Requires explicit user resume. No auto-retry.
        """
        task = self._store.get(task_id)
        if task is None:
            logger.warning(
                "[SCHEDULER] on_completion for unknown task: %s", task_id,
            )
            return

        if success:
            self._handle_success(task)
        else:
            self._handle_failure(task, error)

    # ─────────────────────────────────────────────────────────
    # Boot recovery
    # ─────────────────────────────────────────────────────────

    def recover(self) -> None:
        """Boot-time recovery. Must be called BEFORE event loop starts.

        For each task:
            RUNNING → reset to PENDING (process may have died mid-execution)
            PENDING + overdue → apply missed_policy
            RECURRING + overdue → fast-forward (execute ONCE, align next_run)

        Critical: recurring fast-forward prevents boot-time explosion.
        If system was down for 3 hours with a 1-minute recurring job,
        do NOT execute 180 times. Execute once, set next_run to aligned future.
        """
        now = _time.time()
        all_tasks = self._store.get_all()
        recovered = 0

        for task in all_tasks:
            # Reset interrupted tasks
            if task.status == TaskStatus.RUNNING:
                logger.info(
                    "[SCHEDULER] Recovery: %s was RUNNING, resetting to PENDING",
                    task.short_id,
                )
                task.status = TaskStatus.PENDING
                self._store.update_task(task)
                recovered += 1
                continue

            # Skip non-pending tasks
            if task.status != TaskStatus.PENDING:
                continue

            # Skip tasks that aren't overdue
            if task.next_run is None or task.next_run > now:
                continue

            overdue_seconds = now - task.next_run
            schedule = task.schedule

            # Recurring: fast-forward
            if (task.type == TaskType.RECURRING
                    and schedule
                    and schedule.repeat_interval):
                task.next_run = self._fast_forward_recurring(
                    task.next_run, schedule.repeat_interval, now,
                )
                self._store.update_task(task)
                logger.info(
                    "[SCHEDULER] Recovery: fast-forwarded recurring %s → next_run=%.1f",
                    task.short_id, task.next_run,
                )
                recovered += 1
                continue

            # Non-recurring overdue: apply missed_policy
            missed_policy = "execute"
            if schedule:
                missed_policy = schedule.missed_policy

            if overdue_seconds > self._staleness_threshold:
                if missed_policy == "skip":
                    self._store.update_status(
                        task.id, TaskStatus.CANCELLED,
                        error="Skipped: overdue by %.0fs (policy=skip)" % overdue_seconds,
                    )
                    logger.info(
                        "[SCHEDULER] Recovery: skipped stale task %s (overdue %.0fs)",
                        task.short_id, overdue_seconds,
                    )
                elif missed_policy == "report":
                    self._store.update_status(
                        task.id, TaskStatus.FAILED,
                        error="Missed: overdue by %.0fs (policy=report)" % overdue_seconds,
                    )
                    logger.info(
                        "[SCHEDULER] Recovery: reported missed task %s (overdue %.0fs)",
                        task.short_id, overdue_seconds,
                    )
                else:
                    # missed_policy == "execute" — leave task as PENDING
                    # It will be picked up by the next tick()
                    logger.info(
                        "[SCHEDULER] Recovery: will execute overdue task %s (overdue %.0fs)",
                        task.short_id, overdue_seconds,
                    )
                recovered += 1

        if recovered:
            logger.info("[SCHEDULER] Boot recovery: processed %d tasks", recovered)

    # ─────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────

    def _handle_success(self, task: Task) -> None:
        """Mark task as completed. Handle recurring respawn."""
        self._store.update_status(task.id, TaskStatus.COMPLETED)
        logger.info(
            "[SCHEDULER] Completed %s: '%s'",
            task.short_id, task.query[:40],
        )

        # Recurring: respawn if repeats remaining
        if (task.type == TaskType.RECURRING
                and task.schedule
                and task.schedule.repeat_interval):
            schedule = task.schedule
            # Track total repeats via metadata (attempts resets on each respawn)
            total_repeats = task.metadata.get("total_repeats", 0) + 1
            if schedule.max_repeats > 0 and total_repeats >= schedule.max_repeats:
                logger.info(
                    "[SCHEDULER] Recurring %s reached max_repeats=%d (total=%d)",
                    task.short_id, schedule.max_repeats, total_repeats,
                )
                return

            self._respawn_recurring(task, total_repeats)

    def _handle_failure(self, task: Task, error: Optional[str]) -> None:
        """Handle execution failure with retry backoff."""
        if task.attempts < task.max_retries:
            # Retry: transition back to PENDING with backoff delay
            backoff = _RETRY_BACKOFF_BASE * (2 ** (task.attempts - 1))
            task.status = TaskStatus.PENDING
            task.next_run = _time.time() + backoff
            task.error = error
            self._store.update_task(task)
            logger.info(
                "[SCHEDULER] Retry %s (attempt %d/%d) in %.1fs: %s",
                task.short_id, task.attempts, task.max_retries,
                backoff, error or "unknown error",
            )
        else:
            # Terminal failure: mark FAILED, require explicit resume
            self._store.update_status(task.id, TaskStatus.FAILED, error=error)
            logger.warning(
                "[SCHEDULER] FAILED %s after %d attempts: %s",
                task.short_id, task.attempts, error or "unknown error",
            )

    def _respawn_recurring(self, task: Task, total_repeats: int) -> None:
        """Create a new PENDING task for the next recurring interval.

        next_run is aligned to wall-clock intervals, not last_run + interval.
        This prevents drift accumulation.

        total_repeats: cumulative execution count across all respawns.
        """
        schedule = task.schedule
        now = _time.time()

        # Compute aligned next_run
        next_run = self._fast_forward_recurring(
            task.next_run or task.created_at,
            schedule.repeat_interval,
            now,
        )

        # Create new task (inherits from parent, carries total_repeats)
        new_task = Task(
            id=f"{task.id}_r{total_repeats + 1}",
            type=TaskType.RECURRING,
            query=task.query,
            mission_data=task.mission_data,
            schedule=task.schedule,
            next_run=next_run,
            max_retries=task.max_retries,
            priority=task.priority,
            metadata={**task.metadata, "total_repeats": total_repeats},
        )

        self._store.create(new_task)
        logger.info(
            "[SCHEDULER] Respawned recurring %s → %s (next_run=%.1f, repeats=%d)",
            task.short_id, new_task.short_id, next_run, total_repeats,
        )

    @staticmethod
    def _compute_next_run(task: Task) -> Optional[float]:
        """Compute initial next_run from task.schedule.

        Returns None if no schedule is set (should not happen for
        properly constructed tasks).
        """
        if task.schedule is None:
            return None

        schedule = task.schedule
        now = _time.time()

        if schedule.delay_seconds is not None:
            return now + schedule.delay_seconds

        if schedule.schedule_at is not None:
            return schedule.schedule_at

        if schedule.repeat_interval is not None:
            return now + schedule.repeat_interval

        return None

    @staticmethod
    def _fast_forward_recurring(
        last_scheduled: float,
        interval: int,
        now: float,
    ) -> float:
        """Compute aligned future next_run for a recurring job.

        If system was down for N intervals:
            missed = (now - last_scheduled) // interval
            next_run = last_scheduled + (missed + 1) * interval

        This ensures:
            - Execute ONCE, not N times
            - next_run is aligned to the original schedule
            - No drift accumulation
        """
        if interval <= 0:
            return now + 60  # Safety fallback

        elapsed = now - last_scheduled
        if elapsed <= 0:
            return last_scheduled + interval

        missed = int(elapsed / interval)
        return last_scheduled + (missed + 1) * interval
