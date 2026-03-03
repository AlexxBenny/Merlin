# runtime/scheduler.py

"""
SchedulerManager — Protocol for deterministic task scheduling.

Today:  Protocol definition only (Phase 2 seam).
Future: Implementation with priority queue + tick integration.

This is INFRASTRUCTURE, not cognition.
No LLM. No cortex imports. No IR imports.

Ownership invariant:
    SchedulerManager is the ONLY component that may transition
    task status (PENDING → RUNNING → COMPLETED/FAILED).
    tick() internally manages these transitions.

Integration design:
    tick() is called by RuntimeEventLoop._run() on each cycle.
    No new threads. No async. Cooperative tick scheduler.
    Best-effort timing within ±1 tick interval.
"""

from abc import ABC, abstractmethod
from typing import List

from runtime.task_store import Task


class SchedulerManager(ABC):
    """Deterministic scheduler for persistent jobs.

    No LLM inside tick loop. No cortex. No IR.
    Pure infrastructure — deterministic execution timing.

    Lifecycle contract:
        submit()  → stores task, adds to scheduling queue
        tick()    → checks due tasks, transitions PENDING→RUNNING,
                    returns tasks ready for execution
        cancel()  → transitions PENDING→CANCELLED

    tick() owns ALL status transitions:
        PENDING → RUNNING    (when due time reached)
        Does NOT transition RUNNING → COMPLETED/FAILED.
        That is done by the execution callback after skill completes.

    Time guarantees:
        MERLIN scheduler provides best-effort timing.
        Actual execution may lag by ±1 tick interval
        (currently ~100ms per RuntimeEventLoop.tick_interval).
        This is acceptable for a voice assistant.
    """

    @abstractmethod
    def submit(self, task: Task) -> str:
        """Accept a task for scheduling. Returns task.id.

        The scheduler stores the task and adds it to its
        internal timing queue. Does NOT execute immediately.
        """
        ...

    @abstractmethod
    def cancel(self, task_id: str) -> bool:
        """Cancel a pending task.

        Returns True if cancelled, False if not found or not cancellable.
        Only PENDING tasks can be cancelled.
        """
        ...

    @abstractmethod
    def tick(self) -> List[Task]:
        """Called by RuntimeEventLoop on each cycle.

        Checks if any pending tasks are due.
        Internally transitions due tasks: PENDING → RUNNING.
        Returns the list of tasks now ready for execution.

        The caller is responsible for executing the returned tasks
        and reporting completion/failure back via the TaskStore.

        Must be non-blocking. Must be deterministic.
        Must never call LLM or perform IO beyond TaskStore reads.
        """
        ...

    @abstractmethod
    def pending_count(self) -> int:
        """Number of tasks in the scheduling queue."""
        ...
