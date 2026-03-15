# runtime/task_store.py

"""
TaskStore — Persistent job identity and lifecycle.

Today:  InMemoryTaskStore (simple dict-backed, lost on restart).
        JsonTaskStore (write-aside JSON, survives restart).
Future: SQLiteTaskStore (full persistence + indexed queries).

This is INFRASTRUCTURE, not cognition.
No LLM. No coordinator. No cortex imports. No IR imports.

Domain rule:
    Immediate missions NEVER enter TaskStore.
    Only persistent jobs (delayed, scheduled, recurring, triggered)
    have task identity.

Ownership invariant:
    Only SchedulerManager may call update_status().
    No other component may mutate task lifecycle.
    This invariant is enforced by convention in Phase 2
    and by interface splitting in Phase 3+.
"""

import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


# ─────────────────────────────────────────────────────────────
# Task Types (persistent jobs only — NO immediate missions)
# ─────────────────────────────────────────────────────────────

class TaskType(str, Enum):
    """Types of persistent jobs.

    Immediate missions are NOT modeled here.
    They execute synchronously, have no identity, and die with request.
    """
    DELAYED = "delayed"        # "pause after 10 seconds"
    SCHEDULED = "scheduled"    # "mute at 3pm"
    RECURRING = "recurring"    # "check battery every hour"
    TRIGGERED = "triggered"    # "mute when battery < 20%"


class TaskStatus(str, Enum):
    """Lifecycle states for a persistent job.

    State transitions are owned EXCLUSIVELY by SchedulerManager:
        PENDING → RUNNING (when due)
        RUNNING → COMPLETED (on success)
        RUNNING → FAILED (on error)
        PENDING → CANCELLED (on user cancel)
        PENDING → PAUSED (user pause via API)
        PAUSED  → PENDING (user resume via API)

    Disallowed:
        RUNNING → PAUSED (would leave task in inconsistent state)
    """
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


# ─────────────────────────────────────────────────────────────
# Task Schedule (temporal metadata)
# ─────────────────────────────────────────────────────────────

class TaskSchedule(BaseModel):
    """Temporal metadata for persistent jobs.

    Stores BOTH the logical schedule (human expression) AND
    computed timestamps. Never derive schedule solely from next_run.

    All epoch times use UTC float.
    """
    model_config = ConfigDict(extra="forbid")

    # ── Logical schedule (immutable after creation) ──
    delay_seconds: Optional[int] = None       # DELAYED: seconds from creation
    schedule_at: Optional[float] = None       # SCHEDULED: UTC epoch timestamp
    repeat_interval: Optional[int] = None     # RECURRING: seconds between repeats
    max_repeats: int = 1                      # Bounded iteration — NO unbounded loops

    # ── Original expression (for display and re-resolution) ──
    time_expression: str = ""                 # Original human text: "4 PM", "10 seconds"
    timezone: str = ""                        # IANA timezone at creation time

    # ── Missed job policy ──
    missed_policy: str = "execute"            # "execute" | "skip" | "report"


# ─────────────────────────────────────────────────────────────
# Task (persistent job record)
# ─────────────────────────────────────────────────────────────

class Task(BaseModel):
    """A persistent job with identity and lifecycle.

    This is a DATA RECORD, not an execution context.
    It contains no execution logic, no LLM references,
    and no IR model objects.

    mission_data stores a JSON-safe dict representation
    of the compiled plan — NOT an IR model object.
    This keeps runtime/ completely cognition-agnostic.
    """
    model_config = ConfigDict(extra="forbid")

    id: str
    type: TaskType
    status: TaskStatus = TaskStatus.PENDING
    query: str                                      # Original user query
    mission_data: Optional[Dict[str, Any]] = None   # JSON-safe plan (no IR imports)
    schedule: Optional[TaskSchedule] = None
    created_at: float = Field(default_factory=time.time)
    completed_at: Optional[float] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # ── Job engine fields ──
    short_id: str = ""                              # Human-friendly ID (J-1, J-2)
    next_run: Optional[float] = None                # Computed epoch when job should fire
    attempts: int = 0                               # Execution attempt counter
    max_retries: int = 1                            # Max retry attempts (1 = no retries)
    priority: str = "normal"                        # "low" | "normal" | "high"


# ─────────────────────────────────────────────────────────────
# TaskStore Protocol
# ─────────────────────────────────────────────────────────────

class TaskStore(ABC):
    """Pure data layer for persistent jobs.

    Imports NOTHING from cortex, brain, models, or ir.
    Lives in runtime/ — infrastructure, not cognition.

    WARNING: update_status() must only be invoked by SchedulerManager.
    No other component should mutate task lifecycle.
    """

    @abstractmethod
    def create(self, task: Task) -> str:
        """Store a task. Returns task.id."""
        ...

    @abstractmethod
    def get(self, task_id: str) -> Optional[Task]:
        """Retrieve a task by ID. Returns None if not found."""
        ...

    @abstractmethod
    def update_status(
        self,
        task_id: str,
        status: TaskStatus,
        error: Optional[str] = None,
    ) -> None:
        """Transition task status.

        WARNING: This method must ONLY be called by SchedulerManager.
        Architectural invariant — enforced by convention (Phase 2)
        and by interface splitting (Phase 3+).
        """
        ...

    @abstractmethod
    def list_by_status(self, status: TaskStatus) -> List[Task]:
        """Return all tasks with the given status."""
        ...

    @abstractmethod
    def cancel(self, task_id: str) -> bool:
        """Cancel a pending task. Returns True if cancelled, False if not found/already done."""
        ...

    @abstractmethod
    def get_due(self, now: float) -> List[Task]:
        """Return all PENDING tasks whose next_run <= now.

        Used by SchedulerManager.tick() to find jobs ready to dispatch.
        """
        ...

    @abstractmethod
    def get_all(self) -> List[Task]:
        """Return all tasks regardless of status.

        Used for boot recovery and job listing.
        """
        ...

    @abstractmethod
    def delete(self, task_id: str) -> bool:
        """Remove a task from the store.

        Used for cleanup of completed/cancelled tasks.
        Returns True if deleted, False if not found.
        """
        ...

    @abstractmethod
    def update_task(self, task: Task) -> None:
        """Replace the full task record.

        Used by SchedulerManager for updating next_run, attempts, etc.
        WARNING: Only SchedulerManager should call this.
        """
        ...


# ─────────────────────────────────────────────────────────────
# InMemoryTaskStore (Phase 2 — swappable to SQLite later)
# ─────────────────────────────────────────────────────────────

class InMemoryTaskStore(TaskStore):
    """Dict-backed task store. Lost on restart.

    Sufficient for Phase 2 (protocol definition + simple testing).
    Phase 3+: replace with SQLiteTaskStore for persistence.
    """

    def __init__(self) -> None:
        self._tasks: Dict[str, Task] = {}

    def create(self, task: Task) -> str:
        self._tasks[task.id] = task
        return task.id

    def get(self, task_id: str) -> Optional[Task]:
        return self._tasks.get(task_id)

    def update_status(
        self,
        task_id: str,
        status: TaskStatus,
        error: Optional[str] = None,
    ) -> None:
        task = self._tasks.get(task_id)
        if task is None:
            return
        task.status = status
        if error is not None:
            task.error = error
        if status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
            task.completed_at = time.time()

    def list_by_status(self, status: TaskStatus) -> List[Task]:
        return [t for t in self._tasks.values() if t.status == status]

    def cancel(self, task_id: str) -> bool:
        task = self._tasks.get(task_id)
        if task is None or task.status != TaskStatus.PENDING:
            return False
        task.status = TaskStatus.CANCELLED
        return True

    def get_due(self, now: float) -> List[Task]:
        return [
            t for t in self._tasks.values()
            if t.status == TaskStatus.PENDING
            and t.next_run is not None
            and t.next_run <= now
        ]

    def get_all(self) -> List[Task]:
        return list(self._tasks.values())

    def delete(self, task_id: str) -> bool:
        if task_id in self._tasks:
            del self._tasks[task_id]
            return True
        return False

    def update_task(self, task: Task) -> None:
        if task.id in self._tasks:
            self._tasks[task.id] = task

    @property
    def task_count(self) -> int:
        return len(self._tasks)
