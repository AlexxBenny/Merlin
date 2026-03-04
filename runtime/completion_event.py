# runtime/completion_event.py

"""
CompletionEvent + CompletionQueue + ExecutionContext.

Models for tracking scheduled job completion and execution source.

This is INFRASTRUCTURE, not cognition.
No LLM. No cortex. No IR imports.

CompletionEvent:
    Emitted when a scheduled job finishes (success or failure).
    Consumed by ReportBuilder during REPORTING phase.

CompletionQueue:
    Thread-safe queue for CompletionEvents.
    Decouples job execution from report delivery.

ExecutionContext:
    Tags execution with source (user vs scheduler), job_id, priority.
    - Allows AttentionManager to make source-aware decisions
    - Allows logs to correlate failures to specific jobs
    - Allows priority-based escalation
"""

import logging
import threading
from collections import deque
from typing import Deque, List, Optional

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Execution Context
# ─────────────────────────────────────────────────────────────

class ExecutionContext(BaseModel):
    """Tags an execution with its source and metadata.

    Used to distinguish user-triggered missions from
    scheduler-triggered jobs. Influences:
        - AttentionManager decisions (QUEUE vs INTERRUPT)
        - Log correlation
        - Priority escalation

    source="user":      interactive user mission
    source="scheduler": background scheduled job
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

    source: str = "user"                # "user" | "scheduler"
    job_id: Optional[str] = None        # Task.short_id for scheduler jobs
    priority: str = "normal"            # "low" | "normal" | "high"

    @property
    def is_scheduled(self) -> bool:
        """True if this execution was triggered by the scheduler."""
        return self.source == "scheduler"


# Default contexts (avoid allocations)
USER_CONTEXT = ExecutionContext(source="user")


# ─────────────────────────────────────────────────────────────
# Completion Event
# ─────────────────────────────────────────────────────────────

class CompletionEvent(BaseModel):
    """Record of a scheduled job completion.

    Emitted by the event loop after a scheduled job executes.
    Consumed by ReportBuilder / AttentionManager for proactive reporting.

    Immutable after creation.
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str                        # Task.id
    short_id: str                       # Task.short_id (J-1, J-2)
    query: str                          # Original user query
    status: str                         # "completed" | "failed"
    error: Optional[str] = None         # Error message if failed
    output: Optional[str] = None        # Natural summary for user delivery
    completed_at: float                 # UTC epoch


# ─────────────────────────────────────────────────────────────
# Completion Queue (thread-safe)
# ─────────────────────────────────────────────────────────────

class CompletionQueue:
    """Thread-safe queue for CompletionEvents.

    Producer: event loop (after scheduled job finishes)
    Consumer: ReportBuilder / MissionOrchestrator (during REPORTING)

    push()  → add event (from event loop thread)
    drain() → remove and return all events (from main thread)
    peek()  → view without removing

    Bounded: drops oldest if exceeds max_size.
    """

    def __init__(self, max_size: int = 50) -> None:
        self._queue: Deque[CompletionEvent] = deque(maxlen=max_size)
        self._lock = threading.Lock()

    def push(self, event: CompletionEvent) -> None:
        """Add a completion event. Thread-safe."""
        with self._lock:
            self._queue.append(event)
            logger.debug(
                "[COMPLETION_QUEUE] Pushed %s (%s): %s",
                event.short_id, event.status, event.query[:40],
            )

    def drain(self) -> List[CompletionEvent]:
        """Remove and return all queued events. Thread-safe."""
        with self._lock:
            events = list(self._queue)
            self._queue.clear()
            return events

    def peek(self) -> List[CompletionEvent]:
        """View all queued events without removing. Thread-safe."""
        with self._lock:
            return list(self._queue)

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._queue)

    @property
    def is_empty(self) -> bool:
        with self._lock:
            return len(self._queue) == 0
