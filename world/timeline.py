# world/timeline.py

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, ConfigDict, Field
import time
import threading


class WorldEvent(BaseModel):
    timestamp: float = Field(..., description="Unix timestamp")
    source: str = Field(..., description="Skill / perception / system")
    type: str = Field(..., description="Event type identifier")
    payload: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class WorldTimeline:
    """
    Append-only event log.

    This is the ONLY mutable world structure.
    Everything else derives from this.

    MUTATION GATE INVARIANT (Phase 3C):
    All writes are serialized via _lock.
    Parallel compute nodes may run concurrently,
    but world mutations are always sequential.

    BOOTSTRAP INVARIANT (Phase P0):
    mark_bootstrapped() is called exactly once after transactional
    bootstrap commit. No missions may execute before this.
    """

    def __init__(self):
        self._events: List[WorldEvent] = []
        self._lock = threading.Lock()
        self._bootstrapped: bool = False

    def emit(
        self,
        source: str,
        event_type: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> WorldEvent:
        event = WorldEvent(
            timestamp=time.time(),
            source=source,
            type=event_type,
            payload=payload or {},
        )
        with self._lock:
            self._events.append(event)
        return event

    def emit_batch(self, events: List[WorldEvent]) -> None:
        """Atomically append multiple events.

        Used by bootstrap to commit all initial state in a single
        transaction. No partial world is ever visible.
        """
        if not events:
            return
        with self._lock:
            self._events.extend(events)

    @property
    def bootstrapped(self) -> bool:
        """Read-only: has the world been authoritatively initialized?"""
        return self._bootstrapped

    def mark_bootstrapped(self) -> None:
        """Mark the world as authoritatively initialized.

        Called exactly once by RuntimeEventLoop after transactional
        bootstrap commit. Not reversible.
        """
        self._bootstrapped = True

    def all_events(self) -> List[WorldEvent]:
        with self._lock:
            return list(self._events)

    def event_count(self) -> int:
        """Atomic event count read."""
        with self._lock:
            return len(self._events)

    def events_since_index(self, index: int) -> List[WorldEvent]:
        """Return events emitted after a given index (for contract enforcement)."""
        with self._lock:
            return list(self._events[index:])

    def since(self, timestamp: float) -> List[WorldEvent]:
        with self._lock:
            return [e for e in self._events if e.timestamp >= timestamp]
