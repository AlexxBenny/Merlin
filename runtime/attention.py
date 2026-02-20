# runtime/attention.py

"""
AttentionManager — Arbitrates between proactive notifications and active missions.

This is the layer that transforms MERLIN from concurrent to intelligent.
Without it, proactive notifications fire whenever events happen (Type A — chaos).
With it, proactive notifications are timed to feel intentional (Type B — judgment).

Design principles:
    - Centralized: ONE place for all attention arbitration decisions
    - Config-driven: thresholds and policies from execution.yaml
    - Zero-LLM: purely deterministic — event priority × mission state
    - Queuing: deferred notifications delivered after mission completes
    - Anti-spam: cooldown between notifications

Decision matrix:
    CRITICAL + any state       → INTERRUPT
    INFO     + IDLE            → INTERRUPT
    INFO     + COMPILING       → QUEUE
    INFO     + EXECUTING       → QUEUE
    INFO     + REPORTING       → QUEUE
    BACKGROUND                 → SUPPRESS (always)
"""

import logging
import threading
import time
from enum import Enum
from typing import Callable, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Mission lifecycle states
# ─────────────────────────────────────────────────────────────

class MissionState(str, Enum):
    """Lifecycle state of the cognitive pipeline."""
    IDLE = "idle"
    COMPILING = "compiling"
    EXECUTING = "executing"
    REPORTING = "reporting"


# ─────────────────────────────────────────────────────────────
# Attention decisions
# ─────────────────────────────────────────────────────────────

class AttentionDecision(str, Enum):
    """What to do with a notification."""
    INTERRUPT = "interrupt"    # Deliver immediately
    QUEUE = "queue"            # Hold, deliver after mission
    SUPPRESS = "suppress"      # Drop silently


# ─────────────────────────────────────────────────────────────
# Queued notification
# ─────────────────────────────────────────────────────────────

class QueuedNotification(BaseModel):
    """A notification held for deferred delivery."""
    model_config = ConfigDict(extra="forbid")

    text: str
    priority: str             # "critical" | "info" | "background"
    event_type: str
    queued_at: float = Field(default_factory=time.time)


# ─────────────────────────────────────────────────────────────
# AttentionManager
# ─────────────────────────────────────────────────────────────

class AttentionConfig(BaseModel):
    """Config-driven attention thresholds."""
    model_config = ConfigDict(extra="forbid")

    # Minimum seconds between delivered notifications (anti-spam)
    cooldown_seconds: float = 10.0

    # Maximum queued notifications (oldest dropped on overflow)
    max_queue_size: int = 10

    # Whether to merge queued notifications of the same type
    merge_duplicates: bool = True


class AttentionManager:
    """
    Arbitrates between proactive notifications and active missions.

    Thread-safe. Called from:
    - RuntimeEventLoop thread (proactive notifications)
    - Main thread (mission state transitions)

    Usage:
        manager = AttentionManager(config, deliver_fn=output_channel.send)
        manager.set_mission_state(MissionState.EXECUTING)  # From merlin.py
        decision = manager.decide("info", "memory_pressure")  # From event_loop
        if decision == AttentionDecision.INTERRUPT:
            manager.deliver(text)
        elif decision == AttentionDecision.QUEUE:
            manager.enqueue(text, priority, event_type)
        manager.set_mission_state(MissionState.IDLE)  # Flushes queue
    """

    def __init__(
        self,
        config: Optional[AttentionConfig] = None,
        deliver_fn: Optional[Callable[[str], None]] = None,
    ):
        self._config = config or AttentionConfig()
        self._deliver_fn = deliver_fn
        self._mission_state = MissionState.IDLE
        self._queue: List[QueuedNotification] = []
        self._last_delivery_time: float = 0.0
        self._lock = threading.Lock()

    @classmethod
    def from_config(cls, config: dict, deliver_fn=None) -> "AttentionManager":
        """Build from YAML config dict."""
        attention_cfg = config.get("attention", {})
        return cls(
            config=AttentionConfig(
                cooldown_seconds=attention_cfg.get("cooldown_seconds", 10.0),
                max_queue_size=attention_cfg.get("max_queue_size", 10),
                merge_duplicates=attention_cfg.get("merge_duplicates", True),
            ),
            deliver_fn=deliver_fn,
        )

    # ─────────────────────────────────────────────────────────
    # Mission state management
    # ─────────────────────────────────────────────────────────

    @property
    def mission_state(self) -> MissionState:
        with self._lock:
            return self._mission_state

    def set_mission_state(self, state: MissionState) -> None:
        """
        Update mission state. Flushes queue when returning to IDLE.

        Called from Merlin.handle_percept() at each phase transition:
          Entry     → COMPILING
          Post-comp → EXECUTING
          Post-exec → REPORTING
          Done      → IDLE (triggers flush)
        """
        with self._lock:
            old_state = self._mission_state
            self._mission_state = state

            logger.debug(
                "AttentionManager: state %s → %s",
                old_state, state,
            )

            # Flush queue when returning to IDLE
            if state == MissionState.IDLE and old_state != MissionState.IDLE:
                self._flush_queue_locked()

    # ─────────────────────────────────────────────────────────
    # Decision logic
    # ─────────────────────────────────────────────────────────

    def decide(self, priority: str, event_type: str) -> AttentionDecision:
        """
        Deterministic attention decision.

        Decision matrix:
            CRITICAL + any state       → INTERRUPT
            INFO     + IDLE            → INTERRUPT (if not on cooldown)
            INFO     + COMPILING       → QUEUE
            INFO     + EXECUTING       → QUEUE
            INFO     + REPORTING       → QUEUE
            BACKGROUND                 → SUPPRESS
        """
        with self._lock:
            # BACKGROUND → always suppress
            if priority == "background":
                return AttentionDecision.SUPPRESS

            # CRITICAL → always interrupt (safety-first)
            if priority == "critical":
                return AttentionDecision.INTERRUPT

            # INFO priority behavior depends on mission state
            if self._mission_state == MissionState.IDLE:
                # Check cooldown
                now = time.time()
                if now - self._last_delivery_time < self._config.cooldown_seconds:
                    logger.debug(
                        "AttentionManager: QUEUE (cooldown) event=%s",
                        event_type,
                    )
                    return AttentionDecision.QUEUE
                return AttentionDecision.INTERRUPT

            # Mid-mission → queue
            return AttentionDecision.QUEUE

    # ─────────────────────────────────────────────────────────
    # Delivery and queueing
    # ─────────────────────────────────────────────────────────

    def deliver(self, text: str) -> None:
        """Immediately deliver a notification and update cooldown."""
        with self._lock:
            self._last_delivery_time = time.time()

        if self._deliver_fn:
            try:
                self._deliver_fn(text)
            except Exception:
                logger.debug(
                    "AttentionManager: delivery failed",
                    exc_info=True,
                )

    def enqueue(self, text: str, priority: str, event_type: str) -> None:
        """Add a notification to the deferred queue."""
        with self._lock:
            # Merge duplicates: if same event_type already queued, update text
            if self._config.merge_duplicates:
                for i, item in enumerate(self._queue):
                    if item.event_type == event_type:
                        self._queue[i] = QueuedNotification(
                            text=text,
                            priority=priority,
                            event_type=event_type,
                        )
                        logger.debug(
                            "AttentionManager: merged queued %s",
                            event_type,
                        )
                        return

            # Enforce max queue size (drop oldest)
            if len(self._queue) >= self._config.max_queue_size:
                dropped = self._queue.pop(0)
                logger.debug(
                    "AttentionManager: queue full, dropped %s",
                    dropped.event_type,
                )

            self._queue.append(QueuedNotification(
                text=text,
                priority=priority,
                event_type=event_type,
            ))
            logger.debug(
                "AttentionManager: queued %s (%d in queue)",
                event_type, len(self._queue),
            )

    def _flush_queue_locked(self) -> None:
        """Deliver all queued notifications. Must hold _lock."""
        if not self._queue:
            return

        logger.info(
            "AttentionManager: flushing %d queued notification(s)",
            len(self._queue),
        )

        # Deliver in order (oldest first)
        to_deliver = list(self._queue)
        self._queue.clear()

        for item in to_deliver:
            self._last_delivery_time = time.time()
            if self._deliver_fn:
                try:
                    self._deliver_fn(item.text)
                except Exception:
                    logger.debug(
                        "AttentionManager: flush delivery failed for %s",
                        item.event_type,
                        exc_info=True,
                    )

    def drain_queue(self) -> List[QueuedNotification]:
        """Return and clear queued notifications for merging into reports.

        Called by the orchestrator during REPORTING state, BEFORE
        transitioning to IDLE. This allows proactive insights to be
        merged into the final report rather than dumped separately.

        Returns:
            List of queued notifications (oldest first). Queue is cleared.
        """
        with self._lock:
            items = list(self._queue)
            self._queue.clear()
            if items:
                logger.debug(
                    "AttentionManager: drained %d queued notification(s)",
                    len(items),
                )
            return items

    # ─────────────────────────────────────────────────────────
    # Introspection
    # ─────────────────────────────────────────────────────────

    @property
    def queue_size(self) -> int:
        with self._lock:
            return len(self._queue)

    @property
    def is_busy(self) -> bool:
        with self._lock:
            return self._mission_state != MissionState.IDLE
