# reporting/notification_policy.py

"""
NotificationPolicy — Deterministic rules for proactive speech.

This module answers ONE question:
    "Should MERLIN speak about this event?"

Design rules (guardrails from architecture review):
- Must stay stupid: match event type, check priority, check user focus.
- Must NEVER infer intent, look at history deeply, or reason semantically.
- Must be config-driven (YAML) for easy tuning without code changes.
- Silence is the default. Speech is the exception.
"""

from enum import Enum
from typing import Dict, Optional, Set

from pydantic import BaseModel, ConfigDict, Field

from world.timeline import WorldEvent
from world.snapshot import WorldSnapshot


# ─────────────────────────────────────────────────────────────
# Policy types
# ─────────────────────────────────────────────────────────────

class NotificationAction(str, Enum):
    """What to do when an event occurs."""
    NOTIFY = "notify"       # Tell the user
    LOG = "log"             # Log silently, do not speak
    IGNORE = "ignore"       # Completely ignore


class EventPriority(str, Enum):
    """How urgent the event is."""
    CRITICAL = "critical"   # Always notify, even if user is focused
    INFO = "info"           # Notify unless user is deeply focused
    BACKGROUND = "background"  # Log only, never notify


class EventRule(BaseModel):
    """
    Single rule mapping an event type to a notification action.
    """
    model_config = ConfigDict(extra="forbid")

    event_type: str
    action: NotificationAction = NotificationAction.NOTIFY
    priority: EventPriority = EventPriority.INFO


# ─────────────────────────────────────────────────────────────
# Notification Policy
# ─────────────────────────────────────────────────────────────

class NotificationPolicy:
    """
    Deterministic event → action mapper.

    Usage:
        policy = NotificationPolicy.from_rules([...])
        decision = policy.should_notify(event, snapshot)

    The policy is pure data. No LLM. No inference.
    """

    def __init__(self, rules: Dict[str, EventRule]):
        self._rules = rules

    @classmethod
    def from_rules(cls, rules_list: list[EventRule]) -> "NotificationPolicy":
        """Build from a list of EventRule objects."""
        rules = {r.event_type: r for r in rules_list}
        return cls(rules)

    @classmethod
    def from_config(cls, config: list[dict]) -> "NotificationPolicy":
        """
        Build from YAML-loaded config.

        Expected format:
        - event_type: meeting_starting
          action: notify
          priority: critical
        - event_type: download_progress
          action: ignore
          priority: background
        """
        rules = {}
        for entry in config:
            rule = EventRule(**entry)
            rules[rule.event_type] = rule
        return cls(rules)

    def should_notify(
        self,
        event: WorldEvent,
        snapshot: WorldSnapshot,
    ) -> bool:
        """
        Deterministic decision: should the user hear about this event?

        Logic (in order):
        1. Look up event type in rules
        2. If no rule → default to IGNORE (silence by default)
        3. If action is IGNORE → False
        4. If action is LOG → False
        5. If action is NOTIFY:
           a. CRITICAL priority → always True
           b. INFO priority → True unless user is deeply focused
           c. BACKGROUND priority → False (log only)
        """

        rule = self._rules.get(event.type)

        # Unknown event types → silence
        if rule is None:
            return False

        if rule.action == NotificationAction.IGNORE:
            return False

        if rule.action == NotificationAction.LOG:
            return False

        # action == NOTIFY
        if rule.priority == EventPriority.CRITICAL:
            return True

        if rule.priority == EventPriority.INFO:
            # Respect user focus: if user is deeply engaged, defer
            # For now, always notify for INFO. User focus gating
            # can be refined when WorldState tracks focus depth.
            return True

        # BACKGROUND priority → never notify, only log
        return False

    def get_action(self, event_type: str) -> NotificationAction:
        """Get the configured action for an event type."""
        rule = self._rules.get(event_type)
        if rule is None:
            return NotificationAction.IGNORE
        return rule.action

    @property
    def registered_event_types(self) -> Set[str]:
        """All event types this policy knows about."""
        return set(self._rules.keys())

    # ─────────────────────────────────────────────────────────
    # Factory: sensible defaults
    # ─────────────────────────────────────────────────────────

    @classmethod
    def default(cls) -> "NotificationPolicy":
        """
        Built-in defaults. Overridden by YAML config when available.
        """
        return cls.from_rules([
            # Critical: always notify
            EventRule(event_type="meeting_starting", action=NotificationAction.NOTIFY, priority=EventPriority.CRITICAL),
            EventRule(event_type="error_attention", action=NotificationAction.NOTIFY, priority=EventPriority.CRITICAL),

            # Info: notify when appropriate
            EventRule(event_type="ad_muted", action=NotificationAction.NOTIFY, priority=EventPriority.INFO),
            EventRule(event_type="download_completed", action=NotificationAction.NOTIFY, priority=EventPriority.INFO),
            EventRule(event_type="task_completed", action=NotificationAction.NOTIFY, priority=EventPriority.INFO),

            # Background: log only
            EventRule(event_type="download_progress", action=NotificationAction.LOG, priority=EventPriority.BACKGROUND),
            EventRule(event_type="network_fluctuation", action=NotificationAction.IGNORE, priority=EventPriority.BACKGROUND),
            EventRule(event_type="app_focused", action=NotificationAction.IGNORE, priority=EventPriority.BACKGROUND),
        ])
