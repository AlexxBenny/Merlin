from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Literal, Optional, TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field
import time

if TYPE_CHECKING:
    from conversation.outcome import MissionOutcome


# ---------------------------------------------------------------
# Constants
# ---------------------------------------------------------------

HISTORY_CAP: int = 20
INTENT_CAP: int = 10
GOAL_CAP: int = 5
ENTITY_CAP: int = 50


# ---------------------------------------------------------------
# ContextFrame — domain-scoped semantic continuity
# ---------------------------------------------------------------

@dataclass(frozen=True)
class ContextFrame:
    """Immutable semantic context frame.

    Produced by Cortex or Skill, consumed by downstream nodes
    in the mission DAG via the Orchestrator.

    v1: Type freeze only. Dataflow propagation is a v2 concern.

    Fields:
        domain: Domain of the context (e.g., "browser", "file", "media")
        data: Small typed mapping of keys → scalar or small structured values
        produced_by: Optional identifier (node_id) that produced the frame
    """
    domain: str
    data: Dict[str, Any]
    produced_by: Optional[str] = None

    def get(self, key: str, default: Optional[Any] = None) -> Optional[Any]:
        return self.data.get(key, default)


# ---------------------------------------------------------------
# EntityRecord — typed entry in the entity registry
# ---------------------------------------------------------------

class EntityRecord(BaseModel):
    """Typed record for a tracked entity in conversation context.

    Every entity stored in the registry is wrapped in this record
    to preserve provenance, type, and creation time.

    Fields:
        type:            Semantic type (e.g. "search_results", "folder", "file")
        value:           The entity data (list of results, path string, etc.)
        source_mission:  Mission ID that produced this entity
        created_at:      When this entity was registered
    """
    model_config = ConfigDict(extra="forbid")

    type: str
    value: Any
    source_mission: str
    created_at: float = Field(default_factory=time.time)


# ---------------------------------------------------------------
# GoalState — tracked user goal with lifecycle
# ---------------------------------------------------------------

class GoalState(BaseModel):
    """Tracked user goal with lifecycle.

    Goals are created when a multi-step user objective is detected.
    They persist across missions and are updated as progress is made.

    Lifecycle: active → completed | failed | stalled

    Hierarchy fields (Phase 5B foundation):
        parent_goal:  ID of parent goal (None for top-level goals)
        subgoal_ids:  IDs of child goals
        priority:     Higher = more urgent (0 = default)
        progress:     0.0–1.0, computed from subgoal completion
    """
    model_config = ConfigDict(extra="forbid")

    id: str
    description: str
    status: Literal["active", "completed", "failed", "stalled"] = "active"
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    mission_ids: List[str] = Field(default_factory=list)

    # ── Hierarchy fields (Phase 5B GoalGraph foundation) ──
    parent_goal: Optional[str] = None
    subgoal_ids: List[str] = Field(default_factory=list)
    priority: int = Field(default=0)
    progress: float = Field(default=0.0)


# ---------------------------------------------------------------
# ConversationTurn — single exchange in conversation history
# ---------------------------------------------------------------

class ConversationTurn(BaseModel):
    """Single turn in conversation history.

    Appended only at system boundaries:
    - User turn: immediately after percept enters handle_percept()
    - Assistant turn: immediately after report delivery

    Never written mid-execution.
    """
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    text: str
    timestamp: float = Field(default_factory=time.time)
    mission_id: Optional[str] = None


# ---------------------------------------------------------------
# ConversationFrame — global session working context
# ---------------------------------------------------------------

class ConversationFrame(BaseModel):
    """
    Represents the current conversational focus.
    This is NOT memory. It is working context + structured history.

    The ConversationFrame is the single source of truth for:
    - What the user said (history)
    - What was produced (outcomes)
    - What entities exist in context (entity_registry)
    - What the user intended (recent_intents)
    - What goals are being pursued (active_goals)
    - What the last mission produced (last_results)
    """
    model_config = ConfigDict(extra="forbid")

    created_at: float = Field(default_factory=time.time)

    active_domain: Optional[str] = None
    active_entity: Optional[str] = None

    last_mission_id: Optional[str] = None

    # Domain-scoped context frames carried forward across missions
    context_frames: Dict[str, ContextFrame] = Field(default_factory=dict)

    unresolved_references: Dict[str, Any] = Field(default_factory=dict)

    # Structured conversation history (capped at HISTORY_CAP turns)
    history: List[ConversationTurn] = Field(default_factory=list)

    # Mission outcomes — what was produced (enables reference resolution)
    # Uses Any to break circular import; runtime type is List[MissionOutcome]
    outcomes: List[Any] = Field(default_factory=list)

    # ── Structured Conversation State (v2) ──────────────────────

    # Typed entity registry — keyed by semantic name, wrapped in EntityRecord
    # Replaces flat active_entity with multi-entity tracking
    # Survives domain switches — "search YouTube" then "create folder" won't erase search results
    entity_registry: Dict[str, EntityRecord] = Field(default_factory=dict)

    # Recent user intents — what the user wanted (not raw text)
    # Capped at INTENT_CAP, most recent last
    recent_intents: List[str] = Field(default_factory=list)

    # Active goals — tracked multi-turn user objectives
    # Capped at GOAL_CAP
    active_goals: List[GoalState] = Field(default_factory=list)

    # Last mission results — preserves full node→output structure
    # Keys are node IDs, values are their output dicts
    last_results: Dict[str, Any] = Field(default_factory=dict)

    # ── History Management ──────────────────────────────────────

    def append_turn(self, role: Literal["user", "assistant"], text: str,
                    mission_id: Optional[str] = None) -> None:
        """Append a turn and enforce history cap.

        Drops oldest turns when history exceeds HISTORY_CAP.
        """
        self.history.append(ConversationTurn(
            role=role,
            text=text,
            mission_id=mission_id,
        ))
        # Enforce cap — drop oldest
        if len(self.history) > HISTORY_CAP:
            self.history[:] = self.history[-HISTORY_CAP:]

    # ── Entity Registry ─────────────────────────────────────────

    def register_entity(
        self,
        key: str,
        value: Any,
        entity_type: str,
        source_mission: str,
    ) -> None:
        """Register a named entity in the conversation context.

        Overwrites if key already exists (latest wins).
        Enforces ENTITY_CAP — drops oldest entries by created_at.
        """
        self.entity_registry[key] = EntityRecord(
            type=entity_type,
            value=value,
            source_mission=source_mission,
        )
        # Enforce cap — drop oldest by created_at
        if len(self.entity_registry) > ENTITY_CAP:
            sorted_keys = sorted(
                self.entity_registry.keys(),
                key=lambda k: self.entity_registry[k].created_at,
            )
            for k in sorted_keys[: len(self.entity_registry) - ENTITY_CAP]:
                del self.entity_registry[k]

    def get_entity(self, key: str) -> Optional[EntityRecord]:
        """Retrieve a typed entity from the registry."""
        return self.entity_registry.get(key)

    # ── Intent Tracking ─────────────────────────────────────────

    def push_intent(self, intent: str) -> None:
        """Record a recent user intent. Enforces INTENT_CAP."""
        self.recent_intents.append(intent)
        if len(self.recent_intents) > INTENT_CAP:
            self.recent_intents[:] = self.recent_intents[-INTENT_CAP:]

    # ── Goal Lifecycle ──────────────────────────────────────────

    def add_goal(self, goal_id: str, description: str) -> GoalState:
        """Track a new active goal. Enforces GOAL_CAP.

        When cap is exceeded, oldest completed/failed goals are
        evicted first. If all goals are active, oldest active is dropped.
        """
        goal = GoalState(id=goal_id, description=description)
        self.active_goals.append(goal)
        if len(self.active_goals) > GOAL_CAP:
            # Evict non-active goals first (completed/failed/stalled)
            active = [g for g in self.active_goals if g.status == "active"]
            inactive = [g for g in self.active_goals if g.status != "active"]
            # Drop oldest inactive, then oldest active if needed
            kept = inactive[-(GOAL_CAP - len(active)):] if inactive else []
            kept = active + kept
            self.active_goals[:] = kept[-GOAL_CAP:]
        return goal

    def complete_goal(self, goal_id: str) -> None:
        """Mark a goal as completed."""
        for g in self.active_goals:
            if g.id == goal_id:
                g.status = "completed"
                g.updated_at = time.time()
                return

    def fail_goal(self, goal_id: str) -> None:
        """Mark a goal as failed."""
        for g in self.active_goals:
            if g.id == goal_id:
                g.status = "failed"
                g.updated_at = time.time()
                return

    def get_active_goals(self) -> List[GoalState]:
        """Return only active (in-progress) goals."""
        return [g for g in self.active_goals if g.status == "active"]

    # ── Last Results ────────────────────────────────────────────

    def store_results(self, results: Dict[str, Any]) -> None:
        """Store last mission results, preserving full structure.

        Keys are node IDs, values are their output dicts.
        Replaces previous results entirely (only last mission matters).
        """
        self.last_results = results

    # ── Goal–Mission Attachment (Phase 5B foundation) ───────────

    def attach_mission_to_goal(self, mission_id: str, goal_id: str) -> bool:
        """Attach a completed mission to a goal.

        Returns True if the goal was found and updated, False otherwise.
        """
        for g in self.active_goals:
            if g.id == goal_id:
                if mission_id not in g.mission_ids:
                    g.mission_ids.append(mission_id)
                g.updated_at = time.time()
                return True
        return False

    def update_goal_progress(self, goal_id: str) -> None:
        """Recalculate goal progress from subgoal completion ratios.

        Progress = fraction of subgoals in completed status.
        If no subgoals, progress stays at its current value.
        """
        target = None
        for g in self.active_goals:
            if g.id == goal_id:
                target = g
                break
        if target is None or not target.subgoal_ids:
            return

        # Count completed subgoals
        sub_map = {g.id: g for g in self.active_goals}
        completed = sum(
            1 for sid in target.subgoal_ids
            if sid in sub_map and sub_map[sid].status == "completed"
        )
        target.progress = completed / len(target.subgoal_ids)
        target.updated_at = time.time()

