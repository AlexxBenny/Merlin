# cortex/world_state_provider.py

"""
WorldStateProvider — Abstraction seam for how world state enters the LLM prompt.

Architecture:
    WorldState (complete, internal, authoritative)
            ↓
    WorldStateProvider.build_schema()
            ↓
    WorldStateView (query-scoped projection for LLM)

Implementations:
    SimpleWorldStateProvider — raw model_dump passthrough (Phase 1-2)
    FilteredWorldStateProvider — query-scoped view projection (Phase 3A)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Set

from world.snapshot import WorldSnapshot


class WorldStateProvider(ABC):
    """How world state enters the LLM prompt.

    Cortex calls build_schema() and gets back a dict.
    The provider decides what to include and how to format it.

    The full WorldState remains authoritative internally for proactivity,
    event processing, and monitoring. This only controls the LLM view.
    """

    @abstractmethod
    def build_schema(
        self,
        snapshot: WorldSnapshot,
        query: Optional[str] = None,
        candidate_skills: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        """Return formatted world state for the LLM prompt.

        Args:
            snapshot: Immutable world snapshot (full state).
            query: User query text (for domain detection).
            candidate_skills: Skill names from SkillDiscovery (for domain extraction).

        Must be serializable to JSON. Must be bounded in size.
        """
        ...


class SimpleWorldStateProvider(WorldStateProvider):
    """Direct model_dump passthrough. No filtering.

    Returns the complete WorldState. Safe default for small skill sets.
    """

    def build_schema(
        self,
        snapshot: WorldSnapshot,
        query: Optional[str] = None,
        candidate_skills: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        return snapshot.state.model_dump()
