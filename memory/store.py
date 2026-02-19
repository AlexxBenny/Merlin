# memory/store.py

"""
MemoryStore — Abstraction seam for episodic memory storage and retrieval.

Today:  ListMemoryStore (simple in-memory list, recency-based retrieval).
Future: VectorMemoryStore (FAISS/ChromaDB-backed semantic search).

This seam prevents future memory integration from refactoring
half of cortex and orchestrator. Create the interface now,
swap the backend later.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class MemoryStore(ABC):
    """Episodic memory: store what happened, retrieve what's relevant.

    The orchestrator stores episodes after each mission.
    The context provider may retrieve episodes for the LLM prompt.
    """

    @abstractmethod
    def store_episode(
        self,
        mission_id: str,
        query: str,
        outcome_summary: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store a mission episode for future retrieval."""
        ...

    @abstractmethod
    def retrieve_relevant(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Retrieve episodes relevant to a query.

        Today: recency-based.
        Future: semantic similarity-based.
        """
        ...


class ListMemoryStore(MemoryStore):
    """Simple list-backed memory. No vector search.

    Retrieval is recency-based (returns last top_k episodes).
    Capped at max_episodes to prevent unbounded growth.
    """

    def __init__(self, max_episodes: int = 100):
        self._episodes: List[Dict[str, Any]] = []
        self._max = max_episodes

    def store_episode(
        self,
        mission_id: str,
        query: str,
        outcome_summary: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._episodes.append({
            "mission_id": mission_id,
            "query": query,
            "outcome_summary": outcome_summary,
            "metadata": metadata or {},
        })
        # Enforce cap
        if len(self._episodes) > self._max:
            self._episodes = self._episodes[-self._max:]

    def retrieve_relevant(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Return last top_k episodes (recency-based)."""
        return self._episodes[-top_k:]

    @property
    def episode_count(self) -> int:
        return len(self._episodes)
