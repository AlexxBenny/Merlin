# conversation/outcome.py

"""
MissionOutcome — What a mission produced.

This is the keystone that makes reference resolution possible:
- "play the second video"   → visible_lists["search_results"][1]
- "open the folder you made" → artifacts["created_folder"]
- "summarize that"           → active_entity

Design rules:
- Created ONCE per mission, AFTER execution completes
- artifacts:     singular referents (one thing)
- visible_lists: ordinally referenceable collections (many things)
- These two are NEVER mixed
- nodes_executed/skipped enable replay + debugging
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field
import time


class MissionOutcome(BaseModel):
    """Immutable record of what a mission produced.

    Appended to ConversationFrame.outcomes after each mission.
    Consumed by WorldResolver for reference resolution.
    """
    model_config = ConfigDict(extra="forbid")

    mission_id: str
    timestamp: float = Field(default_factory=time.time)

    # What ran (enables deterministic replay + debugging)
    nodes_executed: List[str]
    nodes_skipped: List[str]
    nodes_failed: List[str] = Field(default_factory=list)
    nodes_timed_out: List[str] = Field(default_factory=list)

    # Singular referents — "the folder", "the file"
    # Keys are semantic names, values are the referent data
    artifacts: Dict[str, Any] = Field(default_factory=dict)

    # Ordinally referenceable collections — "the second video"
    # Keys are collection names, values are ordered lists
    visible_lists: Dict[str, list] = Field(default_factory=dict)

    # What's now "active" in conversation context
    active_entity: Optional[str] = None
    active_domain: Optional[str] = None
