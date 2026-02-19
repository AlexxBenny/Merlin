# world/snapshot.py

from typing import Dict, Any, List
from pydantic import BaseModel, ConfigDict, Field
import time

from world.state import WorldState
from world.timeline import WorldEvent


class WorldSnapshot(BaseModel):
    """
    Immutable view of the world for reasoning.
    """

    created_at: float = Field(default_factory=time.time)
    state: WorldState
    recent_events: List[WorldEvent]

    model_config = ConfigDict(frozen=True, extra="forbid")

    @staticmethod
    def build(
        state: WorldState,
        recent_events: List[WorldEvent],
    ) -> "WorldSnapshot":
        return WorldSnapshot(
            state=state,
            recent_events=recent_events,
        )
