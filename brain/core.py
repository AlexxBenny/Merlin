# brain/core.py

"""
BrainCore — The Circuit Breaker.

This is NOT an intelligence module.
It answers ONE binary question:
    "Is this safe to handle reflexively, or do we need full cognition?"

Design rules (ARCHITECTURE.md §3.2):
- Constant-time routing
- Minimal pattern checks
- No reasoning, planning, skill awareness, or environment access
- Config-driven: keywords come from config/routing.yaml
- This layer should ALMOST NEVER change

Routing invariant:
    False positives (escalating to MISSION unnecessarily) = SAFE
    False negatives (handling complex task as REFLEX) = CATASTROPHIC
    Therefore: default bias → MISSION
"""

from dataclasses import dataclass
from typing import Any, List
import time


@dataclass(frozen=True)
class Percept:
    modality: str       # "text", "speech", "vision"
    payload: str        # raw content
    confidence: float   # 1.0 for text, lower for speech/vision
    timestamp: float


class CognitiveRoute:
    REFLEX = "reflex"
    MISSION = "mission"
    REFUSE = "refuse"


class BrainCore:
    """
    FINAL AUTHORITY ON ROUTING.
    DO NOT ADD LOGIC HERE.

    Config-driven circuit breaker:
    - refuse_indicators → REFUSE
    - reflex_engine.try_match() → REFLEX (template-authoritative)
    - mission_indicators → MISSION (keyword scan)
    - default → MISSION (safe bias)
    """

    def __init__(
        self,
        mission_indicators: List[str] | None = None,
        refuse_indicators: List[str] | None = None,
        reflex_engine: Any | None = None,
    ):
        self._mission_indicators = [
            k.lower() for k in (mission_indicators or [])
        ]
        self._refuse_indicators = [
            k.lower() for k in (refuse_indicators or [])
        ]
        self._reflex_engine = reflex_engine

    def route(self, percept: Percept) -> str:
        """
        Route a percept to the correct cognitive path.

        Order matters:
        1. REFUSE — dangerous commands (safety first)
        2. REFLEX — template match via ReflexEngine (authoritative)
        3. MISSION — structural complexity keywords
        4. Default → MISSION (safe bias)
        """
        text = percept.payload.lower().strip()

        # 1. Safety gate: refuse dangerous commands
        if any(k in text for k in self._refuse_indicators):
            return CognitiveRoute.REFUSE

        # 2. Template-authoritative reflex detection
        #    ReflexEngine.try_match() is cheap (regex only, no execution)
        if self._reflex_engine and self._reflex_engine.try_match(text):
            return CognitiveRoute.REFLEX

        # 3. Structural complexity → needs mission compilation
        if any(k in text for k in self._mission_indicators):
            return CognitiveRoute.MISSION

        # 4. Default: MISSION (safe bias — false positive is harmless)
        return CognitiveRoute.MISSION
