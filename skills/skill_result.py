# skills/skill_result.py

"""
SkillResult — Structured return type for all skills.

Separates contract-validated outputs from side-channel metadata.
Executor validates only `outputs` against the skill contract.
Metadata is consumed by MissionOrchestrator for entity/domain tracking.

Design rules:
- outputs:  MUST match contract.outputs keys exactly
- metadata: side-channel, never validated against contract
- status:   optional semantic status override (e.g. "no_op" for ambiguity)
- Immutable after creation (frozen dataclass)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class SkillResult:
    """Structured skill return value.

    outputs:  Contract-validated keys only. These are what downstream
              nodes can reference via OutputReference.
    metadata: Side-channel data (entity, domain, trace IDs, latency).
              Never validated against contract. Consumed by orchestrator.
    status:   Optional semantic status override. When set, executor uses
              this instead of inferring from outputs. Values: "no_op",
              "completed" (default if None). Skills use "no_op" with
              metadata.reason to signal ambiguity or idempotent skips.
    """
    outputs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: Optional[str] = None

