from enum import Enum
from typing import Dict, Set, List

from pydantic import BaseModel, ConfigDict, Field

from ir.mission import ExecutionMode


class FailurePolicy(str, Enum):
    """
    What happens when a skill fails under a given execution mode.
    The executor reads this — it does NOT hardcode mode→failure mapping.
    """
    FAIL = "fail"           # Propagate failure (mission fails)
    CONTINUE = "continue"   # Log and continue (mission proceeds)
    IGNORE = "ignore"       # Silently ignore


class SkillContract(BaseModel):
    """
    Frozen guarantees a skill declares.

    This is a schema + validator + enforcement rule.
    NOT a new subsystem.

    The executor reads this contract at runtime to enforce:
    - allowed execution modes
    - failure semantics per mode
    - event emission guarantees
    - world mutation permission
    - focus and conflict constraints
    """
    model_config = ConfigDict(extra="forbid")

    name: str                                       # domain.action[.variant]
    description: str = ""                           # For LLM skill manifest

    # ── Capability metadata (for intent matching) ──
    action: str = ""                                # Canonical action: "set_volume", "open_app"
    target_type: str = ""                           # What this skill acts on: "volume", "app"

    inputs: Dict[str, str]                          # key → semantic type (REQUIRED)
    optional_inputs: Dict[str, str] = {}            # key → semantic type (OPTIONAL, have defaults)
    outputs: Dict[str, str]                         # key → semantic type

    allowed_modes: Set[ExecutionMode]                # Which modes this skill permits
    failure_policy: Dict[ExecutionMode, FailurePolicy]  # Mode → what to do on failure

    emits_events: List[str] = []                    # Event types this skill may emit
    mutates_world: bool = False                     # Whether skill may emit world events

    idempotent: bool = False                        # Safe to retry without side effects

    # ── Domain & scheduling metadata ──
    domain: str = ""                                # "fs", "system", "browser", "media"
    requires_focus: bool = False                    # Needs foreground window control
    resource_cost: str = "low"                      # "low" | "medium" | "high"
    conflicts_with: List[str] = Field(default_factory=list)  # Skill names that cannot run in parallel

    # ── Narration metadata (Phase 8) ──
    # Controls how NarrationPolicy announces this skill during execution.
    # narration_visibility:
    #   "foreground" — included in pre-narration (default)
    #   "background" — omitted from narration but logged
    #   "silent"     — completely invisible to narration
    # narration_verb:
    #   Optional human-action phrase override, e.g. "Setting volume".
    #   Empty = auto-generate from description field.
    narration_visibility: str = "foreground"
    narration_verb: str = ""
