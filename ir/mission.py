"""
IR v1 — Frozen Specification

Status: FROZEN
Change Policy:
  - No field removals
  - No semantic reinterpretation
  - No execution logic embedded
  - Future versions must be IR v2, v3, etc.
"""

import re
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ---------------------------
# IR VERSION (Frozen)
# ---------------------------

IR_VERSION = "1.0"


# ---------------------------
# FROZEN REGEX PATTERNS
# ---------------------------

# domain.action[.variant] — lowercase alphanumeric + underscores only
SKILL_NAME_PATTERN = re.compile(
    r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)?$"
)

# ConditionExpr.source: valid node id OR world.* namespace
# Node ids: lowercase alphanumeric + underscores
CONDITION_SOURCE_PATTERN = re.compile(
    r"^([a-z][a-z0-9_]*|world\.[a-z][a-z0-9_.]*)+$"
)


# ---------------------------
# INPUT VALUES
# ---------------------------

class OutputReference(BaseModel):
    """
    Typed reference to another node's output.

    This is the ONLY legal way to reference inter-node data.
    $ref is the external wire format (LLM boundary) — it is stripped
    by the parser before reaching this model. OutputReference has
    NO $ref field.

    Resolution order (strict, one-level only):
      value = results[node][output]
      if index is not None: value = value[index]   # list access
      if field is not None: value = value[field]    # dict access

    Constraints:
      - index requires output to be a list at runtime
      - field requires value (after index) to be a dict at runtime
      - No nested index/field chains (one level only)
      - No computed indices, no slices, no expressions
    """
    node: str
    output: str
    index: Optional[int] = None   # 0-based list element access
    field: Optional[str] = None   # single-level dict field access

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _validate_bounded_access(self) -> "OutputReference":
        # index must be non-negative if provided
        if self.index is not None and self.index < 0:
            raise ValueError(
                f"OutputReference.index must be >= 0, got {self.index}"
            )
        return self


# LiteralValue: any JSON-serializable value that is NOT an OutputReference
# and does NOT start with '$'.
# The actual $-string rejection happens via MissionNode.inputs validator.
LiteralValue = Any

InputValue = Union[OutputReference, LiteralValue]


# ---------------------------
# OUTPUT SPEC
# ---------------------------

class OutputSpec(BaseModel):
    """
    Namespaced, typed output declaration.
    Type is semantic/descriptive, not executable.
    """
    name: str          # e.g. research.findings.v1
    type: str          # semantic descriptor only

    model_config = ConfigDict(extra="forbid")


# ---------------------------
# EXECUTION MODE (Frozen Semantics)
# ---------------------------

class ExecutionMode(str, Enum):
    """
    foreground: blocks mission completion, failure fails mission
    background: non-blocking, failure logged only
    side_effect: non-blocking, failure ignored
    """
    foreground = "foreground"
    background = "background"
    side_effect = "side_effect"


# ---------------------------
# CONDITION (Frozen, Minimal)
# ---------------------------

class ConditionExpr(BaseModel):
    """
    Evaluated once at node scheduling time.
    If condition fails → node is SKIPPED.
    Skipped node: produces no outputs, does not fail mission, is marked SKIPPED.

    No expressions. No boolean logic. No chaining.
    Complex logic belongs in skills.
    """
    source: str        # node id OR world snapshot key (must match namespace)
    equals: Any

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _validate_source_format(self) -> "ConditionExpr":
        if not CONDITION_SOURCE_PATTERN.match(self.source):
            raise ValueError(
                f"ConditionExpr.source '{self.source}' must be a valid node id "
                f"or start with 'world.' namespace"
            )
        return self


# ---------------------------
# MISSION NODE (Atomic Intent Unit)
# ---------------------------

class MissionNode(BaseModel):
    """
    The smallest executable unit. Nothing smaller exists.

    Frozen invariants:
    - skill must be domain.action[.variant]
    - inputs must not contain $-prefixed strings
    - No extra fields allowed
    """
    id: str
    skill: str

    inputs: Dict[str, InputValue] = Field(default_factory=dict)
    outputs: Dict[str, OutputSpec] = Field(default_factory=dict)

    depends_on: List[str] = Field(default_factory=list)

    mode: ExecutionMode = ExecutionMode.foreground
    condition_on: Optional[ConditionExpr] = None

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _validate_frozen_constraints(self) -> "MissionNode":
        # 1. Skill name format (frozen regex)
        if not SKILL_NAME_PATTERN.match(self.skill):
            raise ValueError(
                f"Skill name '{self.skill}' does not match required format "
                f"'domain.action[.variant]' (lowercase, no hyphens)"
            )

        # 2. $-string rejection — enforcement point 1 of 2 (IR layer)
        for key, value in self.inputs.items():
            if isinstance(value, str) and value.startswith("$"):
                raise ValueError(
                    f"Input '{key}' contains banned $-prefixed string "
                    f"'{value}'. Use OutputReference instead."
                )

        return self


# ---------------------------
# MISSION PLAN (Immutable)
# ---------------------------

class MissionPlan(BaseModel):
    """
    id: globally unique per user request (deterministic, no clocks)
    nodes: closed set (no dynamic insertion)
    metadata: non-executable, informational only — must contain ir_version

    No runtime state. No progress tracking. No retry counters.
    """
    id: str
    nodes: List[MissionNode]
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _validate_ir_version(self) -> "MissionPlan":
        version = self.metadata.get("ir_version")
        if version is None:
            raise ValueError(
                "MissionPlan.metadata must contain 'ir_version'. "
                f"Expected '{IR_VERSION}'."
            )
        if version != IR_VERSION:
            raise ValueError(
                f"Unsupported IR version '{version}'. "
                f"Expected '{IR_VERSION}'."
            )
        return self

    @property
    def ir_version(self) -> str:
        return self.metadata.get("ir_version", IR_VERSION)
