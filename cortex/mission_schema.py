"""Compatibility shim — cortex used to define mission schema here.
The canonical definitions live in the neutral IR package `ir.mission`.
Keeping this module ensures existing imports (`cortex.mission_schema`)
continue to work while ownership is moved to `ir`."""

from ir.mission import (
    IR_VERSION,
    ConditionExpr,
    ExecutionMode,
    MissionNode,
    MissionPlan,
    OutputReference,
    OutputSpec,
)

__all__ = [
    "IR_VERSION",
    "ConditionExpr",
    "ExecutionMode",
    "MissionNode",
    "MissionPlan",
    "OutputReference",
    "OutputSpec",
]
