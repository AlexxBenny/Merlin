# errors.py

"""
Typed error hierarchy for MERLIN compilation pipeline.

Design rules:
- CompilationError is the base — all compilation failures inherit from it.
- ParseError: JSON extraction/decode failed (retryable once).
- LLMUnavailableError: LLM not reachable (not retryable).
- MalformedPlanError: Valid JSON, but invalid plan structure (not retryable).
- FailureIR: Structured failure object — conforms to IR boundary.
  Executor never sees this; orchestrator routes around it.

error_type is Literal, not str. Supervision logic depends on this.
"""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


# ──────────────────────────────────────────────
# Exception hierarchy
# ──────────────────────────────────────────────

class CompilationError(Exception):
    """Base for all compilation failures."""
    pass


class ParseError(CompilationError):
    """JSON extraction or decode failed. Retryable once."""
    pass


class LLMUnavailableError(CompilationError):
    """LLM provider is not reachable. Not retryable."""
    pass


class MalformedPlanError(CompilationError):
    """Valid JSON, but invalid plan structure. Not retryable."""
    pass


# ──────────────────────────────────────────────
# FailureIR — Structured failure at the IR boundary
# ──────────────────────────────────────────────

class FailureIR(BaseModel):
    """Structured failure — conforms to IR boundary.

    compile() returns MissionPlan | FailureIR.
    Executor never receives FailureIR.
    Orchestrator inspects isinstance() and routes to error reporting.

    error_type is Literal — not arbitrary str.
    Supervision logic will inspect this downstream.
    """
    model_config = ConfigDict(extra="forbid")

    error_type: Literal[
        "parse_error",
        "llm_unavailable",
        "malformed_plan",
        "incomplete_coverage",
    ]
    error_message: str
    user_query: str
    retryable: bool = False

    # Distinguishes system failure from user-caused failure.
    # System failures: LLM down, JSON decode bug, internal error.
    # User failures: ambiguous query that can't map to skills.
    internal_error: bool = False
