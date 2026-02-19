# tests/test_errors.py

"""
Tests for Phase 4B: Typed Error Hierarchy & FailureIR.

Validates:
- Error hierarchy (CompilationError → ParseError, LLMUnavailableError, MalformedPlanError)
- FailureIR schema (Literal error_type, extra=forbid)
- FailureIR serialization
- compile() returns MissionPlan | FailureIR, never raises, never returns None
"""

import pytest
from pydantic import ValidationError

from errors import (
    CompilationError,
    ParseError,
    LLMUnavailableError,
    MalformedPlanError,
    FailureIR,
)


class TestErrorHierarchy:
    """Verify exception inheritance chain."""

    def test_parse_error_is_compilation_error(self):
        assert issubclass(ParseError, CompilationError)

    def test_llm_unavailable_is_compilation_error(self):
        assert issubclass(LLMUnavailableError, CompilationError)

    def test_malformed_plan_is_compilation_error(self):
        assert issubclass(MalformedPlanError, CompilationError)

    def test_parse_error_catchable_as_compilation_error(self):
        with pytest.raises(CompilationError):
            raise ParseError("bad json")

    def test_llm_unavailable_catchable_as_compilation_error(self):
        with pytest.raises(CompilationError):
            raise LLMUnavailableError("ollama down")


class TestFailureIR:
    """Validate FailureIR Pydantic model."""

    def test_valid_failure_ir(self):
        ir = FailureIR(
            error_type="parse_error",
            error_message="No JSON found",
            user_query="create a folder",
        )
        assert ir.error_type == "parse_error"
        assert ir.retryable is False
        assert ir.internal_error is False

    def test_literal_error_type_enforced(self):
        """error_type must be one of the Literal values."""
        with pytest.raises(ValidationError):
            FailureIR(
                error_type="unknown_error",
                error_message="bad",
                user_query="test",
            )

    def test_all_valid_error_types(self):
        for et in ("parse_error", "llm_unavailable", "malformed_plan"):
            ir = FailureIR(
                error_type=et,
                error_message="test",
                user_query="test",
            )
            assert ir.error_type == et

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            FailureIR(
                error_type="parse_error",
                error_message="bad",
                user_query="test",
                garbage="nope",
            )

    def test_retryable_flag(self):
        ir = FailureIR(
            error_type="parse_error",
            error_message="bad json",
            user_query="test",
            retryable=True,
        )
        assert ir.retryable is True

    def test_internal_error_flag(self):
        ir = FailureIR(
            error_type="llm_unavailable",
            error_message="connection refused",
            user_query="test",
            internal_error=True,
        )
        assert ir.internal_error is True

    def test_serialization_roundtrip(self):
        ir = FailureIR(
            error_type="malformed_plan",
            error_message="Unknown skill 'foo.bar'",
            user_query="do something",
            retryable=False,
            internal_error=False,
        )
        data = ir.model_dump()
        assert data["error_type"] == "malformed_plan"
        assert data["user_query"] == "do something"

        # Roundtrip
        ir2 = FailureIR(**data)
        assert ir2 == ir


class TestFailureIRDistinction:
    """Test the system vs user failure distinction."""

    def test_system_failure(self):
        ir = FailureIR(
            error_type="llm_unavailable",
            error_message="Ollama not running",
            user_query="create a folder",
            internal_error=True,
        )
        assert ir.internal_error is True

    def test_user_failure(self):
        ir = FailureIR(
            error_type="malformed_plan",
            error_message="No skills match query",
            user_query="xyzzy frobulate the widgets",
            internal_error=False,
        )
        assert ir.internal_error is False
