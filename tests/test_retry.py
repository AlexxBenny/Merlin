# tests/test_retry.py

"""
Tests for Phase 4C: Retry Discipline.

Validates:
- Malformed JSON on first attempt → retry → success
- Malformed JSON on first attempt → retry → still malformed → FailureIR
- Invalid skill name → no retry → FailureIR
- LLM unavailable → no retry → FailureIR
- Retry changes at least one parameter (temperature + strict_json)
- Second attempt never triggers further retry (explicit flag)
"""

import json
import pytest
from unittest.mock import MagicMock

from cortex.mission_cortex import MissionCortex
from errors import FailureIR
from ir.mission import MissionPlan, ExecutionMode


class _StubSkill:
    name = "fs.create_folder"

    class contract:
        name = "fs.create_folder"
        description = "Create a folder on the filesystem."
        action = "create_folder"
        target_type = "folder"
        domain = "fs"
        inputs = {"path": "string"}
        optional_inputs = {}
        outputs = {}
        allowed_modes = {ExecutionMode.foreground}
        failure_policy = {}
        emits_events = []
        mutates_world = True


class _StubRegistry:
    """Minimal registry for compilation tests."""
    def all_names(self):
        return {"fs.create_folder"}

    def get(self, name):
        if name == "fs.create_folder":
            return _StubSkill()
        raise KeyError(f"Missing skill '{name}'")

    def all_skills(self):
        return [_StubSkill()]


def _valid_json():
    """Return a valid mission plan JSON string."""
    return json.dumps({
        "nodes": [
            {
                "id": "n1",
                "skill": "fs.create_folder",
                "inputs": {"path": "/tmp/test"},
            }
        ]
    })


def _make_cortex():
    """Build a cortex with a mock LLM."""
    llm = MagicMock()
    registry = _StubRegistry()
    return MissionCortex(llm, registry), llm


class TestRetryOnParseError:
    """Retry fires exactly once on parse_error, then succeeds."""

    def test_garbage_then_valid_json(self):
        cortex, llm = _make_cortex()
        # First call: garbage. Second call: valid JSON.
        llm.complete.side_effect = ["NOT JSON AT ALL", _valid_json()]

        result = cortex.compile("create a folder", {})

        assert isinstance(result, MissionPlan)
        assert llm.complete.call_count == 2

        # Second call must have different temperature
        second_call = llm.complete.call_args_list[1]
        assert second_call.kwargs.get("temperature") == 0.1

    def test_garbage_then_garbage_returns_failure_ir(self):
        cortex, llm = _make_cortex()
        # Both calls return garbage
        llm.complete.side_effect = ["NOT JSON", "STILL NOT JSON"]

        result = cortex.compile("create a folder", {})

        assert isinstance(result, FailureIR)
        assert result.error_type == "parse_error"
        assert result.retryable is False  # Exhausted retry
        assert llm.complete.call_count == 2


class TestNoRetryOnMalformedPlan:
    """Invalid plan structure → no retry → FailureIR."""

    def test_unknown_skill_no_retry(self):
        cortex, llm = _make_cortex()
        # Valid JSON but references a skill that doesn't exist
        llm.complete.return_value = json.dumps({
            "nodes": [
                {"id": "n1", "skill": "nonexistent.skill", "inputs": {}}
            ]
        })

        result = cortex.compile("do something", {})

        assert isinstance(result, FailureIR)
        assert result.error_type == "malformed_plan"
        # Only ONE LLM call — no retry for malformed plans
        assert llm.complete.call_count == 1


class TestNoRetryOnLLMUnavailable:
    """LLM down → no retry → FailureIR."""

    def test_connection_error_no_retry(self):
        cortex, llm = _make_cortex()
        llm.complete.side_effect = ConnectionError("Ollama not running")

        result = cortex.compile("create a folder", {})

        assert isinstance(result, FailureIR)
        assert result.error_type == "llm_unavailable"
        assert result.internal_error is True
        # Only ONE LLM call — no retry for connection errors
        assert llm.complete.call_count == 1


class TestRetryChangesParameters:
    """Second attempt must differ from first (temperature + strict prefix)."""

    def test_second_call_has_lower_temperature(self):
        cortex, llm = _make_cortex()
        llm.complete.side_effect = ["GARBAGE", _valid_json()]

        cortex.compile("test", {})

        first_call = llm.complete.call_args_list[0]
        second_call = llm.complete.call_args_list[1]

        # First call: default temperature (None)
        assert first_call.kwargs.get("temperature") is None
        # Second call: explicit lower temperature
        assert second_call.kwargs.get("temperature") == 0.1

    def test_second_call_has_strict_json_prefix(self):
        cortex, llm = _make_cortex()
        llm.complete.side_effect = ["GARBAGE", _valid_json()]

        cortex.compile("test", {})

        second_prompt = llm.complete.call_args_list[1].args[0]
        assert second_prompt.startswith("CRITICAL: You must respond with ONLY")


class TestRetryDoesNotRecurse:
    """Explicit _retry_attempted flag prevents more than one retry."""

    def test_exactly_two_calls_max(self):
        cortex, llm = _make_cortex()
        # Return garbage forever — should still stop at 2
        llm.complete.return_value = "GARBAGE FOREVER"

        result = cortex.compile("test", {})

        assert isinstance(result, FailureIR)
        assert llm.complete.call_count == 2
