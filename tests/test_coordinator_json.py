# tests/test_coordinator_json.py

"""
Tests for CognitiveCoordinator JSON parsing resilience.

Verifies that _parse_response handles:
- Clean JSON
- JSON wrapped in markdown fences
- JSON with preamble text
- Completely invalid responses (fallback)
"""

import json
from unittest.mock import MagicMock

import pytest

from cortex.cognitive_coordinator import (
    LLMCognitiveCoordinator,
    CoordinatorMode,
    CoordinatorResult,
    FALLBACK_RESULT,
)


def _make_coordinator():
    """Create coordinator with mock LLM."""
    llm = MagicMock()
    return LLMCognitiveCoordinator(llm=llm)


class TestParseResponseResilience:
    """_parse_response handles various LLM output formats."""

    def test_clean_json(self):
        coord = _make_coordinator()
        raw = json.dumps({
            "mode": "SKILL_PLAN",
            "reasoning": "skills cover this",
        })
        result = coord._parse_response(raw, "test query")
        assert result.mode == CoordinatorMode.SKILL_PLAN

    def test_json_with_markdown_fences(self):
        coord = _make_coordinator()
        raw = '```json\n{"mode": "SKILL_PLAN", "reasoning": "test"}\n```'
        result = coord._parse_response(raw, "test query")
        assert result.mode == CoordinatorMode.SKILL_PLAN

    def test_json_with_preamble_text(self):
        """LLM outputs reasoning text BEFORE JSON — extract_json_block handles this."""
        coord = _make_coordinator()
        raw = (
            "Step 1: The user wants to set volume.\n"
            "Step 2: system.set_volume skill exists.\n\n"
            '{"mode": "SKILL_PLAN", "reasoning": "skills handle this"}'
        )
        result = coord._parse_response(raw, "set volume to 50")
        assert result.mode == CoordinatorMode.SKILL_PLAN

    def test_json_with_trailing_text(self):
        """JSON followed by explanation text."""
        coord = _make_coordinator()
        raw = (
            '{"mode": "DIRECT_ANSWER", "answer": "42", "reasoning": "math"}\n\n'
            "I hope that helps!"
        )
        result = coord._parse_response(raw, "what is 6*7")
        assert result.mode == CoordinatorMode.DIRECT_ANSWER
        assert result.answer == "42"

    def test_completely_invalid_falls_back(self):
        """Pure text with no JSON → FALLBACK_RESULT."""
        coord = _make_coordinator()
        raw = "I think you should use SKILL_PLAN because the skills are available."
        result = coord._parse_response(raw, "test")
        assert result.mode == FALLBACK_RESULT.mode

    def test_empty_response_falls_back(self):
        coord = _make_coordinator()
        result = coord._parse_response("", "test")
        assert result.mode == FALLBACK_RESULT.mode

    def test_direct_answer_parsed(self):
        coord = _make_coordinator()
        raw = json.dumps({
            "mode": "DIRECT_ANSWER",
            "answer": "Python was created by Guido van Rossum",
            "reasoning": "knowledge question",
        })
        result = coord._parse_response(raw, "who created python")
        assert result.mode == CoordinatorMode.DIRECT_ANSWER
        assert "Guido" in result.answer

    def test_reasoned_plan_parsed(self):
        coord = _make_coordinator()
        raw = json.dumps({
            "mode": "REASONED_PLAN",
            "computed": {"volume_level": 35},
            "refined_query": "set volume to 35",
            "reasoning": "derived from user pref",
        })
        result = coord._parse_response(raw, "set volume to preferred")
        assert result.mode == CoordinatorMode.REASONED_PLAN
        assert result.computed_vars == {"volume_level": 35}
        assert result.refined_query == "set volume to 35"

    def test_unsupported_parsed(self):
        coord = _make_coordinator()
        raw = json.dumps({
            "mode": "UNSUPPORTED",
            "missing": ["email_send"],
            "suggestion": "Try using your email client",
            "reasoning": "no email skill",
        })
        result = coord._parse_response(raw, "send an email")
        assert result.mode == CoordinatorMode.UNSUPPORTED
        assert "email_send" in result.missing_capabilities

    def test_unknown_mode_falls_back(self):
        coord = _make_coordinator()
        raw = json.dumps({"mode": "INVENTED_MODE", "reasoning": "test"})
        result = coord._parse_response(raw, "test")
        assert result.mode == FALLBACK_RESULT.mode


class TestCapabilityAwareness:
    """Coordinator prompt includes MERLIN system capabilities."""

    def test_prompt_contains_capabilities(self):
        coord = _make_coordinator()
        from world.snapshot import WorldSnapshot
        from world.state import WorldState

        state = WorldState()
        snapshot = WorldSnapshot.build(state, [])

        prompt = coord._build_prompt(
            query="do you have memory?",
            snapshot=snapshot,
            skill_manifest={},
        )

        assert "Episodic memory" in prompt
        assert "Persistent job scheduler" in prompt
        assert "Do NOT claim lack of memory" in prompt
