# tests/test_fallback.py

"""
Tests for Phase 4D: Fallback Compiler.

Validates:
- Simple command + LLM down → fallback plan
- Complex command + LLM down → FailureIR
- Fallback plan passes IR validation
- Fallback plan passes executor contract
- Keyword matching for known skills
- Various phrasing patterns for create_folder
- Integration with MissionCortex.compile() when LLM is unavailable
"""

import pytest
from unittest.mock import MagicMock

from cortex.fallback import FallbackCompiler
from cortex.mission_cortex import MissionCortex
from cortex.validators import validate_mission_plan
from errors import FailureIR
from execution.registry import SkillRegistry
from ir.mission import MissionPlan, ExecutionMode, IR_VERSION
from skills.contract import SkillContract, FailurePolicy


# ── Test skill setup ──

class _TestCreateFolderSkill:
    """Minimal create_folder skill for testing."""
    name = "fs.create_folder"
    contract = SkillContract(
        name="fs.create_folder",
        description="Create a directory at a location anchor",
        inputs={"name": "folder_name"},
        optional_inputs={"anchor": "anchor_name", "parent": "relative_path"},
        outputs={"created": "filesystem_path"},
        allowed_modes={ExecutionMode.foreground, ExecutionMode.side_effect},
        failure_policy={
            ExecutionMode.foreground: FailurePolicy.FAIL,
            ExecutionMode.side_effect: FailurePolicy.IGNORE,
        },
        emits_events=["folder_created"],
        mutates_world=True,
    )


def _make_registry():
    registry = SkillRegistry()
    registry.register(_TestCreateFolderSkill(), validate_types=False)
    return registry


# ── Direct FallbackCompiler tests ──

class TestFallbackMatchesSimpleCommands:
    """Keyword matching for known skill patterns."""

    def test_create_folder_basic(self):
        fc = FallbackCompiler(_make_registry())
        result = fc.compile("create a folder named hello")
        assert isinstance(result, MissionPlan)
        assert result.nodes[0].skill == "fs.create_folder"
        assert result.nodes[0].inputs["name"] == "hello"

    def test_make_folder(self):
        fc = FallbackCompiler(_make_registry())
        result = fc.compile("make a folder called test")
        assert isinstance(result, MissionPlan)
        assert result.nodes[0].inputs["name"] == "test"

    def test_new_directory(self):
        fc = FallbackCompiler(_make_registry())
        result = fc.compile("new folder myproject")
        assert isinstance(result, MissionPlan)
        assert result.nodes[0].inputs["name"] == "myproject"

    def test_create_folder_on_desktop(self):
        fc = FallbackCompiler(_make_registry())
        result = fc.compile("create folder named hello on desktop")
        assert isinstance(result, MissionPlan)
        assert result.nodes[0].inputs["anchor"] == "DESKTOP"

    def test_create_folder_in_documents(self):
        fc = FallbackCompiler(_make_registry())
        result = fc.compile("create a folder named data in documents")
        assert isinstance(result, MissionPlan)
        assert result.nodes[0].inputs["anchor"] == "DOCUMENTS"


class TestFallbackRejectsComplexCommands:
    """Queries too complex for keyword matching → FailureIR."""

    def test_unrelated_query(self):
        fc = FallbackCompiler(_make_registry())
        result = fc.compile("what is the weather today")
        assert isinstance(result, FailureIR)
        assert result.error_type == "llm_unavailable"

    def test_multi_step_query(self):
        fc = FallbackCompiler(_make_registry())
        result = fc.compile("create a folder and then search for files in it")
        # Contains "create" and "folder" but also multi-step intent
        # The fallback might match this — that's OK for single node.
        # What matters is it never attempts DAG inference.
        if isinstance(result, MissionPlan):
            assert len(result.nodes) == 1  # Never multi-node

    def test_ambiguous_query_no_name(self):
        fc = FallbackCompiler(_make_registry())
        result = fc.compile("create a folder")
        # No folder name specified
        assert isinstance(result, FailureIR)


class TestFallbackPlanIsIRValid:
    """Fallback plans must pass the same validation as LLM plans."""

    def test_passes_ir_validation(self):
        fc = FallbackCompiler(_make_registry())
        result = fc.compile("create a folder named test")
        assert isinstance(result, MissionPlan)

        # Must pass the same validator
        available_skills = {"fs.create_folder"}
        validate_mission_plan(result, available_skills)

    def test_has_ir_version_metadata(self):
        fc = FallbackCompiler(_make_registry())
        result = fc.compile("create a folder named test")
        assert isinstance(result, MissionPlan)
        assert result.metadata["ir_version"] == IR_VERSION
        assert result.metadata["source"] == "fallback_compiler"

    def test_deterministic_id(self):
        fc = FallbackCompiler(_make_registry())
        r1 = fc.compile("create a folder named test")
        r2 = fc.compile("create a folder named test")
        assert isinstance(r1, MissionPlan) and isinstance(r2, MissionPlan)
        assert r1.id == r2.id  # Deterministic

    def test_single_node_foreground_mode(self):
        fc = FallbackCompiler(_make_registry())
        result = fc.compile("create a folder named test")
        assert isinstance(result, MissionPlan)
        assert len(result.nodes) == 1
        assert result.nodes[0].mode == ExecutionMode.foreground


# ── Integration with MissionCortex ──

class TestFallbackIntegrationWithCortex:
    """When LLM is down, cortex routes through fallback."""

    def test_llm_down_simple_command_uses_fallback(self):
        llm = MagicMock()
        llm.complete.side_effect = ConnectionError("Ollama not running")

        cortex = MissionCortex(llm, _make_registry())
        result = cortex.compile("create a folder named hello", {})

        assert isinstance(result, MissionPlan)
        assert result.metadata["source"] == "fallback_compiler"
        assert result.nodes[0].inputs["name"] == "hello"

    def test_llm_down_complex_command_returns_failure(self):
        llm = MagicMock()
        llm.complete.side_effect = ConnectionError("Ollama not running")

        cortex = MissionCortex(llm, _make_registry())
        result = cortex.compile("analyze the stock market", {})

        assert isinstance(result, FailureIR)
        assert result.error_type == "llm_unavailable"
