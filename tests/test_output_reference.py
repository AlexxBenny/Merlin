# tests/test_output_reference.py

"""
Tests for OutputReference extension: bounded index + field accessors.

Covers:
1. IR model construction (index/field, validation)
2. Executor resolution (index, field, index+field, error cases)
3. Compiler parsing ($ref with index/field)
4. Integration (end-to-end list→open chain)
"""

import pytest
from ir.mission import (
    OutputReference,
    MissionPlan,
    MissionNode,
    OutputSpec,
    ExecutionMode,
    IR_VERSION,
)


# ==================================================================
# 1. IR Model Construction
# ==================================================================


class TestOutputReferenceModel:
    """OutputReference with optional index/field."""

    def test_flat_reference_unchanged(self):
        """Existing flat references (no index/field) remain valid."""
        ref = OutputReference(node="node_0", output="apps")
        assert ref.node == "node_0"
        assert ref.output == "apps"
        assert ref.index is None
        assert ref.field is None

    def test_with_index_only(self):
        ref = OutputReference(node="node_0", output="apps", index=1)
        assert ref.index == 1
        assert ref.field is None

    def test_with_field_only(self):
        ref = OutputReference(node="node_0", output="status", field="name")
        assert ref.field == "name"
        assert ref.index is None

    def test_with_index_and_field(self):
        ref = OutputReference(node="node_0", output="apps", index=1, field="name")
        assert ref.index == 1
        assert ref.field == "name"

    def test_negative_index_rejected(self):
        with pytest.raises(ValueError, match="must be >= 0"):
            OutputReference(node="node_0", output="apps", index=-1)

    def test_zero_index_valid(self):
        ref = OutputReference(node="node_0", output="apps", index=0)
        assert ref.index == 0

    def test_extra_fields_rejected(self):
        """extra='forbid' still enforced."""
        with pytest.raises(Exception):
            OutputReference(node="n", output="o", slice="0:3")

    def test_mission_node_with_indexed_ref(self):
        """OutputReference with index/field works inside MissionNode."""
        node = MissionNode(
            id="node_1",
            skill="system.open_app",
            inputs={
                "app_name": OutputReference(
                    node="node_0", output="apps", index=1, field="name"
                )
            },
            depends_on=["node_0"],
        )
        ref = node.inputs["app_name"]
        assert isinstance(ref, OutputReference)
        assert ref.index == 1
        assert ref.field == "name"


# ==================================================================
# 2. Executor Resolution
# ==================================================================


class TestExecutorOutputResolution:
    """Test _resolve_input with index and field accessors."""

    @pytest.fixture
    def executor(self):
        """Create a minimal MissionExecutor for testing."""
        from unittest.mock import MagicMock
        from execution.executor import MissionExecutor

        registry = MagicMock()
        timeline = MagicMock()
        return MissionExecutor(registry, timeline)

    @pytest.fixture
    def list_results(self):
        """Simulated results from a list_apps node."""
        return {
            "node_0": {
                "apps": [
                    {"name": "Chrome", "pid": 1234, "title": "Google Chrome"},
                    {"name": "VSCode", "pid": 5678, "title": "Visual Studio Code"},
                    {"name": "Notepad", "pid": 9012, "title": "Untitled"},
                ],
                "count": 3,
            }
        }

    def test_flat_reference_returns_entire_value(self, executor, list_results):
        """Flat ref (no index/field) returns the entire output value."""
        ref = OutputReference(node="node_0", output="apps")
        ok, value, err, _ = executor._resolve_input(ref, list_results)
        assert ok
        assert len(value) == 3
        assert value[0]["name"] == "Chrome"

    def test_index_on_list(self, executor, list_results):
        """Index on list returns correct element."""
        ref = OutputReference(node="node_0", output="apps", index=1)
        ok, value, err, _ = executor._resolve_input(ref, list_results)
        assert ok
        assert value == {"name": "VSCode", "pid": 5678, "title": "Visual Studio Code"}

    def test_field_on_dict(self, executor, list_results):
        """Field on a single dict value."""
        # count is an int, so test on a dict output
        results = {"node_0": {"info": {"status": "active", "count": 3}}}
        ref = OutputReference(node="node_0", output="info", field="status")
        ok, value, err, _ = executor._resolve_input(ref, results)
        assert ok
        assert value == "active"

    def test_index_plus_field(self, executor, list_results):
        """Index + field returns specific field from list element."""
        ref = OutputReference(node="node_0", output="apps", index=1, field="name")
        ok, value, err, _ = executor._resolve_input(ref, list_results)
        assert ok
        assert value == "VSCode"

    def test_index_zero(self, executor, list_results):
        """Index 0 returns first element."""
        ref = OutputReference(node="node_0", output="apps", index=0, field="name")
        ok, value, err, _ = executor._resolve_input(ref, list_results)
        assert ok
        assert value == "Chrome"

    def test_index_last_element(self, executor, list_results):
        """Index 2 (last) returns last element."""
        ref = OutputReference(node="node_0", output="apps", index=2, field="pid")
        ok, value, err, _ = executor._resolve_input(ref, list_results)
        assert ok
        assert value == 9012

    def test_index_on_non_list_fails(self, executor, list_results):
        """Index on non-list value → deterministic error."""
        ref = OutputReference(node="node_0", output="count", index=0)
        ok, value, err, _ = executor._resolve_input(ref, list_results)
        assert not ok
        assert "not list" in err

    def test_index_out_of_bounds_fails(self, executor, list_results):
        """Index beyond list length → deterministic error with length info."""
        ref = OutputReference(node="node_0", output="apps", index=10)
        ok, value, err, _ = executor._resolve_input(ref, list_results)
        assert not ok
        assert "out of bounds" in err
        assert "length=3" in err

    def test_field_on_non_dict_fails(self, executor, list_results):
        """Field on non-dict value → deterministic error."""
        ref = OutputReference(node="node_0", output="count", field="name")
        ok, value, err, _ = executor._resolve_input(ref, list_results)
        assert not ok
        assert "not dict" in err

    def test_missing_field_fails(self, executor, list_results):
        """Field not found in dict → deterministic error with available keys."""
        ref = OutputReference(node="node_0", output="apps", index=0, field="email")
        ok, value, err, _ = executor._resolve_input(ref, list_results)
        assert not ok
        assert "email" in err
        assert "name" in err  # available keys listed

    def test_missing_node_fails(self, executor):
        """Reference to non-existent node → error."""
        ref = OutputReference(node="node_99", output="apps")
        ok, value, err, _ = executor._resolve_input(ref, {})
        assert not ok
        assert "not produced outputs" in err

    def test_missing_output_key_fails(self, executor, list_results):
        """Reference to non-existent output key → error."""
        ref = OutputReference(node="node_0", output="nonexistent")
        ok, value, err, _ = executor._resolve_input(ref, list_results)
        assert not ok
        assert "missing" in err


# ==================================================================
# 3. Compiler $ref Parsing
# ==================================================================


class TestCompilerRefParsing:
    """_parse_inputs handles $ref with index/field."""

    @pytest.fixture
    def cortex(self):
        """Minimal MissionCortex for testing _parse_inputs."""
        from unittest.mock import MagicMock
        from cortex.mission_cortex import MissionCortex
        from execution.registry import SkillRegistry

        return MissionCortex(
            llm_client=MagicMock(),
            registry=SkillRegistry(),
            location_config=None,
        )

    def test_ref_without_index_field(self, cortex):
        """Existing $ref format (no index/field) still works."""
        inputs = {
            "data": {"$ref": {"node": 0, "output": "results"}}
        }
        id_map = {0: "node_0"}
        parsed = cortex._parse_inputs(inputs, id_map)
        ref = parsed["data"]
        assert isinstance(ref, OutputReference)
        assert ref.node == "node_0"
        assert ref.output == "results"
        assert ref.index is None
        assert ref.field is None

    def test_ref_with_index_and_field(self, cortex):
        """$ref with index and field parsed correctly."""
        inputs = {
            "app_name": {
                "$ref": {
                    "node": 0,
                    "output": "apps",
                    "index": 1,
                    "field": "name",
                }
            }
        }
        id_map = {0: "node_0"}
        parsed = cortex._parse_inputs(inputs, id_map)
        ref = parsed["app_name"]
        assert isinstance(ref, OutputReference)
        assert ref.index == 1
        assert ref.field == "name"

    def test_literal_values_unchanged(self, cortex):
        """Non-$ref values pass through unchanged."""
        inputs = {"name": "notepad", "count": 5}
        parsed = cortex._parse_inputs(inputs, None)
        assert parsed["name"] == "notepad"
        assert parsed["count"] == 5
