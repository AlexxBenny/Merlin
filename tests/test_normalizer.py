# tests/test_normalizer.py

"""
Tests for cortex/normalizer.py — LLM output normalization.

Validates every coercion rule at the LLM → IR trust boundary.
The normalizer must handle every shape a 7B+ model might emit
without letting malformed data reach Pydantic.
"""

import pytest

from cortex.normalizer import (
    normalize_plan,
    normalize_node,
    _coerce_list,
    _coerce_dict,
    _flatten_parent_into_path,
)


# ==================================================================
# Plan-level normalization
# ==================================================================


class TestNormalizePlan:
    """Top-level plan shape normalization."""

    def test_valid_plan_passes_through(self):
        payload = {
            "nodes": [
                {
                    "id": "n1",
                    "skill": "fs.create_folder",
                    "inputs": {"name": "X"},
                    "depends_on": [],
                    "mode": "foreground",
                }
            ]
        }
        result = normalize_plan(payload)
        assert isinstance(result["nodes"], list)
        assert len(result["nodes"]) == 1
        assert result["nodes"][0]["id"] == "n1"

    def test_null_nodes_becomes_empty_list(self):
        result = normalize_plan({"nodes": None})
        assert result["nodes"] == []

    def test_missing_nodes_becomes_empty_list(self):
        result = normalize_plan({})
        assert result["nodes"] == []

    def test_nodes_as_dict_raises_type_error(self):
        with pytest.raises(TypeError, match="must be a list"):
            normalize_plan({"nodes": {"n1": {}}})

    def test_nodes_as_string_raises_type_error(self):
        with pytest.raises(TypeError, match="must be a list"):
            normalize_plan({"nodes": "not a list"})

    def test_empty_nodes_list_passes(self):
        result = normalize_plan({"nodes": []})
        assert result["nodes"] == []


# ==================================================================
# Node-level normalization
# ==================================================================


class TestNormalizeNode:
    """Per-node field coercion and null handling."""

    def test_full_valid_node_unchanged(self):
        """A fully valid node should pass through with no mutation."""
        raw = {
            "id": "n1",
            "skill": "fs.create_folder",
            "inputs": {"name": "X"},
            "outputs": {"created": {"name": "path.v1", "type": "path"}},
            "depends_on": ["n0"],
            "mode": "foreground",
            "condition_on": None,
        }
        result = normalize_node(raw)
        assert result["id"] == "n1"
        assert result["skill"] == "fs.create_folder"
        assert result["inputs"] == {"name": "X"}
        assert result["depends_on"] == ["n0"]
        assert result["mode"] == "foreground"
        assert result["condition_on"] is None

    def test_null_depends_on_becomes_list(self):
        raw = {"id": "n1", "skill": "fs.create_folder", "depends_on": None}
        result = normalize_node(raw)
        assert result["depends_on"] == []

    def test_missing_depends_on_becomes_list(self):
        raw = {"id": "n1", "skill": "fs.create_folder"}
        result = normalize_node(raw)
        assert result["depends_on"] == []

    def test_string_depends_on_becomes_list(self):
        raw = {"id": "n1", "skill": "fs.create_folder", "depends_on": "n0"}
        result = normalize_node(raw)
        assert result["depends_on"] == ["n0"]

    def test_valid_depends_on_passes(self):
        raw = {"id": "n1", "skill": "fs.create_folder", "depends_on": ["n0", "n1"]}
        result = normalize_node(raw)
        assert result["depends_on"] == ["n0", "n1"]

    def test_null_inputs_becomes_dict(self):
        raw = {"id": "n1", "skill": "fs.create_folder", "inputs": None}
        result = normalize_node(raw)
        assert result["inputs"] == {}

    def test_missing_inputs_becomes_dict(self):
        raw = {"id": "n1", "skill": "fs.create_folder"}
        result = normalize_node(raw)
        assert result["inputs"] == {}

    def test_null_outputs_becomes_dict(self):
        raw = {"id": "n1", "skill": "fs.create_folder", "outputs": None}
        result = normalize_node(raw)
        assert result["outputs"] == {}

    def test_null_mode_becomes_foreground(self):
        raw = {"id": "n1", "skill": "fs.create_folder", "mode": None}
        result = normalize_node(raw)
        assert result["mode"] == "foreground"

    def test_missing_mode_becomes_foreground(self):
        raw = {"id": "n1", "skill": "fs.create_folder"}
        result = normalize_node(raw)
        assert result["mode"] == "foreground"

    def test_valid_mode_passes(self):
        raw = {"id": "n1", "skill": "fs.create_folder", "mode": "background"}
        result = normalize_node(raw)
        assert result["mode"] == "background"

    def test_null_id_passes_through(self):
        """Null id is NOT defaulted — Pydantic will reject it downstream."""
        raw = {"id": None, "skill": "fs.create_folder"}
        result = normalize_node(raw)
        assert result["id"] is None

    def test_null_skill_passes_through(self):
        """Null skill is NOT defaulted — Pydantic will reject it downstream."""
        raw = {"id": "n1", "skill": None}
        result = normalize_node(raw)
        assert result["skill"] is None

    def test_condition_on_null_passes(self):
        raw = {"id": "n1", "skill": "fs.create_folder", "condition_on": None}
        result = normalize_node(raw)
        assert result["condition_on"] is None

    def test_condition_on_dict_passes(self):
        raw = {
            "id": "n1",
            "skill": "fs.create_folder",
            "condition_on": {"source": "n0", "equals": True},
        }
        result = normalize_node(raw)
        assert result["condition_on"] == {"source": "n0", "equals": True}

    def test_unknown_keys_preserved(self):
        """Unknown keys pass through — Pydantic's extra='forbid' handles them."""
        raw = {"id": "n1", "skill": "fs.create_folder", "priority": "high"}
        result = normalize_node(raw)
        assert result["priority"] == "high"


# ==================================================================
# Coercion helpers
# ==================================================================


class TestCoerceList:
    """_coerce_list: strict type coercion for list fields."""

    def test_none_returns_empty_list(self):
        assert _coerce_list(None, "test") == []

    def test_list_passes_through(self):
        assert _coerce_list(["a", "b"], "test") == ["a", "b"]

    def test_string_becomes_single_element_list(self):
        assert _coerce_list("node_1", "test") == ["node_1"]

    def test_int_becomes_single_element_list(self):
        """Integer becomes [int] for index-based depends_on."""
        assert _coerce_list(42, "depends_on") == [42]

    def test_dict_raises_type_error(self):
        with pytest.raises(TypeError, match="must be a list"):
            _coerce_list({"a": 1}, "depends_on")

    def test_bool_raises_type_error(self):
        with pytest.raises(TypeError, match="must be a list"):
            _coerce_list(True, "depends_on")


class TestCoerceDict:
    """_coerce_dict: strict type coercion for dict fields."""

    def test_none_returns_empty_dict(self):
        assert _coerce_dict(None, "test") == {}

    def test_dict_passes_through(self):
        assert _coerce_dict({"a": 1}, "test") == {"a": 1}

    def test_list_raises_type_error(self):
        with pytest.raises(TypeError, match="must be a dict"):
            _coerce_dict([1, 2], "inputs")

    def test_string_raises_type_error(self):
        with pytest.raises(TypeError, match="must be a dict"):
            _coerce_dict("not a dict", "inputs")

    def test_int_raises_type_error(self):
        with pytest.raises(TypeError, match="must be a dict"):
            _coerce_dict(42, "outputs")


# ==================================================================
# Parent → path flattening
# ==================================================================


class TestParentPathFlattening:
    """_flatten_parent_into_path: hierarchy flattening for file ops."""

    def test_write_file_parent_folded(self):
        """write_file(path="test.py", parent="alex") → path="alex/test.py"."""
        inputs = {"path": "test.py", "parent": "alex", "content": ""}
        result = _flatten_parent_into_path("fs.write_file", inputs)
        assert result["path"] == "alex/test.py"
        assert "parent" not in result
        assert result["content"] == ""  # other keys untouched

    def test_read_file_parent_folded(self):
        """read_file(path="file.txt", parent="docs/notes") → path="docs/notes/file.txt"."""
        inputs = {"path": "file.txt", "parent": "docs/notes"}
        result = _flatten_parent_into_path("fs.read_file", inputs)
        assert result["path"] == "docs/notes/file.txt"
        assert "parent" not in result

    def test_create_folder_not_folded(self):
        """create_folder uses 'name', not 'path' — parent stays untouched."""
        inputs = {"name": "foo", "parent": "bar", "anchor": "WORKSPACE"}
        result = _flatten_parent_into_path("fs.create_folder", inputs)
        # parent NOT folded because there is no 'path' key
        assert result["parent"] == "bar"
        assert result["name"] == "foo"

    def test_no_parent_passthrough(self):
        """No parent in inputs → nothing changes."""
        inputs = {"path": "test.py", "content": "hello"}
        result = _flatten_parent_into_path("fs.write_file", inputs)
        assert result["path"] == "test.py"
        assert "parent" not in result

    def test_parent_without_path_left_for_validator(self):
        """parent without path → left untouched for validator to reject."""
        inputs = {"parent": "alex", "content": "hello"}
        result = _flatten_parent_into_path("fs.write_file", inputs)
        # parent stays — validator will reject as unexpected input
        assert result["parent"] == "alex"

    def test_trailing_slash_handled(self):
        """PurePosixPath normalizes trailing slashes."""
        inputs = {"path": "test.py", "parent": "alex/"}
        result = _flatten_parent_into_path("fs.write_file", inputs)
        assert result["path"] == "alex/test.py"  # no double slash
        assert "parent" not in result

    def test_deep_nested_parent(self):
        """Multi-level parent path folds correctly."""
        inputs = {"path": "test.py", "parent": "alex/projects/merlin"}
        result = _flatten_parent_into_path("fs.write_file", inputs)
        assert result["path"] == "alex/projects/merlin/test.py"
        assert "parent" not in result


# ==================================================================
# Integration: normalize_plan → normalize_node pipeline
# ==================================================================


class TestNormalizationPipeline:
    """End-to-end normalization of realistic LLM outputs."""

    def test_typical_7b_output_with_nulls(self):
        """Typical 7B model output: correct structure but null optionals."""
        payload = {
            "nodes": [
                {
                    "id": "create_x",
                    "skill": "fs.create_folder",
                    "inputs": {"name": "X", "anchor": "WORKSPACE"},
                    "outputs": None,
                    "depends_on": None,
                    "mode": "foreground",
                    "condition_on": None,
                },
                {
                    "id": "create_y",
                    "skill": "fs.create_folder",
                    "inputs": {"name": "Y", "anchor": "WORKSPACE"},
                    "depends_on": None,
                    "mode": None,
                    "condition_on": None,
                },
            ]
        }
        result = normalize_plan(payload)
        assert len(result["nodes"]) == 2
        assert result["nodes"][0]["depends_on"] == []
        assert result["nodes"][0]["outputs"] == {}
        assert result["nodes"][1]["depends_on"] == []
        assert result["nodes"][1]["mode"] == "foreground"

    def test_minimal_node_all_defaults(self):
        """LLM emits only id and skill — everything else defaults."""
        payload = {
            "nodes": [
                {"id": "n1", "skill": "fs.create_folder"}
            ]
        }
        result = normalize_plan(payload)
        node = result["nodes"][0]
        assert node["inputs"] == {}
        assert node["outputs"] == {}
        assert node["depends_on"] == []
        assert node["mode"] == "foreground"
        assert node["id"] == "n1"
        assert "condition_on" not in node

    def test_id_free_node_normalized(self):
        """New schema: LLM emits no id or condition_on."""
        payload = {
            "nodes": [
                {"skill": "system.unmute", "inputs": {}, "depends_on": [], "mode": "foreground"},
                {"skill": "system.set_brightness", "inputs": {"level": 10}, "depends_on": [], "mode": "foreground"},
            ]
        }
        result = normalize_plan(payload)
        assert len(result["nodes"]) == 2
        assert "id" not in result["nodes"][0]
        assert "id" not in result["nodes"][1]
        assert "condition_on" not in result["nodes"][0]
        assert result["nodes"][0]["skill"] == "system.unmute"
        assert result["nodes"][1]["inputs"] == {"level": 10}

    def test_integer_depends_on_preserved(self):
        """New schema: depends_on uses integer indices."""
        payload = {
            "nodes": [
                {"skill": "fs.create_folder", "inputs": {"name": "A"}, "depends_on": [], "mode": "foreground"},
                {"skill": "fs.create_folder", "inputs": {"name": "B"}, "depends_on": [0], "mode": "foreground"},
            ]
        }
        result = normalize_plan(payload)
        assert result["nodes"][1]["depends_on"] == [0]

    def test_string_depends_on_in_chain(self):
        """LLM emits depends_on as single string instead of list."""
        payload = {
            "nodes": [
                {"id": "n1", "skill": "fs.create_folder"},
                {"id": "n2", "skill": "fs.create_folder", "depends_on": "n1"},
            ]
        }
        result = normalize_plan(payload)
        assert result["nodes"][1]["depends_on"] == ["n1"]

    def test_complex_multi_node_dag(self):
        """Complex query with dependencies, conditions, and outputs."""
        payload = {
            "nodes": [
                {
                    "id": "create_x",
                    "skill": "fs.create_folder",
                    "inputs": {"name": "X", "anchor": "WORKSPACE"},
                    "outputs": {"created": {"name": "path.x.v1", "type": "path"}},
                    "depends_on": [],
                    "mode": "foreground",
                },
                {
                    "id": "create_y",
                    "skill": "fs.create_folder",
                    "inputs": {"name": "Y", "anchor": "WORKSPACE"},
                    "outputs": {"created": {"name": "path.y.v1", "type": "path"}},
                    "depends_on": [],
                    "mode": "foreground",
                },
                {
                    "id": "create_z",
                    "skill": "fs.create_folder",
                    "inputs": {"name": "Z", "anchor": "WORKSPACE", "parent": "X"},
                    "depends_on": ["create_x"],
                    "mode": "foreground",
                    "condition_on": None,
                },
                {
                    "id": "create_test",
                    "skill": "fs.create_file",
                    "inputs": {"name": "test.py", "anchor": "WORKSPACE", "parent": "Y"},
                    "depends_on": ["create_y"],
                    "mode": "foreground",
                },
            ]
        }
        result = normalize_plan(payload)
        assert len(result["nodes"]) == 4
        assert result["nodes"][2]["depends_on"] == ["create_x"]
        assert result["nodes"][3]["depends_on"] == ["create_y"]

    def test_parent_folded_in_pipeline(self):
        """End-to-end: LLM emits parent on write_file → normalize folds it.

        Exact reproduction of the failing query:
        'create two folders named ann and alex. Inside alex create test.py'
        """
        payload = {
            "nodes": [
                {
                    "skill": "fs.create_folder",
                    "inputs": {"name": "ann", "anchor": "WORKSPACE"},
                    "depends_on": [],
                    "mode": "foreground",
                },
                {
                    "skill": "fs.create_folder",
                    "inputs": {"name": "alex", "anchor": "WORKSPACE"},
                    "depends_on": [],
                    "mode": "foreground",
                },
                {
                    "skill": "fs.write_file",
                    "inputs": {"path": "test.py", "parent": "alex", "content": ""},
                    "depends_on": [1],
                    "mode": "foreground",
                },
            ]
        }
        result = normalize_plan(payload)
        write_node = result["nodes"][2]
        # parent folded into path
        assert write_node["inputs"]["path"] == "alex/test.py"
        assert "parent" not in write_node["inputs"]
        # other inputs untouched
        assert write_node["inputs"]["content"] == ""
        # create_folder nodes unchanged — parent stays if present
        assert result["nodes"][0]["inputs"] == {"name": "ann", "anchor": "WORKSPACE"}
