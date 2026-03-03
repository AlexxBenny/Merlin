# tests/test_prompt_architecture.py

"""
Tests for the prompt architecture components:

1. Semantic type registry — completeness, direction enforcement, assertion
2. Anchor validation — reject invalid, pass valid
3. Enriched manifest — semantic types exposed, type docs generated
4. Few-shot examples — present in prompt
5. Self-check instruction — present in prompt
"""

import json
import pytest

from cortex.semantic_types import (
    SEMANTIC_TYPES,
    SemanticType,
    assert_types_registered,
)
from cortex.normalizer import validate_anchors


# ==================================================================
# 1. Semantic Type Registry
# ==================================================================


class TestSemanticTypeRegistry:
    """SEMANTIC_TYPES must be complete and internally consistent."""

    def test_all_production_input_types_registered(self):
        """Every input type used by production skills must be in the registry."""
        input_types = {
            "anchor_name", "relative_path", "folder_name",
            "application_name", "cli_arguments",
            "volume_percentage", "brightness_percentage",
        }
        for t in input_types:
            assert t in SEMANTIC_TYPES, f"Missing input type: {t}"
            entry = SEMANTIC_TYPES[t]
            assert entry.direction in ("input", "both"), (
                f"Type '{t}' is used as input but has direction={entry.direction}"
            )

    def test_all_production_output_types_registered(self):
        """Every output type used by production skills must be in the registry."""
        output_types = {
            "filesystem_path", "process_id",
            "actual_volume", "actual_brightness",
            "mute_state", "nightlight_state",
            "media_key_sent", "whether_playback_state_was_changed",
            "application_list", "application_name",
        }
        for t in output_types:
            assert t in SEMANTIC_TYPES, f"Missing output type: {t}"
            entry = SEMANTIC_TYPES[t]
            assert entry.direction in ("output", "both"), (
                f"Type '{t}' is used as output but has direction={entry.direction}"
            )

    def test_every_type_has_nonempty_description(self):
        for name, entry in SEMANTIC_TYPES.items():
            assert entry.description, f"Type '{name}' has empty description"

    def test_invalid_direction_rejected(self):
        with pytest.raises(ValueError, match="direction"):
            SemanticType("desc", direction="sideways")


# ==================================================================
# 2. assert_types_registered
# ==================================================================


class TestAssertTypesRegistered:
    """Fail-fast on unregistered or direction-violated types."""

    def test_valid_types_pass(self):
        """Should not raise for valid types."""
        assert_types_registered(
            "test.skill",
            {"name": "folder_name", "anchor": "anchor_name"},
            {"created": "filesystem_path"},
        )

    def test_unregistered_input_fails(self):
        with pytest.raises(ValueError, match="unregistered.*bogus_type"):
            assert_types_registered(
                "test.bad",
                {"x": "bogus_type"},
                {},
            )

    def test_unregistered_output_fails(self):
        with pytest.raises(ValueError, match="unregistered.*made_up"):
            assert_types_registered(
                "test.bad",
                {},
                {"y": "made_up"},
            )

    def test_output_only_type_as_input_fails(self):
        """filesystem_path is output-only — using as input should fail."""
        with pytest.raises(ValueError, match="output-only.*filesystem_path"):
            assert_types_registered(
                "test.bad",
                {"path": "filesystem_path"},
                {},
            )

    def test_input_only_type_as_output_fails(self):
        """anchor_name is input-only — using as output should fail."""
        with pytest.raises(ValueError, match="input-only.*anchor_name"):
            assert_types_registered(
                "test.bad",
                {},
                {"loc": "anchor_name"},
            )

    def test_empty_contracts_pass(self):
        """Skills with no inputs/outputs should pass."""
        assert_types_registered("test.empty", {}, {})

    def test_both_direction_works_in_either_position(self):
        """application_name is direction='both' — valid in inputs and outputs."""
        assert_types_registered(
            "test.app",
            {"app_name": "application_name"},
            {"focused": "application_name"},
        )


# ==================================================================
# 3. Anchor Validation (normalizer)
# ==================================================================


class TestAnchorValidation:
    """validate_anchors rejects invalid anchors, passes valid ones."""

    VALID = {"WORKSPACE", "DESKTOP", "DOCUMENTS", "DOWNLOADS"}

    def test_valid_anchor_passes(self):
        payload = {"nodes": [
            {"id": "1", "inputs": {"anchor": "WORKSPACE", "name": "X"}},
        ]}
        validate_anchors(payload, self.VALID)  # should not raise

    def test_no_anchor_passes(self):
        payload = {"nodes": [
            {"id": "1", "inputs": {"name": "X"}},
        ]}
        validate_anchors(payload, self.VALID)

    def test_invalid_anchor_raises(self):
        payload = {"nodes": [
            {"id": "1", "inputs": {"anchor": "D:\\ALEX\\X", "name": "Z"}},
        ]}
        with pytest.raises(TypeError, match="invalid anchor.*D:\\\\ALEX\\\\X"):
            validate_anchors(payload, self.VALID)

    def test_unknown_name_raises(self):
        payload = {"nodes": [
            {"id": "1", "inputs": {"anchor": "MY_CUSTOM_PATH", "name": "Z"}},
        ]}
        with pytest.raises(TypeError, match="invalid anchor.*MY_CUSTOM_PATH"):
            validate_anchors(payload, self.VALID)

    def test_empty_anchors_skips_validation(self):
        """If no valid anchors configured, skip validation."""
        payload = {"nodes": [
            {"id": "1", "inputs": {"anchor": "ANYTHING"}},
        ]}
        validate_anchors(payload, set())  # should not raise

    def test_multiple_nodes_first_invalid_caught(self):
        payload = {"nodes": [
            {"id": "1", "inputs": {"anchor": "WORKSPACE", "name": "X"}},
            {"id": "2", "inputs": {"anchor": "C:\\bad\\path", "name": "Y"}},
        ]}
        with pytest.raises(TypeError, match="Node '2'"):
            validate_anchors(payload, self.VALID)

    def test_no_nodes_key_passes(self):
        validate_anchors({}, self.VALID)

    def test_empty_nodes_passes(self):
        validate_anchors({"nodes": []}, self.VALID)


# ==================================================================
# 4. Enriched Manifest & Type Docs
# ==================================================================


class TestEnrichedManifest:
    """_build_skill_manifest and _build_semantic_type_docs produce correct shapes."""

    def test_manifest_exposes_semantic_types(self):
        """Manifest must have inputs as dict of {key: semantic_type}."""
        from unittest.mock import MagicMock
        from cortex.mission_cortex import MissionCortex
        from skills.contract import SkillContract, FailurePolicy
        from ir.mission import ExecutionMode
        from execution.registry import SkillRegistry
        from skills.base import Skill
        from skills.skill_result import SkillResult

        class DummySkill(Skill):
            contract = SkillContract(
                name="fs.create_folder",
                description="Create folder",
                inputs={"name": "folder_name", "anchor": "anchor_name"},
                outputs={"created": "filesystem_path"},
                allowed_modes={ExecutionMode.foreground},
                failure_policy={ExecutionMode.foreground: FailurePolicy.FAIL},
            )
            def execute(self, inputs, world, snapshot=None):
                return SkillResult(outputs={"created": "/tmp/x"})

        registry = SkillRegistry()
        registry.register(DummySkill())

        cortex = MissionCortex(
            llm_client=MagicMock(),
            registry=registry,
            location_config=None,
        )
        manifest = cortex._build_skill_manifest()

        entry = manifest["fs.create_folder"]
        assert "inputs" in entry
        assert entry["inputs"] == {"name": "folder_name", "anchor": "anchor_name"}
        assert "input_keys" not in entry

    def test_type_docs_only_include_input_types(self):
        """_build_semantic_type_docs should only document input/both types."""
        from cortex.mission_cortex import MissionCortex

        manifest = {
            "fs.create_folder": {
                "inputs": {"name": "folder_name", "anchor": "anchor_name"},
            },
        }
        cortex = MissionCortex.__new__(MissionCortex)
        docs = cortex._build_semantic_type_docs(manifest)

        assert "anchor_name" in docs
        assert "folder_name" in docs
        # Output-only types should NOT appear
        assert "filesystem_path" not in docs
        assert "process_id" not in docs

    def test_type_docs_empty_for_no_inputs(self):
        from cortex.mission_cortex import MissionCortex

        manifest = {
            "system.mute": {"inputs": {}},
        }
        cortex = MissionCortex.__new__(MissionCortex)
        docs = cortex._build_semantic_type_docs(manifest)
        assert docs == ""


# ==================================================================
# 5. Prompt Content
# ==================================================================


class TestPromptContent:
    """Prompt must include examples, self-check, and type docs."""

    def _make_prompt(self):
        from unittest.mock import MagicMock
        from cortex.mission_cortex import MissionCortex
        from execution.registry import SkillRegistry

        cortex = MissionCortex(
            llm_client=MagicMock(),
            registry=SkillRegistry(),
            location_config=None,
        )
        manifest = {
            "fs.create_folder": {
                "description": "Create folder",
                "inputs": {"name": "folder_name", "anchor": "anchor_name"},
                "outputs": {"created": "filesystem_path"},
                "allowed_modes": ["foreground"],
            },
        }
        return cortex._build_prompt(
            user_query="test query",
            skill_manifest=manifest,
            world_state_schema={},
        )

    def test_prompt_contains_few_shot_examples(self):
        prompt = self._make_prompt()
        assert "Examples:" in prompt
        assert '"anchor": "DESKTOP"' in prompt
        assert '"anchor": "WORKSPACE"' in prompt

    def test_prompt_contains_nested_example(self):
        """Must include the 3-level nesting example (A → B → C)."""
        prompt = self._make_prompt()
        assert '"parent": "A"' in prompt
        assert '"parent": "A/B"' in prompt

    def test_prompt_contains_self_check(self):
        prompt = self._make_prompt()
        assert "Before producing JSON, verify:" in prompt
        assert "symbolic names" in prompt
        assert "No raw filesystem paths" in prompt

    def test_prompt_contains_type_docs(self):
        prompt = self._make_prompt()
        assert "Semantic Input Types:" in prompt
        assert "anchor_name:" in prompt
        assert "folder_name:" in prompt

    def test_prompt_contains_cross_domain_example(self):
        prompt = self._make_prompt()
        assert "system.open_app" in prompt
        assert "notepad" in prompt

    def test_prompt_contains_ref_example(self):
        """Compiler prompt must include $ref example with index/field."""
        prompt = self._make_prompt()
        assert '"$ref"' in prompt
        assert '"index"' in prompt
        assert '"field"' in prompt
        assert "system.list_apps" in prompt

    def test_prompt_documents_ref_negative_examples(self):
        """Compiler prompt must forbid string-path $ref syntax."""
        prompt = self._make_prompt()
        assert 'Do NOT use string paths' in prompt


    def test_prompt_contains_clarification_rules(self):
        """Compiler prompt must include clarification precedence + intent replacement rules."""
        prompt = self._make_prompt()
        assert "CLARIFICATION RULES:" in prompt
        assert "overrides any conflicting parameters" in prompt
        assert "discard earlier conflicting parameter values" in prompt


# ==================================================================
# 6. Decomposer Prompt Content
# ==================================================================


class TestDecomposerPromptContent:
    """Decomposer prompt must include NL dependency and ordinal rules."""

    def _make_decomposer_prompt(self):
        from unittest.mock import MagicMock
        from cortex.mission_cortex import MissionCortex
        from execution.registry import SkillRegistry

        cortex = MissionCortex(
            llm_client=MagicMock(),
            registry=SkillRegistry(),
            location_config=None,
        )
        # Access the prompt by calling decompose_intents internals
        # The prompt is built inside decompose_intents, so we check
        # the prompt template string in the source
        import inspect
        source = inspect.getsource(cortex.decompose_intents)
        return source

    def test_decomposer_has_nl_dependency_rule(self):
        """Decomposer must teach NL dependency expression."""
        src = self._make_decomposer_prompt()
        assert "depends on the OUTPUT of a previous action" in src

    def test_decomposer_has_ordinal_scope(self):
        """Decomposer must constrain ordinals to numeric only."""
        src = self._make_decomposer_prompt()
        assert "numeric ordinals" in src.lower() or "Ambiguous ordinals" in src

    def test_decomposer_rejects_predicate_selection(self):
        """Decomposer must reject attribute-based selection."""
        src = self._make_decomposer_prompt()
        assert "most memory" in src or "Selection by attribute" in src


