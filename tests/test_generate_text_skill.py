# tests/test_generate_text_skill.py

"""
Tests for P1a: reasoning.generate_text skill.

Covers:
- Skill contract validation (semantic types, action, domain)
- execute() behavior with mock LLM
- Style parameter flow
- Empty response handling
- Fail-fast guard in load_skills()
- Semantic type registration
"""

import inspect
import unittest
from unittest.mock import MagicMock

from skills.reasoning.generate_text import GenerateTextSkill
from skills.skill_result import SkillResult
from skills.contract import SkillContract
from cortex.semantic_types import SEMANTIC_TYPES, assert_types_registered
from execution.registry import SkillRegistry


class TestGenerateTextContract(unittest.TestCase):
    """Contract metadata validation."""

    def test_contract_is_skill_contract(self):
        assert isinstance(GenerateTextSkill.contract, SkillContract)

    def test_name_follows_convention(self):
        assert GenerateTextSkill.contract.name == "reasoning.generate_text"
        assert GenerateTextSkill.contract.action == "generate_text"
        assert GenerateTextSkill.contract.domain == "reasoning"

    def test_semantic_types_registered(self):
        """All types in contract must exist in SEMANTIC_TYPES."""
        contract = GenerateTextSkill.contract
        # Should not raise
        assert_types_registered(
            contract.name,
            {**contract.inputs, **contract.optional_inputs},
            contract.outputs,
        )

    def test_text_prompt_type_exists(self):
        assert "text_prompt" in SEMANTIC_TYPES
        assert SEMANTIC_TYPES["text_prompt"].direction == "input"

    def test_generated_text_type_exists(self):
        assert "generated_text" in SEMANTIC_TYPES
        assert SEMANTIC_TYPES["generated_text"].direction == "output"

    def test_does_not_mutate_world(self):
        assert GenerateTextSkill.contract.mutates_world is False

    def test_is_idempotent(self):
        assert GenerateTextSkill.contract.idempotent is True

    def test_intent_verbs_not_empty(self):
        assert len(GenerateTextSkill.contract.intent_verbs) > 0

    def test_intent_keywords_not_empty(self):
        assert len(GenerateTextSkill.contract.intent_keywords) > 0


class TestGenerateTextExecution(unittest.TestCase):
    """Execute behavior with mock LLM."""

    def _make_skill(self, llm_response="Once upon a time..."):
        llm = MagicMock()
        llm.complete.return_value = llm_response
        skill = GenerateTextSkill(content_llm=llm)
        return skill, llm

    def test_basic_generation(self):
        skill, llm = self._make_skill("The sky is blue.")
        result = skill.execute(
            inputs={"prompt": "describe the sky"},
            world=MagicMock(),
        )
        assert isinstance(result, SkillResult)
        assert result.outputs["text"] == "The sky is blue."
        llm.complete.assert_called_once()

    def test_prompt_flows_to_llm(self):
        skill, llm = self._make_skill("result")
        skill.execute(
            inputs={"prompt": "a poem about rain"},
            world=MagicMock(),
        )
        call_args = llm.complete.call_args[0][0]
        assert "a poem about rain" in call_args

    def test_style_modifies_prompt(self):
        skill, llm = self._make_skill("result")
        skill.execute(
            inputs={"prompt": "a story", "style": "formal"},
            world=MagicMock(),
        )
        call_args = llm.complete.call_args[0][0]
        assert "formal" in call_args

    def test_no_style_still_works(self):
        skill, llm = self._make_skill("result")
        skill.execute(
            inputs={"prompt": "hello"},
            world=MagicMock(),
        )
        # Should not raise, should not contain style instruction
        call_args = llm.complete.call_args[0][0]
        assert "style" not in call_args.lower() or "Write in a" not in call_args

    def test_output_is_stripped(self):
        skill, _ = self._make_skill("  padded text  \n")
        result = skill.execute(
            inputs={"prompt": "test"},
            world=MagicMock(),
        )
        assert result.outputs["text"] == "padded text"

    def test_empty_response_raises(self):
        skill, _ = self._make_skill("")
        with self.assertRaises(RuntimeError):
            skill.execute(
                inputs={"prompt": "test"},
                world=MagicMock(),
            )

    def test_whitespace_only_response_raises(self):
        skill, _ = self._make_skill("   \n  ")
        with self.assertRaises(RuntimeError):
            skill.execute(
                inputs={"prompt": "test"},
                world=MagicMock(),
            )

    def test_metadata_has_domain(self):
        skill, _ = self._make_skill("content")
        result = skill.execute(
            inputs={"prompt": "test"},
            world=MagicMock(),
        )
        assert result.metadata["domain"] == "reasoning"

    def test_no_special_entity_metadata(self):
        """Must NOT have a special entity key — orchestrator handles via node_id."""
        skill, _ = self._make_skill("content")
        result = skill.execute(
            inputs={"prompt": "test"},
            world=MagicMock(),
        )
        assert "entity" not in result.metadata


class TestGenerateTextRegistry(unittest.TestCase):
    """Registration and namespace audit."""

    def test_skill_passes_namespace_audit(self):
        """Skill must pass action namespace governance rules."""
        llm = MagicMock()
        skill = GenerateTextSkill(content_llm=llm)
        registry = SkillRegistry()
        # Should not raise
        registry.register(skill)
        assert "reasoning.generate_text" in registry.all_names()

    def test_action_uniqueness(self):
        """Cannot register two skills with the same action."""
        llm = MagicMock()
        skill1 = GenerateTextSkill(content_llm=llm)
        skill2 = GenerateTextSkill(content_llm=llm)
        registry = SkillRegistry()
        registry.register(skill1)
        with self.assertRaises(ValueError):
            registry.register(skill2)


class TestLoadSkillsFailFast(unittest.TestCase):
    """Fail-fast guard for None dependencies."""

    def test_skill_skipped_when_dep_is_none(self):
        """load_skills must skip skill when required dep is None."""
        from main import load_skills

        registry = SkillRegistry()
        config = {
            "skills": [{
                "name": "reasoning.generate_text",
                "module": "skills.reasoning.generate_text",
                "class": "GenerateTextSkill",
            }]
        }
        # content_llm is None — skill should be skipped
        load_skills(registry, config, deps={"content_llm": None})
        assert "reasoning.generate_text" not in registry.all_names()

    def test_skill_loaded_when_dep_available(self):
        """load_skills registers skill when dep is available."""
        from main import load_skills

        registry = SkillRegistry()
        config = {
            "skills": [{
                "name": "reasoning.generate_text",
                "module": "skills.reasoning.generate_text",
                "class": "GenerateTextSkill",
            }]
        }
        llm = MagicMock()
        load_skills(registry, config, deps={"content_llm": llm})
        assert "reasoning.generate_text" in registry.all_names()


if __name__ == "__main__":
    unittest.main()
