# tests/test_identity_and_memory.py

"""
Tests for identity grounding and memory architecture fixes.

Covers:
1. GENERATION_VERBS: reasoning verbs removed (explain, summarize, describe)
2. Generation-verb guard: embedded question-word detection
3. Generation-verb guard: META speech-act bypass
4. GenerateTextSkill: MERLIN identity in system prompt
5. ReportBuilder: natural narration framing
6. UserKnowledgeStore.retrieve_memory_context()
7. Coordinator prompt: memory block injection
8. Coordinator: _try_knowledge_answer removed
9. Memory skill intent_verbs: semantic correctness
"""

import json
from unittest.mock import MagicMock

import pytest

from cortex.cognitive_coordinator import (
    LLMCognitiveCoordinator,
    CoordinatorMode,
    CoordinatorResult,
    GENERATION_VERBS,
    FALLBACK_RESULT,
)
from memory.user_knowledge import UserKnowledgeStore
from skills.reasoning.generate_text import GenerateTextSkill
from skills.memory.memory_skills import (
    GetPreferenceSkill,
    SetFactSkill,
    SetPreferenceSkill,
)
from brain.structural_classifier import SpeechActType


def _make_coordinator():
    """Create coordinator with mock LLM."""
    llm = MagicMock()
    return LLMCognitiveCoordinator(llm=llm)


def _make_snapshot():
    from world.snapshot import WorldSnapshot
    from world.state import WorldState
    return WorldSnapshot.build(WorldState(), [])


def _make_knowledge_store(**facts):
    """Create a UserKnowledgeStore with pre-populated facts."""
    store = UserKnowledgeStore()
    for key, value in facts.items():
        store.set_fact(key, value)
    return store


# ═══════════════════════════════════════════════════
# 1. GENERATION_VERBS — reasoning verbs removed
# ═══════════════════════════════════════════════════


class TestGenerationVerbsCleanup:
    """Reasoning verbs must NOT be in GENERATION_VERBS."""

    def test_explain_not_in_generation_verbs(self):
        assert "explain" not in GENERATION_VERBS

    def test_summarize_not_in_generation_verbs(self):
        assert "summarize" not in GENERATION_VERBS

    def test_describe_not_in_generation_verbs(self):
        assert "describe" not in GENERATION_VERBS

    def test_content_generation_verbs_present(self):
        """Content generation verbs must still be present."""
        for verb in ("tell", "compose", "draft", "generate", "narrate", "outline"):
            assert verb in GENERATION_VERBS, f"'{verb}' missing from GENERATION_VERBS"


# ═══════════════════════════════════════════════════
# 2. Generation-verb guard: embedded question words
# ═══════════════════════════════════════════════════


class TestEmbeddedQuestionWordDetection:
    """Queries with embedded question words should not trigger generation guard."""

    def _process_with_direct_answer(self, query, speech_act=None):
        """Process a query where the LLM returns DIRECT_ANSWER."""
        coord = _make_coordinator()
        coord._llm.complete.return_value = json.dumps({
            "mode": "DIRECT_ANSWER",
            "answer": "test answer",
            "reasoning": "test",
        })
        snapshot = _make_snapshot()
        return coord.process(
            query=query,
            snapshot=snapshot,
            skill_manifest={},
            speech_act=speech_act,
        )

    def test_tell_me_what_you_can_do_stays_direct(self):
        """Embedded 'what' prevents generation guard from firing."""
        result = self._process_with_direct_answer("tell me what you can do")
        assert result.mode == CoordinatorMode.DIRECT_ANSWER

    def test_tell_me_who_you_are_stays_direct(self):
        """Embedded 'who' prevents generation guard from firing."""
        result = self._process_with_direct_answer("tell me who you are")
        assert result.mode == CoordinatorMode.DIRECT_ANSWER

    def test_tell_me_a_story_gets_overridden(self):
        """No embedded question words → guard fires → SKILL_PLAN."""
        result = self._process_with_direct_answer("tell me a story")
        assert result.mode == CoordinatorMode.SKILL_PLAN

    def test_compose_a_poem_gets_overridden(self):
        """Content generation → guard fires → SKILL_PLAN."""
        result = self._process_with_direct_answer("compose a poem about rain")
        assert result.mode == CoordinatorMode.SKILL_PLAN

    def test_tell_me_how_this_works_stays_direct(self):
        """Embedded 'how' prevents guard."""
        result = self._process_with_direct_answer("tell me how this works")
        assert result.mode == CoordinatorMode.DIRECT_ANSWER


# ═══════════════════════════════════════════════════
# 3. Generation-verb guard: META speech-act bypass
# ═══════════════════════════════════════════════════


class TestMetaSpeechActBypass:
    """META speech act skips the generation guard entirely."""

    def _process_with_direct_answer(self, query, speech_act=None):
        coord = _make_coordinator()
        coord._llm.complete.return_value = json.dumps({
            "mode": "DIRECT_ANSWER",
            "answer": "I am MERLIN",
            "reasoning": "identity question",
        })
        snapshot = _make_snapshot()
        return coord.process(
            query=query,
            snapshot=snapshot,
            skill_manifest={},
            speech_act=speech_act,
        )

    def test_meta_speech_act_bypasses_generation_guard(self):
        """'tell me about yourself' with META → guard skipped."""
        result = self._process_with_direct_answer(
            "tell me about yourself",
            speech_act=SpeechActType.META,
        )
        assert result.mode == CoordinatorMode.DIRECT_ANSWER

    def test_command_speech_act_triggers_generation_guard(self):
        """'tell me a story' with COMMAND → guard fires."""
        result = self._process_with_direct_answer(
            "tell me a story",
            speech_act=SpeechActType.COMMAND,
        )
        assert result.mode == CoordinatorMode.SKILL_PLAN

    def test_none_speech_act_triggers_generation_guard(self):
        """No speech_act → guard fires for generation verbs."""
        result = self._process_with_direct_answer(
            "tell me a story",
            speech_act=None,
        )
        assert result.mode == CoordinatorMode.SKILL_PLAN


# ═══════════════════════════════════════════════════
# 4. GenerateTextSkill: MERLIN identity
# ═══════════════════════════════════════════════════


class TestGenerateTextIdentity:
    """GenerateTextSkill must use MERLIN identity, not generic persona."""

    def test_system_prompt_contains_merlin(self):
        llm = MagicMock()
        llm.complete.return_value = "Generated content."
        skill = GenerateTextSkill(content_llm=llm)
        skill.execute(inputs={"prompt": "test"}, world=MagicMock())
        prompt = llm.complete.call_args[0][0]
        assert "MERLIN" in prompt

    def test_system_prompt_not_generic(self):
        llm = MagicMock()
        llm.complete.return_value = "Generated content."
        skill = GenerateTextSkill(content_llm=llm)
        skill.execute(inputs={"prompt": "test"}, world=MagicMock())
        prompt = llm.complete.call_args[0][0]
        assert "helpful writing assistant" not in prompt


# ═══════════════════════════════════════════════════
# 5. UserKnowledgeStore.retrieve_memory_context()
# ═══════════════════════════════════════════════════


class TestRetrieveMemoryContext:
    """retrieve_memory_context returns structured memory for LLM prompt injection."""

    def test_empty_store_returns_no_stored_knowledge(self):
        store = UserKnowledgeStore()
        ctx = store.retrieve_memory_context("any query")
        assert "no stored knowledge" in ctx

    def test_facts_included(self):
        store = _make_knowledge_store(name="alex", location="mumbai")
        ctx = store.retrieve_memory_context("what is my name")
        assert "fact: name = alex" in ctx
        assert "fact: location = mumbai" in ctx

    def test_preferences_included(self):
        store = UserKnowledgeStore()
        store.set_preference("volume", 80)
        ctx = store.retrieve_memory_context("what is my volume")
        assert "preference: volume = 80" in ctx

    def test_traits_included(self):
        store = UserKnowledgeStore()
        store._traits["occupation"] = MagicMock(value="engineer")
        ctx = store.retrieve_memory_context("what do you know about me")
        assert "trait: occupation = engineer" in ctx

    def test_query_parameter_accepted(self):
        """Query param must be accepted (seam for future semantic retrieval)."""
        store = _make_knowledge_store(name="alex")
        # Should not raise
        ctx = store.retrieve_memory_context("do you know my name")
        assert isinstance(ctx, str)

    def test_max_entries_respected(self):
        store = UserKnowledgeStore()
        for i in range(60):
            store.set_fact(f"key_{i}", f"value_{i}")
        ctx = store.retrieve_memory_context("all", max_entries=10)
        lines = [l for l in ctx.split("\n") if l.strip()]
        assert len(lines) <= 11  # 10 entries + truncation notice


# ═══════════════════════════════════════════════════
# 6. Coordinator prompt: memory block injection
# ═══════════════════════════════════════════════════


class TestCoordinatorMemoryInjection:
    """Coordinator LLM prompt must include user memory block."""

    def test_prompt_contains_memory_section(self):
        coord = _make_coordinator()
        store = _make_knowledge_store(name="alex")
        snapshot = _make_snapshot()
        prompt = coord._build_prompt(
            query="do you know my name",
            snapshot=snapshot,
            skill_manifest={},
            user_knowledge=store,
        )
        assert "USER MEMORY" in prompt
        assert "fact: name = alex" in prompt

    def test_prompt_contains_memory_authority_rule(self):
        coord = _make_coordinator()
        snapshot = _make_snapshot()
        prompt = coord._build_prompt(
            query="any query",
            snapshot=snapshot,
            skill_manifest={},
        )
        assert "MEMORY AUTHORITY" in prompt

    def test_prompt_shows_no_memory_when_store_absent(self):
        coord = _make_coordinator()
        snapshot = _make_snapshot()
        prompt = coord._build_prompt(
            query="test",
            snapshot=snapshot,
            skill_manifest={},
            user_knowledge=None,
        )
        assert "no memory system" in prompt

    def test_prompt_shows_empty_memory_when_store_empty(self):
        coord = _make_coordinator()
        snapshot = _make_snapshot()
        prompt = coord._build_prompt(
            query="test",
            snapshot=snapshot,
            skill_manifest={},
            user_knowledge=UserKnowledgeStore(),
        )
        assert "no stored knowledge" in prompt


# ═══════════════════════════════════════════════════
# 7. _try_knowledge_answer removed
# ═══════════════════════════════════════════════════


class TestTryKnowledgeAnswerRemoved:
    """The regex-based _try_knowledge_answer must no longer exist."""

    def test_method_does_not_exist(self):
        coord = _make_coordinator()
        assert not hasattr(coord, '_try_knowledge_answer')


# ═══════════════════════════════════════════════════
# 8. Memory skill intent_verbs: semantic correctness
# ═══════════════════════════════════════════════════


class TestMemorySkillIntentVerbs:
    """Intent verbs must match retrieval vs storage semantics."""

    def test_know_in_get_preference(self):
        """'know' = retrieval → must be in GetPreferenceSkill."""
        assert "know" in GetPreferenceSkill.contract.intent_verbs

    def test_recall_in_get_preference(self):
        """'recall' = retrieval → must be in GetPreferenceSkill."""
        assert "recall" in GetPreferenceSkill.contract.intent_verbs

    def test_know_not_in_set_fact(self):
        """'know' must NOT be in SetFactSkill (it's retrieval, not storage)."""
        assert "know" not in SetFactSkill.contract.intent_verbs

    def test_memorize_in_set_fact(self):
        """'memorize' = storage → must be in SetFactSkill."""
        assert "memorize" in SetFactSkill.contract.intent_verbs

    def test_remember_in_set_preference(self):
        """'remember' stays in SetPreferenceSkill (declarations caught by speech_act)."""
        assert "remember" in SetPreferenceSkill.contract.intent_verbs
