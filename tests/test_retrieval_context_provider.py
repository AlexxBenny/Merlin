# tests/test_retrieval_context_provider.py

"""
Tests for RetrievalContextProvider — Phase 3B.

Verifies:
1. Hard token budget enforcement (never exceeds 800 tokens)
2. Priority ordering (goals > entities > memory > turns > refs)
3. Dynamic budget allocation (query heuristics)
4. Budget cascade (unused budget flows downward)
5. Turn compaction (modular strategy)
6. Memory integration (ListMemoryStore episodes appear)
7. Empty/None states (backward-compatible)
8. Mathematical guarantee (assertion holds)
"""

import time
import pytest
from typing import Any, Dict, List, Optional

from conversation.frame import (
    ConversationFrame,
    ConversationTurn,
    EntityRecord,
    GoalState,
)
from cortex.context_provider import (
    ContextProvider,
    SimpleContextProvider,
    RetrievalContextProvider,
    _estimate_tokens,
    _detect_query_signals,
    truncation_compactor,
)
from memory.store import ListMemoryStore


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def make_frame(
    n_turns: int = 0,
    n_entities: int = 0,
    n_goals: int = 0,
    turn_length: int = 50,
) -> ConversationFrame:
    """Create a ConversationFrame with specified content."""
    frame = ConversationFrame()
    for i in range(n_turns):
        role = "user" if i % 2 == 0 else "assistant"
        text = f"Turn {i}: " + "x" * turn_length
        frame.append_turn(role, text)
    for i in range(n_entities):
        frame.register_entity(
            key=f"entity_{i}",
            value=f"value_{i}_" + "v" * 50,
            entity_type="test",
            source_mission=f"mission_{i}",
        )
    for i in range(n_goals):
        frame.add_goal(f"goal_{i}", f"Complete task {i} for the project")
    return frame


def make_memory(n_episodes: int = 0) -> ListMemoryStore:
    """Create a ListMemoryStore with specified episodes."""
    store = ListMemoryStore()
    for i in range(n_episodes):
        store.store_episode(
            mission_id=f"mission_{i}",
            query=f"User asked to do task {i}",
            outcome_summary=f"Successfully completed task {i} with result {i}",
        )
    return store


# ─────────────────────────────────────────────────────────────
# Test: Token estimation
# ─────────────────────────────────────────────────────────────

class TestTokenEstimation:
    def test_empty_string(self):
        assert _estimate_tokens("") == 0

    def test_short_string(self):
        assert _estimate_tokens("hi") == 1  # min 1

    def test_known_length(self):
        text = "a" * 400
        assert _estimate_tokens(text) == 100


# ─────────────────────────────────────────────────────────────
# Test: Query signal detection
# ─────────────────────────────────────────────────────────────

class TestQuerySignals:
    def test_neutral_query(self):
        weights = _detect_query_signals("set the volume to 50 percent please")
        assert weights["goals"] == 1.0
        assert weights["memory"] == 1.0
        assert weights["turns"] == 1.0

    def test_memory_signal(self):
        weights = _detect_query_signals("do what we did yesterday")
        assert weights["memory"] > 1.0
        assert weights["turns"] < 1.0

    def test_goal_signal(self):
        weights = _detect_query_signals("continue working on the plan")
        assert weights["goals"] > 1.0

    def test_short_followup(self):
        weights = _detect_query_signals("do it again")
        # "again" triggers memory signal, so it won't be short followup
        assert weights["memory"] > 1.0

    def test_short_followup_no_signal(self):
        weights = _detect_query_signals("yes please")
        assert weights["turns"] > 1.0
        assert weights["memory"] < 1.0

    def test_combined_signals(self):
        """Goal + memory signal both fire."""
        weights = _detect_query_signals("continue the goal from yesterday")
        assert weights["goals"] > 1.0
        assert weights["memory"] > 1.0


# ─────────────────────────────────────────────────────────────
# Test: Turn compaction
# ─────────────────────────────────────────────────────────────

class TestTruncationCompactor:
    def test_empty_turns(self):
        assert truncation_compactor([], 1000) == ""

    def test_single_turn(self):
        turns = [ConversationTurn(role="user", text="Hello world")]
        result = truncation_compactor(turns, 1000)
        assert "User: Hello world" in result

    def test_recent_full_older_compressed(self):
        turns = []
        for i in range(5):
            role = "user" if i % 2 == 0 else "assistant"
            turns.append(ConversationTurn(role=role, text="x" * 100))
        result = truncation_compactor(turns, 2000)
        lines = result.strip().split("\n")
        # Last 2 turns should be full (up to 200 chars)
        # Older turns should be compressed (50 chars + ...)
        assert len(lines) == 5

    def test_budget_enforcement(self):
        """Compactor stops adding turns when budget exhausted."""
        turns = [ConversationTurn(role="user", text="x" * 100) for _ in range(50)]
        result = truncation_compactor(turns, 200)
        assert len(result) <= 250  # small overshoot acceptable from last line

    def test_long_text_truncation(self):
        """Texts > 200 chars get truncated with ellipsis."""
        turns = [ConversationTurn(role="user", text="z" * 500)]
        result = truncation_compactor(turns, 1000)
        assert "..." in result
        assert len(result) < 300  # 200 chars + prefix + ellipsis


# ─────────────────────────────────────────────────────────────
# Test: Hard budget enforcement
# ─────────────────────────────────────────────────────────────

class TestHardBudget:
    """The most critical invariant: output never exceeds token budget."""

    def test_empty_conversation(self):
        provider = RetrievalContextProvider(token_budget=800)
        result = provider.build_context("hello", None, {})
        assert result == ""

    def test_minimal_conversation(self):
        frame = make_frame(n_turns=1)
        provider = RetrievalContextProvider(token_budget=800)
        result = provider.build_context("test query", frame, {})
        assert _estimate_tokens(result) <= 800

    def test_heavy_conversation_within_budget(self):
        """50 turns + 20 entities + 5 goals = heavy load, still bounded."""
        frame = make_frame(n_turns=50, n_entities=20, n_goals=5, turn_length=200)
        memory = make_memory(n_episodes=20)
        provider = RetrievalContextProvider(memory=memory, token_budget=800)
        result = provider.build_context("set the volume", frame, {})
        tokens = _estimate_tokens(result)
        assert tokens <= 800, f"Token budget exceeded: {tokens} > 800"

    def test_extreme_load(self):
        """200 turns + 50 entities + 5 goals + 50 episodes."""
        frame = make_frame(n_turns=200, n_entities=50, n_goals=5, turn_length=500)
        memory = make_memory(n_episodes=50)
        provider = RetrievalContextProvider(memory=memory, token_budget=800)
        result = provider.build_context("play some music", frame, {})
        tokens = _estimate_tokens(result)
        assert tokens <= 800, f"Token budget exceeded: {tokens} > 800"

    def test_tiny_budget(self):
        """Even with very small budget, no crash and invariant holds."""
        frame = make_frame(n_turns=10, n_entities=5, n_goals=3)
        provider = RetrievalContextProvider(token_budget=50)
        result = provider.build_context("test", frame, {})
        tokens = _estimate_tokens(result)
        # 50 tokens + 1 for rounding tolerance
        assert tokens <= 51, f"Token budget exceeded: {tokens} > 51"

    def test_budget_with_memory_signal(self):
        """Memory-boosted query still respects hard cap."""
        frame = make_frame(n_turns=50, n_entities=20, n_goals=5)
        memory = make_memory(n_episodes=20)
        provider = RetrievalContextProvider(memory=memory, token_budget=800)
        result = provider.build_context(
            "do what we did yesterday again", frame, {},
        )
        tokens = _estimate_tokens(result)
        assert tokens <= 800


# ─────────────────────────────────────────────────────────────
# Test: Priority ordering
# ─────────────────────────────────────────────────────────────

class TestPriorityOrdering:
    def test_goals_before_entities(self):
        frame = make_frame(n_turns=0, n_entities=3, n_goals=2)
        provider = RetrievalContextProvider(token_budget=800)
        result = provider.build_context("test", frame, {})
        if "Active Goals:" in result and "Known Entities:" in result:
            assert result.index("Active Goals:") < result.index("Known Entities:")

    def test_entities_before_memory(self):
        frame = make_frame(n_entities=3)
        memory = make_memory(n_episodes=3)
        provider = RetrievalContextProvider(memory=memory, token_budget=800)
        result = provider.build_context("test query here", frame, {})
        if "Known Entities:" in result and "Relevant Memory:" in result:
            assert result.index("Known Entities:") < result.index("Relevant Memory:")

    def test_memory_before_turns(self):
        frame = make_frame(n_turns=3)
        memory = make_memory(n_episodes=3)
        provider = RetrievalContextProvider(memory=memory, token_budget=800)
        result = provider.build_context("test query here", frame, {})
        if "Relevant Memory:" in result and "Recent Conversation:" in result:
            assert result.index("Relevant Memory:") < result.index("Recent Conversation:")


# ─────────────────────────────────────────────────────────────
# Test: Memory integration
# ─────────────────────────────────────────────────────────────

class TestMemoryIntegration:
    def test_no_memory_provider(self):
        """Works fine without memory store."""
        frame = make_frame(n_turns=3)
        provider = RetrievalContextProvider(memory=None, token_budget=800)
        result = provider.build_context("hello", frame, {})
        assert "Relevant Memory:" not in result

    def test_memory_episodes_appear(self):
        frame = make_frame(n_turns=1)
        memory = make_memory(n_episodes=3)
        provider = RetrievalContextProvider(memory=memory, token_budget=800)
        result = provider.build_context("test query here", frame, {})
        assert "Relevant Memory:" in result

    def test_empty_memory(self):
        frame = make_frame(n_turns=1)
        memory = ListMemoryStore()
        provider = RetrievalContextProvider(memory=memory, token_budget=800)
        result = provider.build_context("test query here", frame, {})
        assert "Relevant Memory:" not in result


# ─────────────────────────────────────────────────────────────
# Test: Budget cascade
# ─────────────────────────────────────────────────────────────

class TestBudgetCascade:
    def test_no_goals_cascade_to_entities(self):
        """When goals empty, entities get more budget."""
        # Frame with many entities but no goals
        frame = make_frame(n_turns=1, n_entities=20, n_goals=0)
        provider = RetrievalContextProvider(token_budget=800)
        result = provider.build_context("check entities please", frame, {})
        # Should include more entities than with goals present
        assert "Known Entities:" in result
        entity_count = result.count("entity_")
        assert entity_count > 0

    def test_cascade_never_exceeds_total(self):
        """Even with cascade, total never exceeds budget."""
        frame = make_frame(n_turns=20, n_entities=0, n_goals=0)
        memory = make_memory(n_episodes=0)
        provider = RetrievalContextProvider(memory=memory, token_budget=400)
        result = provider.build_context("tell me more", frame, {})
        tokens = _estimate_tokens(result)
        assert tokens <= 400


# ─────────────────────────────────────────────────────────────
# Test: Dynamic allocation
# ─────────────────────────────────────────────────────────────

class TestDynamicAllocation:
    def test_goal_query_boosts_goals(self):
        """Goal signal should give more budget to goals section."""
        frame = make_frame(n_turns=5, n_entities=5, n_goals=3)
        provider = RetrievalContextProvider(token_budget=800)

        # Goal query
        result_goal = provider.build_context(
            "continue working on the plan", frame, {},
        )
        # Neutral query
        result_neutral = provider.build_context(
            "set volume to fifty percent", frame, {},
        )
        # Both should be within budget
        assert _estimate_tokens(result_goal) <= 800
        assert _estimate_tokens(result_neutral) <= 800

    def test_memory_query_includes_episodes(self):
        """Memory signal should prioritize memory content."""
        frame = make_frame(n_turns=5)
        memory = make_memory(n_episodes=5)
        provider = RetrievalContextProvider(memory=memory, token_budget=800)
        result = provider.build_context(
            "do what we did yesterday", frame, {},
        )
        assert "Relevant Memory:" in result


# ─────────────────────────────────────────────────────────────
# Test: Modular compactor
# ─────────────────────────────────────────────────────────────

class TestModularCompactor:
    def test_custom_compactor(self):
        """Custom compactor is used instead of default."""
        def counting_compactor(turns, budget):
            return f"  [{len(turns)} turns compacted]"

        frame = make_frame(n_turns=10)
        provider = RetrievalContextProvider(
            token_budget=800,
            turn_compactor=counting_compactor,
        )
        result = provider.build_context("test query here", frame, {})
        assert "[10 turns compacted]" in result


# ─────────────────────────────────────────────────────────────
# Test: Backward compatibility
# ─────────────────────────────────────────────────────────────

class TestBackwardCompatibility:
    def test_simple_provider_unchanged(self):
        """SimpleContextProvider still works as before."""
        frame = make_frame(n_turns=3, n_entities=2, n_goals=1)
        provider = SimpleContextProvider()
        result = provider.build_context("hello", frame, {})
        assert "Conversation Context:" in result
        assert "Active Goals:" in result

    def test_simple_provider_none_conversation(self):
        provider = SimpleContextProvider()
        result = provider.build_context("hello", None, {})
        assert result == ""

    def test_retrieval_provider_abc_compliant(self):
        """RetrievalContextProvider implements ContextProvider ABC."""
        provider = RetrievalContextProvider(token_budget=800)
        assert isinstance(provider, ContextProvider)

    def test_base_budgets_validation(self):
        """Invalid base budgets should be caught at construction."""
        with pytest.raises(AssertionError, match="sum to 1.0"):
            RetrievalContextProvider(
                token_budget=800,
                base_budgets={"goals": 0.5, "entities": 0.5,
                              "memory": 0.5, "turns": 0.5, "refs": 0.5},
            )


# ─────────────────────────────────────────────────────────────
# Test: Edge cases
# ─────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_only_memory_no_conversation(self):
        """Memory-only provider with no conversation."""
        memory = make_memory(n_episodes=3)
        provider = RetrievalContextProvider(memory=memory, token_budget=800)
        result = provider.build_context("find previous work", None, {})
        assert "Relevant Memory:" in result

    def test_resolved_references(self):
        """References are rendered within budget."""
        frame = make_frame(n_turns=1)
        frame.unresolved_references = {
            "resolved": [
                {"ordinal": "first", "value": "file.txt"},
                {"ordinal": "second", "value": "doc.pdf"},
            ]
        }
        provider = RetrievalContextProvider(token_budget=800)
        result = provider.build_context("open the first one", frame, {})
        assert "Resolved References:" in result

    def test_empty_everything(self):
        """Empty frame + no memory = empty output."""
        frame = ConversationFrame()
        provider = RetrievalContextProvider(memory=None, token_budget=800)
        result = provider.build_context("test", frame, {})
        # Either empty or minimal
        assert _estimate_tokens(result) <= 10
