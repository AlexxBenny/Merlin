# tests/test_cortex_context.py

"""
Tests for Phase 1C: Context Flow into Cortex.

Validates:
- Prompt contains summaries, not raw list payloads
- Context omitted cleanly when no history exists
- summarize_visible_lists produces correct output
- MAX_CONTEXT_TURNS bounds prompt growth
- Only last outcome injected
"""

import pytest

from conversation.frame import ConversationFrame
from conversation.outcome import MissionOutcome
from cortex.mission_cortex import MissionCortex
from execution.registry import SkillRegistry


class TestBuildContextSection:
    """Validate _build_context_section behavior."""

    def _make_cortex(self):
        """Create a MissionCortex for testing context methods."""
        registry = SkillRegistry()
        return MissionCortex(
            llm_client=None,  # not needed for context tests
            registry=registry,
        )

    def test_none_conversation_returns_empty(self):
        cortex = self._make_cortex()
        result = cortex._build_context_section(None)
        assert result == ""

    def test_empty_conversation_returns_empty(self):
        cortex = self._make_cortex()
        frame = ConversationFrame()
        result = cortex._build_context_section(frame)
        assert result == ""

    def test_active_domain_included(self):
        cortex = self._make_cortex()
        frame = ConversationFrame(active_domain="filesystem")
        result = cortex._build_context_section(frame)
        assert "domain: filesystem" in result

    def test_active_entity_included(self):
        cortex = self._make_cortex()
        frame = ConversationFrame(active_entity="folder 'hello'")
        result = cortex._build_context_section(frame)
        assert "entity: folder 'hello'" in result

    def test_history_summarized_in_context(self):
        cortex = self._make_cortex()
        frame = ConversationFrame()
        frame.append_turn("user", "create a folder named hello")
        frame.append_turn("assistant", "Done. (fs.create_folder)")

        result = cortex._build_context_section(frame)
        assert "User: create a folder" in result
        assert "Assistant: Done." in result

    def test_history_bounded_by_max_turns(self):
        cortex = self._make_cortex()
        frame = ConversationFrame()
        # Add more turns than MAX_CONTEXT_TURNS
        for i in range(10):
            frame.append_turn("user", f"message_{i}")

        result = cortex._build_context_section(frame)
        # Should only contain last MAX_CONTEXT_TURNS messages
        assert "message_0" not in result
        assert "message_4" not in result
        assert "message_5" in result
        assert "message_9" in result

    def test_long_text_truncated(self):
        cortex = self._make_cortex()
        frame = ConversationFrame()
        long_text = "x" * 500
        frame.append_turn("user", long_text)

        result = cortex._build_context_section(frame)
        # Should be truncated with ...
        assert "..." in result
        # Should not contain the full 500-char text
        assert "x" * 500 not in result

    def test_only_last_outcome_injected(self):
        cortex = self._make_cortex()
        frame = ConversationFrame()

        # Add two outcomes
        outcome1 = MissionOutcome(
            mission_id="m_1",
            nodes_executed=["n_0"],
            nodes_skipped=[],
            visible_lists={"n_0.results": [{"title": "old_result"}]},
        )
        outcome2 = MissionOutcome(
            mission_id="m_2",
            nodes_executed=["n_0"],
            nodes_skipped=[],
            visible_lists={"n_0.results": [{"title": "new_result"}]},
        )
        frame.outcomes.extend([outcome1, outcome2])

        result = cortex._build_context_section(frame)
        # Should refer to second outcome, not first
        assert "new_result" in result
        assert "old_result" not in result

    def test_no_raw_lists_in_context(self):
        cortex = self._make_cortex()
        frame = ConversationFrame()
        # Outcome with a real list of items
        outcome = MissionOutcome(
            mission_id="m_1",
            nodes_executed=["n_0"],
            nodes_skipped=[],
            visible_lists={
                "n_0.results": [
                    {"title": "video1", "url": "http://example.com/1"},
                    {"title": "video2", "url": "http://example.com/2"},
                    {"title": "video3", "url": "http://example.com/3"},
                ]
            },
        )
        frame.outcomes.append(outcome)

        result = cortex._build_context_section(frame)
        # Summary should exist
        assert "3 items" in result
        # Raw list content should NOT be in the prompt
        assert "http://example.com/2" not in result
        assert "http://example.com/3" not in result


class TestSummarizeVisibleLists:
    """Validate summarize_visible_lists behavior."""

    def test_empty_visible_lists(self):
        outcome = MissionOutcome(
            mission_id="m_1",
            nodes_executed=[],
            nodes_skipped=[],
        )
        result = MissionCortex.summarize_visible_lists(outcome)
        assert result == ""

    def test_empty_list_labeled(self):
        outcome = MissionOutcome(
            mission_id="m_1",
            nodes_executed=["n_0"],
            nodes_skipped=[],
            visible_lists={"n_0.results": []},
        )
        result = MissionCortex.summarize_visible_lists(outcome)
        assert "empty list" in result

    def test_dict_items_summarized(self):
        outcome = MissionOutcome(
            mission_id="m_1",
            nodes_executed=["n_0"],
            nodes_skipped=[],
            visible_lists={
                "n_0.results": [
                    {"title": "Python tutorial"},
                    {"title": "JS tutorial"},
                ]
            },
        )
        result = MissionCortex.summarize_visible_lists(outcome)
        assert "2 items" in result
        assert "Python tutorial" in result

    def test_scalar_items_summarized(self):
        outcome = MissionOutcome(
            mission_id="m_1",
            nodes_executed=["n_0"],
            nodes_skipped=[],
            visible_lists={"n_0.files": ["file1.txt", "file2.txt", "file3.txt"]},
        )
        result = MissionCortex.summarize_visible_lists(outcome)
        assert "3 items" in result
        assert "file1.txt" in result
