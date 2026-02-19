# tests/test_conversation_history.py

"""
Tests for Phase 1A: Conversation History.

Validates:
- Turn append order (user before assistant)
- History cap enforcement (oldest turns dropped)
- Role validation (Literal["user", "assistant"] only)
- mission_id linkage
"""

import pytest
from pydantic import ValidationError

from conversation.frame import (
    ConversationFrame,
    ConversationTurn,
    HISTORY_CAP,
)


class TestConversationTurn:
    """Validate ConversationTurn type enforcement."""

    def test_valid_user_turn(self):
        turn = ConversationTurn(role="user", text="hello")
        assert turn.role == "user"
        assert turn.text == "hello"
        assert turn.timestamp > 0
        assert turn.mission_id is None

    def test_valid_assistant_turn(self):
        turn = ConversationTurn(role="assistant", text="hi there")
        assert turn.role == "assistant"

    def test_invalid_role_rejected(self):
        """Literal type must reject anything other than 'user'/'assistant'."""
        with pytest.raises(ValidationError):
            ConversationTurn(role="system", text="bad")

    def test_invalid_role_empty_string(self):
        with pytest.raises(ValidationError):
            ConversationTurn(role="", text="bad")

    def test_invalid_role_typo(self):
        with pytest.raises(ValidationError):
            ConversationTurn(role="User", text="bad")  # case matters

    def test_mission_id_linkage(self):
        turn = ConversationTurn(
            role="assistant",
            text="done",
            mission_id="mission_123",
        )
        assert turn.mission_id == "mission_123"

    def test_extra_fields_rejected(self):
        """extra='forbid' must prevent monkey-patching."""
        with pytest.raises(ValidationError):
            ConversationTurn(role="user", text="hi", garbage="nope")


class TestConversationFrameHistory:
    """Validate history management on ConversationFrame."""

    def test_empty_history_on_create(self):
        frame = ConversationFrame()
        assert frame.history == []

    def test_append_user_turn(self):
        frame = ConversationFrame()
        frame.append_turn("user", "hello")
        assert len(frame.history) == 1
        assert frame.history[0].role == "user"
        assert frame.history[0].text == "hello"

    def test_append_assistant_turn(self):
        frame = ConversationFrame()
        frame.append_turn("assistant", "hi back")
        assert len(frame.history) == 1
        assert frame.history[0].role == "assistant"

    def test_append_order_preserved(self):
        frame = ConversationFrame()
        frame.append_turn("user", "first")
        frame.append_turn("assistant", "second")
        frame.append_turn("user", "third")
        assert [t.role for t in frame.history] == ["user", "assistant", "user"]
        assert [t.text for t in frame.history] == ["first", "second", "third"]

    def test_append_with_mission_id(self):
        frame = ConversationFrame()
        frame.append_turn("assistant", "done", mission_id="m_42")
        assert frame.history[0].mission_id == "m_42"

    def test_invalid_role_rejected_via_append(self):
        frame = ConversationFrame()
        with pytest.raises(ValidationError):
            frame.append_turn("system", "bad")  # type: ignore

    def test_history_cap_enforcement(self):
        """History must drop oldest turns when exceeding HISTORY_CAP."""
        frame = ConversationFrame()
        for i in range(HISTORY_CAP + 5):
            role = "user" if i % 2 == 0 else "assistant"
            frame.append_turn(role, f"turn_{i}")

        assert len(frame.history) == HISTORY_CAP
        # First turn should be turn_5 (0-4 dropped)
        assert frame.history[0].text == "turn_5"
        # Last turn should be the most recent
        assert frame.history[-1].text == f"turn_{HISTORY_CAP + 4}"

    def test_history_cap_exact_boundary(self):
        """At exactly HISTORY_CAP, no truncation should happen."""
        frame = ConversationFrame()
        for i in range(HISTORY_CAP):
            frame.append_turn("user", f"turn_{i}")
        assert len(frame.history) == HISTORY_CAP
        assert frame.history[0].text == "turn_0"

    def test_outcomes_list_starts_empty(self):
        frame = ConversationFrame()
        assert frame.outcomes == []

    def test_extra_fields_rejected_on_frame(self):
        """ConversationFrame extra='forbid' still works."""
        with pytest.raises(ValidationError):
            ConversationFrame(bad_field="nope")
