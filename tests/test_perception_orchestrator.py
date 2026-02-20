# tests/test_perception_orchestrator.py

"""
Tests for PerceptionOrchestrator.

Tests cover text-only mode, voice-only mode, and the
CancellationToken mechanics.
"""

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from brain.core import Percept
from perception.text import TextPerception
from perception.speech import SpeechPerception
from perception.engines.mock_stt import MockSTT
from perception.perception_orchestrator import (
    PerceptionOrchestrator,
    CancellationToken,
)


class TestCancellationToken:
    def test_initial_state_not_cancelled(self):
        token = CancellationToken()
        assert token.is_cancelled is False

    def test_cancel_sets_flag(self):
        token = CancellationToken()
        token.cancel()
        assert token.is_cancelled is True

    def test_cancel_is_idempotent(self):
        token = CancellationToken()
        token.cancel()
        token.cancel()
        assert token.is_cancelled is True


class TestPerceptionOrchestratorTextOnly:
    def test_text_only_returns_text_percept(self):
        text = TextPerception()
        orch = PerceptionOrchestrator(text=text)

        with patch("builtins.input", return_value="hello"):
            percept = orch.next_percept()

        assert percept.modality == "text"
        assert percept.payload == "hello"

    def test_text_only_strips_whitespace(self):
        text = TextPerception()
        orch = PerceptionOrchestrator(text=text)

        with patch("builtins.input", return_value="  hello  "):
            percept = orch.next_percept()

        assert percept.payload == "hello"


class TestPerceptionOrchestratorVoiceOnly:
    def test_voice_only_returns_speech_percept(self):
        stt = MockSTT(["set volume to 50"])
        recorder = MagicMock()
        recorder.sample_rate = 16000
        recorder.record_until_silence.return_value = np.zeros(
            16000, dtype=np.float32,
        )
        speech = SpeechPerception(stt, recorder)
        orch = PerceptionOrchestrator(text=None, speech=speech)

        percept = orch.next_percept()

        assert percept.modality == "speech"
        assert percept.payload == "set volume to 50"


class TestPerceptionOrchestratorValidation:
    def test_requires_at_least_one_channel(self):
        with pytest.raises(ValueError, match="at least one channel"):
            PerceptionOrchestrator(text=None, speech=None)
