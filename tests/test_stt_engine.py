# tests/test_stt_engine.py

"""
Tests for STTEngine abstraction and MockSTT.
"""

import numpy as np
import pytest

from perception.stt_engine import STTEngine, TranscriptionResult
from perception.engines.mock_stt import MockSTT


class TestTranscriptionResult:
    def test_immutable(self):
        r = TranscriptionResult(text="hello", confidence=0.95, language="en")
        with pytest.raises(AttributeError):
            r.text = "changed"

    def test_fields(self):
        r = TranscriptionResult(text="test", confidence=0.8, language="fr")
        assert r.text == "test"
        assert r.confidence == 0.8
        assert r.language == "fr"


class TestMockSTT:
    def test_returns_predefined_transcripts(self):
        stt = MockSTT(["hello world", "set volume to 50"])
        audio = np.zeros(16000, dtype=np.float32)

        r1 = stt.transcribe(audio, 16000)
        assert r1.text == "hello world"
        assert r1.confidence == 1.0

        r2 = stt.transcribe(audio, 16000)
        assert r2.text == "set volume to 50"

    def test_returns_empty_after_exhausted(self):
        stt = MockSTT(["only one"])
        audio = np.zeros(16000, dtype=np.float32)

        stt.transcribe(audio, 16000)
        r = stt.transcribe(audio, 16000)
        assert r.text == ""

    def test_is_available(self):
        stt = MockSTT([])
        assert stt.is_available() is True

    def test_does_not_support_streaming(self):
        stt = MockSTT([])
        assert stt.supports_streaming() is False

    def test_custom_confidence(self):
        stt = MockSTT(["test"], confidence=0.7)
        audio = np.zeros(16000, dtype=np.float32)
        r = stt.transcribe(audio, 16000)
        assert r.confidence == 0.7

    def test_is_abstract(self):
        """STTEngine cannot be instantiated directly."""
        with pytest.raises(TypeError):
            STTEngine()
