# tests/test_speech_perception.py

"""
Tests for SpeechPerception using MockSTT.

No hardware required — AudioRecorder is mocked.
"""

import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from perception.speech import SpeechPerception
from perception.engines.mock_stt import MockSTT


def _make_mock_recorder(audio_data=None):
    """Create a mock AudioRecorder that returns predefined audio."""
    recorder = MagicMock()
    recorder.sample_rate = 16000
    if audio_data is None:
        audio_data = np.zeros(16000, dtype=np.float32)  # 1s silence
    recorder.record_until_silence.return_value = audio_data
    return recorder


class TestSpeechPerception:
    def test_produces_speech_percept(self):
        stt = MockSTT(["hello merlin"])
        recorder = _make_mock_recorder()
        sp = SpeechPerception(stt, recorder)

        percept = sp.listen()

        assert percept.modality == "speech"
        assert percept.payload == "hello merlin"
        assert percept.confidence == 1.0
        assert percept.timestamp <= time.time()

    def test_empty_audio_returns_empty_payload(self):
        stt = MockSTT(["should not reach this"])
        recorder = _make_mock_recorder(np.array([], dtype=np.float32))
        sp = SpeechPerception(stt, recorder)

        percept = sp.listen()

        assert percept.payload == ""
        assert percept.confidence == 0.0

    def test_strips_whitespace(self):
        stt = MockSTT(["  set volume to 50  "])
        recorder = _make_mock_recorder()
        sp = SpeechPerception(stt, recorder)

        percept = sp.listen()
        assert percept.payload == "set volume to 50"

    def test_multiple_listens(self):
        stt = MockSTT(["play music", "pause"])
        recorder = _make_mock_recorder()
        sp = SpeechPerception(stt, recorder)

        p1 = sp.listen()
        p2 = sp.listen()

        assert p1.payload == "play music"
        assert p2.payload == "pause"

    def test_calls_recorder_and_stt(self):
        stt = MockSTT(["test"])
        recorder = _make_mock_recorder()
        sp = SpeechPerception(stt, recorder)

        sp.listen()

        recorder.record_until_silence.assert_called_once()
