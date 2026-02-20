# tests/test_tts_engine.py

"""
Tests for TTSEngine abstraction and SilentTTS.
"""

import pytest

from reporting.tts_engine import TTSEngine
from reporting.engines.silent_tts import SilentTTS


class TestSilentTTS:
    def test_speak_does_not_raise(self):
        tts = SilentTTS()
        tts.speak("Hello world")  # should not raise

    def test_is_available(self):
        tts = SilentTTS()
        assert tts.is_available() is True

    def test_speak_empty_string(self):
        tts = SilentTTS()
        tts.speak("")  # should not raise

    def test_speak_long_text(self):
        tts = SilentTTS()
        tts.speak("x" * 10000)  # should not raise


class TestTTSEngineAbstract:
    def test_is_abstract(self):
        """TTSEngine cannot be instantiated directly."""
        with pytest.raises(TypeError):
            TTSEngine()
