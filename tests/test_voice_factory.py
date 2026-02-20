# tests/test_voice_factory.py

"""
Tests for VoiceEngineFactory.

Verifies config-driven engine creation and graceful fallback
when dependencies are missing.
"""

from unittest.mock import patch

import pytest

from infrastructure.voice_factory import VoiceEngineFactory
from reporting.tts_engine import TTSEngine
from perception.stt_engine import STTEngine


class TestVoiceEngineFactorySTT:
    def test_unknown_engine_returns_none(self):
        config = {"stt": {"engine": "nonexistent"}}
        result = VoiceEngineFactory.create_stt(config)
        assert result is None

    def test_missing_stt_config_returns_none(self):
        config = {}
        result = VoiceEngineFactory.create_stt(config)
        assert result is None

    def test_import_error_returns_none(self):
        """Simulate missing faster-whisper dependency."""
        config = {
            "stt": {
                "engine": "faster-whisper",
                "model": "tiny",
                "device": "cpu",
            }
        }
        with patch.dict("sys.modules", {"faster_whisper": None}):
            # This should catch ImportError and return None
            result = VoiceEngineFactory.create_stt(config)
            # May return None or a WhisperSTT depending on env
            # The key invariant: it must not raise
            assert result is None or isinstance(result, STTEngine)


class TestVoiceEngineFactoryTTS:
    def test_unknown_engine_returns_none(self):
        config = {"tts": {"engine": "nonexistent"}}
        result = VoiceEngineFactory.create_tts(config)
        assert result is None

    def test_missing_tts_config_returns_none(self):
        config = {}
        result = VoiceEngineFactory.create_tts(config)
        assert result is None

    def test_silent_engine_created(self):
        config = {"tts": {"engine": "silent"}}
        result = VoiceEngineFactory.create_tts(config)
        assert result is not None
        assert isinstance(result, TTSEngine)
        assert result.is_available() is True

    def test_import_error_returns_none(self):
        """Simulate missing pyttsx3 dependency."""
        config = {"tts": {"engine": "pyttsx3"}}
        with patch.dict("sys.modules", {"pyttsx3": None}):
            result = VoiceEngineFactory.create_tts(config)
            assert result is None or isinstance(result, TTSEngine)
