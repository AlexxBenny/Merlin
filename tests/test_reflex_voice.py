# tests/test_reflex_voice.py

"""
Tests that reflex templates match voice-transcribed input.

These test the full chain: normalize → regex match.
No hardware required — uses reflex engine with loaded templates.
"""

import pytest
import yaml
from pathlib import Path

from runtime.reflex_engine import ReflexEngine


@pytest.fixture
def reflex():
    """Build ReflexEngine with templates from routing.yaml."""
    config_path = Path(__file__).parent.parent / "config" / "routing.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    templates = ReflexEngine.load_templates(config.get("reflex_templates", []))
    engine = ReflexEngine(timeline=None, registry=None)
    for t in templates:
        engine.register_template(t)
    return engine


class TestVoiceTranscriptionMatchesMute:
    """Whisper transcribes 'mute' as 'Mute.' — must still match reflex."""

    def test_mute_with_period(self, reflex):
        m = reflex.try_match("Mute.")
        assert m is not None
        assert m.skill == "system.mute"

    def test_mute_with_exclamation(self, reflex):
        m = reflex.try_match("Mute!")
        assert m is not None
        assert m.skill == "system.mute"

    def test_mute_uppercase(self, reflex):
        m = reflex.try_match("MUTE")
        assert m is not None
        assert m.skill == "system.mute"

    def test_mute_plain(self, reflex):
        m = reflex.try_match("mute")
        assert m is not None
        assert m.skill == "system.mute"


class TestVoiceTranscriptionMatchesVolume:
    def test_volume_with_period(self, reflex):
        m = reflex.try_match("Set volume to 50.")
        assert m is not None
        assert m.skill == "system.set_volume"
        assert m.params["level"] == "50"

    def test_volume_with_filler(self, reflex):
        m = reflex.try_match("um set volume to 70")
        assert m is not None
        assert m.skill == "system.set_volume"


class TestVoiceTranscriptionMatchesMedia:
    def test_play_with_period(self, reflex):
        m = reflex.try_match("Play.")
        assert m is not None
        assert m.skill == "system.media_play"

    def test_pause_with_period(self, reflex):
        m = reflex.try_match("Pause.")
        assert m is not None
        assert m.skill == "system.media_pause"

    def test_next_with_period(self, reflex):
        m = reflex.try_match("Next.")
        assert m is not None
        assert m.skill == "system.media_next"

    def test_skip_with_period(self, reflex):
        m = reflex.try_match("Skip.")
        assert m is not None
        assert m.skill == "system.media_next"


class TestVoiceTranscriptionMatchesBrightness:
    def test_brightness_with_period(self, reflex):
        m = reflex.try_match("Set brightness to 80.")
        assert m is not None
        assert m.skill == "system.set_brightness"
        assert m.params["level"] == "80"


class TestVoiceTranscriptionMatchesNightlight:
    def test_nightlight_with_period(self, reflex):
        m = reflex.try_match("Toggle night light.")
        assert m is not None
        assert m.skill == "system.toggle_nightlight"


class TestComplexQueriesStillMiss:
    """Complex queries must NOT match reflex — they go to MISSION."""

    def test_multi_step_no_match(self, reflex):
        assert reflex.try_match(
            "Mute the volume and then set brightness to 50."
        ) is None

    def test_ambiguous_no_match(self, reflex):
        assert reflex.try_match(
            "What's the current volume level?"
        ) is None
