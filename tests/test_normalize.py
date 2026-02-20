# tests/test_normalize.py

"""
Tests for perception.normalize — the single normalization choke point.
"""

import pytest
from perception.normalize import normalize_for_matching


class TestTrailingPunctuation:
    def test_strip_period(self):
        assert normalize_for_matching("Mute.") == "mute"

    def test_strip_exclamation(self):
        assert normalize_for_matching("Exit!") == "exit"

    def test_strip_question_mark(self):
        assert normalize_for_matching("What?") == "what"

    def test_strip_multiple_trailing(self):
        assert normalize_for_matching("Really?!") == "really"

    def test_preserve_internal_period(self):
        assert normalize_for_matching("Dr. Smith") == "dr. smith"

    def test_preserve_internal_decimal(self):
        assert normalize_for_matching("set to 2.5") == "set to 2.5"

    def test_trailing_on_longer_text(self):
        assert normalize_for_matching("Set volume to 50.") == "set volume to 50"


class TestWhitespace:
    def test_leading_trailing_whitespace(self):
        assert normalize_for_matching("  mute  ") == "mute"

    def test_collapse_internal_whitespace(self):
        assert normalize_for_matching("set  volume  to  50") == "set volume to 50"

    def test_tabs_and_newlines(self):
        assert normalize_for_matching("set\tvolume\nto 50") == "set volume to 50"


class TestCaseFolding:
    def test_uppercase(self):
        assert normalize_for_matching("MUTE") == "mute"

    def test_mixed_case(self):
        assert normalize_for_matching("Set Volume To 50") == "set volume to 50"


class TestFillerRemoval:
    def test_um(self):
        assert normalize_for_matching("um mute") == "mute"

    def test_uh(self):
        assert normalize_for_matching("uh set volume to 50") == "set volume to 50"

    def test_hey(self):
        assert normalize_for_matching("hey play music") == "play music"

    def test_okay(self):
        assert normalize_for_matching("okay pause") == "pause"

    def test_ok(self):
        assert normalize_for_matching("ok skip") == "skip"

    def test_so(self):
        assert normalize_for_matching("so mute the volume") == "mute the volume"

    def test_well(self):
        assert normalize_for_matching("well unmute") == "unmute"

    def test_filler_only_removed_at_start(self):
        # "um" in the middle should NOT be stripped
        assert normalize_for_matching("hum along") == "hum along"

    def test_filler_with_trailing_punct(self):
        assert normalize_for_matching("Um, mute.") == "mute"


class TestEdgeCases:
    def test_empty_string(self):
        assert normalize_for_matching("") == ""

    def test_whitespace_only(self):
        assert normalize_for_matching("   ") == ""

    def test_punctuation_only(self):
        assert normalize_for_matching("...") == ""

    def test_none_like(self):
        assert normalize_for_matching("") == ""

    def test_complex_voice_command(self):
        assert normalize_for_matching(
            "Um, set the brightness to 80!"
        ) == "set the brightness to 80"
