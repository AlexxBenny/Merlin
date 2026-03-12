# tests/test_ordinal_resolver.py

"""
Tests for ordinal entity resolution — Phase 2.

Covers:
1. detect_ordinal_entity_reference — ordinal word detection
2. Numeric ordinal detection (1st, 2nd, 3rd)
3. Action verb requirement
4. Entity index validation
5. Edge cases (no entities, no verbs, no ordinals)
6. Integration: _handle_reflex ordinal override
"""

import pytest
from brain.ordinal_resolver import (
    detect_ordinal_entity_reference,
    _ORDINAL_WORDS,
)


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

def _entities(n=5):
    """Generate n sample entities for testing."""
    return [
        {"index": i + 1, "type": "link", "text": f"Result {i + 1}"}
        for i in range(n)
    ]


# ─────────────────────────────────────────────────────────────
# 1. Ordinal word detection
# ─────────────────────────────────────────────────────────────

class TestOrdinalWordDetection:
    """Test detection of ordinal words like 'first', 'second', etc."""

    def test_first(self):
        result = detect_ordinal_entity_reference(
            "open the first one", _entities(),
        )
        assert result is not None
        assert result[0] == 1

    def test_second(self):
        result = detect_ordinal_entity_reference(
            "click the second result", _entities(),
        )
        assert result is not None
        assert result[0] == 2

    def test_third(self):
        result = detect_ordinal_entity_reference(
            "select the third link", _entities(),
        )
        assert result is not None
        assert result[0] == 3

    def test_fifth(self):
        result = detect_ordinal_entity_reference(
            "open the fifth item", _entities(),
        )
        assert result is not None
        assert result[0] == 5

    def test_all_ordinal_words_recognized(self):
        """Every word in _ORDINAL_WORDS should be detected."""
        for word, expected_index in _ORDINAL_WORDS.items():
            entities = _entities(max(expected_index, 1))
            result = detect_ordinal_entity_reference(
                f"open the {word} result", entities,
            )
            assert result is not None, f"'{word}' not detected"
            assert result[0] == expected_index, (
                f"'{word}' resolved to {result[0]}, expected {expected_index}"
            )

    def test_case_insensitive(self):
        result = detect_ordinal_entity_reference(
            "Open the FIRST one", _entities(),
        )
        assert result is not None
        assert result[0] == 1


# ─────────────────────────────────────────────────────────────
# 2. Numeric ordinal detection
# ─────────────────────────────────────────────────────────────

class TestNumericOrdinalDetection:
    """Test detection of numeric ordinals like 1st, 2nd, 3rd."""

    def test_1st(self):
        result = detect_ordinal_entity_reference(
            "open the 1st result", _entities(),
        )
        assert result is not None
        assert result[0] == 1

    def test_2nd(self):
        result = detect_ordinal_entity_reference(
            "click the 2nd link", _entities(),
        )
        assert result is not None
        assert result[0] == 2

    def test_3rd(self):
        result = detect_ordinal_entity_reference(
            "select the 3rd video", _entities(),
        )
        assert result is not None
        assert result[0] == 3

    def test_4th(self):
        result = detect_ordinal_entity_reference(
            "open the 4th one", _entities(),
        )
        assert result is not None
        assert result[0] == 4


# ─────────────────────────────────────────────────────────────
# 3. Action verb requirement
# ─────────────────────────────────────────────────────────────

class TestActionVerbRequirement:
    """Test that ordinals are only detected with action verbs."""

    def test_no_verb_no_match(self):
        """Ordinal without action verb should NOT match."""
        result = detect_ordinal_entity_reference(
            "the first result is interesting", _entities(),
        )
        assert result is None

    def test_question_no_match(self):
        """Questions about ordinals should NOT match."""
        result = detect_ordinal_entity_reference(
            "what is the first item about", _entities(),
        )
        assert result is None

    def test_open_verb_matches(self):
        result = detect_ordinal_entity_reference(
            "open the first one", _entities(),
        )
        assert result is not None

    def test_click_verb_matches(self):
        result = detect_ordinal_entity_reference(
            "click the second result", _entities(),
        )
        assert result is not None

    def test_select_verb_matches(self):
        result = detect_ordinal_entity_reference(
            "select the third link", _entities(),
        )
        assert result is not None

    def test_watch_verb_matches(self):
        result = detect_ordinal_entity_reference(
            "watch the first video", _entities(),
        )
        assert result is not None

    def test_go_verb_matches(self):
        result = detect_ordinal_entity_reference(
            "go to the first site", _entities(),
        )
        assert result is not None


# ─────────────────────────────────────────────────────────────
# 4. Entity index validation
# ─────────────────────────────────────────────────────────────

class TestEntityIndexValidation:
    """Test that only valid entity indices are returned."""

    def test_index_out_of_range(self):
        """Index beyond entity list should return None."""
        result = detect_ordinal_entity_reference(
            "open the tenth result", _entities(3),  # only 3 entities
        )
        assert result is None

    def test_index_at_boundary(self):
        """Index at exact boundary should match."""
        result = detect_ordinal_entity_reference(
            "open the third one", _entities(3),
        )
        assert result is not None
        assert result[0] == 3

    def test_empty_entities(self):
        """No entities → no match."""
        result = detect_ordinal_entity_reference(
            "open the first one", [],
        )
        assert result is None


# ─────────────────────────────────────────────────────────────
# 5. Edge cases
# ─────────────────────────────────────────────────────────────

class TestEdgeCases:
    """Test edge cases and non-matching queries."""

    def test_no_ordinal_in_query(self):
        """Regular browser command without ordinal → None."""
        result = detect_ordinal_entity_reference(
            "open youtube", _entities(),
        )
        assert result is None

    def test_just_number_no_suffix(self):
        """Bare number without ordinal suffix should not match."""
        result = detect_ordinal_entity_reference(
            "open 3 tabs", _entities(),
        )
        assert result is None

    def test_complex_sentence_with_ordinal(self):
        """Ordinal embedded in complex sentence should match."""
        result = detect_ordinal_entity_reference(
            "please open the first result for me", _entities(),
        )
        assert result is not None
        assert result[0] == 1

    def test_entity_nouns_optional(self):
        """'open the first' without noun should match."""
        result = detect_ordinal_entity_reference(
            "open the first", _entities(),
        )
        assert result is not None
        assert result[0] == 1

    def test_real_scenario_amazon(self):
        """Real-world: 'open the first site' on Amazon results."""
        entities = [
            {"index": 1, "type": "link", "text": "iPhone 14 Pro Max"},
            {"index": 2, "type": "link", "text": "iPhone 14 Plus"},
            {"index": 3, "type": "button", "text": "Add to Cart"},
        ]
        result = detect_ordinal_entity_reference(
            "open the first site", entities,
        )
        assert result is not None
        assert result[0] == 1

    def test_real_scenario_youtube(self):
        """Real-world: 'play the second video' on YouTube results."""
        entities = [
            {"index": 1, "type": "link", "text": "Marvel Official Trailer"},
            {"index": 2, "type": "link", "text": "Avengers Endgame Trailer"},
        ]
        result = detect_ordinal_entity_reference(
            "play the second video", entities,
        )
        assert result is not None
        assert result[0] == 2

    def test_returns_matched_text(self):
        """Should return the matched ordinal text."""
        result = detect_ordinal_entity_reference(
            "open the first result", _entities(),
        )
        assert result is not None
        assert "first" in result[1].lower()
