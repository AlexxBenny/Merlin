# tests/test_tier_classification.py

"""
Tests for Phase 5A: CognitiveTier enum and HeuristicTierClassifier.

Validates:
- Enum values and membership
- Keyword index construction from SkillRegistry
- Single-domain queries → Tier 1 (SIMPLE)
- Cross-domain queries → Tier 2 (MULTI_INTENT)
- Within-domain multi-verb queries → Tier 2
- Classification is deterministic and O(n) in query tokens
"""

import pytest
from unittest.mock import MagicMock

from brain.escalation_policy import (
    CognitiveTier,
    HeuristicTierClassifier,
    TierClassifier,
)


# ── Helpers ──────────────────────────────────────────────────

def _make_registry(*skill_defs):
    """Build a mock SkillRegistry from (name, domain, description, inputs) tuples."""
    registry = MagicMock()
    names = [s[0] for s in skill_defs]
    registry.all_names.return_value = names

    def _get(name):
        for s in skill_defs:
            if s[0] == name:
                skill = MagicMock()
                skill.name = s[0]
                skill.contract.name = s[0]
                skill.contract.domain = s[1]
                skill.contract.description = s[2]
                skill.contract.inputs = s[3]
                skill.contract.optional_inputs = s[4] if len(s) > 4 else {}
                return skill
        return None

    registry.get.side_effect = _get
    return registry


# Standard skill set matching MERLIN's actual skills
STANDARD_SKILLS = [
    ("fs.create_folder", "fs", "Create a folder", {"name": "folder_name", "anchor": "anchor_name"}, {"parent": "relative_path"}),
    ("system.open_app", "system", "Open an application", {"app_name": "application_name"}),
    ("system.set_brightness", "system", "Set screen brightness", {"level": "brightness_percentage"}),
    ("system.media_play", "system", "Play media playback", {}),
    ("system.unmute", "system", "Unmute system audio", {}),
]


# ── Enum Tests ───────────────────────────────────────────────

class TestCognitiveTierEnum:
    def test_values(self):
        assert CognitiveTier.SIMPLE == "simple"
        assert CognitiveTier.MULTI_INTENT == "multi"
        assert CognitiveTier.HIERARCHICAL == "hierarchical"

    def test_is_string_enum(self):
        assert isinstance(CognitiveTier.SIMPLE, str)

    def test_all_members(self):
        names = {t.name for t in CognitiveTier}
        assert names == {"SIMPLE", "MULTI_INTENT", "HIERARCHICAL", "REASONING"}


# ── Protocol Tests ───────────────────────────────────────────

class TestTierClassifierProtocol:
    def test_heuristic_satisfies_protocol(self):
        """HeuristicTierClassifier must satisfy the TierClassifier protocol."""
        registry = _make_registry(*STANDARD_SKILLS)
        classifier = HeuristicTierClassifier(registry)
        assert isinstance(classifier, TierClassifier)


# ── Simple Query Tests ───────────────────────────────────────

class TestSimpleQueries:
    @pytest.fixture
    def classifier(self):
        registry = _make_registry(*STANDARD_SKILLS)
        return HeuristicTierClassifier(registry)

    def test_single_action_simple(self, classifier):
        """Single-action queries should be Tier 1."""
        assert classifier.classify("create a folder called docs") == CognitiveTier.SIMPLE

    def test_single_action_with_location(self, classifier):
        """Single action with modifier stays Tier 1."""
        assert classifier.classify("create folder docs on desktop") == CognitiveTier.SIMPLE

    def test_open_app_simple(self, classifier):
        """Opening a single app is Tier 1."""
        assert classifier.classify("open notepad") == CognitiveTier.SIMPLE

    def test_set_brightness_simple(self, classifier):
        """Single system command is Tier 1."""
        assert classifier.classify("set brightness to 80") == CognitiveTier.SIMPLE

    def test_play_music_simple(self, classifier):
        """Single media command is Tier 1."""
        assert classifier.classify("play music") == CognitiveTier.SIMPLE

    def test_empty_query(self, classifier):
        """Empty queries default to Tier 1."""
        assert classifier.classify("") == CognitiveTier.SIMPLE

    def test_unknown_words(self, classifier):
        """Queries with no recognized keywords default to Tier 1."""
        assert classifier.classify("hello world") == CognitiveTier.SIMPLE


# ── Multi-Intent Query Tests ─────────────────────────────────

class TestMultiIntentQueries:
    @pytest.fixture
    def classifier(self):
        registry = _make_registry(*STANDARD_SKILLS)
        return HeuristicTierClassifier(registry)

    def test_cross_domain_two_actions(self, classifier):
        """Two actions spanning different domains → Tier 2."""
        result = classifier.classify("create folder docs and play music")
        assert result == CognitiveTier.MULTI_INTENT

    def test_cross_domain_with_open(self, classifier):
        """Open app + create folder → Tier 2 (cross-domain)."""
        result = classifier.classify("open notepad and create folder logs on desktop")
        assert result == CognitiveTier.MULTI_INTENT

    def test_same_domain_multi_verb(self, classifier):
        """Multiple verbs within same domain + conjunction → Tier 2."""
        result = classifier.classify("unmute and play music")
        assert result == CognitiveTier.MULTI_INTENT

    def test_three_actions_cross_domain(self, classifier):
        """Three actions crossing domains with high conjunction density → Tier 2 or 3."""
        result = classifier.classify(
            "create a folder called docs, open spotify, and set brightness to 50"
        )
        # 3 verbs + 2 conjunctions: crosses the verb/conjunction thresholds
        assert result in (CognitiveTier.MULTI_INTENT, CognitiveTier.HIERARCHICAL)


# ── Determinism Tests ────────────────────────────────────────

class TestDeterminism:
    def test_same_input_same_output(self):
        """Classifier must be deterministic — same input always produces same tier."""
        registry = _make_registry(*STANDARD_SKILLS)
        c1 = HeuristicTierClassifier(registry)
        c2 = HeuristicTierClassifier(registry)

        query = "create a folder and play music"
        assert c1.classify(query) == c2.classify(query)

    def test_case_insensitive(self):
        """Classification should be case-insensitive."""
        registry = _make_registry(*STANDARD_SKILLS)
        classifier = HeuristicTierClassifier(registry)

        result_lower = classifier.classify("create folder and play music")
        result_upper = classifier.classify("CREATE FOLDER AND PLAY MUSIC")
        assert result_lower == result_upper


# ── Edge Cases ───────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_registry(self):
        """Classifier with empty registry defaults all queries to Tier 1."""
        registry = _make_registry()
        classifier = HeuristicTierClassifier(registry)
        assert classifier.classify("create a folder and play music") == CognitiveTier.SIMPLE

    def test_conjunction_without_verbs(self):
        """Conjunctions alone don't trigger Tier 2."""
        registry = _make_registry(*STANDARD_SKILLS)
        classifier = HeuristicTierClassifier(registry)
        assert classifier.classify("the cat and the dog") == CognitiveTier.SIMPLE
