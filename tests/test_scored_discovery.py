# tests/test_scored_discovery.py

"""
Tests for DomainScoredDiscovery.

Covers:
- Fallback: returns all skills when total < threshold
- Phase 1: intent verb/keyword scoring
- Phase 2: bounded domain expansion
- Domain cap: max_per_domain respected
- Global cap: max_manifest hard ceiling
- Deterministic ordering: (score desc, name asc)
- Zero-score exclusion: unmatched skills not in Phase 1
- Cross-domain expansion: sibling skills included
"""

import unittest
from unittest.mock import MagicMock

from cortex.scored_discovery import DomainScoredDiscovery
from cortex.skill_discovery import AllSkillsDiscovery
from execution.registry import SkillRegistry
from skills.contract import SkillContract, FailurePolicy
from ir.mission import ExecutionMode


def _make_skill(name, domain, verbs, keywords, action=None):
    """Create a minimal mock skill with contract."""
    skill = MagicMock()
    skill.name = name
    skill.contract = SkillContract(
        name=name,
        action=action or name.split(".")[-1],
        target_type="test",
        description=f"Test skill {name}",
        narration_template=f"do {name}",
        intent_verbs=verbs,
        intent_keywords=keywords,
        verb_specificity="generic",
        domain=domain,
        inputs={"input": "application_name"},
        outputs={"output": "info_string"},
        allowed_modes={ExecutionMode.foreground},
        failure_policy={
            ExecutionMode.foreground: FailurePolicy.FAIL,
        },
    )
    return skill


def _make_registry(skills):
    """Create a SkillRegistry with given skills."""
    registry = SkillRegistry()
    for skill in skills:
        registry.register(skill)
    return registry


class TestFallbackBehavior(unittest.TestCase):
    """Below fallback_threshold, returns all skills."""

    def test_returns_all_when_below_threshold(self):
        skills = [
            _make_skill("fs.write", "fs", ["write"], ["file"]),
            _make_skill("fs.read", "fs", ["read"], ["file"]),
        ]
        registry = _make_registry(skills)
        discovery = DomainScoredDiscovery(fallback_threshold=30)

        result = discovery.find_candidates("write a file", registry)
        assert len(result) == 2
        assert "fs.write" in result
        assert "fs.read" in result


class TestScoring(unittest.TestCase):
    """Phase 1: intent verb + keyword scoring."""

    def _build_large_registry(self):
        """Build a registry with >30 skills to bypass fallback."""
        skills = []
        # Target skills
        skills.append(_make_skill("fs.write", "fs", ["write", "save"], ["file", "content"]))
        skills.append(_make_skill("fs.read", "fs", ["read", "load"], ["file"]))
        skills.append(_make_skill("reasoning.generate", "reasoning", ["write", "generate"], ["story", "poem"]))
        # Filler skills (unrelated, fill to >30)
        for i in range(30):
            skills.append(_make_skill(
                f"filler.skill_{i}", "filler",
                [f"verb{i}"], [f"keyword{i}"],
            ))
        return _make_registry(skills)

    def test_verb_match_scores_higher(self):
        discovery = DomainScoredDiscovery(fallback_threshold=5)
        registry = self._build_large_registry()
        scores = discovery._score_skills("write a poem", registry)

        # "write" matches fs.write (verb=2.0) and reasoning.generate (verb=2.0)
        assert scores["fs.write"] >= 2.0
        assert scores["reasoning.generate"] >= 2.0
        # "poem" matches reasoning.generate (keyword=1.0)
        assert scores["reasoning.generate"] >= 3.0  # verb + keyword
        # Filler skills should score 0
        assert scores["filler.skill_0"] == 0.0

    def test_zero_score_not_in_phase1(self):
        discovery = DomainScoredDiscovery(fallback_threshold=5, top_k=5)
        registry = self._build_large_registry()
        result = discovery.find_candidates("write a poem", registry)

        # Zero-scored fillers should not be in result
        # (unless domain-expanded, which won't happen for "filler" domain)
        for i in range(30):
            assert f"filler.skill_{i}" not in result


class TestDomainExpansion(unittest.TestCase):
    """Phase 2: bounded domain expansion."""

    def _build_registry_with_many_fs(self):
        """30+ filler skills + multiple fs skills."""
        skills = []
        skills.append(_make_skill("fs.write", "fs", ["write"], ["file"]))
        skills.append(_make_skill("fs.read", "fs", ["read"], ["file"]))
        skills.append(_make_skill("fs.delete", "fs", ["delete"], ["file"]))
        skills.append(_make_skill("fs.copy", "fs", ["copy"], ["file"]))
        skills.append(_make_skill("fs.create_folder", "fs", ["create"], ["folder"]))
        # Filler
        for i in range(30):
            skills.append(_make_skill(
                f"other.skill_{i}", "other",
                [f"xverb{i}"], [f"xkw{i}"],
            ))
        return _make_registry(skills)

    def test_sibling_skills_expanded(self):
        discovery = DomainScoredDiscovery(
            fallback_threshold=5, top_k=3, max_per_domain=12,
        )
        registry = self._build_registry_with_many_fs()

        result = discovery.find_candidates("write a file", registry)
        # fs.write matched directly, fs.read/delete/copy/create_folder expanded
        assert "fs.write" in result
        assert "fs.read" in result  # Sibling in fs domain

    def test_domain_cap_respected(self):
        """Expansion limited to max_per_domain."""
        skills = []
        skills.append(_make_skill("big.target", "big", ["write"], ["file"]))
        # 25 siblings in "big" domain
        for i in range(25):
            skills.append(_make_skill(
                f"big.extra_{i}", "big",
                [f"x{i}"], [f"y{i}"],
            ))
        # Filler
        for i in range(30):
            skills.append(_make_skill(
                f"filler.s_{i}", "filler",
                [f"f{i}"], [f"g{i}"],
            ))
        registry = _make_registry(skills)

        discovery = DomainScoredDiscovery(
            fallback_threshold=5, top_k=5, max_per_domain=10,
        )
        result = discovery.find_candidates("write a file", registry)

        # big.target matched + up to 10 siblings = 11 max from "big" domain
        big_count = sum(1 for k in result if k.startswith("big."))
        assert big_count <= 11  # 1 direct + 10 expanded


class TestGlobalCap(unittest.TestCase):
    """Hard global ceiling."""

    def test_max_manifest_enforced(self):
        skills = []
        # Every skill matches "write"
        for i in range(80):
            skills.append(_make_skill(
                f"d{i}.write_{i}", f"d{i}",
                ["write"], ["data"],
            ))
        registry = _make_registry(skills)

        discovery = DomainScoredDiscovery(
            fallback_threshold=5, top_k=30,
            max_per_domain=12, max_manifest=40,
        )
        result = discovery.find_candidates("write data", registry)
        assert len(result) <= 40


class TestDeterministicOrdering(unittest.TestCase):
    """Output should be stable across calls."""

    def test_same_query_same_result(self):
        skills = []
        for i in range(35):
            skills.append(_make_skill(
                f"test.skill_{i:02d}", "test",
                ["action"], ["thing"],
            ))
        registry = _make_registry(skills)

        discovery = DomainScoredDiscovery(fallback_threshold=5, top_k=10)

        r1 = discovery.find_candidates("do the action thing", registry)
        r2 = discovery.find_candidates("do the action thing", registry)
        assert list(r1.keys()) == list(r2.keys())


class TestManifestFormat(unittest.TestCase):
    """Output format matches AllSkillsDiscovery."""

    def test_manifest_has_required_keys(self):
        skills = [
            _make_skill("fs.write", "fs", ["write"], ["file"]),
        ]
        registry = _make_registry(skills)
        discovery = DomainScoredDiscovery(fallback_threshold=30)

        result = discovery.find_candidates("write file", registry)
        entry = result["fs.write"]

        assert "description" in entry
        assert "action" in entry
        assert "inputs" in entry
        assert "outputs" in entry
        assert "allowed_modes" in entry

class TestEmptyPhase1Fallback(unittest.TestCase):
    """When Phase 1 scores all zeros, fall back to AllSkills (never empty manifest)."""

    def test_unrecognized_query_returns_all(self):
        """Query with zero verb/keyword matches must not produce empty manifest."""
        skills = []
        for i in range(35):
            skills.append(_make_skill(
                f"domain.skill_{i}", "domain",
                [f"specificverb{i}"], [f"specifickw{i}"],
            ))
        registry = _make_registry(skills)

        discovery = DomainScoredDiscovery(fallback_threshold=5, top_k=10)
        # Query matches NOTHING
        result = discovery.find_candidates(
            "xyzzy blorp quux", registry,
        )
        # Must NOT be empty — fallback to all
        assert len(result) == 35

    def test_partial_gibberish_with_one_match(self):
        """If at least one skill matches, no fallback needed."""
        skills = []
        skills.append(_make_skill("fs.write", "fs", ["write"], ["file"]))
        for i in range(34):
            skills.append(_make_skill(
                f"other.s_{i}", "other",
                [f"v{i}"], [f"k{i}"],
            ))
        registry = _make_registry(skills)

        discovery = DomainScoredDiscovery(fallback_threshold=5, top_k=10)
        result = discovery.find_candidates(
            "xyzzy write blorp", registry,
        )
        # fs.write matched — should be in result, no full fallback
        assert "fs.write" in result
        assert len(result) < 35  # Not all skills


if __name__ == "__main__":
    unittest.main()
