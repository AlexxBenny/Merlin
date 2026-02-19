# tests/test_contract_invariants.py

"""
SkillContract Invariant Tests — Step 2 readiness.

Enforces structural invariants on SkillContract across all
registered skills. These tests must pass before AND after
the required/optional split.

Run: python -m pytest tests/test_contract_invariants.py -v
"""

import pytest

from skills.contract import SkillContract


# ── Collect all skill classes ──

def _all_skill_classes():
    """Import all skill classes and return them."""
    from skills.system.unmute import UnmuteSkill
    from skills.system.mute import MuteSkill
    from skills.system.set_volume import SetVolumeSkill
    from skills.system.set_brightness import SetBrightnessSkill
    from skills.system.open_app import OpenAppSkill
    from skills.system.close_app import CloseAppSkill
    from skills.system.focus_app import FocusAppSkill
    from skills.system.list_apps import ListAppsSkill
    from skills.system.toggle_nightlight import ToggleNightlightSkill
    from skills.system.media_play import MediaPlaySkill
    from skills.system.media_pause import MediaPauseSkill
    from skills.system.media_next import MediaNextSkill
    from skills.system.media_previous import MediaPreviousSkill
    from skills.fs.create_folder import CreateFolderSkill

    return [
        UnmuteSkill, MuteSkill, SetVolumeSkill, SetBrightnessSkill,
        OpenAppSkill, CloseAppSkill, FocusAppSkill, ListAppsSkill,
        ToggleNightlightSkill, MediaPlaySkill, MediaPauseSkill,
        MediaNextSkill, MediaPreviousSkill, CreateFolderSkill,
    ]


# ── Tests ──

class TestContractInvariants:
    """Structural invariants on SkillContract — read from class attribute."""

    @pytest.fixture(scope="class")
    def contracts(self):
        """Extract contract from each skill class (class attribute, no instantiation)."""
        return {cls.contract.name: cls.contract for cls in _all_skill_classes()}

    def test_all_skills_have_contract(self, contracts):
        """Every skill class must have a SkillContract."""
        for name, contract in contracts.items():
            assert isinstance(contract, SkillContract), (
                f"Skill '{name}' has no SkillContract"
            )

    def test_contract_names_unique(self, contracts):
        """No two skills share the same contract name."""
        names = [cls.contract.name for cls in _all_skill_classes()]
        assert len(names) == len(set(names)), (
            f"Duplicate contract names: "
            f"{[n for n in names if names.count(n) > 1]}"
        )

    def test_required_optional_disjoint(self, contracts):
        """required_inputs ∩ optional_inputs must be ∅.

        Forward-looking: once optional_inputs is added to SkillContract,
        this prevents accidental dual declaration.
        """
        for name, contract in contracts.items():
            required = set(contract.inputs.keys())
            optional = set(
                getattr(contract, "optional_inputs", {}).keys()
            )

            overlap = required & optional
            assert overlap == set(), (
                f"Skill '{name}' has inputs declared as both required "
                f"AND optional: {sorted(overlap)}"
            )

    def test_inputs_have_semantic_types(self, contracts):
        """All declared inputs must have non-empty semantic types."""
        for name, contract in contracts.items():
            for key, stype in contract.inputs.items():
                assert isinstance(stype, str) and len(stype) > 0, (
                    f"Skill '{name}' input '{key}' has invalid "
                    f"semantic type: {stype!r}"
                )

    def test_outputs_have_semantic_types(self, contracts):
        """All declared outputs must have non-empty semantic types."""
        for name, contract in contracts.items():
            for key, stype in contract.outputs.items():
                assert isinstance(stype, str) and len(stype) > 0, (
                    f"Skill '{name}' output '{key}' has invalid "
                    f"semantic type: {stype!r}"
                )

    def test_skill_count(self, contracts):
        """Sanity check: we should have 14 skills."""
        assert len(contracts) == 14, (
            f"Expected 14 skills, got {len(contracts)}: "
            f"{sorted(contracts.keys())}"
        )
