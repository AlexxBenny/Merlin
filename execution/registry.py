from typing import Dict
from skills.base import Skill
from cortex.semantic_types import assert_types_registered


class SkillRegistry:
    def __init__(self):
        self._skills: Dict[str, Skill] = {}

    def register(self, skill: Skill, *, validate_types: bool = True) -> None:
        if skill.name in self._skills:
            raise ValueError(f"Duplicate skill '{skill.name}'")
        # Fail loudly if skill declares a type not in SEMANTIC_TYPES
        if validate_types:
            assert_types_registered(
                skill.name,
                skill.contract.inputs,
                skill.contract.outputs,
            )
        self._skills[skill.name] = skill

    def get(self, name: str) -> Skill:
        if name not in self._skills:
            raise KeyError(f"Missing skill '{name}'")
        return self._skills[name]

    def all_names(self) -> set[str]:
        return set(self._skills.keys())
