import logging
from typing import Dict, List, Set
from skills.base import Skill
from cortex.semantic_types import assert_types_registered


logger = logging.getLogger(__name__)


class SkillRegistry:
    def __init__(self):
        self._skills: Dict[str, Skill] = {}
        self._registered_actions: Set[str] = set()

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

        # ── Capability integrity checks ──
        action = skill.contract.action
        if action:
            # Naming consistency: action must match skill name convention
            parts = skill.name.split(".", 1)
            expected_action = parts[1] if len(parts) >= 2 else parts[0]
            if action != expected_action:
                raise ValueError(
                    f"Skill '{skill.name}': action '{action}' does not match "
                    f"expected '{expected_action}' derived from name"
                )
            # Global uniqueness: no two skills may share the same action
            if action in self._registered_actions:
                raise ValueError(
                    f"Skill '{skill.name}': action '{action}' is already "
                    f"registered by another skill"
                )
            self._registered_actions.add(action)

        # ── Description style enforcement (Phase 8B) ──
        # description is a governed UX-visible field — not documentation.
        # Style contract: imperative verb phrase, ≤6 words, no parens/commas.
        desc = skill.contract.description
        if desc:
            _desc_violations = []
            if not desc[0].isupper():
                _desc_violations.append("must start with uppercase verb")
            if "(" in desc or ")" in desc:
                _desc_violations.append("must not contain parentheses")
            if "," in desc:
                _desc_violations.append("must not contain commas")
            if desc.endswith("."):
                _desc_violations.append("must not end with period")
            if len(desc.split()) > 6:
                _desc_violations.append(f"exceeds 6 words ({len(desc.split())})")
            if _desc_violations:
                logger.warning(
                    "Skill '%s' description style: %s — '%s'",
                    skill.name, "; ".join(_desc_violations), desc,
                )

        self._skills[skill.name] = skill

    def get(self, name: str) -> Skill:
        if name not in self._skills:
            raise KeyError(f"Missing skill '{name}'")
        return self._skills[name]

    def all_names(self) -> set[str]:
        return set(self._skills.keys())

    def audit_action_namespace(self) -> List[str]:
        """Startup action namespace governance audit.

        Validates:
        1. Every registered skill declares an action (no empty actions)
        2. All actions are lowercase
        3. All (action, target_type) pairs are globally unique
        4. No skills share the same action

        Returns list of violation messages. Empty = healthy.
        Called at startup after all skills are registered.
        """
        violations: List[str] = []
        seen_actions: Dict[str, str] = {}  # action → skill_name
        seen_pairs: Dict[tuple, str] = {}  # (action, target_type) → skill_name

        for name in sorted(self._skills.keys()):
            skill = self._skills[name]
            action = skill.contract.action
            target_type = skill.contract.target_type

            # Rule 1: every skill must declare an action
            if not action:
                violations.append(
                    f"Skill '{name}': missing action declaration"
                )
                continue

            # Rule 2: actions must be lowercase
            if action != action.lower():
                violations.append(
                    f"Skill '{name}': action '{action}' is not lowercase"
                )

            # Rule 3: action uniqueness
            if action in seen_actions:
                violations.append(
                    f"Skill '{name}': action '{action}' collides with "
                    f"'{seen_actions[action]}'"
                )
            else:
                seen_actions[action] = name

            # Rule 4: (action, target_type) pair uniqueness
            pair = (action, target_type)
            if pair in seen_pairs:
                violations.append(
                    f"Skill '{name}': (action={action}, target_type={target_type}) "
                    f"collides with '{seen_pairs[pair]}'"
                )
            else:
                seen_pairs[pair] = name

        if violations:
            for v in violations:
                logger.error("[AUDIT] %s", v)
        else:
            logger.info(
                "[AUDIT] Action namespace audit passed: %d skills, "
                "%d unique actions, 0 violations",
                len(self._skills), len(seen_actions),
            )

        return violations
