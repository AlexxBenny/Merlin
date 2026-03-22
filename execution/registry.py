import logging
from typing import Dict, List, Set, Tuple
from skills.base import Skill
from cortex.semantic_types import assert_types_registered


logger = logging.getLogger(__name__)


class SkillRegistry:
    def __init__(self):
        self._skills: Dict[str, Skill] = {}
        self._registered_actions: Set[Tuple[str, str]] = set()  # (domain, action)

    def register(self, skill: Skill, *, validate_types: bool = True) -> None:
        if skill.name in self._skills:
            logger.debug(
                "Skill '%s' already registered — skipping (idempotent)",
                skill.name,
            )
            return
        # Fail loudly if skill declares a type not in SEMANTIC_TYPES
        if validate_types:
            assert_types_registered(
                skill.name,
                skill.contract.inputs,
                skill.contract.outputs,
            )

        # ── Capability integrity checks ──
        action = skill.contract.action
        domain = skill.contract.domain
        if action:
            # Structural invariant: name must be "domain.action"
            if "." not in skill.name:
                raise ValueError(
                    f"Skill '{skill.name}': name must contain '.'"
                )
            domain_from_name, action_from_name = skill.name.split(".", 1)

            # Auto-derive domain from name when not explicitly set
            if not domain:
                domain = domain_from_name
            elif domain_from_name != domain:
                # Cross-check: explicit domain must match name prefix
                raise ValueError(
                    f"Skill '{skill.name}': domain '{domain}' does not match "
                    f"domain '{domain_from_name}' derived from name"
                )
            if action_from_name != action:
                raise ValueError(
                    f"Skill '{skill.name}': action '{action}' does not match "
                    f"expected '{action_from_name}' derived from name"
                )
            # (domain, action) uniqueness — allows same action across domains
            key = (domain, action)
            if key in self._registered_actions:
                raise ValueError(
                    f"Skill '{skill.name}': (domain='{domain}', "
                    f"action='{action}') is already registered"
                )
            self._registered_actions.add(key)

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
        3. (domain, action) pairs are globally unique
        4. No duplicate full skill names

        Returns list of violation messages. Empty = healthy.
        Called at startup after all skills are registered.
        """
        violations: List[str] = []
        seen_keys: Dict[Tuple[str, str], str] = {}  # (domain, action) → skill_name
        seen_names: set = set()

        for name in sorted(self._skills.keys()):
            skill = self._skills[name]
            action = skill.contract.action
            domain = skill.contract.domain or name.split(".", 1)[0]

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

            # Rule 3: (domain, action) uniqueness
            key = (domain, action)
            if key in seen_keys:
                violations.append(
                    f"Skill '{name}': (domain='{domain}', action='{action}') "
                    f"collides with '{seen_keys[key]}'"
                )
            else:
                seen_keys[key] = name

            # Rule 4: full name uniqueness
            if name in seen_names:
                violations.append(
                    f"Skill '{name}': duplicate full name"
                )
            seen_names.add(name)

        if violations:
            for v in violations:
                logger.error("[AUDIT] %s", v)
        else:
            logger.info(
                "[AUDIT] Action namespace audit passed: %d skills, "
                "%d unique (domain, action) pairs, 0 violations",
                len(self._skills), len(seen_keys),
            )

        return violations
