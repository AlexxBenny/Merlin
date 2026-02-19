# skills/system/set_brightness.py

import logging
from typing import Any, Dict

from skills.skill_result import SkillResult

from skills.base import Skill
from skills.contract import SkillContract, FailurePolicy
from ir.mission import ExecutionMode
from world.timeline import WorldTimeline
from infrastructure.system_controller import SystemController


logger = logging.getLogger(__name__)


class SetBrightnessSkill(Skill):
    """
    Set display brightness to a percentage.

    Inputs:
        level: Brightness percentage (will be clamped to 0-100)

    Validates and clamps input before calling controller.
    Emits brightness_changed with actual confirmed value.
    """

    contract = SkillContract(
        name="system.set_brightness",
        description="Set display brightness percentage",
        domain="system",
        requires_focus=False,
        resource_cost="low",
        inputs={"level": "brightness_percentage"},
        outputs={"brightness": "actual_brightness"},
        allowed_modes={ExecutionMode.foreground, ExecutionMode.side_effect},
        failure_policy={
            ExecutionMode.foreground: FailurePolicy.FAIL,
            ExecutionMode.side_effect: FailurePolicy.CONTINUE,
        },
        emits_events=["brightness_changed"],
        mutates_world=True,
        idempotent=True,
    )

    def __init__(self, system_controller: SystemController):
        self._controller = system_controller

    def execute(self, inputs: Dict[str, Any], world: WorldTimeline, snapshot=None) -> SkillResult:
        # Defensive parsing — reflex bypasses Cortex, can't trust regex alone
        try:
            level = int(inputs["level"])
        except (ValueError, TypeError) as e:
            raise RuntimeError(f"Invalid brightness level: {inputs.get('level')!r}") from e

        # Clamp to valid range
        original = level
        level = max(0, min(100, level))
        if level != original:
            logger.info("Brightness clamped: %d → %d", original, level)

        result = self._controller.set_brightness(level)

        if not result.success:
            raise RuntimeError(f"Failed to set brightness: {result.error}")

        actual = result.actual_value

        world.emit("skill.system", "brightness_changed", {
            "value": actual,
            "requested": original,
        })

        return SkillResult(
            outputs={"brightness": actual},
            metadata={"entity": f"brightness {actual}%", "domain": "system"},
        )
