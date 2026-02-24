# skills/system/set_volume.py

import logging
from typing import Any, Dict

from skills.skill_result import SkillResult

from skills.base import Skill
from skills.contract import SkillContract, FailurePolicy
from ir.mission import ExecutionMode
from world.timeline import WorldTimeline
from infrastructure.system_controller import SystemController


logger = logging.getLogger(__name__)


class SetVolumeSkill(Skill):
    """
    Set system master volume to a percentage.

    Inputs:
        level: Volume percentage (will be clamped to 0-100)

    Validates and clamps input before calling controller.
    Emits volume_changed with actual confirmed value.
    """

    contract = SkillContract(
        name="system.set_volume",
        action="set_volume",
        target_type="volume",
        description="Adjust the volume",
        narration_template="set volume to {level}%",
        intent_verbs=["set", "adjust", "change"],
        intent_keywords=["volume", "sound", "audio", "loudness"],
        domain="system",
        requires_focus=False,
        resource_cost="low",
        inputs={"level": "volume_percentage"},
        outputs={"volume": "actual_volume"},
        allowed_modes={ExecutionMode.foreground, ExecutionMode.side_effect},
        failure_policy={
            ExecutionMode.foreground: FailurePolicy.FAIL,
            ExecutionMode.side_effect: FailurePolicy.CONTINUE,
        },
        emits_events=["volume_changed"],
        mutates_world=True,
        idempotent=True,
    )

    def __init__(self, system_controller: SystemController):
        self._controller = system_controller

    def execute(self, inputs: Dict[str, Any], world: WorldTimeline, snapshot=None) -> SkillResult:
        # Defensive parsing — reflex bypasses Cortex
        try:
            level = int(inputs["level"])
        except (ValueError, TypeError) as e:
            raise RuntimeError(f"Invalid volume level: {inputs.get('level')!r}") from e

        # Clamp to valid range
        original = level
        level = max(0, min(100, level))
        if level != original:
            logger.info("Volume clamped: %d → %d", original, level)

        result = self._controller.set_volume(level)

        if not result.success:
            raise RuntimeError(f"Failed to set volume: {result.error}")

        actual = result.actual_value

        world.emit("skill.system", "volume_changed", {
            "value": actual,
            "requested": original,
        })

        return SkillResult(
            outputs={"volume": actual},
            metadata={"entity": f"volume {actual}%", "domain": "system"},
        )
