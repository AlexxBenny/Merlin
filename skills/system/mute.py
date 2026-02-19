# skills/system/mute.py

from typing import Any, Dict

from skills.skill_result import SkillResult

from skills.base import Skill
from skills.contract import SkillContract, FailurePolicy
from ir.mission import ExecutionMode
from world.timeline import WorldTimeline
from infrastructure.system_controller import SystemController


class MuteSkill(Skill):
    """
    Mute system audio.

    No inputs required.
    Emits mute_toggled with actual muted state.
    """

    contract = SkillContract(
        name="system.mute",
        description="Mute system audio",
        domain="system",
        requires_focus=False,
        resource_cost="low",
        inputs={},
        outputs={"muted": "mute_state"},
        allowed_modes={ExecutionMode.foreground, ExecutionMode.side_effect},
        failure_policy={
            ExecutionMode.foreground: FailurePolicy.FAIL,
            ExecutionMode.side_effect: FailurePolicy.CONTINUE,
        },
        emits_events=["mute_toggled"],
        mutates_world=True,
        idempotent=True,
    )

    def __init__(self, system_controller: SystemController):
        self._controller = system_controller

    def execute(self, inputs: Dict[str, Any], world: WorldTimeline, snapshot=None) -> SkillResult:
        result = self._controller.mute()

        if not result.success:
            raise RuntimeError(f"Failed to mute: {result.error}")

        world.emit("skill.system", "mute_toggled", {
            "muted": result.actual_value,
        })

        return SkillResult(
            outputs={"muted": result.actual_value},
            metadata={"domain": "system"},
        )
