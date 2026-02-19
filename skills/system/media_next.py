# skills/system/media_next.py

from typing import Any, Dict

from skills.skill_result import SkillResult

from skills.base import Skill
from skills.contract import SkillContract, FailurePolicy
from ir.mission import ExecutionMode
from world.timeline import WorldTimeline
from infrastructure.system_controller import SystemController


class MediaNextSkill(Skill):
    """
    Send media next track virtual key.

    No inputs required.
    """

    contract = SkillContract(
        name="system.media_next",
        description="Skip to next media track",
        domain="system",
        requires_focus=False,
        resource_cost="low",
        inputs={},
        outputs={"sent": "media_key_sent"},
        allowed_modes={ExecutionMode.foreground, ExecutionMode.side_effect},
        failure_policy={
            ExecutionMode.foreground: FailurePolicy.FAIL,
            ExecutionMode.side_effect: FailurePolicy.CONTINUE,
        },
        emits_events=[],
        mutates_world=False,
        idempotent=False,
    )

    def __init__(self, system_controller: SystemController):
        self._controller = system_controller

    def execute(self, inputs: Dict[str, Any], world: WorldTimeline, snapshot=None) -> SkillResult:
        result = self._controller.media_next()

        if not result.success:
            raise RuntimeError(f"Failed to send media next: {result.error}")

        return SkillResult(
            outputs={"sent": True},
            metadata={"domain": "system"},
        )
