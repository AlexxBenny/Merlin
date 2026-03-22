# skills/system/media_previous.py

from typing import Any, Dict

from skills.skill_result import SkillResult

from skills.base import Skill
from skills.contract import SkillContract, FailurePolicy
from ir.mission import ExecutionMode
from world.timeline import WorldTimeline
from infrastructure.system_controller import SystemController


class MediaPreviousSkill(Skill):
    """
    Send media previous track virtual key.

    No inputs required.
    """

    contract = SkillContract(
        name="system.media_previous",
        action="media_previous",
        target_type="media",
        description="Go to the previous track",
        intent_verbs=["previous", "prev", "back", "last"],
        intent_keywords=["track", "song", "music"],
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
        output_style="terse",
        requires=["media_session_active"],
    )

    def __init__(self, system_controller: SystemController):
        self._controller = system_controller

    def execute(self, inputs: Dict[str, Any], world: WorldTimeline, snapshot=None) -> SkillResult:
        result = self._controller.media_previous()

        if not result.success:
            raise RuntimeError(f"Failed to send media previous: {result.error}")

        return SkillResult(
            outputs={"sent": True},
            metadata={"domain": "system"},
        )
