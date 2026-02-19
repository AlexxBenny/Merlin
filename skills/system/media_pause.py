# skills/system/media_pause.py

"""
State-aware media PAUSE skill.

Reads WorldSnapshot.state.media to determine current playback state.
Only sends the play/pause toggle key if media IS currently playing.
Idempotent: calling "pause" when already paused is a no-op, not a resume.
"""

from typing import Any, Dict

from skills.skill_result import SkillResult
from skills.base import Skill
from skills.contract import SkillContract, FailurePolicy
from ir.mission import ExecutionMode
from world.timeline import WorldTimeline
from infrastructure.system_controller import SystemController


class MediaPauseSkill(Skill):
    """
    Pause media playback — state-guarded.

    Reads snapshot.state.media to determine current state:
    - If no media session detected → no-op (reason: no_media_session)
    - If already paused → no-op (reason: already_paused)
    - If playing → sends VK_MEDIA_PLAY_PAUSE toggle to pause

    Idempotent: safe to call repeatedly.
    """

    contract = SkillContract(
        name="system.media_pause",
        description="Pause media playback (state-aware, idempotent)",
        domain="system",
        requires_focus=False,
        resource_cost="low",
        inputs={},
        outputs={"changed": "whether_playback_state_was_changed"},
        allowed_modes={ExecutionMode.foreground, ExecutionMode.side_effect},
        failure_policy={
            ExecutionMode.foreground: FailurePolicy.FAIL,
            ExecutionMode.side_effect: FailurePolicy.CONTINUE,
        },
        emits_events=[],
        mutates_world=False,
        idempotent=True,
    )

    def __init__(self, system_controller: SystemController):
        self._controller = system_controller

    def execute(self, inputs: Dict[str, Any], world: WorldTimeline, snapshot=None) -> SkillResult:
        # ── State guard ──────────────────────────────────────
        media = snapshot.state.media if snapshot and snapshot.state else None

        if media is None:
            return SkillResult(
                outputs={"changed": False},
                metadata={"domain": "system", "reason": "no_media_session"},
            )

        if not media.is_playing:
            return SkillResult(
                outputs={"changed": False},
                metadata={"domain": "system", "reason": "already_paused"},
            )

        # ── Action: pause playback ───────────────────────────
        result = self._controller.media_play_pause()

        if not result.success:
            raise RuntimeError(f"Failed to send media key: {result.error}")

        return SkillResult(
            outputs={"changed": True},
            metadata={"domain": "system"},
        )
