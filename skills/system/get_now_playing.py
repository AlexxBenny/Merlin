# skills/system/get_now_playing.py

"""
GetNowPlayingSkill — Current media query from WorldSnapshot.

Returns currently playing track title, artist, and platform from
MediaState maintained by MediaSource. O(1), zero LLM.
"""

from typing import Any, Dict

from skills.skill_result import SkillResult
from skills.base import Skill
from skills.contract import SkillContract, FailurePolicy
from ir.mission import ExecutionMode
from world.timeline import WorldTimeline


class GetNowPlayingSkill(Skill):
    """
    Get currently playing media info.

    Reads directly from WorldSnapshot — no OS calls, no LLM.
    Zero inputs required.
    """

    contract = SkillContract(
        name="system.get_now_playing",
        action="get_now_playing",
        target_type="media",
        description="Get currently playing media info",
        intent_verbs=["playing", "listening"],
        intent_keywords=["playing", "listening", "song", "track", "now playing", "current"],
        domain="system",
        requires_focus=False,
        resource_cost="low",
        inputs={},
        outputs={
            "title": "info_string",
            "artist": "info_string",
            "platform": "info_string",
            "is_playing": "info_string",
        },
        allowed_modes={ExecutionMode.foreground},
        failure_policy={
            ExecutionMode.foreground: FailurePolicy.FAIL,
        },
        emits_events=[],
        mutates_world=False,
        idempotent=True,
        output_style="templated",
        requires=["media_session_active"],
    )

    def execute(self, inputs: Dict[str, Any], world: WorldTimeline, snapshot=None) -> SkillResult:
        if snapshot is None or snapshot.state.media is None:
            return SkillResult(
                outputs={
                    "title": "nothing",
                    "artist": "unknown",
                    "platform": "unknown",
                    "is_playing": "no",
                },
                metadata={"domain": "system"},
            )

        m = snapshot.state.media
        if not m.is_playing:
            return SkillResult(
                outputs={
                    "title": "nothing playing",
                    "artist": "n/a",
                    "platform": m.platform or "unknown",
                    "is_playing": "no",
                },
                metadata={
                    "domain": "system",
                    "response_template": "Nothing is playing right now.",
                },
            )

        return SkillResult(
            outputs={
                "title": m.title or "unknown track",
                "artist": m.artist or "unknown artist",
                "platform": m.platform or "unknown",
                "is_playing": "yes",
            },
            metadata={
                "domain": "system",
                "response_template": "Now playing {title} by {artist} on {platform}.",
            },
        )
