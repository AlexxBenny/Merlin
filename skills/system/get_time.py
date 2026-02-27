# skills/system/get_time.py

"""
GetTimeSkill — Deterministic clock query from WorldSnapshot.

Returns current time, date, and day of week from the TimeState
maintained by TimeSource. O(1), zero LLM, zero OS calls.

Time is sensor data — it must never go through an LLM.
"""

from typing import Any, Dict

from skills.skill_result import SkillResult
from skills.base import Skill
from skills.contract import SkillContract, FailurePolicy
from ir.mission import ExecutionMode
from world.timeline import WorldTimeline


class GetTimeSkill(Skill):
    """
    Get current time, date, and day of week.

    Reads directly from WorldSnapshot — no OS calls, no LLM.
    Zero inputs required.
    """

    contract = SkillContract(
        name="system.get_time",
        action="get_time",
        target_type="time",
        description="Get current time and date",
        intent_verbs=["time", "date", "day", "clock"],
        intent_keywords=["time", "clock", "date", "day"],
        domain="system",
        requires_focus=False,
        resource_cost="low",
        inputs={},
        outputs={
            "time": "info_string",
            "date": "info_string",
            "day": "info_string",
        },
        allowed_modes={ExecutionMode.foreground},
        failure_policy={
            ExecutionMode.foreground: FailurePolicy.FAIL,
        },
        emits_events=[],
        mutates_world=False,
        idempotent=True,
    )

    def execute(self, inputs: Dict[str, Any], world: WorldTimeline, snapshot=None) -> SkillResult:
        if snapshot is None or snapshot.state.time is None:
            return SkillResult(
                outputs={"time": "unknown", "date": "unknown", "day": "unknown"},
                metadata={
                    "domain": "system",
                    "response_template": "I don't have the time right now.",
                },
            )

        t = snapshot.state.time
        # Format time as human-readable (e.g., "9:05 PM")
        hour_12 = t.hour % 12 or 12
        ampm = "AM" if t.hour < 12 else "PM"
        time_str = f"{hour_12}:{t.minute:02d} {ampm}"

        return SkillResult(
            outputs={
                "time": time_str,
                "date": t.date or "unknown",
                "day": t.day_of_week or "unknown",
            },
            metadata={
                "domain": "system",
                "response_template": "It's {time}, {day}, {date}.",
            },
        )
