# skills/system/get_time.py

"""
GetTimeSkill — Live clock query.

Returns current time, date, and day of week from datetime.now().
O(1), zero LLM, zero latency. Authoritative source: stdlib.

data_freshness="live" — time is ephemeral telemetry.
The snapshot is passed but NOT used for the primary output.
"""

from datetime import datetime
from typing import Any, Dict

from skills.skill_result import SkillResult
from skills.base import Skill
from skills.contract import SkillContract, FailurePolicy
from ir.mission import ExecutionMode
from world.timeline import WorldTimeline


class GetTimeSkill(Skill):
    """
    Get current time, date, and day of week.

    Reads live from datetime.now() — never stale.
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
        data_freshness="live",
        output_style="templated",
    )

    def execute(self, inputs: Dict[str, Any], world: WorldTimeline, snapshot=None) -> SkillResult:
        dt = datetime.now()

        hour_12 = dt.hour % 12 or 12
        ampm = "AM" if dt.hour < 12 else "PM"
        time_str = f"{hour_12}:{dt.minute:02d} {ampm}"

        return SkillResult(
            outputs={
                "time": time_str,
                "date": dt.strftime("%Y-%m-%d"),
                "day": dt.strftime("%A"),
            },
            metadata={
                "domain": "system",
                "response_template": "It's {time}, {day}, {date}.",
            },
        )
