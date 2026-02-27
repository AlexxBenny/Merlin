# skills/system/get_battery.py

"""
GetBatterySkill — Deterministic battery query from WorldSnapshot.

Returns battery percentage, charging status, and health classification
from HardwareState maintained by SystemSource. O(1), zero LLM.

Battery is sensor data — it must never go through an LLM.
"""

from typing import Any, Dict

from skills.skill_result import SkillResult
from skills.base import Skill
from skills.contract import SkillContract, FailurePolicy
from ir.mission import ExecutionMode
from world.timeline import WorldTimeline


class GetBatterySkill(Skill):
    """
    Get current battery status.

    Reads directly from WorldSnapshot — no OS calls, no LLM.
    Zero inputs required.
    """

    contract = SkillContract(
        name="system.get_battery",
        action="get_battery",
        target_type="battery",
        description="Get battery percentage and charging status",
        intent_verbs=["battery", "charge", "power"],
        intent_keywords=["battery", "charge", "charging", "power"],
        domain="system",
        requires_focus=False,
        resource_cost="low",
        inputs={},
        outputs={
            "percent": "info_string",
            "charging": "info_string",
            "status": "info_string",
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
        if snapshot is None:
            return SkillResult(
                outputs={"percent": "unknown", "charging": "unknown", "status": "unknown"},
                metadata={"domain": "system"},
            )

        hw = snapshot.state.system.hardware
        percent = hw.battery_percent
        charging = hw.battery_charging
        status = hw.battery_status or "unknown"

        if percent is None:
            return SkillResult(
                outputs={"percent": "unknown", "charging": "unknown", "status": "no battery detected"},
                metadata={
                    "domain": "system",
                    "response_template": "No battery detected.",
                },
            )

        charging_text = ", charging" if charging else ""
        return SkillResult(
            outputs={
                "percent": f"{int(percent)}%",
                "charging": "yes" if charging else "no",
                "status": status,
            },
            metadata={
                "domain": "system",
                "response_template": f"Battery is at {{percent}}{charging_text}.",
            },
        )
