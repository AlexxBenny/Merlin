# skills/system/get_battery.py

"""
GetBatterySkill — Live battery query.

Returns battery percentage, charging status, and health classification
from psutil.sensors_battery(). Authoritative source: OS sensor.

data_freshness="live" — battery is ephemeral telemetry.
The snapshot is passed but NOT used for the primary output.
"""

import logging
from typing import Any, Dict

from skills.skill_result import SkillResult
from skills.base import Skill
from skills.contract import SkillContract, FailurePolicy
from ir.mission import ExecutionMode
from world.timeline import WorldTimeline

logger = logging.getLogger(__name__)

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False
    logger.warning("psutil not installed — GetBatterySkill will return unknown")


class GetBatterySkill(Skill):
    """
    Get current battery status.

    Reads live from psutil.sensors_battery() — never stale.
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
        data_freshness="live",
        output_style="templated",
    )

    def execute(self, inputs: Dict[str, Any], world: WorldTimeline, snapshot=None) -> SkillResult:
        if not _HAS_PSUTIL:
            return SkillResult(
                outputs={"percent": "unknown", "charging": "unknown", "status": "psutil not installed"},
                metadata={"domain": "system"},
            )

        battery = psutil.sensors_battery()

        if battery is None:
            return SkillResult(
                outputs={"percent": "unknown", "charging": "unknown", "status": "no battery detected"},
                metadata={
                    "domain": "system",
                    "response_template": "No battery detected.",
                },
            )

        percent = battery.percent
        charging = battery.power_plugged

        if charging:
            status = "charging"
        elif percent <= 10:
            status = "critical"
        elif percent <= 20:
            status = "low"
        else:
            status = "normal"

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
