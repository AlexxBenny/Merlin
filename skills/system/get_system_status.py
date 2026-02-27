# skills/system/get_system_status.py

"""
GetSystemStatusSkill — Aggregate system health from WorldSnapshot.

Returns CPU, RAM, disk, battery, and brightness/volume in one snapshot.
Reads from SystemState maintained by SystemSource. O(1), zero LLM.
"""

from typing import Any, Dict

from skills.skill_result import SkillResult
from skills.base import Skill
from skills.contract import SkillContract, FailurePolicy
from ir.mission import ExecutionMode
from world.timeline import WorldTimeline


class GetSystemStatusSkill(Skill):
    """
    Get aggregate system status — CPU, RAM, disk, battery, display, audio.

    Reads directly from WorldSnapshot. Zero inputs required.
    """

    contract = SkillContract(
        name="system.get_system_status",
        action="get_system_status",
        target_type="system",
        description="Get aggregate system health status",
        intent_verbs=["status", "health", "info", "diagnostics"],
        intent_keywords=["system", "status", "health", "cpu", "ram", "memory", "disk", "diagnostics"],
        domain="system",
        requires_focus=False,
        resource_cost="low",
        inputs={},
        outputs={
            "cpu": "info_string",
            "memory": "info_string",
            "disk": "info_string",
            "battery": "info_string",
            "brightness": "info_string",
            "volume": "info_string",
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
                outputs={k: "unknown" for k in self.contract.outputs},
                metadata={"domain": "system"},
            )

        sys = snapshot.state.system
        res = sys.resources
        hw = sys.hardware

        def fmt_pct(val, suffix=""):
            if val is None:
                return "unknown"
            return f"{int(val)}%{suffix}"

        cpu = fmt_pct(res.cpu_percent)
        if res.cpu_status and res.cpu_status != "normal":
            cpu += f" ({res.cpu_status})"

        memory = fmt_pct(res.memory_percent)
        if res.memory_status and res.memory_status != "normal":
            memory += f" ({res.memory_status})"

        disk = fmt_pct(res.disk_percent)

        battery = "no battery"
        if hw.battery_percent is not None:
            charging = ", charging" if hw.battery_charging else ""
            battery = f"{int(hw.battery_percent)}%{charging}"

        brightness = fmt_pct(hw.brightness_percent)
        volume = fmt_pct(hw.volume_percent)
        if hw.muted:
            volume += " (muted)"

        return SkillResult(
            outputs={
                "cpu": cpu,
                "memory": memory,
                "disk": disk,
                "battery": battery,
                "brightness": brightness,
                "volume": volume,
            },
            metadata={
                "domain": "system",
                "response_template": (
                    "CPU {cpu}, memory {memory}, disk {disk}, "
                    "battery {battery}, brightness {brightness}, volume {volume}."
                ),
            },
        )
