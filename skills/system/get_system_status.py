# skills/system/get_system_status.py

"""
GetSystemStatusSkill — Live aggregate system health query.

Returns CPU, RAM, disk, battery, and brightness/volume from live OS reads.
Authoritative sources: psutil (CPU, RAM, disk, battery), SystemController
(brightness, volume).

data_freshness="live" — system telemetry is ephemeral.
The snapshot is passed but NOT used for the primary output.

Note on psutil.cpu_percent():
    The first call with interval=0 returns 0.0 (meaningless).
    We use interval=0.1 to get a real reading. This adds ~100ms latency
    to this skill — acceptable for a diagnostic query.
"""

import logging
from typing import Any, Dict

from skills.skill_result import SkillResult
from skills.base import Skill
from skills.contract import SkillContract, FailurePolicy
from ir.mission import ExecutionMode
from world.timeline import WorldTimeline
from infrastructure.system_controller import SystemController

logger = logging.getLogger(__name__)

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False
    logger.warning("psutil not installed — GetSystemStatusSkill resources will be unknown")


class GetSystemStatusSkill(Skill):
    """
    Get aggregate system status — CPU, RAM, disk, battery, display, audio.

    Reads live from psutil + SystemController. Zero inputs required.
    """

    contract = SkillContract(
        name="system.get_system_status",
        action="get_system_status",
        target_type="system",
        description="Get aggregate system health status",
        intent_verbs=["diagnostics"],
        intent_keywords=["system", "status", "health", "info", "cpu", "ram", "memory", "disk", "diagnostics"],
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
        data_freshness="live",
        output_style="templated",
    )

    def __init__(self, system_controller: SystemController):
        self._controller = system_controller

    def execute(self, inputs: Dict[str, Any], world: WorldTimeline, snapshot=None) -> SkillResult:
        def fmt_pct(val, suffix=""):
            if val is None:
                return "unknown"
            return f"{int(val)}%{suffix}"

        # ── CPU (live) ──
        cpu_str = "unknown"
        if _HAS_PSUTIL:
            try:
                # interval=0.1 avoids the 0.0 first-call problem
                cpu = psutil.cpu_percent(interval=0.1)
                cpu_str = fmt_pct(cpu)
            except Exception as e:
                logger.warning("GetSystemStatus: CPU read failed: %s", e)

        # ── Memory (live) ──
        memory_str = "unknown"
        if _HAS_PSUTIL:
            try:
                mem = psutil.virtual_memory()
                memory_str = fmt_pct(mem.percent)
            except Exception as e:
                logger.warning("GetSystemStatus: memory read failed: %s", e)

        # ── Disk (live) ──
        disk_str = "unknown"
        if _HAS_PSUTIL:
            try:
                disk = psutil.disk_usage("/")
                disk_str = fmt_pct(disk.percent)
            except Exception as e:
                logger.warning("GetSystemStatus: disk read failed: %s", e)

        # ── Battery (live) ──
        battery_str = "no battery"
        if _HAS_PSUTIL:
            try:
                battery = psutil.sensors_battery()
                if battery is not None:
                    charging = ", charging" if battery.power_plugged else ""
                    battery_str = f"{int(battery.percent)}%{charging}"
            except Exception as e:
                logger.warning("GetSystemStatus: battery read failed: %s", e)

        # ── Brightness (live from SystemController) ──
        brightness_str = "unknown"
        try:
            brightness = self._controller.get_brightness()
            brightness_str = fmt_pct(brightness)
        except Exception as e:
            logger.warning("GetSystemStatus: brightness read failed: %s", e)

        # ── Volume (live from SystemController) ──
        volume_str = "unknown"
        try:
            vol, muted = self._controller.get_volume()
            volume_str = fmt_pct(vol)
            if muted:
                volume_str += " (muted)"
        except Exception as e:
            logger.warning("GetSystemStatus: volume read failed: %s", e)

        return SkillResult(
            outputs={
                "cpu": cpu_str,
                "memory": memory_str,
                "disk": disk_str,
                "battery": battery_str,
                "brightness": brightness_str,
                "volume": volume_str,
            },
            metadata={
                "domain": "system",
                "response_template": (
                    "CPU {cpu}, memory {memory}, disk {disk}, "
                    "battery {battery}, brightness {brightness}, volume {volume}."
                ),
            },
        )
