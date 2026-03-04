# skills/system/toggle_nightlight.py

from typing import Any, Dict

from skills.skill_result import SkillResult

from skills.base import Skill
from skills.contract import SkillContract, FailurePolicy
from ir.mission import ExecutionMode
from world.timeline import WorldTimeline
from infrastructure.system_controller import SystemController


class ToggleNightlightSkill(Skill):
    """
    Toggle Windows Night Light on/off.

    No inputs required.
    Emits nightlight_toggled with actual enabled state.

    Note: Night Light behavior varies by Windows version.
    This is best-effort — controller returns actual state
    from registry read after toggle.
    """

    contract = SkillContract(
        name="system.toggle_nightlight",
        action="toggle_nightlight",
        target_type="display",
        description="Toggle night light",
        intent_verbs=["toggle", "turn", "switch"],
        intent_keywords=["nightlight", "night light", "blue light"],
        verb_specificity="generic",
        domain="system",
        requires_focus=False,
        resource_cost="low",
        inputs={},
        outputs={"enabled": "nightlight_state"},
        allowed_modes={ExecutionMode.foreground, ExecutionMode.side_effect},
        failure_policy={
            ExecutionMode.foreground: FailurePolicy.FAIL,
            ExecutionMode.side_effect: FailurePolicy.CONTINUE,
        },
        emits_events=["nightlight_toggled"],
        mutates_world=True,
        idempotent=False,  # Toggle is NOT idempotent
        output_style="terse",
    )

    def __init__(self, system_controller: SystemController):
        self._controller = system_controller

    def execute(self, inputs: Dict[str, Any], world: WorldTimeline, snapshot=None) -> SkillResult:
        result = self._controller.toggle_nightlight()

        if not result.success:
            raise RuntimeError(f"Failed to toggle night light: {result.error}")

        state_str = "enabled" if result.actual_value else "disabled"

        world.emit("skill.system", "nightlight_toggled", {
            "enabled": result.actual_value,
            "state": state_str,
        })

        return SkillResult(
            outputs={"enabled": result.actual_value},
            metadata={"entity": f"night light {state_str}", "domain": "system"},
        )
