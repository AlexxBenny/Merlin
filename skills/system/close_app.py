# skills/system/close_app.py

from typing import Any, Dict

from skills.skill_result import SkillResult

from skills.base import Skill
from skills.contract import SkillContract, FailurePolicy
from ir.mission import ExecutionMode
from world.timeline import WorldTimeline
from infrastructure.system_controller import SystemController


class CloseAppSkill(Skill):
    """
    Request an application to close gracefully.

    Inputs:
        app_name: Name of the app to close

    Delegates to SystemController.close_app() (sends WM_CLOSE).
    Emits app_closed on success.
    """

    contract = SkillContract(
        name="system.close_app",
        action="close_app",
        target_type="app",
        description="Close an application",
        narration_template="close {app_name}",
        intent_verbs=["close", "quit", "exit", "kill"],
        intent_keywords=["app", "application", "program"],
        domain="system",
        requires_focus=True,
        inputs={"app_name": "application_name"},
        outputs={"closed": "application_name"},
        allowed_modes={ExecutionMode.foreground, ExecutionMode.side_effect},
        failure_policy={
            ExecutionMode.foreground: FailurePolicy.FAIL,
            ExecutionMode.side_effect: FailurePolicy.IGNORE,
        },
        emits_events=["app_closed"],
        mutates_world=True,
    )

    def __init__(self, system_controller: SystemController):
        self._controller = system_controller

    def execute(self, inputs: Dict[str, Any], world: WorldTimeline, snapshot=None) -> SkillResult:
        app_name = inputs["app_name"]

        success = self._controller.close_app(app_name)

        if not success:
            raise RuntimeError(f"Failed to close '{app_name}': no matching window found")

        world.emit("skill.system", "app_closed", {
            "app": app_name,
        })

        return SkillResult(
            outputs={"closed": app_name},
            metadata={"entity": f"app '{app_name}'", "domain": "system"},
        )
