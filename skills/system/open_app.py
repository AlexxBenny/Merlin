# skills/system/open_app.py

from typing import Any, Dict

from skills.skill_result import SkillResult

from skills.base import Skill
from skills.contract import SkillContract, FailurePolicy
from ir.mission import ExecutionMode
from world.timeline import WorldTimeline
from infrastructure.system_controller import SystemController


class OpenAppSkill(Skill):
    """
    Open an application by name.

    Inputs:
        app_name: Name or command of the app (e.g., "notepad", "chrome")
        args:     Optional list of CLI arguments

    Delegates to SystemController.open_app().
    Emits app_launched on success.
    """

    contract = SkillContract(
        name="system.open_app",
        description="Open an application by name",
        domain="system",
        requires_focus=True,
        inputs={
            "app_name": "application_name",
        },
        optional_inputs={
            "args": "cli_arguments",
        },
        outputs={
            "opened": "application_name",
            "pid": "process_id",
        },
        allowed_modes={ExecutionMode.foreground, ExecutionMode.side_effect},
        failure_policy={
            ExecutionMode.foreground: FailurePolicy.FAIL,
            ExecutionMode.side_effect: FailurePolicy.CONTINUE,
        },
        emits_events=["app_launched"],
        mutates_world=True,
    )

    def __init__(self, system_controller: SystemController):
        self._controller = system_controller

    def execute(self, inputs: Dict[str, Any], world: WorldTimeline, snapshot=None) -> SkillResult:
        app_name = inputs["app_name"]
        args = inputs.get("args")

        handle = self._controller.open_app(app_name, args=args)

        if not handle.success:
            raise RuntimeError(
                f"Failed to open '{app_name}': {handle.error}"
            )

        world.emit("skill.system", "app_launched", {
            "app": handle.app_name,
            "pid": handle.pid,
        })

        return SkillResult(
            outputs={"opened": handle.app_name, "pid": handle.pid},
            metadata={"entity": f"app '{handle.app_name}'", "domain": "system"},
        )
