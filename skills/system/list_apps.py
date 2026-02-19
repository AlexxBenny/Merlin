# skills/system/list_apps.py

from typing import Any, Dict

from skills.skill_result import SkillResult

from skills.base import Skill
from skills.contract import SkillContract, FailurePolicy
from ir.mission import ExecutionMode
from world.timeline import WorldTimeline
from infrastructure.system_controller import SystemController


class ListAppsSkill(Skill):
    """
    List currently running visible applications.

    No inputs required.

    Delegates to SystemController.list_running_apps().
    Read-only: does NOT mutate world.
    """

    contract = SkillContract(
        name="system.list_apps",
        action="list_apps",
        target_type="app",
        description="List currently running visible applications",
        domain="system",
        requires_focus=False,
        inputs={},
        outputs={"apps": "application_list"},
        allowed_modes={ExecutionMode.foreground},
        failure_policy={
            ExecutionMode.foreground: FailurePolicy.FAIL,
        },
        emits_events=[],
        mutates_world=False,
        idempotent=True,
    )

    def __init__(self, system_controller: SystemController):
        self._controller = system_controller

    def execute(self, inputs: Dict[str, Any], world: WorldTimeline, snapshot=None) -> SkillResult:
        apps = self._controller.list_running_apps()

        app_list = [
            {"name": app.name, "pid": app.pid, "title": app.title}
            for app in apps
        ]

        return SkillResult(
            outputs={"apps": app_list},
            metadata={"entity": "running apps", "domain": "system"},
        )
