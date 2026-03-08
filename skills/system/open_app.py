# skills/system/open_app.py

from typing import Any, Dict, Optional

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
    Creates an AppSession on success via SessionManager.
    Emits app_launched on success.
    """

    contract = SkillContract(
        name="system.open_app",
        action="open_app",
        target_type="app",
        description="Open an application",
        narration_template="open {app_name}",
        intent_verbs=["open", "launch", "start", "run"],
        intent_keywords=["app", "application", "program"],
        verb_specificity="generic",
        domain="system",
        requires_focus=True,
        inputs={
            "app_name": "application_name",
        },
        entity_params=["app_name"],
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
        output_style="terse",
    )

    def __init__(self, system_controller: SystemController,
                 session_manager=None, app_registry=None):
        self._controller = system_controller
        self._session_mgr = session_manager
        self._app_registry = app_registry  # Optional: for entity-based launch

    def execute(self, inputs: Dict[str, Any], world: WorldTimeline, snapshot=None) -> SkillResult:
        app_name = inputs["app_name"]
        app_id = inputs.get("app_id")  # Injected by EntityResolver (Phase 9C)
        args = inputs.get("args")

        # ── Entity-based launch (preferred) ──
        if app_id and self._app_registry:
            entity = self._app_registry.get(app_id)
            if entity:
                handle = self._controller.launch(entity, args=args)
            else:
                # app_id not in registry — fall back to legacy
                handle = self._controller.open_app(app_name, args=args)
        else:
            # ── Legacy fallback (no entity resolver) ──
            handle = self._controller.open_app(app_name, args=args)

        if not handle.success:
            raise RuntimeError(
                f"Failed to open '{app_name}': {handle.error}"
            )

        # ── Create session handle ──
        session_id = None
        if self._session_mgr is not None:
            session = self._session_mgr.create_app_session(
                app_name=handle.app_name,
                pid=handle.pid,
            )
            session_id = session.id

        world.emit("skill.system", "app_launched", {
            "app": handle.app_name,
            "pid": handle.pid,
        })

        metadata = {"entity": f"app '{handle.app_name}'", "domain": "system"}
        if session_id:
            metadata["session_id"] = session_id

        return SkillResult(
            outputs={"opened": handle.app_name, "pid": handle.pid},
            metadata=metadata,
        )

