# skills/system/focus_app.py

from typing import Any, Dict

from skills.skill_result import SkillResult

from skills.base import Skill
from skills.contract import SkillContract, FailurePolicy
from ir.mission import ExecutionMode
from world.timeline import WorldTimeline
from infrastructure.system_controller import SystemController


class FocusAppSkill(Skill):
    """
    Bring an application window to the foreground.

    Inputs:
        app_name: Name of the app to focus

    Delegates to SystemController.focus_app().
    Updates AppSession focus state and pushes SessionStack.
    Emits app_focused on success.
    """

    contract = SkillContract(
        name="system.focus_app",
        action="focus_app",
        target_type="app",
        description="Focus an application",
        narration_template="focus {app_name}",
        intent_verbs=["focus", "switch", "bring"],
        intent_keywords=["app", "application", "window"],
        verb_specificity="generic",
        domain="system",
        requires_focus=True,
        inputs={"app_name": "application_name"},
        entity_params=["app_name"],
        optional_inputs={"app_id": "canonical_entity_id"},  # Injected by EntityResolver
        outputs={"focused": "application_name"},
        allowed_modes={ExecutionMode.foreground, ExecutionMode.side_effect},
        failure_policy={
            ExecutionMode.foreground: FailurePolicy.FAIL,
            ExecutionMode.side_effect: FailurePolicy.IGNORE,
        },
        emits_events=["app_focused"],
        mutates_world=True,
        output_style="terse",
        requires=["app_running"],
        produces=["app_focused"],
        effect_type="create",
    )

    def __init__(self, system_controller: SystemController,
                 session_manager=None):
        self._controller = system_controller
        self._session_mgr = session_manager

    def execute(self, inputs: Dict[str, Any], world: WorldTimeline, snapshot=None) -> SkillResult:
        app_name = inputs["app_name"]

        success = self._controller.focus_app(app_name)

        if not success:
            raise RuntimeError(
                f"Failed to focus '{app_name}': OS denied focus change "
                f"or no matching window found"
            )

        # ── Update session state ──
        if self._session_mgr is not None:
            session = self._session_mgr.get_session_by_app(app_name)
            if session:
                # Clear focus on all other app sessions
                for s in self._session_mgr.get_sessions_by_type(
                    session.type
                ):
                    if s.id != session.id and getattr(s, "is_focused", False):
                        self._session_mgr.update_session(
                            s.id, is_focused=False,
                        )
                # Set focus on this session
                self._session_mgr.update_session(
                    session.id, is_focused=True,
                )
                # Push to session stack
                self._session_mgr.session_stack.push(session.id)

        world.emit("skill.system", "app_focused", {
            "app": app_name,
        })

        return SkillResult(
            outputs={"focused": app_name},
            metadata={"entity": f"app '{app_name}'", "domain": "system"},
        )
