# skills/browser/browser_go_forward.py

"""
BrowserGoForwardSkill — Go forward in browser history.

Delegates to BrowserController.go_forward().
Zero inputs — same pattern as system.mute.
"""

from typing import Any, Dict

from skills.base import Skill
from skills.contract import SkillContract, FailurePolicy
from skills.skill_result import SkillResult
from ir.mission import ExecutionMode
from world.timeline import WorldTimeline


class BrowserGoForwardSkill(Skill):
    """Go forward in browser history."""

    contract = SkillContract(
        name="browser.go_forward",
        action="go_forward",
        target_type="browser_page",
        description="Go forward in browser history",
        narration_template="go forward",
        intent_verbs=["forward", "go forward", "next"],
        intent_keywords=["browser", "page", "forward"],
        verb_specificity="specific",
        domain="browser",
        requires_focus=True,
        inputs={},
        outputs={
            "url": "url_string",
        },
        allowed_modes={ExecutionMode.foreground},
        failure_policy={
            ExecutionMode.foreground: FailurePolicy.FAIL,
        },
        emits_events=["browser_action_completed"],
        mutates_world=True,
        output_style="terse",
    )

    def __init__(self, browser_controller):
        self._controller = browser_controller

    def execute(self, inputs: Dict[str, Any], world: WorldTimeline, snapshot=None) -> SkillResult:
        result = self._controller.go_forward()

        if not result.success:
            raise RuntimeError(f"Go forward failed: {result.error}")

        url = result.snapshot.url if result.snapshot else ""

        world.emit("skill.browser", "browser_action_completed", {
            "action": "go_forward",
            "url": url,
        })

        return SkillResult(
            outputs={"url": url},
            metadata={"domain": "browser", "entity": "go forward"},
        )
