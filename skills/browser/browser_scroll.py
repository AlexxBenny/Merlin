# skills/browser/browser_scroll.py

"""
BrowserScrollSkill — Scroll the browser page up or down.

Delegates to BrowserController.scroll_page().
"""

from typing import Any, Dict

from skills.base import Skill
from skills.contract import SkillContract, FailurePolicy
from skills.skill_result import SkillResult
from ir.mission import ExecutionMode
from world.timeline import WorldTimeline


class BrowserScrollSkill(Skill):
    """Scroll the browser page."""

    contract = SkillContract(
        name="browser.scroll",
        action="scroll",
        target_type="browser_page",
        description="Scroll the browser page",
        narration_template="scroll {direction}",
        intent_verbs=["scroll"],
        intent_keywords=["page", "down", "up"],
        verb_specificity="specific",
        domain="browser",
        requires_focus=True,
        inputs={
            "direction": "scroll_direction",
        },
        outputs={
            "scrolled": "boolean",
        },
        allowed_modes={ExecutionMode.foreground},
        failure_policy={
            ExecutionMode.foreground: FailurePolicy.FAIL,
        },
        emits_events=[],
        mutates_world=False,
        output_style="terse",
    )

    def __init__(self, browser_controller):
        self._controller = browser_controller

    def execute(self, inputs: Dict[str, Any], world: WorldTimeline, snapshot=None) -> SkillResult:
        direction = str(inputs.get("direction", "down")).lower()
        if direction not in ("up", "down"):
            direction = "down"

        result = self._controller.scroll_page(direction)

        if not result.success:
            raise RuntimeError(f"Scroll failed: {result.error}")

        return SkillResult(
            outputs={"scrolled": True},
            metadata={"domain": "browser", "entity": f"scroll {direction}"},
        )
