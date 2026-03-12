# skills/browser/browser_navigate.py

"""
BrowserNavigateSkill — Navigate browser to a URL.

Delegates to BrowserController.navigate().
"""

from typing import Any, Dict

from skills.base import Skill
from skills.contract import SkillContract, FailurePolicy
from skills.skill_result import SkillResult
from ir.mission import ExecutionMode
from world.timeline import WorldTimeline


class BrowserNavigateSkill(Skill):
    """Navigate the browser to a URL."""

    contract = SkillContract(
        name="browser.navigate",
        action="navigate",
        target_type="browser_page",
        description="Navigate browser to a URL",
        narration_template="navigate to {url}",
        intent_verbs=["navigate", "go", "open", "visit"],
        intent_keywords=["website", "page", "site", "url"],
        verb_specificity="generic",
        domain="browser",
        requires_focus=True,
        inputs={
            "url": "url_string",
        },
        outputs={
            "final_url": "url_string",
            "page_title": "info_string",
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
        url = inputs["url"]

        if not url:
            raise RuntimeError("No URL provided")

        result = self._controller.navigate(url)

        if not result.success:
            raise RuntimeError(f"Navigation failed: {result.error}")

        final_url = result.snapshot.url if result.snapshot else url
        page_title = result.snapshot.title if result.snapshot else ""

        world.emit("skill.browser", "browser_action_completed", {
            "action": "navigate",
            "url": final_url,
        })

        return SkillResult(
            outputs={"final_url": final_url, "page_title": page_title},
            metadata={"domain": "browser", "entity": f"navigate {url[:60]}"},
        )
