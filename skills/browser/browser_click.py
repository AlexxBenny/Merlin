# skills/browser/browser_click.py

"""
BrowserClickSkill — Click a browser entity by index.

Resolves display index → backend_node_id from current snapshot,
then delegates to BrowserController.click().

Follows system.mute pattern: parse → controller → emit → return.
"""

from typing import Any, Dict

from skills.base import Skill
from skills.contract import SkillContract, FailurePolicy
from skills.skill_result import SkillResult
from ir.mission import ExecutionMode
from world.timeline import WorldTimeline


class BrowserClickSkill(Skill):
    """Click a browser entity by its display index."""

    contract = SkillContract(
        name="browser.click",
        action="click",
        target_type="browser_entity",
        description="Click a browser entity",
        narration_template="click entity {entity_index}",
        intent_verbs=["click", "select", "open", "press"],
        intent_keywords=["link", "button", "result", "video", "item", "entity"],
        verb_specificity="specific",
        domain="browser",
        requires_focus=True,
        inputs={
            "entity_index": "entity_index",
        },
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
        index = int(inputs["entity_index"])

        page_snapshot = self._controller.get_snapshot(cached=True)
        entity = next(
            (e for e in page_snapshot.entities if e.index == index),
            None,
        )
        if not entity:
            raise RuntimeError(
                f"No entity at index {index} "
                f"(available: 1–{len(page_snapshot.entities)})"
            )

        result = self._controller.click(entity.backend_node_id)

        if not result.success:
            raise RuntimeError(f"Click failed: {result.error}")

        world.emit("skill.browser", "browser_action_completed", {
            "action": "click",
            "entity_index": index,
            "url": result.snapshot.url if result.snapshot else "",
        })

        return SkillResult(
            outputs={"url": result.snapshot.url if result.snapshot else ""},
            metadata={"domain": "browser", "entity": f"click entity {index}"},
        )
