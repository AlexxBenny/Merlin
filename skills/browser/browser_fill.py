# skills/browser/browser_fill.py

"""
BrowserFillSkill — Fill a browser input field by entity index.

Resolves display index → backend_node_id from current snapshot,
then delegates to BrowserController.fill().
"""

from typing import Any, Dict

from skills.base import Skill
from skills.contract import SkillContract, FailurePolicy
from skills.skill_result import SkillResult
from ir.mission import ExecutionMode
from world.timeline import WorldTimeline


class BrowserFillSkill(Skill):
    """Fill a browser input field with text."""

    contract = SkillContract(
        name="browser.fill",
        action="fill",
        target_type="browser_entity",
        description="Fill a browser input field",
        narration_template="fill entity {entity_index} with text",
        intent_verbs=["fill", "type", "enter", "write"],
        intent_keywords=["input", "field", "text", "search", "box"],
        verb_specificity="generic",
        domain="browser",
        requires_focus=True,
        inputs={
            "entity_index": "entity_index",
            "text": "fill_text",
        },
        outputs={
            "filled": "boolean",
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
        text = inputs["text"]

        if not text:
            raise RuntimeError("No text provided for fill")

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

        result = self._controller.fill(entity.backend_node_id, text)

        if not result.success:
            raise RuntimeError(f"Fill failed: {result.error}")

        world.emit("skill.browser", "browser_action_completed", {
            "action": "fill",
            "entity_index": index,
        })

        return SkillResult(
            outputs={"filled": True},
            metadata={"domain": "browser", "entity": f"fill entity {index}"},
        )
