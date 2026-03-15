# skills/browser/browser_find_element.py

"""
BrowserFindElementSkill — Generic element location with multi-strategy scoring.

Tier 2 reactive controller: finds any element on the page using
CSS selectors, ARIA roles, text matching, and DOM heuristics.
Returns the element's index and backend_node_id for subsequent skills.

No LLM — uses deterministic multi-strategy scoring.
"""

from typing import Any, Dict

from skills.base import Skill
from skills.contract import SkillContract, FailurePolicy
from skills.skill_result import SkillResult
from ir.mission import ExecutionMode
from world.timeline import WorldTimeline

import logging
logger = logging.getLogger(__name__)


class BrowserFindElementSkill(Skill):
    """Find an element on the page — multi-strategy, no LLM."""

    contract = SkillContract(
        name="browser.find_element",
        action="find_element",
        target_type="browser_page",
        description="Find element on the page",
        narration_template="find {text} element",
        intent_verbs=["find", "locate"],
        intent_keywords=["element", "button", "input", "link"],
        verb_specificity="specific",
        domain="browser",
        requires_focus=True,
        inputs={},
        optional_inputs={
            "text": "search_text",
            "role": "aria_role",
            "css": "css_selector",
            "element_type": "element_type",
        },
        input_groups=[{"text", "role", "css", "element_type"}],
        outputs={
            "entity_index": "entity_index",
            "backend_node_id": "node_id",
            "entity_type": "element_type",
            "entity_text": "info_string",
        },
        allowed_modes={ExecutionMode.foreground},
        failure_policy={
            ExecutionMode.foreground: FailurePolicy.FAIL,
        },
        output_style="terse",
    )

    def __init__(self, browser_controller):
        self._controller = browser_controller

    def execute(self, inputs: Dict[str, Any], world: WorldTimeline, snapshot=None) -> SkillResult:
        text = inputs.get("text")
        role = inputs.get("role")
        css = inputs.get("css")
        element_type = inputs.get("element_type")

        if not any([text, role, css, element_type]):
            raise RuntimeError(
                "At least one of text, role, css, or element_type required"
            )

        scored = self._controller.locate_element(
            text=text, role=role, css=css, element_type=element_type,
        )

        if scored is None:
            criteria = ", ".join(
                f"{k}='{v}'" for k, v in [
                    ("text", text), ("role", role),
                    ("css", css), ("type", element_type),
                ] if v
            )
            raise RuntimeError(
                f"Element not found matching: {criteria}"
            )

        entity = scored.entity
        logger.info(
            "[FIND] Found %s idx=%d text='%s' (score=%.1f, strategy=%s)",
            entity.entity_type, entity.index,
            entity.text[:40], scored.score, scored.strategy,
        )

        return SkillResult(
            outputs={
                "entity_index": entity.index,
                "backend_node_id": entity.backend_node_id,
                "entity_type": entity.entity_type,
                "entity_text": entity.text,
            },
            metadata={
                "domain": "browser",
                "entity": f"found {entity.entity_type}: '{entity.text[:30]}'",
                "score": scored.score,
                "strategy": scored.strategy,
            },
        )
