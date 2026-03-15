# skills/browser/browser_select_result.py

"""
BrowserSelectResultSkill — Tier 2 reactive controller.

Selects the nth result from a page using structural DOM clustering.
No LLM — purely deterministic via repeated DOM structure detection.

Example queries:
    "play the first video"    → ordinal=1
    "open the second article" → ordinal=2
    "click the third result"  → ordinal=3

The controller handles all DOM perception:
    detect_semantic_groups() → structural clustering
    click_nth_result()       → scroll + click backend_node_id
"""

import logging
import time
from typing import Any, Dict

from runtime.sources.browser import BrowserSource
from skills.base import Skill
from skills.contract import SkillContract, FailurePolicy
from skills.skill_result import SkillResult
from ir.mission import ExecutionMode
from world.timeline import WorldTimeline

logger = logging.getLogger(__name__)

# Ordinal text → integer mapping
ORDINAL_MAP = {
    "first": 1, "1st": 1,
    "second": 2, "2nd": 2,
    "third": 3, "3rd": 3,
    "fourth": 4, "4th": 4,
    "fifth": 5, "5th": 5,
    "sixth": 6, "6th": 6,
    "seventh": 7, "7th": 7,
    "eighth": 8, "8th": 8,
    "ninth": 9, "9th": 9,
    "tenth": 10, "10th": 10,
    "last": -1,
    "top": 1,
}


class BrowserSelectResultSkill(Skill):
    """Select the nth result on a page — reactive, no LLM.

    Uses structural DOM clustering to detect repeated layout patterns
    (video cards, search results, product tiles) and clicks the nth one.
    """

    contract = SkillContract(
        name="browser.select_result",
        action="select_result",
        target_type="browser_page",
        description="Select the nth result/item on page",
        narration_template="select result #{ordinal}",
        intent_verbs=["play", "open", "click", "select", "watch"],
        intent_keywords=[
            "first", "second", "third", "fourth", "fifth",
            "video", "result", "article", "item", "product",
            "link", "post", "top",
        ],
        verb_specificity="specific",
        domain="browser",
        requires_focus=True,
        inputs={
            "ordinal": "entity_index",
        },
        optional_inputs={
            "hint_text": "search_text",
        },
        outputs={
            "url": "url_string",
            "page_title": "info_string",
        },
        allowed_modes={ExecutionMode.foreground},
        failure_policy={
            ExecutionMode.foreground: FailurePolicy.FAIL,
        },
        emits_events=["browser_page_changed"],
        mutates_world=True,
        output_style="terse",
    )

    def __init__(self, browser_controller):
        self._controller = browser_controller

    def execute(self, inputs: Dict[str, Any], world: WorldTimeline, snapshot=None) -> SkillResult:
        ordinal = inputs.get("ordinal", 1)

        # Handle string ordinals ("first", "second", etc.)
        if isinstance(ordinal, str):
            ordinal = ORDINAL_MAP.get(ordinal.lower(), None)
            if ordinal is None:
                try:
                    ordinal = int(ordinal)
                except (ValueError, TypeError):
                    raise RuntimeError(
                        f"Could not parse ordinal: {inputs.get('ordinal')}"
                    )

        hint_text = inputs.get("hint_text", None)

        logger.info(
            "[SELECT_RESULT] ordinal=%d hint_text=%s",
            ordinal, repr(hint_text),
        )

        # Use controller's structural clustering
        result = self._controller.click_nth_result(
            ordinal=ordinal,
            hint_text=hint_text,
        )

        if not result.success:
            raise RuntimeError(
                f"click_nth_result failed: {result.error}"
            )

        # Get final snapshot
        final_snapshot = self._controller.get_snapshot(cached=False)
        final_url = final_snapshot.url
        page_title = final_snapshot.title

        # Emit world state event
        world.emit("skill.browser", "browser_page_changed", {
            "url": final_url,
            "title": page_title,
            "entity_count": len(final_snapshot.entities),
            "tab_count": final_snapshot.tab_count,
            "top_entities": (
                BrowserSource._extract_top_entities(final_snapshot)
            ),
        })

        return SkillResult(
            outputs={"url": final_url, "page_title": page_title},
            metadata={
                "domain": "browser",
                "entity": f"result #{ordinal}",
            },
        )
