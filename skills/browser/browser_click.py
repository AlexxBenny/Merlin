# skills/browser/browser_click.py

"""
BrowserClickSkill — Click a browser entity by index.

Resolves display index → backend_node_id from current snapshot,
then delegates to BrowserController.click().

Follows system.mute pattern: parse → controller → emit → return.
"""

from typing import Any, Dict
import logging

logger = logging.getLogger(__name__)

from runtime.sources.browser import BrowserSource
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
        inputs={},
        optional_inputs={
            "entity_index": "entity_index",
            "entity_ref": "entity_ref",
        },
        input_groups=[{"entity_index", "entity_ref"}],
        outputs={
            "url": "url_string",
            "page_title": "info_string",
        },
        allowed_modes={ExecutionMode.foreground},
        failure_policy={
            ExecutionMode.foreground: FailurePolicy.FAIL,
        },
        emits_events=["browser_entities_refreshed"],
        mutates_world=True,
        output_style="terse",
    )

    def __init__(self, browser_controller):
        self._controller = browser_controller

    def execute(self, inputs: Dict[str, Any], world: WorldTimeline, snapshot=None) -> SkillResult:
        if "entity_index" not in inputs:
            raise RuntimeError(
                "entity_index is required (provide entity_index directly "
                "or entity_ref for resolver to convert)"
            )
        index = int(inputs["entity_index"])
        # Entity text from resolver — used to verify against index drift.
        # If the resolver resolved entity_ref, it sets _resolved_entity_text.
        # If the compiler set entity_index directly, this is empty.
        expected_text = inputs.pop("_resolved_entity_text", "")

        # ALWAYS use a fresh snapshot — cached snapshot may be stale
        # due to DOM mutations (scroll, dynamic content, SPA navigation)
        # between entity resolution and skill execution.
        page_snapshot = self._controller.get_snapshot(cached=False)

        entity = next(
            (e for e in page_snapshot.entities if e.index == index),
            None,
        )

        # Verify: if resolver provided expected text, confirm the entity
        # at this index still matches. If not, the DOM shifted — try
        # to find the right entity by text match.
        if entity and expected_text:
            if expected_text.lower() not in entity.text.lower():
                # Index drifted — search by text instead
                fallback = next(
                    (e for e in page_snapshot.entities
                     if expected_text.lower() in e.text.lower()),
                    None,
                )
                if fallback:
                    logger.info(
                        "[BrowserClick] Index %d drifted (was '%s', now '%s'). "
                        "Found by text at index %d.",
                        index, expected_text[:30],
                        entity.text[:30], fallback.index,
                    )
                    entity = fallback
                    index = fallback.index

        if not entity:
            raise RuntimeError(
                f"No entity at index {index} "
                f"(available: 1–{len(page_snapshot.entities)})"
            )

        result = self._controller.click(entity.backend_node_id)

        if not result.success:
            raise RuntimeError(f"Click failed: {result.error}")

        # Click may cause DOM mutation — refresh snapshot for world state
        post_snap = self._controller.get_snapshot(cached=False)
        world.emit("skill.browser", "browser_entities_refreshed", {
            "url": post_snap.url if post_snap else "",
            "title": post_snap.title if post_snap else "",
            "entity_count": len(post_snap.entities) if post_snap else 0,
            "tab_count": post_snap.tab_count if post_snap else 0,
            "top_entities": (
                BrowserSource._extract_top_entities(post_snap)
                if post_snap else []
            ),
        })

        return SkillResult(
            outputs={
                "url": result.snapshot.url if result.snapshot else "",
                "page_title": result.snapshot.title if result.snapshot else "",
            },
            metadata={"domain": "browser", "entity": f"click entity {index}"},
        )
