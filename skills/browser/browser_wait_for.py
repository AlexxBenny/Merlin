# skills/browser/browser_wait_for.py

"""
BrowserWaitForSkill — Wait for a DOM condition.

Tier 2 reactive primitive: polls DOM until a CSS-matched element
appears, or until timeout. Used between other skills to handle
async page updates (AJAX, lazy loading, SPA transitions).
"""

from typing import Any, Dict

from runtime.sources.browser import BrowserSource
from skills.base import Skill
from skills.contract import SkillContract, FailurePolicy
from skills.skill_result import SkillResult
from ir.mission import ExecutionMode
from world.timeline import WorldTimeline

import logging
logger = logging.getLogger(__name__)


class BrowserWaitForSkill(Skill):
    """Wait for a DOM element to appear — no LLM."""

    contract = SkillContract(
        name="browser.wait_for",
        action="wait_for",
        target_type="browser_page",
        description="Wait for element to appear",
        narration_template="wait for {css_selector}",
        intent_verbs=["wait"],
        intent_keywords=["element", "appear", "load"],
        verb_specificity="specific",
        domain="browser",
        requires_focus=True,
        inputs={},
        optional_inputs={
            "css_selector": "css_selector",
            "text": "search_text",
            "timeout": "seconds",
        },
        input_groups=[{"css_selector", "text"}],
        outputs={
            "found": "boolean",
            "entity_index": "entity_index",
        },
        allowed_modes={ExecutionMode.foreground},
        failure_policy={
            ExecutionMode.foreground: FailurePolicy.CONTINUE,
        },
        emits_events=["browser_entities_refreshed"],
        mutates_world=False,
        output_style="terse",
    )

    def __init__(self, browser_controller):
        self._controller = browser_controller

    def execute(self, inputs: Dict[str, Any], world: WorldTimeline, snapshot=None) -> SkillResult:
        css_selector = inputs.get("css_selector")
        text = inputs.get("text")
        timeout = float(inputs.get("timeout", 5.0))

        if not css_selector and not text:
            raise RuntimeError(
                "At least one of css_selector or text required"
            )

        found_entity = None

        # Strategy 1: CSS-based wait
        if css_selector:
            found_entity = self._controller.wait_for_element(
                css_selector, timeout=timeout,
            )

        # Strategy 2: text-based wait (poll + locate)
        if found_entity is None and text:
            import time
            start = time.time()
            while time.time() - start < timeout:
                scored = self._controller.locate_element(text=text)
                if scored:
                    found_entity = scored.entity
                    break
                import time as _t
                _t.sleep(0.3)

        if found_entity is None:
            target = css_selector or text
            logger.info("[WAIT_FOR] Element not found after %.1fs: '%s'", timeout, target)

            return SkillResult(
                outputs={"found": False, "entity_index": 0},
                metadata={"domain": "browser", "entity": f"wait timeout for '{target}'"},
            )

        logger.info(
            "[WAIT_FOR] Found %s idx=%d text='%s'",
            found_entity.entity_type, found_entity.index,
            found_entity.text[:40],
        )

        # Emit refreshed entities
        fresh_snapshot = self._controller.get_snapshot(cached=False)
        world.emit("skill.browser", "browser_entities_refreshed", {
            "url": fresh_snapshot.url,
            "title": fresh_snapshot.title,
            "entity_count": len(fresh_snapshot.entities),
            "tab_count": fresh_snapshot.tab_count,
            "top_entities": (
                BrowserSource._extract_top_entities(fresh_snapshot)
            ),
        })

        return SkillResult(
            outputs={
                "found": True,
                "entity_index": found_entity.index,
            },
            metadata={
                "domain": "browser",
                "entity": f"found {found_entity.entity_type}: '{found_entity.text[:30]}'",
            },
        )
