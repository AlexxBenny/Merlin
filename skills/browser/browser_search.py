# skills/browser/browser_search.py

"""
BrowserSearchSkill — Reactive compound search skill.

Tier 2 reactive controller: locates search input via 3-tier resolution
(site-specific IDs → semantic ARIA/type → text heuristic), fills text,
submits, and waits for DOM change. No LLM — purely deterministic.

Submission strategies (in order):
    1. Enter keypress → wait_for_dom_change
    2. Locate search/submit button → click → wait_for_dom_change
    3. form.submit() via JS → wait_for_dom_change

Site coverage:
    Amazon, Bing, DuckDuckGo, eBay, Google, Reddit, Wikipedia, YouTube
    + any site using ARIA roles / type="search" / placeholder patterns
"""

import asyncio
import time
from typing import Any, Dict

from runtime.sources.browser import BrowserSource
from skills.base import Skill
from skills.contract import SkillContract, FailurePolicy
from skills.skill_result import SkillResult
from ir.mission import ExecutionMode
from world.timeline import WorldTimeline

import logging
logger = logging.getLogger(__name__)

# Use shared SearchInputResolver for centralized search input logic
from infrastructure.search_input_resolver import (
    find_search_input,
    submit_search,
)


class BrowserSearchSkill(Skill):
    """Search on the current page — reactive, no LLM.

    Locates search input via CSS selectors with fallback chain,
    fills query text, then tries submission strategies in order
    until DOM changes.
    """

    contract = SkillContract(
        name="browser.search",
        action="search",
        target_type="browser_page",
        description="Search on the current page",
        narration_template="search for {query}",
        intent_verbs=["search", "find", "look"],
        intent_keywords=["search", "query"],
        verb_specificity="specific",
        domain="browser",
        requires_focus=True,
        inputs={
            "query": "search_text",
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
        query = inputs.get("query", "")
        if not query:
            raise RuntimeError("No search query provided")

        # ── Locate search input ──
        search_entity = self._find_search_input()

        if search_entity is None:
            raise RuntimeError(
                "Could not find search input on page. "
                "Tried CSS selectors and text matching."
            )

        # Scroll into view if needed
        vis = self._controller.check_visibility(
            search_entity.backend_node_id,
        )
        if not vis.get("in_viewport", True):
            self._controller.scroll_to_element(
                search_entity.backend_node_id,
            )

        # Interaction cooldown
        time.sleep(0.15)

        # ── Fill search input ──
        fill_result = self._controller.fill(
            search_entity.backend_node_id, query,
        )
        if not fill_result.success:
            raise RuntimeError(f"Fill failed: {fill_result.error}")

        # ── Multi-strategy submission ──
        final_snapshot = self._submit_search(search_entity)

        final_url = final_snapshot.url
        page_title = final_snapshot.title

        # ── Emit world state event ──
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
                "entity": f"search '{query[:40]}'",
            },
        )

    def _find_search_input(self):
        """Locate search input — delegates to shared SearchInputResolver."""
        return find_search_input(self._controller)

    def _submit_search(self, search_entity):
        """Submit search — delegates to shared SearchInputResolver."""
        return submit_search(self._controller, search_entity)
