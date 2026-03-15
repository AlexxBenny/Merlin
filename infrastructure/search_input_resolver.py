# infrastructure/search_input_resolver.py

"""
SearchInputResolver — Robust search input location and submission logic.

Used by:
  - browser_search.py (skill)

Resolution strategy (3-tier):
    Tier 1 — Site-specific selectors (by known ID/name)
        Most reliable. Covers top 20+ sites by traffic.
    Tier 2 — Semantic selectors (ARIA roles, type attributes)
        Standard web patterns that work on well-built sites.
    Tier 3 — Heuristic fallback (text match via locate_element)
        Catches anything Tier 1+2 missed.

All selectors are CSS. Each is tried against the live DOM with
visibility check. First visible match wins.
"""

import logging
import time
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from infrastructure.browser_controller import DOMEntity, PageSnapshot
    from infrastructure.browser_use_controller import BrowserUseController

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Tier 1 — Site-specific selectors (by known ID/name)
#
# These target the #1 search input on major websites.
# Ordered alphabetically by site for maintainability.
# ─────────────────────────────────────────────────────────────

SITE_SPECIFIC_SELECTORS = [
    # Amazon
    '#twotabsearchtextbox',
    'input[name="field-keywords"]',
    # Bing
    '#sb_form_q',
    # DuckDuckGo
    '#search_form_input_homepage',
    '#search_form_input',
    '#searchbox_input',
    # eBay
    '#gh-ac',
    'input[name="_nkw"]',
    # Google
    'input[name="q"]',
    'textarea[name="q"]',
    # Reddit
    '#header-search-bar',
    # Wikipedia
    '#searchInput',
    '#searchform input[name="search"]',
    # YouTube
    'input[name="search_query"]',
    '#search-input input',
]

# ─────────────────────────────────────────────────────────────
# Tier 2 — Semantic selectors (ARIA/HTML standards)
#
# These work on any well-built site that follows web standards.
# Ordered by specificity — most specific first.
# ─────────────────────────────────────────────────────────────

SEMANTIC_SELECTORS = [
    'input[type="search"]',
    '[role="searchbox"]',
    '[role="combobox"]',
    'input[aria-label*="earch"]',
    'input[aria-label*="Search"]',
    'input[placeholder*="earch"]',
    'input[placeholder*="Search"]',
    'textarea[role="searchbox"]',
    'textarea[aria-label*="earch"]',
]

# Combined ordered list: site-specific first, then semantic
ALL_SELECTORS = SITE_SPECIFIC_SELECTORS + SEMANTIC_SELECTORS


def find_search_input(
    controller: "BrowserUseController",
) -> Optional["DOMEntity"]:
    """Locate search input via 3-tier resolution.

    Tier 1+2: CSS selector chain with visibility check
    Tier 3:   locate_element(text="search", type="input") fuzzy match

    Returns:
        DOMEntity for the search input, or None.
    """
    # Tier 1+2: CSS selector chain
    for selector in ALL_SELECTORS:
        entities = controller.find_by_css(selector)
        if entities:
            for entity in entities:
                vis = controller.check_visibility(entity.backend_node_id)
                if vis.get("visible", False):
                    logger.info(
                        "[RESOLVER] Found input via CSS '%s': "
                        "backend_node_id=%d text='%s'",
                        selector, entity.backend_node_id,
                        entity.text[:40],
                    )
                    return entity

    # Tier 3: locate_element fuzzy fallback
    scored = controller.locate_element(
        text="search", element_type="input",
    )
    if scored:
        logger.info(
            "[RESOLVER] Found input via locate(%s): "
            "backend_node_id=%d score=%.1f",
            scored.strategy, scored.entity.backend_node_id,
            scored.score,
        )
        return scored.entity

    return None


def submit_search(
    controller: "BrowserUseController",
    search_entity: "DOMEntity",
) -> "PageSnapshot":
    """Try submission strategies in order until DOM changes.

    1. Enter keypress → wait for DOM change
    2. Click search button → wait for DOM change
    3. form.submit() via JS → wait for DOM change

    Returns:
        PageSnapshot after submission (may be unchanged if all fail).
    """
    # Strategy 1: Enter keypress
    logger.info("[RESOLVER] Submitting via Enter keypress")
    baseline = controller.get_snapshot(cached=True)
    press_result = controller.press_key("Enter")
    if press_result.success:
        time.sleep(0.15)
        snapshot = controller.wait_for_dom_change(timeout=3.0)
        if (snapshot.snapshot_id != baseline.snapshot_id
                or snapshot.url != baseline.url):
            logger.info("[RESOLVER] Enter succeeded — DOM changed")
            return snapshot

    # Strategy 2: Click search/submit button
    logger.info("[RESOLVER] Enter did not trigger change, trying button click")
    button = controller.locate_element(
        text="search", element_type="button",
    )
    if button is None:
        button = controller.locate_element(
            text="submit", element_type="button",
        )

    if button:
        baseline = controller.get_snapshot(cached=True)
        time.sleep(0.15)
        click_result = controller.click(button.entity.backend_node_id)
        if click_result.success:
            snapshot = controller.wait_for_dom_change(timeout=3.0)
            if (snapshot.snapshot_id != baseline.snapshot_id
                    or snapshot.url != baseline.url):
                logger.info("[RESOLVER] Button click succeeded")
                return snapshot

    # Strategy 3: form.submit() via JS
    logger.info("[RESOLVER] Button did not work, trying form.submit()")
    try:
        controller.submit_form(search_entity.backend_node_id)
        time.sleep(0.15)
        snapshot = controller.wait_for_dom_change(timeout=3.0)
        logger.info("[RESOLVER] form.submit() succeeded")
        return snapshot
    except Exception as e:
        logger.warning("[RESOLVER] form.submit() failed: %s", e)

    # All strategies exhausted
    logger.warning("[RESOLVER] All submission strategies exhausted")
    return controller.get_snapshot(cached=False)
