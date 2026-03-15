# infrastructure/browser_use_controller.py

"""
BrowserUseController — Concrete deterministic browser controller.

Implements BrowserController using browser-use's native APIs:
- BrowserSession.get_browser_state_summary() → DOM perception
- Page.get_element(backend_node_id)          → deterministic element access
- Element.click() / fill() / hover()        → deterministic interaction
- BrowserSession.navigate_to()              → deterministic navigation

Design rules:
- Wraps BrowserUseAdapter — accesses adapter._browser (BrowserSession)
  and adapter._loop (asyncio event loop)
- All browser-use APIs are async; uses run_coroutine_threadsafe()
  on the adapter's existing event loop
- Entity filtering: interactive-only (ax_role + tag + js_listener + clickable)
- Entity ordering: sorted by backend_node_id for deterministic indexes
- Snapshot ID: hash(sorted(backend_node_ids)) — DOM-based, not timestamp
- Snapshot cache: invalidated after navigation, tab switch, explicit refresh
- Navigation detection: wait loop (200ms polls up to 3s)
- No LLM calls — semantic ops belong in skills
"""

import asyncio
import hashlib
import logging
import re
import time
from collections import Counter
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
from urllib.parse import quote

from infrastructure.browser_controller import (
    BrowserController,
    BlockingElement,
    BrowserResult,
    DOMEntity,
    ElementScore,
    PageSnapshot,
    TabInfo,
)

if TYPE_CHECKING:
    from infrastructure.browser_use_adapter import BrowserUseAdapter

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Interactive element classification
# ─────────────────────────────────────────────────────────────

# Accessibility roles that indicate interactive elements
INTERACTIVE_AX_ROLES = frozenset({
    "link", "button", "textbox", "checkbox", "radio", "combobox",
    "searchbox", "slider", "spinbutton", "menuitem", "tab", "switch",
    "menuitemcheckbox", "menuitemradio", "option", "treeitem",
    "gridcell",
})

# HTML tags that are inherently interactive
INTERACTIVE_TAGS = frozenset({
    "a", "button", "input", "select", "textarea", "video", "audio",
})

# Accessibility role → entity type mapping
AX_ROLE_MAP = {
    "link": "link",
    "button": "button",
    "textbox": "input",
    "checkbox": "input",
    "radio": "input",
    "combobox": "input",
    "searchbox": "input",
    "slider": "input",
    "spinbutton": "input",
    "menuitem": "button",
    "menuitemcheckbox": "button",
    "menuitemradio": "button",
    "tab": "button",
    "switch": "input",
    "option": "input",
    "treeitem": "button",
    "gridcell": "clickable",
}

# HTML tag → entity type fallback
TAG_TYPE_MAP = {
    "a": "link",
    "button": "button",
    "input": "input",
    "select": "input",
    "textarea": "input",
    "video": "media",
    "audio": "media",
    "img": "image",
}


# ─────────────────────────────────────────────────────────────
# URL Normalization — single authority for all navigation paths
# ─────────────────────────────────────────────────────────────

# Known site aliases → full URLs (shared with browser_goal_parser)
_SITE_ALIAS_MAP = {
    "youtube": "https://www.youtube.com",
    "google": "https://www.google.com",
    "amazon": "https://www.amazon.com",
    "bing": "https://www.bing.com",
    "duckduckgo": "https://duckduckgo.com",
    "reddit": "https://www.reddit.com",
    "twitter": "https://twitter.com",
    "x": "https://x.com",
    "wikipedia": "https://www.wikipedia.org",
    "github": "https://github.com",
    "linkedin": "https://www.linkedin.com",
    "facebook": "https://www.facebook.com",
    "instagram": "https://www.instagram.com",
    "spotify": "https://open.spotify.com",
    "ebay": "https://www.ebay.com",
    "imdb": "https://www.imdb.com",
    "gmail": "https://mail.google.com",
    "netflix": "https://www.netflix.com",
    "whatsapp": "https://web.whatsapp.com",
    "stackoverflow": "https://stackoverflow.com",
    "stack overflow": "https://stackoverflow.com",
}

# Regex: looks like a domain (has TLD, no spaces)
_DOMAIN_RE = re.compile(
    r'^[a-zA-Z0-9][-a-zA-Z0-9]*'
    r'(?:\.[a-zA-Z0-9][-a-zA-Z0-9]*)*'
    r'\.[a-zA-Z]{2,}'
    r'(?:[:/].*)?$'
)


def normalize_url_input(raw: str) -> str:
    """Normalize a raw URL input for CDP navigation.

    CDP Page.navigate requires a fully-qualified URL with scheme.
    This function handles all common input patterns:

    1. Valid URL (has scheme)    → pass through
    2. Known site alias          → map via _SITE_ALIAS_MAP
    3. localhost[:port]          → prepend http://
    4. Bare domain (has TLD)     → prepend https://
    5. Non-domain text           → Google search fallback

    This is the SINGLE normalization authority — all navigation
    flows through here. Do NOT duplicate this logic elsewhere.
    """
    raw = raw.strip()
    if not raw:
        return raw

    # ── 1. Already has scheme → pass through ──
    if raw.startswith(("http://", "https://", "file://", "chrome://")):
        return raw

    lowered = raw.lower()

    # ── 2. Known site alias → direct map ──
    if lowered in _SITE_ALIAS_MAP:
        return _SITE_ALIAS_MAP[lowered]

    # ── 3. Localhost → http (not https) ──
    if lowered == "localhost" or lowered.startswith("localhost:"):
        return "http://" + raw

    # ── 4. Looks like a domain (has TLD, no spaces) → prepend https ──
    if " " not in raw and _DOMAIN_RE.match(raw):
        return "https://" + raw

    # ── 5. Non-domain text → search engine ──
    return "https://www.google.com/search?q=" + quote(raw)


# ─────────────────────────────────────────────────────────────
# Concrete Controller
# ─────────────────────────────────────────────────────────────

class BrowserUseController(BrowserController):
    """Deterministic browser controller backed by browser-use.

    Wraps the adapter's BrowserSession. Does NOT own the browser
    lifecycle — the adapter manages start/stop.

    Usage:
        adapter = BrowserUseAdapter(config, api_key, model_name)
        controller = BrowserUseController(adapter)
        snapshot = controller.get_snapshot()
        controller.click(snapshot.entities[0].backend_node_id)
    """

    def __init__(self, adapter: "BrowserUseAdapter"):
        self._adapter = adapter
        self._cached_snapshot: Optional[PageSnapshot] = None
        self._cached_selector_map: Dict[int, Any] = {}
        self._cached_page_info: Optional[Any] = None

    # ─────────────────────────────────────────────────────────
    # Internal properties
    # ─────────────────────────────────────────────────────────

    @property
    def _browser(self):
        """Access the browser-use BrowserSession via adapter."""
        return self._adapter._browser

    @property
    def _loop(self):
        """Access the adapter's asyncio event loop."""
        return self._adapter._loop

    # ─────────────────────────────────────────────────────────
    # Async bridge (sync → async, deadlock-safe)
    # ─────────────────────────────────────────────────────────

    def _run_async(self, coro, timeout: float = 30.0):
        """Bridge async coroutine to sync using adapter's event loop.

        Guarantees the adapter's loop and browser are initialized before
        scheduling any coroutine. Detects same-loop calls to prevent deadlock.
        """
        self._ensure_ready()

        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None

        if running is not None and running == self._loop:
            raise RuntimeError(
                "BrowserUseController: cannot call sync bridge "
                "from the adapter's own event loop — deadlock"
            )

        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    # ─────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────

    def _ensure_ready(self) -> None:
        """Ensure the adapter's event loop, browser, and page are initialized.

        The adapter lazy-inits its loop and browser inside run_task()
        (the autonomous agent path). Deterministic controller methods
        bypass run_task(), so they must call this guard instead.

        Lifecycle invariant: loop + browser + page must all be valid
        before any action executes.

        Safe to call multiple times — adapter's methods are idempotent.
        """
        # 1. Event loop (sync, creates thread once)
        self._adapter._ensure_loop()

        # 2. Browser instance (async, must schedule on the now-valid loop)
        if self._adapter._browser is None:
            future = asyncio.run_coroutine_threadsafe(
                self._adapter._ensure_browser(),
                self._adapter._loop,
            )
            future.result(timeout=30.0)
            logger.info("[BrowserController] Browser initialized via ensure_ready()")

        # 3. Page/tab (async — browser-use BrowserSession doesn't auto-create a page)
        future = asyncio.run_coroutine_threadsafe(
            self._ensure_page_async(),
            self._adapter._loop,
        )
        future.result(timeout=15.0)

    async def _ensure_page_async(self) -> None:
        """Ensure the browser session is started and has an active page.

        browser-use BrowserSession requires start() to launch Chrome,
        connect via CDP, initialize SessionManager, and discover tabs.
        BrowserSession(...) only creates a config object.

        start() is idempotent — safe to call when already connected.
        This is exactly what the autonomous agent does internally.
        """
        page = await self._browser.get_current_page()
        if page is not None:
            return

        # Session created but never started — start it now
        logger.info("[BrowserController] Starting browser session (CDP connect)...")
        await self._browser.start()
        logger.info("[BrowserController] Browser session started")

    def is_alive(self) -> bool:
        """Check if browser connection is alive."""
        try:
            browser = self._browser
            if browser is None:
                return False
            return getattr(browser, "is_cdp_connected", False)
        except Exception:
            return False

    # ─────────────────────────────────────────────────────────
    # Navigation
    # ─────────────────────────────────────────────────────────

    def navigate(self, url: str) -> BrowserResult:
        """Navigate current tab to URL."""
        try:
            return self._run_async(self._navigate_async(url))
        except Exception as e:
            logger.error("[BrowserController] navigate failed: %s", e)
            return BrowserResult(success=False, error=str(e))

    async def _navigate_async(self, url: str) -> BrowserResult:
        page = await self._browser.get_current_page()
        if page is None:
            return BrowserResult(success=False, error="No active page after session start")

        # Normalize URL — handles bare site names, missing scheme, search fallback
        url = normalize_url_input(url)
        logger.debug("[BrowserController] navigate → normalized URL: %s", url[:100])

        url_before = await self._browser.get_current_page_url()
        await page.goto(url)
        url_after, _ = await self._wait_for_navigation(
            url_before, len(await self._browser.get_pages()),
        )

        self._invalidate_cache()
        snapshot = await self._get_snapshot_async()
        return BrowserResult(
            success=True, snapshot=snapshot,
            navigated=(url_before != url_after),
        )

    def go_back(self) -> BrowserResult:
        """Navigate back."""
        try:
            return self._run_async(self._go_back_async())
        except Exception as e:
            logger.error("[BrowserController] go_back failed: %s", e)
            return BrowserResult(success=False, error=str(e))

    async def _go_back_async(self) -> BrowserResult:
        page = await self._browser.get_current_page()
        if page is None:
            return BrowserResult(success=False, error="No active page")

        url_before = await self._browser.get_current_page_url()
        tabs_before = len(await self._browser.get_pages())
        await page.go_back()
        url_after, tabs_after = await self._wait_for_navigation(
            url_before, tabs_before,
        )

        self._invalidate_cache()
        snapshot = await self._get_snapshot_async()
        return BrowserResult(
            success=True, snapshot=snapshot,
            navigated=(url_before != url_after),
            new_tab_opened=(tabs_after > tabs_before),
        )

    def go_forward(self) -> BrowserResult:
        """Navigate forward."""
        try:
            return self._run_async(self._go_forward_async())
        except Exception as e:
            logger.error("[BrowserController] go_forward failed: %s", e)
            return BrowserResult(success=False, error=str(e))

    async def _go_forward_async(self) -> BrowserResult:
        page = await self._browser.get_current_page()
        if page is None:
            return BrowserResult(success=False, error="No active page")

        url_before = await self._browser.get_current_page_url()
        tabs_before = len(await self._browser.get_pages())
        await page.go_forward()
        url_after, tabs_after = await self._wait_for_navigation(
            url_before, tabs_before,
        )

        self._invalidate_cache()
        snapshot = await self._get_snapshot_async()
        return BrowserResult(
            success=True, snapshot=snapshot,
            navigated=(url_before != url_after),
            new_tab_opened=(tabs_after > tabs_before),
        )

    # ─────────────────────────────────────────────────────────
    # Page Interaction
    # ─────────────────────────────────────────────────────────

    def click(self, backend_node_id: int) -> BrowserResult:
        """Click an element by backend_node_id."""
        try:
            return self._run_async(self._click_async(backend_node_id))
        except Exception as e:
            logger.error("[BrowserController] click(%d) failed: %s",
                         backend_node_id, e)
            return BrowserResult(success=False, error=str(e))

    async def _click_async(self, backend_node_id: int) -> BrowserResult:
        page = await self._browser.get_current_page()
        if page is None:
            return BrowserResult(success=False, error="No active page")

        url_before = await self._browser.get_current_page_url()
        tabs_before = len(await self._browser.get_pages())

        element = await page.get_element(backend_node_id)
        await element.click()

        url_after, tabs_after = await self._wait_for_navigation(
            url_before, tabs_before,
        )

        self._invalidate_cache()
        snapshot = await self._get_snapshot_async()
        return BrowserResult(
            success=True, snapshot=snapshot,
            navigated=(url_before != url_after),
            new_tab_opened=(tabs_after > tabs_before),
        )

    def fill(self, backend_node_id: int, text: str) -> BrowserResult:
        """Fill an input element with text."""
        try:
            return self._run_async(self._fill_async(backend_node_id, text))
        except Exception as e:
            logger.error("[BrowserController] fill(%d) failed: %s",
                         backend_node_id, e)
            return BrowserResult(success=False, error=str(e))

    async def _fill_async(
        self, backend_node_id: int, text: str,
    ) -> BrowserResult:
        page = await self._browser.get_current_page()
        if page is None:
            return BrowserResult(success=False, error="No active page")

        element = await page.get_element(backend_node_id)
        await element.fill(text)

        self._invalidate_cache()
        snapshot = await self._get_snapshot_async()
        return BrowserResult(success=True, snapshot=snapshot)

    def press_key(self, key: str) -> BrowserResult:
        """Press a keyboard key on the current page."""
        try:
            return self._run_async(self._press_key_async(key))
        except Exception as e:
            logger.error("[BrowserController] press_key(%s) failed: %s",
                         key, e)
            return BrowserResult(success=False, error=str(e))

    async def _press_key_async(self, key: str) -> BrowserResult:
        page = await self._browser.get_current_page()
        if page is None:
            return BrowserResult(success=False, error="No active page")

        url_before = await self._browser.get_current_page_url()
        tabs_before = len(await self._browser.get_pages())

        await page.press(key)

        # Wait for potential navigation (Enter on search box → new page)
        url_after, tabs_after = await self._wait_for_navigation(
            url_before, tabs_before,
        )

        self._invalidate_cache()
        snapshot = await self._get_snapshot_async()
        return BrowserResult(
            success=True, snapshot=snapshot,
            navigated=(url_before != url_after),
            new_tab_opened=(tabs_after > tabs_before),
        )

    # ─────────────────────────────────────────────────────────
    # Scrolling
    # ─────────────────────────────────────────────────────────

    def scroll_page(self, direction: str, amount: int = 3) -> BrowserResult:
        """Scroll the page viewport."""
        try:
            return self._run_async(
                self._scroll_page_async(direction, amount),
            )
        except Exception as e:
            logger.error("[BrowserController] scroll_page failed: %s", e)
            return BrowserResult(success=False, error=str(e))

    async def _scroll_page_async(
        self, direction: str, amount: int = 3,
    ) -> BrowserResult:
        page = await self._browser.get_current_page()
        if page is None:
            return BrowserResult(success=False, error="No active page")

        key = "PageDown" if direction == "down" else "PageUp"
        for _ in range(amount):
            await page.press(key)
            await asyncio.sleep(0.1)  # Brief pause between presses

        self._invalidate_cache()
        snapshot = await self._get_snapshot_async()
        return BrowserResult(success=True, snapshot=snapshot)

    def scroll_element(
        self, backend_node_id: int, direction: str,
    ) -> BrowserResult:
        """Scroll a specific scrollable container element."""
        try:
            return self._run_async(
                self._scroll_element_async(backend_node_id, direction),
            )
        except Exception as e:
            logger.error("[BrowserController] scroll_element failed: %s", e)
            return BrowserResult(success=False, error=str(e))

    async def _scroll_element_async(
        self, backend_node_id: int, direction: str,
    ) -> BrowserResult:
        page = await self._browser.get_current_page()
        if page is None:
            return BrowserResult(success=False, error="No active page")

        element = await page.get_element(backend_node_id)
        # Focus the element, then use keyboard scrolling
        await element.focus()
        key = "PageDown" if direction == "down" else "PageUp"
        for _ in range(3):
            await page.press(key)
            await asyncio.sleep(0.1)

        self._invalidate_cache()
        snapshot = await self._get_snapshot_async()
        return BrowserResult(success=True, snapshot=snapshot)

    # ─────────────────────────────────────────────────────────
    # Tab Management
    # ─────────────────────────────────────────────────────────

    def list_tabs(self) -> List[TabInfo]:
        """List all open tabs."""
        try:
            return self._run_async(self._list_tabs_async())
        except Exception as e:
            logger.error("[BrowserController] list_tabs failed: %s", e)
            return []

    async def _list_tabs_async(self) -> List[TabInfo]:
        pages = await self._browser.get_pages()
        tabs = []
        for page in pages:
            try:
                url = await page.url() if hasattr(page, "url") else ""
                title = await page.title() if hasattr(page, "title") else ""
                tab_id = getattr(page, "_target_id", str(id(page)))
                tabs.append(TabInfo(tab_id=str(tab_id), url=url, title=title))
            except Exception:
                pass
        return tabs

    def switch_tab(self, tab_id: str) -> BrowserResult:
        """Switch agent focus to a different tab."""
        try:
            return self._run_async(self._switch_tab_async(tab_id))
        except Exception as e:
            logger.error("[BrowserController] switch_tab failed: %s", e)
            return BrowserResult(success=False, error=str(e))

    async def _switch_tab_async(self, tab_id: str) -> BrowserResult:
        await self._browser.switch_tab(tab_id)
        self._invalidate_cache()
        snapshot = await self._get_snapshot_async()
        return BrowserResult(success=True, snapshot=snapshot)

    # ─────────────────────────────────────────────────────────
    # State — Snapshot
    # ─────────────────────────────────────────────────────────

    def get_snapshot(self, cached: bool = True) -> PageSnapshot:
        """Get current page state as a versioned snapshot."""
        if cached and self._cached_snapshot is not None:
            return self._cached_snapshot
        try:
            snapshot = self._run_async(self._get_snapshot_async())
            self._cached_snapshot = snapshot
            return snapshot
        except Exception as e:
            logger.error("[BrowserController] get_snapshot failed: %s", e)
            # Return empty snapshot on failure
            return PageSnapshot(
                snapshot_id="error", url="", title="",
                entities=(), entity_count=0, tab_count=0,
                timestamp=time.time(),
            )

    async def _get_snapshot_async(self) -> PageSnapshot:
        """Build a PageSnapshot from browser-use's DOM state."""
        state = await self._browser.get_browser_state_summary(
            include_screenshot=False, cached=False,
        )

        url = state.url or ""
        title = state.title or ""
        selector_map = state.dom_state.selector_map if state.dom_state else {}
        total_count = len(selector_map)

        # Retain full DOM for structural clustering
        self._cached_selector_map = selector_map
        self._cached_page_info = state.page_info

        # Filter interactive-only nodes, sort by backend_node_id
        interactive_nodes: List[Tuple[int, Any]] = []
        for sm_index, node in selector_map.items():
            if self._is_interactive(node):
                interactive_nodes.append((sm_index, node))

        interactive_nodes.sort(key=lambda x: x[1].backend_node_id)

        # Build entities with deterministic indexes
        entities: List[DOMEntity] = []
        backend_node_ids: List[int] = []
        for display_index, (_, node) in enumerate(interactive_nodes, start=1):
            entity_type = self._classify_entity(node)
            text = self._get_entity_text(node)
            url_attr = node.attributes.get("href", "") if node.attributes else ""

            entities.append(DOMEntity(
                index=display_index,
                backend_node_id=node.backend_node_id,
                entity_type=entity_type,
                text=text,
                url=url_attr or None,
                ax_role=(node.ax_node.role if node.ax_node else None),
            ))
            backend_node_ids.append(node.backend_node_id)

        # DOM-based snapshot ID
        snapshot_id = hashlib.md5(
            str(backend_node_ids).encode(),
        ).hexdigest()[:12]

        # Tab info
        tab_infos = []
        for tab in (state.tabs or []):
            tab_infos.append(TabInfo(
                tab_id=str(tab.target_id),
                url=tab.url or "",
                title=tab.title or "",
            ))

        # Scroll percentage
        scroll_pct = None
        if state.page_info:
            page_h = state.page_info.page_height
            vp_h = state.page_info.viewport_height
            if page_h > vp_h:
                scroll_pct = round(
                    state.page_info.scroll_y / (page_h - vp_h) * 100, 1,
                )

        snapshot = PageSnapshot(
            snapshot_id=snapshot_id,
            url=url,
            title=title,
            entities=tuple(entities),
            entity_count=total_count,
            tab_count=len(state.tabs or []),
            tabs=tuple(tab_infos),
            scroll_pct=scroll_pct,
            timestamp=time.time(),
        )
        self._cached_snapshot = snapshot
        return snapshot

    def _invalidate_cache(self) -> None:
        """Invalidate cached snapshot — forces fresh DOM traversal."""
        self._cached_snapshot = None
        self._cached_selector_map = {}
        self._cached_page_info = None

    # ─────────────────────────────────────────────────────────
    # State — Entity Search
    # ─────────────────────────────────────────────────────────

    def find_entities(self, text: str) -> List[DOMEntity]:
        """Search entities by text — tokenized match, no LLM.

        Strategy:
        1. All query tokens ⊆ entity tokens → strong match
        2. Fallback: any token overlap → weak match
        """
        snapshot = self.get_snapshot(cached=True)
        query_tokens = set(text.lower().split())
        if not query_tokens:
            return []

        strong_matches: List[DOMEntity] = []
        weak_matches: List[DOMEntity] = []

        for entity in snapshot.entities:
            entity_tokens = set(entity.text.lower().split())
            if query_tokens.issubset(entity_tokens):
                strong_matches.append(entity)
            elif query_tokens & entity_tokens:
                weak_matches.append(entity)

        return strong_matches if strong_matches else weak_matches

    def get_action_space_text(
        self,
        goal_tokens: Optional[set] = None,
        max_elements: int = 300,
    ) -> str:
        """Produce browser-use style indexed DOM text for LLM escalation.

        Design rules:
            - DOM traversal order (sorted by sm_index, NOT backend_node_id)
            - Filter pipeline: visible → interactive
            - Cap at max_elements (default 300) to avoid prompt explosion
            - Goal-token elements always included (not filtered by cap)
            - No truncation of the element set below 300

        Output format:
            [1]<a href="/home">Home</a>
            [2]<input type="search" placeholder="Search...">
            [3]<button>Submit</button>

        Args:
            goal_tokens: Set of lowercase tokens from the goal text.
                         Elements matching these tokens are prioritized.
            max_elements: Maximum elements to include (default 300).

        Returns:
            Formatted text string for LLM prompt injection.
        """
        # Ensure selector_map is populated
        if not self._cached_selector_map:
            self.get_snapshot(cached=False)

        selector_map = self._cached_selector_map
        if not selector_map:
            return "(no interactive elements found)"

        # Filter interactive nodes — preserve DOM traversal order (sm_index)
        interactive: List[Tuple[int, Any]] = []
        for sm_index in sorted(selector_map.keys()):
            node = selector_map[sm_index]
            if self._is_interactive(node):
                interactive.append((sm_index, node))

        if not interactive:
            return "(no interactive elements found)"

        # Separate goal-relevant elements (always included) from others
        goal_relevant = []
        others = []
        for sm_index, node in interactive:
            if goal_tokens:
                text = self._get_entity_text(node).lower()
                text_tokens = set(text.split())
                if goal_tokens & text_tokens:
                    goal_relevant.append((sm_index, node))
                    continue
            others.append((sm_index, node))

        # Cap: goal-relevant first, then fill remaining from others
        remaining_cap = max(max_elements - len(goal_relevant), 0)
        selected = goal_relevant + others[:remaining_cap]

        # Re-sort by sm_index to maintain DOM order
        selected.sort(key=lambda x: x[0])

        # Format output
        lines = []
        for display_idx, (_, node) in enumerate(selected, start=1):
            tag = getattr(node, "tag_name", "div").lower()
            text = self._get_entity_text(node)[:100]
            attrs = self._format_key_attrs(node)
            if attrs:
                lines.append(f"[{display_idx}]<{tag} {attrs}>{text}</{tag}>")
            else:
                lines.append(f"[{display_idx}]<{tag}>{text}</{tag}>")

        return "\n".join(lines)

    # Key attributes to include in action space output
    _KEY_ATTRS = ("href", "type", "placeholder", "role", "aria-label", "name", "value")

    @staticmethod
    def _format_key_attrs(node) -> str:
        """Format key attributes for action space text."""
        attrs = getattr(node, "attributes", {}) or {}
        parts = []
        for key in BrowserUseController._KEY_ATTRS:
            val = attrs.get(key)
            if val:
                parts.append(f'{key}="{val[:60]}"')
        return " ".join(parts)

    # ─────────────────────────────────────────────────────────
    # Internal — Entity Classification
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _is_interactive(node) -> bool:
        """Check if a DOM node is interactive.

        Uses accessibility role, HTML tag, JS click listeners, and
        snapshot clickability.
        """
        # Must be visible
        if not getattr(node, "is_visible", True):
            return False

        # Check accessibility role
        ax_node = getattr(node, "ax_node", None)
        if ax_node and getattr(ax_node, "role", None):
            if ax_node.role in INTERACTIVE_AX_ROLES:
                return True

        # Check HTML tag
        tag = getattr(node, "tag_name", "").lower()
        if tag in INTERACTIVE_TAGS:
            return True

        # Check JS click listener
        if getattr(node, "has_js_click_listener", False):
            return True

        # Check snapshot clickability
        snap = getattr(node, "snapshot_node", None)
        if snap and getattr(snap, "is_clickable", False):
            return True

        return False

    @staticmethod
    def _classify_entity(node) -> str:
        """Classify entity type using ax_role → tag → JS listener."""
        # 1. Accessibility role (handles div[role="button"], custom elements)
        ax_node = getattr(node, "ax_node", None)
        if ax_node:
            ax_role = getattr(ax_node, "role", None)
            if ax_role and ax_role in AX_ROLE_MAP:
                return AX_ROLE_MAP[ax_role]

        # 2. HTML tag fallback
        tag = getattr(node, "tag_name", "").lower()
        if tag in TAG_TYPE_MAP:
            return TAG_TYPE_MAP[tag]

        # 3. JS click listener → implicit button
        if getattr(node, "has_js_click_listener", False):
            return "button"

        # 4. Clickable by snapshot
        snap = getattr(node, "snapshot_node", None)
        if snap and getattr(snap, "is_clickable", False):
            return "clickable"

        return "other"

    @staticmethod
    def _get_entity_text(node) -> str:
        """Get meaningful text for an entity.

        Uses browser-use's built-in priority:
        value → aria-label → title → placeholder → alt → innerText
        """
        try:
            text = node.get_meaningful_text_for_llm()
            return (text or "").strip()[:120]
        except Exception:
            pass

        # Fallback: try attributes directly
        attrs = getattr(node, "attributes", {}) or {}
        for attr in ("aria-label", "title", "placeholder", "alt", "value"):
            val = attrs.get(attr, "")
            if val:
                return val[:120]

        # Last resort: nested text
        try:
            return (node.get_all_children_text() or "").strip()[:120]
        except Exception:
            return ""

    # ─────────────────────────────────────────────────────────
    # Internal — Navigation Detection
    # ─────────────────────────────────────────────────────────

    async def _wait_for_navigation(
        self,
        url_before: str,
        tabs_before: int,
        max_wait: float = 3.0,
    ) -> Tuple[str, int]:
        """Wait until URL changes, new tab opens, or timeout.

        Polls every 200ms up to max_wait seconds.
        Returns (url_after, tabs_after).
        """
        elapsed = 0.0
        while elapsed < max_wait:
            await asyncio.sleep(0.2)
            elapsed += 0.2
            try:
                url_after = await self._browser.get_current_page_url()
                tabs_after = len(await self._browser.get_pages())
                if url_after != url_before or tabs_after != tabs_before:
                    # Give a brief moment for page to settle
                    await asyncio.sleep(0.3)
                    return (
                        await self._browser.get_current_page_url(),
                        len(await self._browser.get_pages()),
                    )
            except Exception:
                pass

        # Timeout — return current state
        try:
            return (
                await self._browser.get_current_page_url(),
                len(await self._browser.get_pages()),
            )
        except Exception:
            return url_before, tabs_before

    # ─────────────────────────────────────────────────────────
    # Tier 2 — Reactive Controller Methods
    # ─────────────────────────────────────────────────────────

    def find_by_css(self, selector: str) -> List[DOMEntity]:
        """Find elements by CSS selector — deterministic, structural."""
        try:
            return self._run_async(self._find_by_css_async(selector))
        except Exception as e:
            logger.error("[BrowserController] find_by_css failed: %s", e)
            return []

    async def _find_by_css_async(self, selector: str) -> List[DOMEntity]:
        page = await self._browser.get_current_page()
        if page is None:
            return []

        elements = await page.get_elements_by_css_selector(selector)
        result: List[DOMEntity] = []

        for i, el in enumerate(elements, start=1):
            try:
                info = await el.get_basic_info()
                node_name = info.get("nodeName", "unknown").lower()
                attrs = info.get("attributes", {})

                # Classify entity type from tag
                entity_type = TAG_TYPE_MAP.get(node_name, "clickable")

                # Get meaningful text: aria-label > placeholder > value > name
                text = (
                    attrs.get("aria-label", "")
                    or attrs.get("placeholder", "")
                    or attrs.get("value", "")
                    or attrs.get("title", "")
                    or attrs.get("alt", "")
                    or attrs.get("name", "")
                    or ""
                )

                result.append(DOMEntity(
                    index=i,
                    backend_node_id=el._backend_node_id,
                    entity_type=entity_type,
                    text=text[:120],
                    url=attrs.get("href"),
                ))
            except Exception:
                continue

        return result

    def locate_element(
        self,
        *,
        text: Optional[str] = None,
        role: Optional[str] = None,
        css: Optional[str] = None,
        element_type: Optional[str] = None,
    ) -> Optional[ElementScore]:
        """Multi-strategy element location with scoring. No LLM."""
        try:
            return self._run_async(
                self._locate_element_async(
                    text=text, role=role, css=css,
                    element_type=element_type,
                ),
            )
        except Exception as e:
            logger.error("[BrowserController] locate_element failed: %s", e)
            return None

    async def _locate_element_async(
        self,
        *,
        text: Optional[str] = None,
        role: Optional[str] = None,
        css: Optional[str] = None,
        element_type: Optional[str] = None,
    ) -> Optional[ElementScore]:
        """Multi-strategy locate with 10-signal scoring.

        Strategy priority:
          1. CSS selector (structural, most stable)
          2. ARIA role match (accessibility-based)
          3. Attribute match (name, label, placeholder)
          4. Text content match (tokenized)
          5. Type heuristic (largest visible input, etc.)

        Scoring signals (additive):
          input[type=search]       +8
          aria-label match         +6
          placeholder match        +5
          visible                  +5
          in viewport              +4
          inside <form>            +3
          exact text match         +3
          width>200 (input/button) +2
          near top (y<0.33*vh)     +2
          height>20                +1
        """
        candidates: List[ElementScore] = []
        keyword = (text or "").lower().strip()

        # ── Strategy 1: CSS selector ──
        if css:
            css_entities = await self._find_by_css_async(css)
            for entity in css_entities:
                score = 10.0  # CSS is highest confidence
                candidates.append(ElementScore(
                    entity=entity, score=score,
                    strategy="css",
                    details=f"matched '{css}'",
                ))

        # ── Strategies 2–5: use snapshot entities ──
        snapshot = await self._get_snapshot_async()

        for entity in snapshot.entities:
            entity_text_lower = entity.text.lower()
            ax_role = (entity.ax_role or "").lower()

            score = 0.0
            strategy = ""
            details = ""

            # Signal: input[type=search] tag (+8)
            if entity.entity_type == "input" and ax_role in (
                "searchbox", "search",
            ):
                score += 8.0
                strategy = "search_tag"
                details = "input[type=search]"

            # Signal: ARIA role match (+6 for aria-label, +7 for role)
            if role and ax_role == role.lower():
                score += 7.0
                strategy = strategy or "aria_role"
                details += f" role='{ax_role}'"
            elif keyword and keyword in (entity.text or "").lower():
                # aria-label / placeholder contains keyword (+6/+5)
                score += 6.0
                strategy = strategy or "aria_label"
                details += f" label contains '{keyword}'"

            # Signal: exact text match (+3 on top of partial)
            if keyword and entity_text_lower.strip() == keyword:
                score += 3.0
                details += " exact_match"

            # Signal: type match
            if element_type and entity.entity_type == element_type:
                score += 3.0
                strategy = strategy or "type"
                details += f" type='{element_type}'"

            # Signal: heuristic — first input on page
            if (
                element_type == "input"
                and entity.entity_type == "input"
                and not keyword
            ):
                position_bonus = max(0, 2.0 - entity.index * 0.1)
                score += 1.0 + position_bonus
                strategy = strategy or "heuristic"
                details += " first input heuristic"

            if score > 0:
                candidates.append(ElementScore(
                    entity=entity, score=score,
                    strategy=strategy, details=details.strip(),
                ))

        if not candidates:
            return None

        # ── Visibility + context bonuses ──
        # Get viewport dimensions once for position scoring
        try:
            state = await self._browser.get_browser_state_summary(
                include_screenshot=False, cached=True,
            )
            vp_h = state.page_info.viewport_height if state.page_info else 1080
        except Exception:
            vp_h = 1080

        for cand in candidates:
            try:
                vis = await self._check_visibility_async(
                    cand.entity.backend_node_id,
                )
                # Signal: visible (+5)
                if vis.get("visible"):
                    cand.score += 5.0
                # Signal: in viewport (+4)
                if vis.get("in_viewport"):
                    cand.score += 4.0

                bbox = vis.get("bbox")
                if bbox:
                    w = bbox.get("width", 0)
                    h = bbox.get("height", 0)
                    y = bbox.get("y", 9999)

                    # Signal: width>200 for input/button only (+2)
                    if w > 200 and cand.entity.entity_type in (
                        "input", "button",
                    ):
                        cand.score += 2.0

                    # Signal: height>20 (+1)
                    if h > 20:
                        cand.score += 1.0

                    # Signal: near top-third of viewport (+2)
                    target_y = vp_h * 0.33
                    dist = abs(y - target_y) / vp_h
                    position_bonus = max(0, 2.0 * (1.0 - dist))
                    cand.score += position_bonus

                # Signal: inside <form> (+3)
                if vis.get("inside_form"):
                    cand.score += 3.0

            except Exception:
                pass

        # Sort by score descending
        candidates.sort(key=lambda c: -c.score)
        best = candidates[0]

        logger.info(
            "[LOCATE] Best: %s idx=%d score=%.1f strategy=%s (%s)",
            best.entity.entity_type, best.entity.index,
            best.score, best.strategy, best.details,
        )
        return best

    def wait_for_element(
        self, selector: str, timeout: float = 5.0,
    ) -> Optional[DOMEntity]:
        """Poll DOM until element matching CSS selector appears."""
        try:
            return self._run_async(
                self._wait_for_element_async(selector, timeout),
                timeout=timeout + 5.0,
            )
        except Exception as e:
            logger.debug("[BrowserController] wait_for_element: %s", e)
            return None

    async def _wait_for_element_async(
        self, selector: str, timeout: float,
    ) -> Optional[DOMEntity]:
        elapsed = 0.0
        while elapsed < timeout:
            entities = await self._find_by_css_async(selector)
            if entities:
                logger.info(
                    "[WAIT] Element '%s' found after %.1fs",
                    selector, elapsed,
                )
                return entities[0]
            await asyncio.sleep(0.3)
            elapsed += 0.3

        logger.info("[WAIT] Element '%s' not found after %.1fs", selector, timeout)
        return None

    def wait_for_dom_change(self, timeout: float = 5.0) -> PageSnapshot:
        """Wait for DOM to change (snapshot_id differs from current)."""
        try:
            return self._run_async(
                self._wait_for_dom_change_async(timeout),
                timeout=timeout + 5.0,
            )
        except Exception as e:
            logger.debug("[BrowserController] wait_for_dom_change: %s", e)
            return self.get_snapshot(cached=False)

    async def _wait_for_dom_change_async(
        self, timeout: float,
    ) -> PageSnapshot:
        baseline = await self._get_snapshot_async()
        baseline_id = baseline.snapshot_id
        baseline_url = baseline.url

        elapsed = 0.0
        while elapsed < timeout:
            await asyncio.sleep(0.3)
            elapsed += 0.3
            try:
                current = await self._get_snapshot_async()
                # Detect: entity structure change OR URL change (SPA)
                if (
                    current.snapshot_id != baseline_id
                    or current.url != baseline_url
                ):
                    logger.info(
                        "[WAIT] DOM changed after %.1fs "
                        "(id: %s→%s, url: %s→%s)",
                        elapsed,
                        baseline_id[:8], current.snapshot_id[:8],
                        baseline_url[:40], current.url[:40],
                    )
                    return current
            except Exception:
                pass

        logger.debug("[WAIT] No DOM change after %.1fs", timeout)
        return baseline

    def check_visibility(
        self, backend_node_id: int,
    ) -> Dict[str, Any]:
        """Check if an element is truly visible and interactable."""
        try:
            return self._run_async(
                self._check_visibility_async(backend_node_id),
            )
        except Exception as e:
            return {
                "visible": False, "in_viewport": False,
                "bbox": None, "reason": str(e),
            }

    async def _check_visibility_async(
        self, backend_node_id: int,
    ) -> Dict[str, Any]:
        """Full visibility check via JS + CDP.

        Checks: display, visibility, opacity, bounding box,
        viewport intersection, overlay coverage, form proximity.
        """
        page = await self._browser.get_current_page()
        if page is None:
            return {
                "visible": False, "in_viewport": False,
                "bbox": None, "inside_form": False,
                "reason": "no page",
            }

        element = await page.get_element(backend_node_id)

        # Get bounding box
        bbox = await element.get_bounding_box()
        if bbox is None:
            return {
                "visible": False, "in_viewport": False,
                "bbox": None, "inside_form": False,
                "reason": "no bounding box",
            }

        # Combined JS check: CSS visibility + overlay + form proximity
        inside_form = False
        try:
            vis_result = await element.evaluate(
                "() => {"
                "  const s = getComputedStyle(this);"
                "  const rect = this.getBoundingClientRect();"
                "  const cx = rect.left + rect.width / 2;"
                "  const cy = rect.top + rect.height / 2;"
                "  const topEl = document.elementFromPoint(cx, cy);"
                "  const notCovered = !topEl || this === topEl "
                "    || this.contains(topEl) || topEl.contains(this);"
                "  const inForm = !!this.closest('form');"
                "  return JSON.stringify({"
                "    display: s.display,"
                "    visibility: s.visibility,"
                "    opacity: parseFloat(s.opacity),"
                "    notCovered: notCovered,"
                "    inForm: inForm"
                "  });"
                "}"
            )
            import json
            vis_data = json.loads(vis_result)

            inside_form = vis_data.get("inForm", False)

            if vis_data.get("display") == "none":
                return {
                    "visible": False, "in_viewport": False,
                    "bbox": bbox, "inside_form": inside_form,
                    "reason": "display: none",
                }
            if vis_data.get("visibility") == "hidden":
                return {
                    "visible": False, "in_viewport": False,
                    "bbox": bbox, "inside_form": inside_form,
                    "reason": "visibility: hidden",
                }
            if vis_data.get("opacity", 1) <= 0:
                return {
                    "visible": False, "in_viewport": False,
                    "bbox": bbox, "inside_form": inside_form,
                    "reason": "opacity: 0",
                }
            if not vis_data.get("notCovered", True):
                return {
                    "visible": False, "in_viewport": False,
                    "bbox": bbox, "inside_form": inside_form,
                    "reason": "covered by overlay",
                }
        except Exception:
            pass  # If JS fails, assume visible

        # Check viewport intersection
        try:
            state = await self._browser.get_browser_state_summary(
                include_screenshot=False, cached=True,
            )
            vp_w = state.page_info.viewport_width if state.page_info else 1920
            vp_h = state.page_info.viewport_height if state.page_info else 1080

            in_viewport = (
                bbox["x"] + bbox["width"] > 0
                and bbox["y"] + bbox["height"] > 0
                and bbox["x"] < vp_w
                and bbox["y"] < vp_h
            )
        except Exception:
            in_viewport = True  # Assume in viewport if check fails

        return {
            "visible": True,
            "in_viewport": in_viewport,
            "bbox": bbox,
            "inside_form": inside_form,
            "reason": "",
        }

    def scroll_to_element(self, backend_node_id: int) -> None:
        """Scroll element into viewport center via JS."""
        try:
            self._run_async(
                self._scroll_to_element_async(backend_node_id),
            )
        except Exception as e:
            logger.debug("[BrowserController] scroll_to_element: %s", e)

    async def _scroll_to_element_async(
        self, backend_node_id: int,
    ) -> None:
        page = await self._browser.get_current_page()
        if page is None:
            return
        element = await page.get_element(backend_node_id)
        await element.evaluate(
            "() => this.scrollIntoView({block: 'center', behavior: 'smooth'})"
        )
        await asyncio.sleep(0.3)  # Let scroll animation finish

    def submit_form(self, backend_node_id: int) -> None:
        """Submit the form containing the given element via JS."""
        try:
            self._run_async(
                self._submit_form_async(backend_node_id),
            )
        except Exception as e:
            raise RuntimeError(f"form.submit() failed: {e}") from e

    async def _submit_form_async(
        self, backend_node_id: int,
    ) -> None:
        page = await self._browser.get_current_page()
        if page is None:
            raise RuntimeError("No page available")
        element = await page.get_element(backend_node_id)
        result = await element.evaluate(
            "() => {"
            "  const form = this.closest('form');"
            "  if (!form) return 'no_form';"
            "  form.submit();"
            "  return 'ok';"
            "}"
        )
        if result == "no_form":
            raise RuntimeError("Element is not inside a <form>")
    # ─────────────────────────────────────────────────────────
    # Tab Management
    # ─────────────────────────────────────────────────────────

    def switch_tab(self, tab_id: str) -> PageSnapshot:
        """Switch to a tab by its 4-char tab_id.

        Also invalidates snapshot cache and updates BrowserSession/WorldState
        so the rest of MERLIN sees the new active tab.
        """
        try:
            self._run_async(self._switch_tab_async(tab_id))
        except Exception as e:
            logger.error("[BrowserController] switch_tab failed: %s", e)
        self._invalidate_cache()
        return self.get_snapshot(cached=False)

    async def _switch_tab_async(self, tab_id: str) -> None:
        target_id = await self._browser.get_target_id_from_tab_id(tab_id)
        await self._browser.cdp_client.send.Target.activateTarget(
            params={'targetId': target_id},
        )

    def close_tab(self, tab_id: str) -> None:
        """Close a tab by its 4-char tab_id."""
        try:
            self._run_async(self._close_tab_async(tab_id))
        except Exception as e:
            logger.error("[BrowserController] close_tab failed: %s", e)
        self._invalidate_cache()

    async def _close_tab_async(self, tab_id: str) -> None:
        target_id = await self._browser.get_target_id_from_tab_id(tab_id)
        await self._browser.close_page(target_id)

    def get_page_height(self) -> int:
        """Return the current page's total height in pixels.

        Uses cached page_info from the last get_snapshot call.
        Returns 0 if page_info is unavailable.
        """
        if self._cached_page_info:
            return getattr(self._cached_page_info, 'page_height', 0) or 0
        return 0

    # ─────────────────────────────────────────────────────────
    # Structural DOM Clustering
    # ─────────────────────────────────────────────────────────

    def get_full_dom(self) -> Dict[int, Any]:
        """Return raw selector_map from latest browser state.

        Returns the full DOM (all interactive nodes indexed by
        ``backend_node_id``). The map is refreshed automatically
        when ``get_snapshot(cached=False)`` is called.
        """
        if not self._cached_selector_map:
            self.get_snapshot(cached=False)
        return self._cached_selector_map

    # -- helpers --

    @staticmethod
    def _subtree_has_tag(node, tag: str) -> bool:
        """Check if any descendant has the given tag name."""
        for child in (node.children_nodes or []):
            if child.tag_name == tag:
                return True
            if BrowserUseController._subtree_has_tag(child, tag):
                return True
        return False

    @staticmethod
    def _count_subtree_nodes(node) -> int:
        """Count element nodes in subtree (excluding text nodes)."""
        count = 1
        for child in (node.children_nodes or []):
            if hasattr(child, 'node_type') and child.node_type.value == 1:
                count += BrowserUseController._count_subtree_nodes(child)
        return count

    @staticmethod
    def _count_clickable_descendants(node) -> int:
        """Count clickable descendants (a, button, or click-listener)."""
        count = 0
        for child in (node.children_nodes or []):
            if child.tag_name in ("a", "button") or child.has_js_click_listener:
                count += 1
            count += BrowserUseController._count_clickable_descendants(child)
        return count

    @staticmethod
    def _has_clickable_descendant(node) -> bool:
        """Check if any descendant is clickable."""
        for child in (node.children_nodes or []):
            if (child.tag_name in ("a", "button")
                    or child.has_js_click_listener
                    or (child.ax_node and child.ax_node.role in ("link", "button"))):
                return True
            if BrowserUseController._has_clickable_descendant(child):
                return True
        return False

    @staticmethod
    def _find_container(node):
        """Walk ancestors to find the repeating unit (card/tile).

        Finds the first ancestor whose parent has ≥3 children with
        dominant-tag ratio ≥ 0.5. Returns the ancestor itself (the
        repeating child), not the parent list container.

        Example: for a link inside a video card inside a list of cards,
        returns the video card — not the list.
        """
        current = node.parent_node
        while current:
            parent = current.parent_node
            if parent:
                children = [
                    c for c in (parent.children_nodes or [])
                    if hasattr(c, 'node_type') and c.node_type.value == 1
                ]
                if len(children) >= 3:
                    tags = Counter(c.tag_name for c in children)
                    dominant_tag, dominant_count = tags.most_common(1)[0]
                    if dominant_count / len(children) >= 0.5:
                        return current  # current is the repeating child
            current = parent
        return node.parent_node  # fallback: immediate parent

    @staticmethod
    def _container_signature(container) -> str:
        """Build an 11-feature structural signature for a container node.

        Features:
        1. tag_name
        2. depth (xpath segment count)
        3. path_prefix (first 3 xpath segments)
        4. child_tag_histogram
        5. has_image
        6. has_link
        7. text_length_bucket (S/M/L)
        8. aspect_ratio_bucket (W/S/T)
        9. subtree_node_count_bucket (s/m/l)
        10. clickable_descendant_count
        11. container identity (separate from node identity)
        """
        child_tags = Counter(
            c.tag_name for c in (container.children_nodes or [])
            if hasattr(c, 'node_type') and c.node_type.value == 1
        )
        has_img = BrowserUseController._subtree_has_tag(container, "img")
        has_link = BrowserUseController._subtree_has_tag(container, "a")

        text = container.get_all_children_text() if hasattr(container, 'get_all_children_text') else ""
        text_len = len(text or "")
        text_bucket = "S" if text_len < 50 else "M" if text_len < 200 else "L"

        path = container.xpath if hasattr(container, 'xpath') else ""
        path_parts = path.split("/") if path else []
        path_prefix = "/".join(path_parts[:3])
        depth = len(path_parts)

        pos = container.absolute_position
        if pos and pos.height > 0:
            ar = pos.width / pos.height
            ratio = "W" if ar > 2 else "S" if ar > 0.7 else "T"
        else:
            ratio = "?"

        nc = BrowserUseController._count_subtree_nodes(container)
        nc_bucket = "s" if nc < 10 else "m" if nc < 50 else "l"
        cc = BrowserUseController._count_clickable_descendants(container)

        sig = (
            f"{container.tag_name}|d={depth}|p={path_prefix}|"
            f"ct={sorted(child_tags.items())}|"
            f"img={has_img}|link={has_link}|text={text_bucket}|"
            f"ratio={ratio}|nc={nc_bucket}|cc={cc}"
        )
        return hashlib.md5(sig.encode()).hexdigest()[:12]

    @staticmethod
    def _has_consistent_spacing(containers) -> bool:
        """Check if containers have consistent vertical spacing (CV < 0.5)."""
        ys = sorted(
            c.absolute_position.y
            for c in containers
            if c.absolute_position
        )
        if len(ys) < 3:
            return False
        deltas = [ys[i + 1] - ys[i] for i in range(len(ys) - 1)]
        mean = sum(deltas) / len(deltas)
        if mean < 10:
            return False
        variance = sum((d - mean) ** 2 for d in deltas) / len(deltas)
        cv = (variance ** 0.5) / mean
        return cv < 0.5

    def detect_semantic_groups(
        self,
        *,
        min_group_size: int = 3,
        hint_text: Optional[str] = None,
    ) -> List[List[DOMEntity]]:
        """Detect repeated DOM structures via structural clustering.

        Algorithm:
        1. Broad candidate filter (clickable/interactive nodes)
        2. Find container for each candidate (ancestor walk)
        3. Hash 11-feature structural signature per container
        4. Group containers by signature
        5. Deduplicate (one entry per unique container)
        6. Filter group_size >= min_group_size
        7. Spatial validation (vertical rhythm CV < 0.5)
        8. Score groups (size, viewport, spacing, links, images,
           center distance, text similarity)
        9. Sort best group by visual position (y, x)
        10. Return as List[List[DOMEntity]] sorted by score

        Args:
            min_group_size: minimum containers in a group (default 3)
            hint_text: optional search query for text similarity bias

        Returns:
            List of groups, each group being a list of DOMEntity sorted
            by visual position. Groups are sorted by descending score.
        """
        selector_map = self.get_full_dom()
        if not selector_map:
            return []

        # Step 1 — Broad candidate filter
        candidates = []
        for node in selector_map.values():
            if not getattr(node, 'is_visible', False):
                continue
            tag = node.tag_name
            role = node.ax_node.role if node.ax_node else None
            attrs = node.attributes or {}
            if (tag in ("a", "button", "input", "video")
                    or node.has_js_click_listener
                    or role in ("link", "button", "menuitem")
                    or attrs.get("href")
                    or self._has_clickable_descendant(node)):
                candidates.append(node)

        if not candidates:
            return []

        # Step 2 + 3 + 4 — Find containers, compute signatures, group
        # Deduplication: track containers by id() to avoid duplicate entries
        from collections import defaultdict
        container_groups: Dict[str, Dict[int, Any]] = defaultdict(dict)

        for node in candidates:
            container = self._find_container(node)
            if container is None:
                continue
            cid = id(container)
            sig = self._container_signature(container)
            if cid not in container_groups[sig]:
                container_groups[sig][cid] = container

        # Step 5 — Filter by min_group_size
        valid_groups: Dict[str, List[Any]] = {}
        for sig, container_dict in container_groups.items():
            containers = list(container_dict.values())
            if len(containers) >= min_group_size:
                valid_groups[sig] = containers

        if not valid_groups:
            return []

        # Step 6 + 7 — Score groups
        viewport_height = 720
        viewport_width = 1280
        if self._cached_page_info:
            viewport_height = getattr(self._cached_page_info, 'viewport_height', 720)
            viewport_width = getattr(self._cached_page_info, 'viewport_width', 1280)
        scroll_y = 0
        if self._cached_page_info:
            scroll_y = getattr(self._cached_page_info, 'scroll_y', 0)

        page_center_x = viewport_width / 2.0

        hint_tokens = set()
        if hint_text:
            hint_tokens = {t.lower() for t in hint_text.split() if len(t) > 2}

        scored_groups: List[Tuple[float, str, List[Any]]] = []

        # Navigation suppression constants
        _NAV_ANCESTOR_TAGS = {"nav", "header", "footer"}
        _CONTENT_ANCESTOR_TAGS = {"main", "article", "section"}
        _NAV_CLASS_TOKENS = {"nav", "menu", "header", "footer", "sidebar", "toolbar"}

        for sig, containers in valid_groups.items():
            score = 0.0

            # ── Existing signals (retained) ──

            # Group size (cap at 10 for scoring)
            score += min(len(containers), 10) * 4.0

            # Viewport visibility
            visible_count = 0
            for c in containers:
                pos = c.absolute_position
                if pos:
                    cy = pos.y - scroll_y
                    if 0 <= cy <= viewport_height:
                        visible_count += 1
            score += visible_count * 3.0

            # Layout consistency (spatial rhythm)
            if self._has_consistent_spacing(containers):
                score += 3.0 * len(containers)

            # Link presence
            link_count = sum(
                1 for c in containers
                if self._subtree_has_tag(c, "a")
            )
            score += link_count * 2.0

            # Image presence (strengthened: ×1.0 → ×4.0)
            img_count = sum(
                1 for c in containers
                if self._subtree_has_tag(c, "img")
            )
            score += img_count * 4.0

            # Center distance penalty
            for c in containers:
                pos = c.absolute_position
                if pos:
                    cx = pos.x + pos.width / 2.0
                    dist_from_center = abs(cx - page_center_x)
                    penalty = dist_from_center / viewport_width
                    score -= penalty

            # Text similarity to hint
            if hint_tokens:
                for c in containers:
                    text = (c.get_all_children_text() or "").lower()
                    matched = sum(1 for t in hint_tokens if t in text)
                    score += matched * 2.0

            # ── NEW: Content-vs-Navigation discrimination signals ──

            # Signal 1: Semantic ancestor penalty/boost
            # Check first container's ancestry (all share structure)
            sample = containers[0]
            ancestor = sample.parent_node
            while ancestor:
                tag = getattr(ancestor, 'tag_name', '')
                if tag in _NAV_ANCESTOR_TAGS:
                    score -= 15.0  # flat penalty (not ×containers)
                    break
                if tag in _CONTENT_ANCESTOR_TAGS:
                    score += 5.0
                    break
                attrs = getattr(ancestor, 'attributes', {}) or {}
                cls = (attrs.get('class', '') + ' '
                       + attrs.get('id', '')).lower()
                if any(tok in cls for tok in _NAV_CLASS_TOKENS):
                    score -= 10.0  # flat penalty
                    break
                ancestor = getattr(ancestor, 'parent_node', None)

            # Signal 2: Text density (text_length / subtree_nodes)
            # High density = content cards; low density = nav links
            total_text_len = sum(
                len(c.get_all_children_text() or '')
                for c in containers
            )
            total_nodes = sum(
                self._count_subtree_nodes(c) for c in containers
            )
            if total_nodes > 0:
                text_density = total_text_len / total_nodes
                if text_density < 10:      # nav: ~"Home" per node
                    score -= 8.0 * len(containers)
                elif text_density > 30:    # content: rich text per node
                    score += 3.0 * len(containers)

            # Signal 3: Anchor text length
            # Nav links have short anchor text; content cards have long titles
            anchor_texts = []
            for c in containers:
                for child in (c.children_nodes or []):
                    if child.tag_name == "a":
                        t = child.get_all_children_text() if hasattr(
                            child, 'get_all_children_text') else ''
                        anchor_texts.append(len(t or ''))
                    # Recurse one level for nested anchors
                    for grandchild in (child.children_nodes or []):
                        if grandchild.tag_name == "a":
                            t = grandchild.get_all_children_text() if hasattr(
                                grandchild, 'get_all_children_text') else ''
                            anchor_texts.append(len(t or ''))
            if anchor_texts:
                avg_anchor_len = sum(anchor_texts) / len(anchor_texts)
                if avg_anchor_len < 15:
                    score -= 5.0 * len(containers)  # short = nav
                elif avg_anchor_len > 30:
                    score += 3.0 * len(containers)  # long = content

            # Signal 4: Subtree complexity bonus
            avg_nodes = total_nodes / max(len(containers), 1)
            if avg_nodes >= 5:
                score += avg_nodes * 1.5  # deep subtrees = content
            elif avg_nodes <= 2:
                score -= 5.0 * len(containers)  # flat = nav

            # Signal 5: Element height
            heights = [
                c.absolute_position.height for c in containers
                if c.absolute_position and c.absolute_position.height > 0
            ]
            if heights:
                avg_height = sum(heights) / len(heights)
                if avg_height > 80:
                    score += 5.0  # tall = content cards
                elif avg_height < 40:
                    score -= 3.0 * len(containers)  # short = nav items

            scored_groups.append((score, sig, containers))

        # Step 8 — Sort by score descending
        scored_groups.sort(key=lambda x: x[0], reverse=True)

        # Step 9 + 10 — Convert to DOMEntity lists
        result: List[List[DOMEntity]] = []
        for score_val, sig, containers in scored_groups:
            # Sort containers by visual position
            containers.sort(key=lambda c: (
                c.absolute_position.y if c.absolute_position else 9999,
                c.absolute_position.x if c.absolute_position else 9999,
            ))

            entities: List[DOMEntity] = []
            for ordinal, c in enumerate(containers, start=1):
                # Find the primary clickable node within this container
                primary = self._find_primary_clickable(c)
                if primary is None:
                    primary = c  # Self is the clickable

                entity_type = self._classify_entity(primary)
                text = self._get_entity_text(primary)
                url_attr = (primary.attributes.get("href", "")
                            if primary.attributes else "")

                entities.append(DOMEntity(
                    index=ordinal,
                    backend_node_id=primary.backend_node_id,
                    entity_type=entity_type,
                    text=text,
                    url=url_attr or None,
                    ax_role=(primary.ax_node.role if primary.ax_node else None),
                ))

            if entities:
                result.append(entities)

        return result

    @staticmethod
    def _find_primary_clickable(container):
        """Find the most important clickable element in a container.

        Priority: <a> with href > <button> > js click listener > self.
        """
        best = None
        for child in (container.children_nodes or []):
            if child.tag_name == "a" and child.attributes and child.attributes.get("href"):
                return child
            if child.tag_name == "button" and best is None:
                best = child
            if child.has_js_click_listener and best is None:
                best = child
            # Recurse
            deep = BrowserUseController._find_primary_clickable(child)
            if deep is not None:
                if deep.tag_name == "a" and deep.attributes and deep.attributes.get("href"):
                    return deep
                if best is None:
                    best = deep
        return best

    def click_nth_result(
        self,
        ordinal: int,
        hint_text: Optional[str] = None,
    ) -> BrowserResult:
        """Click the nth result in the highest-scoring semantic group.

        Uses structural clustering to detect repeated DOM elements
        (cards, tiles, results), then clicks the nth one by visual order.

        Args:
            ordinal: 1-based index (1 = first result)
            hint_text: optional search query for text similarity bias

        Returns:
            BrowserResult with updated snapshot

        Raises:
            RuntimeError: if no groups found or ordinal out of range
        """
        groups = self.detect_semantic_groups(hint_text=hint_text)
        if not groups:
            raise RuntimeError("No repeated result groups found on page")

        best = groups[0]
        if ordinal < 1 or ordinal > len(best):
            raise RuntimeError(
                f"Ordinal {ordinal} out of range (1-{len(best)} results found)"
            )

        target = best[ordinal - 1]
        logger.info(
            "[CLUSTER] click_nth_result: ordinal=%d backend_node_id=%d "
            "text='%s' entity_type=%s",
            ordinal, target.backend_node_id,
            (target.text or "")[:50], target.entity_type,
        )

        # Scroll into view
        self.scroll_to_element(target.backend_node_id)
        time.sleep(0.15)  # Interaction cooldown

        return self.click(target.backend_node_id)


# ─────────────────────────────────────────────────────────────
# Utility functions (moved from deleted browser_task_runner.py)
# ─────────────────────────────────────────────────────────────

# Cookie/consent dismiss keywords
DISMISS_KEYWORDS = {
    "accept", "agree", "ok", "got it", "i understand",
    "allow", "consent", "continue", "close", "dismiss",
    "reject", "decline", "no thanks",
}

# ARIA roles indicating dialog-type elements
DIALOG_ROLES = {"dialog", "alertdialog"}


def detect_blocking_elements(
    dom_nodes: Dict[str, Any],
    viewport_width: int = 1280,
    viewport_height: int = 720,
) -> List[BlockingElement]:
    """Detect modals, cookie banners, and overlays blocking interaction.

    Detection signals:
    - role="dialog" or role="alertdialog"
    - Text contains cookie/consent keywords + has dismiss button
    - Large overlay covering viewport
    """
    blockers: List[BlockingElement] = []

    for node in dom_nodes.values():
        ax = getattr(node, 'ax_node', None)
        role = ax.role if ax else None
        attrs = getattr(node, 'attributes', {}) or {}
        pos = getattr(node, 'absolute_position', None)
        visible = getattr(node, 'is_visible', False)

        if not visible:
            continue

        # Signal 1: ARIA dialog role
        if role in DIALOG_ROLES:
            dismiss_id, dismiss_text = _find_dismiss_button(node)
            covers = _covers_viewport(pos, viewport_width, viewport_height)
            blockers.append(BlockingElement(
                blocking_type="dialog",
                dismiss_node_id=dismiss_id,
                dismiss_text=dismiss_text,
                covers_viewport=covers,
            ))
            continue

        # Signal 2: Cookie/consent banner detection
        text = ""
        if hasattr(node, 'get_all_children_text'):
            text = (node.get_all_children_text() or "").lower()

        cookie_keywords = {"cookie", "consent", "privacy", "gdpr"}
        if any(kw in text for kw in cookie_keywords) and len(text) < 500:
            dismiss_id, dismiss_text = _find_dismiss_button(node)
            if dismiss_id is not None:
                blockers.append(BlockingElement(
                    blocking_type="cookie_banner",
                    dismiss_node_id=dismiss_id,
                    dismiss_text=dismiss_text,
                    covers_viewport=False,
                ))
                continue

        # Signal 3: Large fixed/absolute overlay
        cls = (attrs.get('class', '') + ' ' + attrs.get('id', '')).lower()
        style = attrs.get('style', '').lower()
        if ('overlay' in cls or 'modal' in cls or 'popup' in cls
                or 'position: fixed' in style or 'position:fixed' in style):
            if pos and _covers_viewport(pos, viewport_width, viewport_height):
                dismiss_id, dismiss_text = _find_dismiss_button(node)
                blockers.append(BlockingElement(
                    blocking_type="overlay",
                    dismiss_node_id=dismiss_id,
                    dismiss_text=dismiss_text,
                    covers_viewport=True,
                ))

    return blockers


def _find_dismiss_button(container) -> tuple:
    """Find a dismiss/close/accept button within a container.

    Returns (backend_node_id, button_text) or (None, "").
    """
    for child in (getattr(container, 'children_nodes', None) or []):
        tag = getattr(child, 'tag_name', '')
        text = ""
        if hasattr(child, 'get_all_children_text'):
            text = (child.get_all_children_text() or "").lower().strip()
        attrs = getattr(child, 'attributes', {}) or {}
        aria_label = attrs.get('aria-label', '').lower()

        is_button = tag in ("button", "a") or getattr(child, 'has_js_click_listener', False)

        if is_button:
            combined_text = text + " " + aria_label
            if any(kw in combined_text for kw in DISMISS_KEYWORDS):
                return (child.backend_node_id, text or aria_label)

        # Recurse
        result = _find_dismiss_button(child)
        if result[0] is not None:
            return result

    return (None, "")


def _covers_viewport(pos, vw: int, vh: int) -> bool:
    """Check if position covers > 50% of viewport."""
    if pos is None:
        return False
    w = getattr(pos, 'width', 0)
    h = getattr(pos, 'height', 0)
    return (w * h) > (vw * vh * 0.5)


def wait_for_stable_dom(
    controller: "BrowserUseController",
    timeout: float = 5.0,
    quiet_ms: int = 300,
) -> PageSnapshot:
    """Wait for DOM to mutate, then wait for it to stop changing.

    Phase 1: Detect first mutation via controller.wait_for_dom_change()
    Phase 2: Quiet-window — ensure rendering is complete (no changes for quiet_ms)

    Uses a single deadline to avoid timeout accounting errors.
    """
    deadline = time.monotonic() + timeout

    # Phase 1: wait for first mutation (delegates to controller)
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return controller.get_snapshot(cached=False)

    snap = controller.wait_for_dom_change(timeout=min(remaining, 3.0))

    # Phase 2: quiet-window — poll until no changes for quiet_ms
    stable_since = time.monotonic()
    last_id = snap.snapshot_id

    while time.monotonic() < deadline:
        elapsed_stable = (time.monotonic() - stable_since) * 1000
        if elapsed_stable >= quiet_ms:
            return snap  # Stable for long enough

        time.sleep(0.1)
        new_snap = controller.get_snapshot(cached=False)
        if new_snap.snapshot_id != last_id:
            # DOM changed again — reset quiet window
            snap = new_snap
            last_id = new_snap.snapshot_id
            stable_since = time.monotonic()

    return snap  # Deadline reached — return latest
