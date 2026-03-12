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
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from infrastructure.browser_controller import (
    BrowserController,
    BrowserResult,
    DOMEntity,
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

        Detects same-loop calls to prevent deadlock.
        """
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
            return BrowserResult(success=False, error="No active page")

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
