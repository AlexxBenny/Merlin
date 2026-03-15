# infrastructure/browser_controller.py

"""
BrowserController — Abstract interface for deterministic browser control.

Architectural boundary:
- SystemController (OS-level app lifecycle: launch, focus, close)
- BrowserController (runtime protocol: CDP, DOM, navigation, entities)

SystemController launches desktop apps.
BrowserController owns a Chromium instance it can fully control.

"open chrome" → SystemController (just launches)
"search genai in youtube" → BrowserController (controlled Chromium instance)

Design rules (same as SystemController):
- No timeline/event imports: returns results, skills emit events
- No WorldState dependency: pure infrastructure
- Timeout-guarded: all operations have bounded execution
- Pure deterministic: NO LLM calls inside the controller
- Semantic operations (extract_content, find_element) belong in SKILLS, not here

Skills using this controller must declare:
    domain = "browser"
    requires_focus = True
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────
# Result types (pure data, no logic)
# ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TabInfo:
    """Lightweight tab descriptor."""
    tab_id: str
    url: str
    title: str


@dataclass(frozen=True)
class DOMEntity:
    """A classified interactive page element.

    Identity:
        backend_node_id is the TRUE identity — stable within a snapshot.
        index is ephemeral display order — assigned by sorting
        backend_node_ids deterministically. Used for LLM readability only.

    Entity types:
        link, button, input, media, image, clickable, other
    """
    index: int                          # Ephemeral display index (NOT identity)
    backend_node_id: int                # TRUE identity — stable within snapshot
    entity_type: str                    # "link", "button", "input", "media", "image", "clickable"
    text: str                           # Meaningful text (ax_name priority)
    url: Optional[str] = None           # href for links
    ax_role: Optional[str] = None       # Raw accessibility role for debug


@dataclass(frozen=True)
class PageSnapshot:
    """Versioned page state.

    snapshot_id is derived from DOM structure (sorted backend_node_ids),
    NOT from timestamps. It changes only when interactive DOM changes.
    """
    snapshot_id: str                    # hash(sorted(backend_node_ids))
    url: str
    title: str
    entities: tuple                     # Tuple[DOMEntity, ...] — immutable
    entity_count: int                   # Total selector_map size (pre-filter)
    tab_count: int
    tabs: tuple = ()                    # Tuple[TabInfo, ...]
    scroll_pct: Optional[float] = None  # Vertical scroll percentage
    timestamp: float = 0.0


@dataclass(frozen=True)
class BrowserResult:
    """Result of any deterministic browser action."""
    success: bool
    snapshot: Optional[PageSnapshot] = None
    error: Optional[str] = None
    navigated: bool = False             # True if URL changed after action
    new_tab_opened: bool = False        # True if tab count increased


@dataclass
class ElementScore:
    """Scored element candidate from multi-strategy resolution.

    Used by locate_element() to rank candidates deterministically.
    Higher score = better match.
    """
    entity: DOMEntity
    score: float
    strategy: str                       # Which strategy matched
    details: str = ""                   # Human-readable match info


# ─────────────────────────────────────────────────────────────
# Blocking element detection
# ─────────────────────────────────────────────────────────────

@dataclass
class BlockingElement:
    """A modal, cookie banner, or overlay blocking page interaction."""
    blocking_type: str                  # "dialog", "cookie_banner", "overlay"
    dismiss_node_id: Optional[int] = None  # backend_node_id of dismiss button
    dismiss_text: str = ""              # Text of the dismiss button
    covers_viewport: bool = False       # True if > 50% viewport
    z_index: int = 0


# ─────────────────────────────────────────────────────────────
# Abstract Interface
# ─────────────────────────────────────────────────────────────

class BrowserController(ABC):
    """
    Abstract deterministic browser controller.

    All methods are pure infrastructure — no LLM calls.
    Semantic operations (extract_content, find_element_by_prompt)
    belong in browser skills, not here.

    Tier 1 — Deterministic primitives:
        navigate, click, fill, press_key, scroll, get_snapshot

    Tier 2 — Reactive controller methods:
        find_by_css, locate_element, wait_for_element, wait_for_dom_change,
        check_visibility

    Skills using this controller must declare:
        domain = "browser"
        requires_focus = True

    Skills must NOT use SystemController for browser automation.
    """

    # ── Navigation ──

    @abstractmethod
    def navigate(self, url: str) -> BrowserResult:
        """Navigate the current tab to URL."""
        ...

    @abstractmethod
    def go_back(self) -> BrowserResult:
        """Navigate back in history."""
        ...

    @abstractmethod
    def go_forward(self) -> BrowserResult:
        """Navigate forward in history."""
        ...

    # ── Page Interaction (by backend_node_id) ──

    @abstractmethod
    def click(self, backend_node_id: int) -> BrowserResult:
        """Click an element by its backend node ID.

        Skills resolve user-facing index → backend_node_id from
        the current snapshot before calling this method.
        """
        ...

    @abstractmethod
    def fill(self, backend_node_id: int, text: str) -> BrowserResult:
        """Fill an input element with text.

        Clears existing content before typing.
        """
        ...

    @abstractmethod
    def press_key(self, key: str) -> BrowserResult:
        """Press a keyboard key on the current page.

        Common keys: Enter, Escape, Tab, ArrowDown, ArrowUp, Space, Backspace.
        Used for form submission (Enter), modal dismissal (Escape), and
        keyboard navigation (Tab, Arrow keys).
        """
        ...

    # ── Scrolling ──

    @abstractmethod
    def scroll_page(self, direction: str, amount: int = 3) -> BrowserResult:
        """Scroll the page viewport.

        Args:
            direction: "up" or "down"
            amount: Number of page-scroll increments
        """
        ...

    @abstractmethod
    def scroll_element(
        self, backend_node_id: int, direction: str,
    ) -> BrowserResult:
        """Scroll a specific scrollable container element.

        Used for virtual-scroll containers (Reddit, Twitter, YouTube comments).
        """
        ...

    # ── State ──

    @abstractmethod
    def get_snapshot(self, cached: bool = True) -> PageSnapshot:
        """Get the current page state as a versioned snapshot.

        Args:
            cached: If True, return cached snapshot if available.
                    If False, force a fresh DOM traversal.

        Returns:
            PageSnapshot with interactive-only filtered entities
            sorted deterministically by backend_node_id.
        """
        ...

    @abstractmethod
    def find_entities(self, text: str) -> List[DOMEntity]:
        """Search entities by text — pure tokenized match, no LLM.

        Matching strategy:
        1. All query tokens are subset of entity tokens → match
        2. Fallback: any token overlap → match
        """
        ...

    # ── Tab Management ──

    @abstractmethod
    def list_tabs(self) -> List[TabInfo]:
        """List all open tabs."""
        ...

    @abstractmethod
    def switch_tab(self, tab_id: str) -> BrowserResult:
        """Switch agent focus to a different tab."""
        ...

    # ── Lifecycle ──

    @abstractmethod
    def is_alive(self) -> bool:
        """Check if the browser connection is alive."""
        ...

    # ── Reactive Controller Methods (Tier 2) ──

    @abstractmethod
    def find_by_css(self, selector: str) -> List[DOMEntity]:
        """Find elements by CSS selector — deterministic, structural.

        Uses Page.get_elements_by_css_selector() for exact DOM queries.
        Returns DOMEntity list with backend_node_ids.

        Example selectors:
            'input[type="search"]'
            'button[aria-label*="Submit"]'
            '[role="searchbox"]'
        """
        ...

    @abstractmethod
    def locate_element(
        self,
        *,
        text: Optional[str] = None,
        role: Optional[str] = None,
        css: Optional[str] = None,
        element_type: Optional[str] = None,
    ) -> Optional[ElementScore]:
        """Multi-strategy element location with scoring. No LLM.

        Tries resolution strategies in priority order:
            1. CSS selector (structural, most stable)
            2. ARIA role match
            3. Attribute match (name, aria-label, placeholder)
            4. Text content match
            5. DOM heuristics (largest visible input, etc.)

        Scores candidates:
            +3  visible in viewport
            +5  aria-label/placeholder contains keyword
            +2  inside <form> element
            +2  large bounding box (prominent element)
            +1  near top of page

        Returns highest-scoring ElementScore or None.
        """
        ...

    @abstractmethod
    def wait_for_element(
        self, selector: str, timeout: float = 5.0,
    ) -> Optional[DOMEntity]:
        """Poll DOM until element matching CSS selector appears.

        Polls every 300ms up to timeout. Returns first match or None.
        Used for: AJAX responses, lazy-loaded content, SPA transitions.
        """
        ...

    @abstractmethod
    def wait_for_dom_change(self, timeout: float = 5.0) -> PageSnapshot:
        """Wait for DOM to change (snapshot_id differs from current).

        Detects: element count changes, node additions/removals,
        URL changes (SPA navigation), content mutations.

        Returns snapshot after change, or current snapshot on timeout.
        """
        ...

    @abstractmethod
    def check_visibility(self, backend_node_id: int) -> Dict[str, Any]:
        """Check if an element is truly visible and interactable.

        Checks: display != none, visibility != hidden, opacity > 0,
        bounding box exists, within viewport bounds, not covered by
        overlay (elementFromPoint), form proximity.

        Returns dict with:
            visible: bool
            in_viewport: bool
            bbox: {x, y, width, height} or None
            inside_form: bool
            reason: str (if not visible)
        """
        ...

    @abstractmethod
    def scroll_to_element(self, backend_node_id: int) -> None:
        """Scroll element into viewport center.

        Uses scrollIntoView({block: 'center'}) via JS.
        Must be called before clicking off-screen elements.
        """
        ...

    @abstractmethod
    def submit_form(self, backend_node_id: int) -> None:
        """Submit the form containing the given element.

        Uses element.closest('form')?.submit() via JS.
        Raises RuntimeError if element is not inside a form.
        """
        ...

    @abstractmethod
    def get_full_dom(self) -> Dict[str, Any]:
        """Return raw selector_map from latest browser state.

        Returns the full DOM (all interactive nodes). The map is
        refreshed when ``get_snapshot(cached=False)`` is called.
        """
        ...

    @abstractmethod
    def detect_semantic_groups(
        self,
        *,
        min_group_size: int = 3,
        hint_text: Optional[str] = None,
    ) -> List[List[DOMEntity]]:
        """Detect repeated DOM structures via structural clustering.

        Returns list of groups sorted by score. Each group is a list
        of DOMEntity sorted by visual position (y, x).
        """
        ...

    @abstractmethod
    def click_nth_result(
        self,
        ordinal: int,
        hint_text: Optional[str] = None,
    ) -> "BrowserResult":
        """Click the nth result in the highest-scoring group.

        Args:
            ordinal: 1-based index (1 = first result)
            hint_text: optional query for text similarity bias

        Returns:
            BrowserResult with updated snapshot
        """
        ...

