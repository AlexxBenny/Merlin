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


# ─────────────────────────────────────────────────────────────
# Abstract Interface
# ─────────────────────────────────────────────────────────────

class BrowserController(ABC):
    """
    Abstract deterministic browser controller.

    All methods are pure infrastructure — no LLM calls.
    Semantic operations (extract_content, find_element_by_prompt)
    belong in browser skills, not here.

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
