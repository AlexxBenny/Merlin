# infrastructure/browser_controller.py

"""
BrowserController — Abstract interface for browser runtime control.

This is the architectural boundary between:
- SystemController (OS-level app lifecycle: launch, focus, close)
- BrowserController (runtime protocol: CDP, tabs, DOM, navigation)

SystemController launches desktop apps.
BrowserController owns a Chromium instance it can fully control.

"open chrome" → SystemController (just launches)
"search genai in youtube" → BrowserController (controlled Chromium instance)

Same word "chrome". Different domain intent.
Cortex uses SkillContract.domain to route correctly.

Implementation deferred. This file defines the contract only.
Future implementation will use Chromium DevTools Protocol (CDP)
via Playwright or direct CDP connection.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────
# Result types
# ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TabInfo:
    """Lightweight tab descriptor."""
    tab_id: str
    url: str
    title: str


@dataclass(frozen=True)
class PageContent:
    """Extracted page content."""
    url: str
    title: str
    text: str
    html: Optional[str] = None


# ─────────────────────────────────────────────────────────────
# Abstract Interface
# ─────────────────────────────────────────────────────────────

class BrowserController(ABC):
    """
    Abstract browser runtime controller.

    Design constraints (same as SystemController):
    - Stateless from MERLIN's perspective (browser state is browser's)
    - No timeline/event imports: returns results, skills emit events
    - No WorldState dependency: pure infrastructure
    - Timeout-guarded: all operations have bounded execution

    Skills using this controller must declare:
        domain = "browser"
        requires_focus = True

    Skills must NOT use SystemController for browser automation.
    """

    # ── Lifecycle ──

    @abstractmethod
    def launch(self) -> None:
        """
        Launch or connect to a controlled browser instance.
        NOT the same as SystemController.open_app("chrome").
        This creates a programmatically-controlled instance.
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Close the controlled browser instance."""
        ...

    @abstractmethod
    def is_alive(self) -> bool:
        """Check if the controlled instance is still running."""
        ...

    # ── Navigation ──

    @abstractmethod
    def new_tab(self, url: str) -> TabInfo:
        """Open a new tab and navigate to URL."""
        ...

    @abstractmethod
    def navigate(self, url: str) -> None:
        """Navigate the current tab to URL."""
        ...

    @abstractmethod
    def close_tab(self, tab_id: Optional[str] = None) -> None:
        """Close a tab. Current tab if tab_id is None."""
        ...

    @abstractmethod
    def list_tabs(self) -> List[TabInfo]:
        """List all open tabs."""
        ...

    # ── Page Interaction ──

    @abstractmethod
    def click(self, selector: str) -> None:
        """Click an element by CSS selector."""
        ...

    @abstractmethod
    def type_text(self, selector: str, text: str) -> None:
        """Type text into an element."""
        ...

    @abstractmethod
    def search(self, query: str, engine: str = "google") -> None:
        """Perform a search using specified engine."""
        ...

    # ── Content Extraction ──

    @abstractmethod
    def get_page_text(self) -> str:
        """Extract visible text content from current page."""
        ...

    @abstractmethod
    def get_page_content(self) -> PageContent:
        """Extract full page content (text + metadata)."""
        ...

    @abstractmethod
    def get_comments(self) -> List[str]:
        """
        Extract comments from current page.
        Platform-aware: YouTube, Reddit, etc.
        """
        ...

    @abstractmethod
    def screenshot(self) -> bytes:
        """Capture screenshot of current page as PNG bytes."""
        ...
