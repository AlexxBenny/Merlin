# infrastructure/observer.py

"""
EnvironmentObserver — Pull-based environment query interface.

The observer provides synchronous, point-in-time queries about the
current OS environment. It is the "eyes" of the ExecutionSupervisor.

Architectural rules:
- Pull-based only — no polling, no threads, no callbacks
- Queries are synchronous and cheap (single win32 call each)
- Reuses SystemController for all OS interaction
- Stateless — no caching, always fresh data
- Used by: ExecutionSupervisor (guard evaluation), SessionManager (TTL cleanup)
- Never imported by skills — skills don't observe, they execute

Protocol pattern: EnvironmentObserver is a Protocol (structural typing).
SystemEnvironmentObserver is the concrete Windows implementation.
Other implementations (Linux, mock) can be added without changing callers.
"""

import logging
import os
from typing import Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# WindowInfo re-export (from system_controller)
# ─────────────────────────────────────────────────────────────

from infrastructure.system_controller import WindowInfo


# ─────────────────────────────────────────────────────────────
# Protocol definition
# ─────────────────────────────────────────────────────────────

@runtime_checkable
class EnvironmentObserver(Protocol):
    """Pull-based environment query interface.

    All methods are synchronous. All methods are safe to call
    from any thread. All methods return clean defaults on failure
    (None, False, empty string) — they never raise.
    """

    def get_active_window(self) -> Optional[WindowInfo]:
        """Return info about the currently focused window, or None."""
        ...

    def is_app_running(self, app_name: str) -> bool:
        """Check if an app has at least one visible window."""
        ...

    def is_app_focused(self, app_name: str) -> bool:
        """Check if an app currently has foreground focus."""
        ...

    def file_exists(self, path: str) -> bool:
        """Check if a file exists at the given path."""
        ...

    def get_clipboard_text(self) -> Optional[str]:
        """Return current clipboard text content, or None."""
        ...


# ─────────────────────────────────────────────────────────────
# Concrete implementation (Windows, via SystemController)
# ─────────────────────────────────────────────────────────────

class SystemEnvironmentObserver:
    """Concrete EnvironmentObserver using SystemController.

    Delegates all OS queries to SystemController methods.
    Adds get_active_window() via win32gui.GetForegroundWindow().

    Never raises. Returns safe defaults on failure.
    """

    def __init__(self, controller):
        """
        Args:
            controller: SystemController instance.
        """
        self._ctrl = controller

    def get_active_window(self) -> Optional[WindowInfo]:
        """Return info about the currently focused window."""
        try:
            import sys
            if sys.platform != "win32":
                return None

            import win32gui
            import win32process
            import psutil

            hwnd = win32gui.GetForegroundWindow()
            if not hwnd:
                return None

            title = win32gui.GetWindowText(hwnd)
            if not title:
                return None

            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            try:
                proc = psutil.Process(pid)
                app_name = proc.name()
            except (psutil.NoSuchProcess, Exception):
                app_name = None

            return WindowInfo(
                title=title,
                hwnd=hwnd,
                pid=pid,
                app_name=app_name,
            )
        except Exception as e:
            logger.debug("get_active_window failed: %s", e)
            return None

    def is_app_running(self, app_name: str) -> bool:
        """Check if an app has at least one visible window."""
        try:
            windows = self._ctrl.find_windows(app_name)
            return len(windows) > 0
        except Exception as e:
            logger.debug("is_app_running failed for '%s': %s", app_name, e)
            return False

    def is_app_focused(self, app_name: str) -> bool:
        """Check if an app currently has foreground focus."""
        try:
            active = self.get_active_window()
            if active is None or active.app_name is None:
                return False
            return app_name.lower() in active.app_name.lower()
        except Exception as e:
            logger.debug("is_app_focused failed for '%s': %s", app_name, e)
            return False

    def file_exists(self, path: str) -> bool:
        """Check if a file exists at the given path."""
        try:
            return os.path.exists(path)
        except Exception:
            return False

    def get_clipboard_text(self) -> Optional[str]:
        """Return current clipboard text content, or None."""
        try:
            import sys
            if sys.platform != "win32":
                return None

            import win32clipboard
            win32clipboard.OpenClipboard()
            try:
                if win32clipboard.IsClipboardFormatAvailable(
                    win32clipboard.CF_UNICODETEXT
                ):
                    data = win32clipboard.GetClipboardData(
                        win32clipboard.CF_UNICODETEXT
                    )
                    return str(data)
                return None
            finally:
                win32clipboard.CloseClipboard()
        except Exception as e:
            logger.debug("get_clipboard_text failed: %s", e)
            return None
