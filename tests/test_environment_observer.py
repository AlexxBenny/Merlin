# tests/test_environment_observer.py

"""
Tests for EnvironmentObserver protocol and SystemEnvironmentObserver.

Tests cover:
- Protocol conformance (structural typing check)
- is_app_running (via mocked SystemController)
- is_app_focused (via mocked get_active_window)
- file_exists (real filesystem)
- get_active_window (mocked win32 calls)
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from infrastructure.observer import (
    EnvironmentObserver,
    SystemEnvironmentObserver,
)
from infrastructure.system_controller import WindowInfo


# ─────────────────────────────────────────────────────────────
# Protocol conformance
# ─────────────────────────────────────────────────────────────


class TestProtocolConformance:

    def test_system_observer_matches_protocol(self):
        """SystemEnvironmentObserver satisfies the EnvironmentObserver protocol."""
        ctrl = MagicMock()
        observer = SystemEnvironmentObserver(controller=ctrl)
        assert isinstance(observer, EnvironmentObserver)


# ─────────────────────────────────────────────────────────────
# is_app_running
# ─────────────────────────────────────────────────────────────


class TestIsAppRunning:

    def test_returns_true_when_windows_found(self):
        ctrl = MagicMock()
        ctrl.find_windows.return_value = [
            WindowInfo(title="Untitled - Notepad", hwnd=12345, pid=1000, app_name="notepad.exe"),
        ]
        observer = SystemEnvironmentObserver(controller=ctrl)
        assert observer.is_app_running("notepad") is True
        ctrl.find_windows.assert_called_once_with("notepad")

    def test_returns_false_when_no_windows(self):
        ctrl = MagicMock()
        ctrl.find_windows.return_value = []
        observer = SystemEnvironmentObserver(controller=ctrl)
        assert observer.is_app_running("notepad") is False

    def test_returns_false_on_exception(self):
        ctrl = MagicMock()
        ctrl.find_windows.side_effect = RuntimeError("win32 error")
        observer = SystemEnvironmentObserver(controller=ctrl)
        assert observer.is_app_running("notepad") is False


# ─────────────────────────────────────────────────────────────
# is_app_focused
# ─────────────────────────────────────────────────────────────


class TestIsAppFocused:

    def test_returns_true_when_focused(self):
        ctrl = MagicMock()
        observer = SystemEnvironmentObserver(controller=ctrl)

        # Mock get_active_window to return a notepad window
        observer.get_active_window = MagicMock(return_value=WindowInfo(
            title="Untitled - Notepad",
            hwnd=12345,
            pid=1000,
            app_name="notepad.exe",
        ))

        assert observer.is_app_focused("notepad") is True

    def test_returns_false_when_different_app_focused(self):
        ctrl = MagicMock()
        observer = SystemEnvironmentObserver(controller=ctrl)

        observer.get_active_window = MagicMock(return_value=WindowInfo(
            title="Google Chrome",
            hwnd=67890,
            pid=2000,
            app_name="chrome.exe",
        ))

        assert observer.is_app_focused("notepad") is False

    def test_returns_false_when_no_active_window(self):
        ctrl = MagicMock()
        observer = SystemEnvironmentObserver(controller=ctrl)
        observer.get_active_window = MagicMock(return_value=None)

        assert observer.is_app_focused("notepad") is False


# ─────────────────────────────────────────────────────────────
# file_exists
# ─────────────────────────────────────────────────────────────


class TestFileExists:

    def test_existing_file(self, tmp_path):
        ctrl = MagicMock()
        observer = SystemEnvironmentObserver(controller=ctrl)

        test_file = tmp_path / "test.txt"
        test_file.write_text("hello")

        assert observer.file_exists(str(test_file)) is True

    def test_nonexistent_file(self):
        ctrl = MagicMock()
        observer = SystemEnvironmentObserver(controller=ctrl)
        assert observer.file_exists("/nonexistent/path/file.txt") is False

    def test_existing_directory(self, tmp_path):
        ctrl = MagicMock()
        observer = SystemEnvironmentObserver(controller=ctrl)
        assert observer.file_exists(str(tmp_path)) is True


# ─────────────────────────────────────────────────────────────
# Integration: observer + session cleanup
# ─────────────────────────────────────────────────────────────


class TestObserverSessionIntegration:

    def test_cleanup_uses_observer(self):
        """SessionManager.cleanup_stale_sessions uses observer.is_app_running."""
        from infrastructure.session import SessionManager, LAUNCH_GRACE_SECONDS
        import time

        mgr = SessionManager()
        mgr.create_app_session(app_name="notepad", pid=1000)
        mgr.create_app_session(app_name="chrome", pid=2000)

        # Backdate sessions so they are past the grace period
        for session in mgr._sessions.values():
            session.created_at = time.time() - LAUNCH_GRACE_SECONDS - 1

        # Mock observer: notepad is running, chrome is not
        ctrl = MagicMock()
        ctrl.find_windows.side_effect = lambda name: (
            [WindowInfo(title="x", hwnd=1, pid=1000, app_name="notepad.exe")]
            if "notepad" in name.lower()
            else []
        )
        observer = SystemEnvironmentObserver(controller=ctrl)

        closed = mgr.cleanup_stale_sessions(observer=observer)
        assert len(closed) == 1
        assert mgr.session_count == 1
        assert mgr.get_session_by_app("notepad") is not None
        assert mgr.get_session_by_app("chrome") is None
