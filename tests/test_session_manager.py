# tests/test_session_manager.py

"""
Tests for SessionManager, AppCapabilityRegistry, SessionStack, TaskStack.

Tests cover:
- Session CRUD (create, get, update, close)
- AppCapabilityRegistry (lookup, defaults, normalization)
- SessionStack (push, pop, peek, dedup, remove)
- TaskStack (push, pop, peek, scoping)
- SessionManager.build_session_context() output
- TTL cleanup (with mock observer)
- Thread safety (concurrent session creation)
- Interaction: create session via open_app, close via close_app,
  update focus via focus_app
"""

import threading
from unittest.mock import MagicMock

import pytest

from infrastructure.app_capabilities import (
    AppCapabilities,
    AppCapabilityRegistry,
)
from infrastructure.session import (
    AppSession,
    BrowserSession,
    SessionManager,
    SessionStack,
    SessionType,
    TaskStack,
)


# ─────────────────────────────────────────────────────────────
# AppCapabilityRegistry
# ─────────────────────────────────────────────────────────────


class TestAppCapabilityRegistry:

    def test_known_app_lookup(self):
        reg = AppCapabilityRegistry.from_dict({
            "notepad": {"supports_typing": True, "supports_copy": True, "supports_save": True},
            "spotify": {"supports_typing": False, "supports_copy": False, "supports_save": False},
        })
        caps = reg.get("notepad")
        assert caps.supports_typing is True
        assert caps.supports_copy is True
        assert caps.supports_save is True

        caps2 = reg.get("spotify")
        assert caps2.supports_typing is False

    def test_unknown_app_returns_default(self):
        reg = AppCapabilityRegistry.from_dict({
            "_default": {"supports_typing": False, "supports_copy": False, "supports_save": False},
        })
        caps = reg.get("unknown_app")
        assert caps.supports_typing is False
        assert caps.supports_copy is False
        assert caps.supports_save is False

    def test_case_insensitive_lookup(self):
        reg = AppCapabilityRegistry.from_dict({
            "Notepad": {"supports_typing": True, "supports_copy": True, "supports_save": True},
        })
        assert reg.get("notepad").supports_typing is True
        assert reg.get("NOTEPAD").supports_typing is True

    def test_strip_exe_suffix(self):
        reg = AppCapabilityRegistry.from_dict({
            "notepad": {"supports_typing": True, "supports_copy": True, "supports_save": True},
        })
        assert reg.get("notepad.exe").supports_typing is True

    def test_supports_method(self):
        reg = AppCapabilityRegistry.from_dict({
            "notepad": {"supports_typing": True, "supports_copy": True, "supports_save": True},
        })
        assert reg.supports("notepad", "supports_typing") is True
        assert reg.supports("notepad", "supports_save") is True
        assert reg.supports("spotify", "supports_typing") is False

    def test_known_apps_property(self):
        reg = AppCapabilityRegistry.from_dict({
            "notepad": {"supports_typing": True, "supports_copy": True, "supports_save": True},
            "chrome": {"supports_typing": True, "supports_copy": True, "supports_save": False},
        })
        assert "notepad" in reg.known_apps
        assert "chrome" in reg.known_apps

    def test_from_yaml_missing_file(self, tmp_path):
        reg = AppCapabilityRegistry.from_yaml(str(tmp_path / "nonexistent.yaml"))
        # Returns empty registry with all-false defaults
        caps = reg.get("anything")
        assert caps.supports_typing is False


# ─────────────────────────────────────────────────────────────
# SessionStack
# ─────────────────────────────────────────────────────────────


class TestSessionStack:

    def test_push_and_peek(self):
        stack = SessionStack()
        stack.push("s1")
        assert stack.peek() == "s1"

    def test_push_dedup(self):
        stack = SessionStack()
        stack.push("s1")
        stack.push("s2")
        stack.push("s1")  # Move to top
        assert stack.peek() == "s1"
        assert len(stack) == 2

    def test_pop(self):
        stack = SessionStack()
        stack.push("s1")
        stack.push("s2")
        assert stack.pop() == "s2"
        assert stack.peek() == "s1"

    def test_pop_empty(self):
        stack = SessionStack()
        assert stack.pop() is None

    def test_remove(self):
        stack = SessionStack()
        stack.push("s1")
        stack.push("s2")
        stack.remove("s1")
        assert len(stack) == 1
        assert stack.peek() == "s2"

    def test_get_stack(self):
        stack = SessionStack()
        stack.push("s1")
        stack.push("s2")
        result = stack.get_stack()
        assert result == ["s1", "s2"]
        # Ensure it's a copy
        result.append("s3")
        assert len(stack) == 2


# ─────────────────────────────────────────────────────────────
# TaskStack
# ─────────────────────────────────────────────────────────────


class TestTaskStack:

    def test_push_and_peek(self):
        stack = TaskStack()
        entry = stack.push("task1", "Write a story")
        assert entry.task_id == "task1"
        assert entry.description == "Write a story"
        assert stack.peek().task_id == "task1"

    def test_pop(self):
        stack = TaskStack()
        stack.push("task1", "First task")
        stack.push("task2", "Second task")
        popped = stack.pop()
        assert popped.task_id == "task2"
        assert stack.peek().task_id == "task1"

    def test_pop_empty(self):
        stack = TaskStack()
        assert stack.pop() is None

    def test_session_ids(self):
        stack = TaskStack()
        entry = stack.push("task1", "Edit document", session_ids=["s1", "s2"])
        assert entry.session_ids == ["s1", "s2"]


# ─────────────────────────────────────────────────────────────
# SessionManager — CRUD
# ─────────────────────────────────────────────────────────────


class TestSessionManagerCRUD:

    def _make_manager(self):
        reg = AppCapabilityRegistry.from_dict({
            "notepad": {"supports_typing": True, "supports_copy": True, "supports_save": True},
        })
        return SessionManager(capability_registry=reg)

    def test_create_app_session(self):
        mgr = self._make_manager()
        session = mgr.create_app_session(app_name="notepad", pid=1234)
        assert isinstance(session, AppSession)
        assert session.app_name == "notepad"
        assert session.pid == 1234
        assert session.type == SessionType.APP
        assert mgr.session_count == 1

    def test_get_session_by_id(self):
        mgr = self._make_manager()
        session = mgr.create_app_session(app_name="notepad", pid=1234)
        retrieved = mgr.get_session(session.id)
        assert retrieved is session

    def test_get_session_by_app(self):
        mgr = self._make_manager()
        mgr.create_app_session(app_name="notepad", pid=1234)
        session = mgr.get_session_by_app("notepad")
        assert session is not None
        assert session.app_name == "notepad"

    def test_get_session_by_app_case_insensitive(self):
        mgr = self._make_manager()
        mgr.create_app_session(app_name="Notepad", pid=1234)
        session = mgr.get_session_by_app("notepad")
        assert session is not None

    def test_update_session(self):
        mgr = self._make_manager()
        session = mgr.create_app_session(app_name="notepad", pid=1234)
        updated = mgr.update_session(session.id, is_focused=True)
        assert updated.is_focused is True

    def test_close_session(self):
        mgr = self._make_manager()
        session = mgr.create_app_session(app_name="notepad", pid=1234)
        closed = mgr.close_session(session.id)
        assert closed is not None
        assert mgr.session_count == 0

    def test_close_session_by_app(self):
        mgr = self._make_manager()
        mgr.create_app_session(app_name="notepad", pid=1234)
        closed = mgr.close_session_by_app("notepad")
        assert closed is not None
        assert mgr.session_count == 0

    def test_dedup_on_create(self):
        """Creating a session for same app updates existing instead of duplicating."""
        mgr = self._make_manager()
        s1 = mgr.create_app_session(app_name="notepad", pid=1234)
        s2 = mgr.create_app_session(app_name="notepad", pid=5678)
        # Should be same session, updated
        assert s1.id == s2.id
        assert mgr.session_count == 1
        assert s2.pid == 5678

    def test_get_sessions_by_type(self):
        mgr = self._make_manager()
        mgr.create_app_session(app_name="notepad", pid=1234)
        mgr.create_app_session(app_name="chrome", pid=5678)
        app_sessions = mgr.get_sessions_by_type(SessionType.APP)
        assert len(app_sessions) == 2

    def test_get_active_sessions(self):
        mgr = self._make_manager()
        mgr.create_app_session(app_name="notepad", pid=1234)
        mgr.create_app_session(app_name="chrome", pid=5678)
        sessions = mgr.get_active_sessions()
        assert len(sessions) == 2


# ─────────────────────────────────────────────────────────────
# SessionManager — build_session_context
# ─────────────────────────────────────────────────────────────


class TestSessionContext:

    def test_empty_context(self):
        mgr = SessionManager()
        context = mgr.build_session_context()
        assert context == {}

    def test_app_session_in_context(self):
        reg = AppCapabilityRegistry.from_dict({
            "notepad": {"supports_typing": True, "supports_copy": True, "supports_save": True},
        })
        mgr = SessionManager(capability_registry=reg)
        mgr.create_app_session(app_name="notepad", pid=1234)

        context = mgr.build_session_context()
        assert "open_applications" in context
        apps = context["open_applications"]
        assert len(apps) == 1
        assert apps[0]["app_name"] == "notepad"
        assert apps[0]["capabilities"]["supports_typing"] is True

    def test_task_stack_in_context(self):
        mgr = SessionManager()
        mgr.task_stack.push("t1", "Write a story")
        context = mgr.build_session_context()
        assert "active_tasks" in context
        assert context["active_tasks"][0]["description"] == "Write a story"


# ─────────────────────────────────────────────────────────────
# SessionManager — TTL cleanup
# ─────────────────────────────────────────────────────────────


class TestSessionCleanup:

    def test_cleanup_stale_sessions(self):
        mgr = SessionManager()
        mgr.create_app_session(app_name="notepad", pid=1234)
        mgr.create_app_session(app_name="chrome", pid=5678)

        # Mock observer: notepad is running, chrome is not
        observer = MagicMock()
        observer.is_app_running.side_effect = lambda name: name.lower() == "notepad"

        closed = mgr.cleanup_stale_sessions(observer=observer)
        assert len(closed) == 1
        assert mgr.session_count == 1
        # Notepad should survive
        assert mgr.get_session_by_app("notepad") is not None

    def test_cleanup_without_observer(self):
        mgr = SessionManager()
        mgr.create_app_session(app_name="notepad", pid=1234)
        closed = mgr.cleanup_stale_sessions(observer=None)
        assert closed == []
        assert mgr.session_count == 1


# ─────────────────────────────────────────────────────────────
# Thread safety
# ─────────────────────────────────────────────────────────────


class TestThreadSafety:

    def test_concurrent_session_creation(self):
        mgr = SessionManager()
        errors = []

        def create_session(i):
            try:
                mgr.create_app_session(app_name=f"app_{i}", pid=1000 + i)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=create_session, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert mgr.session_count == 20

    def test_concurrent_stack_ops(self):
        stack = SessionStack()
        errors = []

        def push_pop(i):
            try:
                stack.push(f"s_{i}")
                stack.peek()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=push_pop, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
