# infrastructure/session.py

"""
SessionManager — MERLIN-managed interactive session handles.

Sessions represent MERLIN's tracked interactive environments.
They are NOT WorldState observations — they are internal handles
that MERLIN creates, updates, and destroys.

Architectural rules:
- Sessions are MERLIN-managed, not OS-observed
- WorldState.SessionState = OS reality (foreground_app, open_apps)
- SessionManager = MERLIN interaction handles (document buffers, unsaved state)
- A session summary is exposed to the compiler via build_session_context()
  as a SEPARATE prompt section, never embedded in WorldState
- SessionStack = which app/session is currently active (auto-updated from OS)
- TaskStack    = which workflow MERLIN is pursuing (never auto-updated)

TTL cleanup:
- SessionManager periodically validates sessions against the observer
  (Phase 2). Stale sessions (app closed externally) are cleaned up.

Thread safety:
- SessionManager is thread-safe. Called from executor threads (skill
  execution) and main thread (mission orchestration).
"""

import logging
import threading
import time
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Session types
# ─────────────────────────────────────────────────────────────

class SessionType(str, Enum):
    """Type of interactive session."""
    APP = "app"
    BROWSER = "browser"
    TERMINAL = "terminal"
    DOCUMENT = "document"


# ─────────────────────────────────────────────────────────────
# Session models
# ─────────────────────────────────────────────────────────────

class Session(BaseModel):
    """Base session — tracked interactive environment handle.

    Sessions are MERLIN-managed objects. They represent
    the state of an interactive environment that MERLIN
    has opened or is interacting with.
    """
    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    type: SessionType
    created_at: float = Field(default_factory=time.time)
    last_active: float = Field(default_factory=time.time)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def touch(self) -> None:
        """Update last_active timestamp."""
        # Pydantic v2 frozen workaround — use model_config validation
        object.__setattr__(self, "last_active", time.time())


class AppSession(Session):
    """Session for a desktop application.

    Tracks the app's process, window, and document state.
    Capability flags are NOT stored here — they live in
    AppCapabilityRegistry (static per app type).
    """
    type: SessionType = SessionType.APP
    app_name: str
    pid: Optional[int] = None
    window_title: Optional[str] = None
    hwnd: Optional[int] = None
    is_focused: bool = False
    has_unsaved_changes: bool = False


class BrowserSession(Session):
    """Session for a browser automation context."""
    type: SessionType = SessionType.BROWSER
    driver_id: str = ""
    current_url: Optional[str] = None
    tab_count: int = 0


class TerminalSession(Session):
    """Session for an interactive terminal."""
    type: SessionType = SessionType.TERMINAL
    shell: str = ""
    cwd: Optional[str] = None


class DocumentSession(Session):
    """Session for a document being edited."""
    type: SessionType = SessionType.DOCUMENT
    file_path: Optional[str] = None
    editor_session_id: Optional[str] = None  # Links to parent AppSession


# ─────────────────────────────────────────────────────────────
# SessionStack — which app/session is currently active
# ─────────────────────────────────────────────────────────────

class SessionStack:
    """Tracks the active session stack (which environment is on top).

    This reflects OS reality — when the observer detects a focus
    change, SessionStack is updated automatically.

    Thread-safe.
    """

    def __init__(self):
        self._stack: List[str] = []
        self._lock = threading.Lock()

    def push(self, session_id: str) -> None:
        """Push a session to the top. If already in stack, move to top."""
        with self._lock:
            # Remove if already present (avoid duplicates)
            if session_id in self._stack:
                self._stack.remove(session_id)
            self._stack.append(session_id)

    def pop(self) -> Optional[str]:
        """Pop and return the top session ID, or None if empty."""
        with self._lock:
            return self._stack.pop() if self._stack else None

    def peek(self) -> Optional[str]:
        """Return the top session ID without removing it."""
        with self._lock:
            return self._stack[-1] if self._stack else None

    def remove(self, session_id: str) -> None:
        """Remove a specific session from the stack (e.g., on close)."""
        with self._lock:
            if session_id in self._stack:
                self._stack.remove(session_id)

    def get_stack(self) -> List[str]:
        """Return a copy of the full stack (top-last)."""
        with self._lock:
            return list(self._stack)

    def __len__(self) -> int:
        with self._lock:
            return len(self._stack)


# ─────────────────────────────────────────────────────────────
# TaskStack — which workflow MERLIN is currently pursuing
# ─────────────────────────────────────────────────────────────

class TaskStackEntry(BaseModel):
    """A tracked workflow on the task stack."""
    model_config = ConfigDict(extra="forbid")

    task_id: str
    description: str
    started_at: float = Field(default_factory=time.time)
    session_ids: List[str] = Field(default_factory=list)


class TaskStack:
    """Tracks MERLIN's active workflow context.

    Unlike SessionStack, TaskStack is NEVER auto-updated from OS.
    It reflects MERLIN's intent, not OS reality.

    When user says:
        "write story in notepad" → push task
        "search google for alien myths" → push new task
        "go back to the story" → pop back to first task

    Thread-safe.
    """

    def __init__(self):
        self._stack: List[TaskStackEntry] = []
        self._lock = threading.Lock()

    def push(self, task_id: str, description: str,
             session_ids: Optional[List[str]] = None) -> TaskStackEntry:
        """Push a new task context."""
        entry = TaskStackEntry(
            task_id=task_id,
            description=description,
            session_ids=session_ids or [],
        )
        with self._lock:
            self._stack.append(entry)
        logger.debug("TaskStack push: %s (%s)", task_id, description)
        return entry

    def pop(self) -> Optional[TaskStackEntry]:
        """Pop and return the top task, or None if empty."""
        with self._lock:
            if self._stack:
                entry = self._stack.pop()
                logger.debug("TaskStack pop: %s", entry.task_id)
                return entry
            return None

    def peek(self) -> Optional[TaskStackEntry]:
        """Return the top task without removing it."""
        with self._lock:
            return self._stack[-1] if self._stack else None

    def get_stack(self) -> List[TaskStackEntry]:
        """Return a copy of the full stack (top-last)."""
        with self._lock:
            return list(self._stack)

    def __len__(self) -> int:
        with self._lock:
            return len(self._stack)


# ─────────────────────────────────────────────────────────────
# SessionManager — central session registry
# ─────────────────────────────────────────────────────────────

class SessionManager:
    """Central registry for MERLIN's interactive sessions.

    Infrastructure service — not cognitive, not a skill.

    Responsibilities:
    - Create/close/update session handles
    - Track session lifecycle
    - Provide session context summary for the compiler
    - TTL cleanup (validate against observer when available)

    Thread-safe. Called from executor threads and main thread.
    """

    def __init__(self, capability_registry=None):
        """
        Args:
            capability_registry: Optional AppCapabilityRegistry for
                                 enriching session context with capabilities.
        """
        self._sessions: Dict[str, Session] = {}
        self._lock = threading.Lock()
        self._session_stack = SessionStack()
        self._task_stack = TaskStack()
        self._capability_registry = capability_registry
        logger.info("SessionManager initialized")

    # ── Session CRUD ──────────────────────────────────────────

    def create_app_session(
        self,
        app_name: str,
        pid: Optional[int] = None,
        window_title: Optional[str] = None,
        hwnd: Optional[int] = None,
    ) -> AppSession:
        """Create and register a new app session.

        If a session for this app already exists (by app_name),
        the existing session is updated rather than creating a duplicate.
        """
        with self._lock:
            # Check for existing session by app_name
            for s in self._sessions.values():
                if isinstance(s, AppSession) and s.app_name.lower() == app_name.lower():
                    # Update existing session
                    object.__setattr__(s, "pid", pid or s.pid)
                    object.__setattr__(s, "window_title", window_title or s.window_title)
                    object.__setattr__(s, "hwnd", hwnd or s.hwnd)
                    s.touch()
                    logger.info(
                        "Session updated: %s (app=%s, pid=%s)",
                        s.id, app_name, pid,
                    )
                    return s

            # Create new session
            session = AppSession(
                app_name=app_name,
                pid=pid,
                window_title=window_title,
                hwnd=hwnd,
            )
            self._sessions[session.id] = session
            self._session_stack.push(session.id)

            logger.info(
                "Session created: %s (app=%s, pid=%s)",
                session.id, app_name, pid,
            )
            return session

    def create_session(self, session: Session) -> Session:
        """Register a pre-built session object."""
        with self._lock:
            self._sessions[session.id] = session
            self._session_stack.push(session.id)
            logger.info("Session created: %s (type=%s)", session.id, session.type)
            return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID."""
        with self._lock:
            return self._sessions.get(session_id)

    def get_sessions_by_type(self, session_type: SessionType) -> List[Session]:
        """Get all sessions of a given type."""
        with self._lock:
            return [s for s in self._sessions.values() if s.type == session_type]

    def get_session_by_app(self, app_name: str) -> Optional[AppSession]:
        """Find an app session by application name (case-insensitive)."""
        app_lower = app_name.lower()
        with self._lock:
            for s in self._sessions.values():
                if isinstance(s, AppSession) and s.app_name.lower() == app_lower:
                    return s
            return None

    def update_session(self, session_id: str, **updates: Any) -> Optional[Session]:
        """Update session fields. Returns the updated session, or None."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                logger.warning("Cannot update unknown session: %s", session_id)
                return None

            for key, value in updates.items():
                if hasattr(session, key):
                    object.__setattr__(session, key, value)
                else:
                    logger.warning(
                        "Session %s has no attribute '%s'", session_id, key,
                    )

            session.touch()
            return session

    def close_session(self, session_id: str) -> Optional[Session]:
        """Close and unregister a session."""
        with self._lock:
            session = self._sessions.pop(session_id, None)
            if session:
                self._session_stack.remove(session_id)
                logger.info("Session closed: %s", session_id)
            else:
                logger.debug("close_session: no session with id %s", session_id)
            return session

    def close_session_by_app(self, app_name: str) -> Optional[Session]:
        """Close the session for a given app name."""
        session = self.get_session_by_app(app_name)
        if session:
            return self.close_session(session.id)
        return None

    def get_active_sessions(self) -> List[Session]:
        """Get all currently active sessions."""
        with self._lock:
            return list(self._sessions.values())

    # ── TTL cleanup ───────────────────────────────────────────

    def cleanup_stale_sessions(self, observer=None) -> List[str]:
        """Remove sessions for apps that are no longer running.

        Args:
            observer: EnvironmentObserver instance (Phase 2).
                      If None, cleanup is skipped.

        Returns:
            List of closed session IDs.
        """
        if observer is None:
            return []

        closed_ids: List[str] = []
        with self._lock:
            stale = []
            for sid, session in self._sessions.items():
                if isinstance(session, AppSession):
                    if not observer.is_app_running(session.app_name):
                        stale.append(sid)

            for sid in stale:
                session = self._sessions.pop(sid)
                self._session_stack.remove(sid)
                closed_ids.append(sid)
                logger.info(
                    "Stale session cleaned up: %s (app=%s)",
                    sid, getattr(session, "app_name", "?"),
                )

        return closed_ids

    # ── Context for compiler ──────────────────────────────────

    def build_session_context(self) -> Dict[str, Any]:
        """Build a session summary for the LLM compiler prompt.

        This is a separate prompt section — NOT WorldState.
        Returns a clean dict suitable for JSON serialization
        into the compiler prompt.
        """
        with self._lock:
            app_sessions = []
            for s in self._sessions.values():
                if isinstance(s, AppSession):
                    entry: Dict[str, Any] = {
                        "session_id": s.id,
                        "app_name": s.app_name,
                        "is_focused": s.is_focused,
                        "has_unsaved_changes": s.has_unsaved_changes,
                    }
                    if s.window_title:
                        entry["window_title"] = s.window_title

                    # Enrich with capability info
                    if self._capability_registry:
                        caps = self._capability_registry.get(s.app_name)
                        entry["capabilities"] = {
                            "supports_typing": caps.supports_typing,
                            "supports_copy": caps.supports_copy,
                            "supports_save": caps.supports_save,
                        }

                    app_sessions.append(entry)

            browser_sessions = []
            for s in self._sessions.values():
                if isinstance(s, BrowserSession):
                    browser_sessions.append({
                        "session_id": s.id,
                        "current_url": s.current_url,
                        "tab_count": s.tab_count,
                    })

            # Task stack summary
            task_stack = []
            for entry in self._task_stack.get_stack():
                task_stack.append({
                    "task_id": entry.task_id,
                    "description": entry.description,
                })

            context: Dict[str, Any] = {}
            if app_sessions:
                context["open_applications"] = app_sessions
            if browser_sessions:
                context["browser_sessions"] = browser_sessions
            if task_stack:
                context["active_tasks"] = task_stack

            return context

    # ── Properties ────────────────────────────────────────────

    @property
    def session_stack(self) -> SessionStack:
        return self._session_stack

    @property
    def task_stack(self) -> TaskStack:
        return self._task_stack

    @property
    def capabilities(self):
        return self._capability_registry

    @property
    def session_count(self) -> int:
        with self._lock:
            return len(self._sessions)
