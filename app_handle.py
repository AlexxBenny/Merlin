"""AppHandle: Application Identity Abstraction

This module provides a time-stable identity abstraction for launched applications.
It solves the fundamental problem where launch PID ≠ runtime PID (especially on
Windows 11 UWP apps, Electron, and shell-registered apps).

IMPORTANT DESIGN PRINCIPLE:
    AppHandle is a BELIEF, not AUTHORITY.
    Reality is always "what windows exist NOW".
    The handle must always: resolve → act → possibly fail.
    Never: assume → act blindly.

Author: AURA System
Phase: 2.5 (Identity Modeling)
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
from datetime import datetime
from enum import Enum
import uuid


class IdentityBasis(str, Enum):
    """How the identity was established/resolved.
    
    This is epistemic transparency—the planner needs to know
    WHY an identity was resolved, not just WHAT it resolved to.
    """
    HWND = "hwnd"              # Direct HWND validation (highest confidence)
    PID = "pid"                # Process ID match (unstable on UWP)
    APP_NAME = "app_name"      # Executable/process name match (medium confidence)
    TITLE_FALLBACK = "title_fallback"  # Title substring match (degraded)
    UNKNOWN = "unknown"        # Initial state or lost


class ResolutionConfidence(str, Enum):
    """How confident we are that the handle points to the right app."""
    HIGH = "high"          # HWND validated, same process
    MEDIUM = "medium"      # App name match, but HWND may have changed
    DEGRADED = "degraded"  # Title fallback only
    LOST = "lost"          # Cannot resolve to any window


@dataclass
class AppHandle:
    """Persistent identity for a launched application.
    
    This is NOT a cache. This is NOT a guess.
    This is a structured belief about which windows belong to
    an application we launched, with explicit uncertainty tracking.
    
    Stable Fields (set at creation, never change):
        - handle_id: Globally unique identifier
        - app_name: Original requested app name
        - launched_at: When we launched it
        - launch_command: What command we executed
        
    Observational Fields (updated on resolution):
        - known_hwnds: Window handles we believe belong to this app
        - known_pids: Process IDs we believe belong to this app
        - last_title: Last known primary window title
        - last_resolved_at: When we last successfully resolved
        - identity_basis: HOW we established current identity
        - resolution_confidence: HOW CERTAIN we are
    """
    
    # === STABLE (set at creation, never changes) ===
    handle_id: str
    app_name: str
    launched_at: datetime
    launch_command: str
    
    # === OBSERVATIONAL (updated on each resolution) ===
    known_hwnds: List[int] = field(default_factory=list)
    known_pids: List[int] = field(default_factory=list)
    last_title: Optional[str] = None
    last_resolved_at: Optional[datetime] = None
    
    # === EPISTEMIC STATE ===
    identity_basis: IdentityBasis = IdentityBasis.UNKNOWN
    resolution_confidence: ResolutionConfidence = ResolutionConfidence.HIGH
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for tool return values."""
        return {
            "handle_id": self.handle_id,
            "app_name": self.app_name,
            "launched_at": self.launched_at.isoformat(),
            "launch_command": self.launch_command,
            "known_hwnds": self.known_hwnds.copy(),
            "known_pids": self.known_pids.copy(),
            "last_title": self.last_title,
            "last_resolved_at": self.last_resolved_at.isoformat() if self.last_resolved_at else None,
            "identity_basis": self.identity_basis.value,
            "resolution_confidence": self.resolution_confidence.value
        }
    
    @classmethod
    def create(cls, app_name: str, launch_command: str) -> 'AppHandle':
        """Create a new handle at launch time."""
        return cls(
            handle_id=str(uuid.uuid4()),
            app_name=app_name,
            launched_at=datetime.now(),
            launch_command=launch_command,
            known_hwnds=[],
            known_pids=[],
            last_title=None,
            last_resolved_at=None,
            identity_basis=IdentityBasis.UNKNOWN,
            resolution_confidence=ResolutionConfidence.HIGH
        )
    
    def bind_window(self, hwnd: int, pid: int, title: str) -> None:
        """Bind this handle to a discovered window.
        
        Called after launch successfully finds a window.
        This is the INITIAL binding with highest confidence.
        """
        self.known_hwnds = [hwnd]
        self.known_pids = [pid]
        self.last_title = title
        self.last_resolved_at = datetime.now()
        self.identity_basis = IdentityBasis.HWND
        self.resolution_confidence = ResolutionConfidence.HIGH
    
    def resolve(
        self, 
        find_windows_fn: Callable,
        is_window_fn: Callable[[int], bool],
        get_window_info_fn: Callable[[int], Dict[str, Any]],
        allow_rebinding: bool = False
    ) -> List[Dict[str, Any]]:
        """Re-resolve this handle to current windows.
        
        Args:
            find_windows_fn: Function to enumerate windows by criteria
            is_window_fn: Function to check if HWND is still valid
            get_window_info_fn: Function to get window details from HWND
            allow_rebinding: If True, allow updating known_hwnds from
                            degraded resolution paths. Default False to
                            prevent silent identity drift.
        
        Returns:
            List of window info dicts that match this handle.
            
        Side Effects:
            Updates observational fields based on resolution.
            
        IMPORTANT: This is NOT retry logic. This is a single-pass resolution.
        """
        matches = []
        
        # === Strategy 1: Check known HWNDs (most precise) ===
        still_valid_hwnds = []
        for hwnd in self.known_hwnds:
            if is_window_fn(hwnd):
                info = get_window_info_fn(hwnd)
                if info:
                    matches.append(info)
                    still_valid_hwnds.append(hwnd)
        
        if matches:
            self.known_hwnds = still_valid_hwnds
            self.known_pids = list(set(m["pid"] for m in matches))
            self.last_title = matches[0].get("title")
            self.last_resolved_at = datetime.now()
            self.identity_basis = IdentityBasis.HWND
            self.resolution_confidence = ResolutionConfidence.HIGH
            return matches
        
        # === Strategy 2: Search by app_name (fallback) ===
        matches = find_windows_fn(app_name=self.app_name)
        
        if matches:
            self.last_resolved_at = datetime.now()
            self.identity_basis = IdentityBasis.APP_NAME
            self.resolution_confidence = ResolutionConfidence.MEDIUM
            
            # Only rebind HWNDs if explicitly allowed
            if allow_rebinding:
                self.known_hwnds = [m["hwnd"] for m in matches]
                self.known_pids = list(set(m["pid"] for m in matches))
                self.last_title = matches[0].get("title")
            
            return matches
        
        # === Strategy 3: Search by last_title (degraded) ===
        if self.last_title:
            # Use only first 30 chars to avoid over-specific matching
            title_hint = self.last_title[:30] if len(self.last_title) > 30 else self.last_title
            matches = find_windows_fn(title_substring=title_hint)
            
            if matches:
                self.last_resolved_at = datetime.now()
                self.identity_basis = IdentityBasis.TITLE_FALLBACK
                self.resolution_confidence = ResolutionConfidence.DEGRADED
                
                # NEVER rebind from title fallback unless explicitly allowed
                # This prevents silent identity poisoning
                if allow_rebinding:
                    self.known_hwnds = [m["hwnd"] for m in matches]
                    self.known_pids = list(set(m["pid"] for m in matches))
                
                return matches
        
        # === Complete loss ===
        self.resolution_confidence = ResolutionConfidence.LOST
        self.identity_basis = IdentityBasis.UNKNOWN
        return []
    
    def is_alive(self) -> bool:
        """Quick check if handle might still be valid."""
        return self.resolution_confidence != ResolutionConfidence.LOST
    
    def invalidate(self) -> None:
        """Mark handle as no longer valid (e.g., after successful close)."""
        self.resolution_confidence = ResolutionConfidence.LOST
        self.identity_basis = IdentityBasis.UNKNOWN
        self.known_hwnds = []
        self.known_pids = []


class HandleRegistry:
    """In-memory registry of active AppHandles.
    
    This allows tools to look up handles by handle_id.
    Handles are registered at launch time and can be retrieved
    by subsequent focus/close operations.
    
    This is NOT persistence. This is session-scoped.
    """
    
    _handles: Dict[str, AppHandle] = {}
    
    @classmethod
    def register(cls, handle: AppHandle) -> None:
        """Register a new handle."""
        cls._handles[handle.handle_id] = handle
    
    @classmethod
    def get(cls, handle_id: str) -> Optional[AppHandle]:
        """Get a handle by ID."""
        return cls._handles.get(handle_id)
    
    @classmethod
    def remove(cls, handle_id: str) -> bool:
        """Remove a handle from registry."""
        if handle_id in cls._handles:
            del cls._handles[handle_id]
            return True
        return False
    
    @classmethod
    def list_all(cls) -> List[AppHandle]:
        """List all registered handles."""
        return list(cls._handles.values())
    
    @classmethod
    def prune_stale(cls, max_age_hours: float = 1.0) -> int:
        """Remove handles older than max_age_hours. Returns count removed."""
        now = datetime.now()
        stale_ids = []
        
        for handle_id, handle in cls._handles.items():
            age = (now - handle.launched_at).total_seconds() / 3600
            if age > max_age_hours:
                stale_ids.append(handle_id)
        
        for handle_id in stale_ids:
            del cls._handles[handle_id]
        
        return len(stale_ids)
    
    @classmethod
    def clear(cls) -> None:
        """Clear all handles (for testing)."""
        cls._handles.clear()
