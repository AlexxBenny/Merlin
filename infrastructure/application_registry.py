# infrastructure/application_registry.py

"""
ApplicationRegistry — Normalized application entity store.

Design rules:
- Data + lookup ONLY. No scanning, no filesystem access.
- Populated at boot by ApplicationDiscoveryService.
- Refresh via atomic swap (build new, replace reference).
- Never mutated during mission execution.
- Thread-safe reads, single-writer at swap time.

Used by:
- EntityResolver (cortex layer) for app_id normalization
- OpenAppSkill for entity lookup at execution time
- SystemController.launch() for strategy selection
- SessionManager for canonical_process_names matching
"""

import logging
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Resolution metadata
# ─────────────────────────────────────────────────────────────

class ResolutionMethod(str, Enum):
    """How a launch strategy was discovered."""
    PROTOCOL = "protocol"
    APP_PATHS = "app_paths"
    START_MENU = "start_menu"
    APPSFOLDER = "appsfolder"
    INSTALL_SEARCH = "install_search"
    CLI_PATH = "cli_path"
    SHELL_FALLBACK = "shell_fallback"


# ─────────────────────────────────────────────────────────────
# LaunchStrategy — one way to launch an app
# ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LaunchStrategy:
    """A single launchable route to an application.

    An entity may have multiple strategies (protocol, exe, shell).
    The executor selects the best one by priority at launch time.

    priority and reliability_score are data-driven — assigned by
    discovery based on strategy type, overridable per-entity.
    """
    type: str                      # "protocol", "executable", "shell", "appsfolder"
    value: str                     # URI ("spotify:"), path, or AppUserModelID
    method: ResolutionMethod       # How this strategy was discovered
    priority: int = 50             # Higher = preferred. Executor sorts descending.
    reliability_score: int = 50    # Confidence that this strategy works
    details: Optional[str] = None  # Diagnostic: registry key, shortcut path, etc.


# ─── Default priority assignments (used by discovery) ─────────

STRATEGY_DEFAULTS = {
    "executable":  {"priority": 100, "reliability": 100},
    "appsfolder":  {"priority": 90,  "reliability": 90},
    "protocol":    {"priority": 80,  "reliability": 70},
    "shell":       {"priority": 10,  "reliability": 30},
}


# ─────────────────────────────────────────────────────────────
# ApplicationEntity — normalized application identity
# ─────────────────────────────────────────────────────────────

@dataclass
class ApplicationEntity:
    """Canonical application identity.

    Represents a single application with potentially multiple
    launch strategies, executables, and protocols.

    Fields:
        app_id:                   Canonical identifier ("chrome", "spotify")
        display_names:            All known display names
        launch_strategies:        All discovered launch routes, sorted by priority
        executables:              All known executables ("chrome.exe", "chrome_proxy.exe")
        protocols:                All known protocol handlers ("http:", "spotify:")
        canonical_process_names:  Process names for session/guard matching
        install_locations:        All discovered install paths
        capabilities:             Static capability flags (from AppCapabilityRegistry)
        is_uwp:                   Whether this is a UWP/Store app
    """
    app_id: str
    display_names: List[str] = field(default_factory=list)
    launch_strategies: List[LaunchStrategy] = field(default_factory=list)
    executables: List[str] = field(default_factory=list)
    protocols: List[str] = field(default_factory=list)
    canonical_process_names: List[str] = field(default_factory=list)
    install_locations: List[str] = field(default_factory=list)
    capabilities: Any = None  # AppCapabilities (avoid circular import)
    is_uwp: bool = False

    def best_strategy(self) -> Optional[LaunchStrategy]:
        """Return the highest-priority launch strategy, or None."""
        if not self.launch_strategies:
            return None
        return max(self.launch_strategies, key=lambda s: s.priority)

    def strategies_by_type(self, stype: str) -> List[LaunchStrategy]:
        """Return strategies of a given type."""
        return [s for s in self.launch_strategies if s.type == stype]

    def has_process_name(self, process_name: str) -> bool:
        """Check if a process name matches this entity (case-insensitive)."""
        pn_lower = process_name.lower()
        return any(cpn.lower() == pn_lower for cpn in self.canonical_process_names)


# ─────────────────────────────────────────────────────────────
# ApplicationRegistry — data + lookup (no scanning)
# ─────────────────────────────────────────────────────────────

class ApplicationRegistry:
    """Immutable entity store populated at boot.

    Responsibilities:
    - Store ApplicationEntity objects
    - Look up by app_id or display name
    - Atomic refresh (swap entire index; never mutate in-place)

    Thread safety:
    - Reads are lock-free (dict lookup on immutable references)
    - refresh() acquires a write lock for the atomic swap
    - Never call register() or refresh() during mission execution

    Usage:
        registry = ApplicationRegistry()
        registry.register(entity)
        entity = registry.get("chrome")
        entity = registry.lookup_by_name("Google Chrome")
    """

    def __init__(self):
        self._by_id: Dict[str, ApplicationEntity] = {}
        self._name_index: Dict[str, str] = {}  # lowercase name → app_id
        self._process_index: Dict[str, str] = {}  # lowercase proc name → app_id
        self._lock = threading.Lock()

    # ── Registration (boot-time only) ─────────────────────────

    def register(self, entity: ApplicationEntity) -> None:
        """Register a single entity. Boot-time only."""
        with self._lock:
            self._by_id[entity.app_id] = entity

            # Index all display names
            for name in entity.display_names:
                self._name_index[name.lower().strip()] = entity.app_id

            # Also index the app_id itself
            self._name_index[entity.app_id.lower().strip()] = entity.app_id

            # Index process names for guard/session matching
            for pn in entity.canonical_process_names:
                self._process_index[pn.lower().strip()] = entity.app_id

    # ── Lookup ────────────────────────────────────────────────

    def get(self, app_id: str) -> Optional[ApplicationEntity]:
        """Look up entity by canonical app_id."""
        return self._by_id.get(app_id)

    def lookup_by_name(self, name: str) -> Optional[ApplicationEntity]:
        """Look up entity by any display name (case-insensitive)."""
        app_id = self._name_index.get(name.lower().strip())
        if app_id:
            return self._by_id.get(app_id)
        return None

    def lookup_by_process(self, process_name: str) -> Optional[ApplicationEntity]:
        """Look up entity by process name (for guard/session matching)."""
        app_id = self._process_index.get(process_name.lower().strip())
        if app_id:
            return self._by_id.get(app_id)
        return None

    def all_ids(self) -> List[str]:
        """Return all registered app_ids."""
        return list(self._by_id.keys())

    def __len__(self) -> int:
        return len(self._by_id)

    def __contains__(self, app_id: str) -> bool:
        return app_id in self._by_id

    # ── Atomic refresh ────────────────────────────────────────

    def refresh(self, entities: List[ApplicationEntity]) -> None:
        """Atomic swap: build new indices, then replace in one step.

        Never call during mission execution. The caller is responsible
        for ensuring no execution is in progress.
        """
        new_by_id: Dict[str, ApplicationEntity] = {}
        new_name_index: Dict[str, str] = {}
        new_process_index: Dict[str, str] = {}

        for entity in entities:
            new_by_id[entity.app_id] = entity
            for name in entity.display_names:
                new_name_index[name.lower().strip()] = entity.app_id
            new_name_index[entity.app_id.lower().strip()] = entity.app_id
            for pn in entity.canonical_process_names:
                new_process_index[pn.lower().strip()] = entity.app_id

        with self._lock:
            self._by_id = new_by_id
            self._name_index = new_name_index
            self._process_index = new_process_index

        logger.info(
            "ApplicationRegistry refreshed: %d entities, %d names indexed, "
            "%d process names indexed",
            len(new_by_id), len(new_name_index), len(new_process_index),
        )

    # ── Diagnostics ───────────────────────────────────────────

    def summary(self) -> Dict[str, Any]:
        """Return diagnostic summary."""
        by_type = {"uwp": 0, "desktop": 0}
        strategy_counts: Dict[str, int] = {}
        for entity in self._by_id.values():
            if entity.is_uwp:
                by_type["uwp"] += 1
            else:
                by_type["desktop"] += 1
            for s in entity.launch_strategies:
                strategy_counts[s.type] = strategy_counts.get(s.type, 0) + 1

        return {
            "total_entities": len(self._by_id),
            "total_names_indexed": len(self._name_index),
            "total_process_names_indexed": len(self._process_index),
            "by_type": by_type,
            "strategy_counts": strategy_counts,
        }
