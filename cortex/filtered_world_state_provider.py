# cortex/filtered_world_state_provider.py

"""
FilteredWorldStateProvider — Query-scoped view projection.

Architecture:
    WorldState (complete, internal, authoritative)
            ↓
    FilteredWorldStateProvider.build_schema()
            ↓
    WorldStateView (query-scoped projection for LLM)

This does NOT remove state. It projects a view.
The full WorldState remains authoritative internally for proactivity,
event processing, and monitoring (all deterministic, no LLM needed).

Domain-to-state mapping is config-driven via skills.yaml.
"""

import logging
from typing import Any, Dict, List, Optional, Set

from cortex.world_state_provider import WorldStateProvider
from world.snapshot import WorldSnapshot

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Default domain → state section mapping
# ─────────────────────────────────────────────────────────────
# Overridden by config/skills.yaml domain_state_mapping.
# Keys are skill domain prefixes (e.g. "system", "fs").
# Values are dot-paths into WorldState (e.g. "system.hardware").

DEFAULT_DOMAIN_STATE_MAP: Dict[str, List[str]] = {
    "system": ["system.hardware", "system.session"],
    "fs":     ["cwd"],
    "media":  ["media", "system.hardware"],
    "browser": ["system.session"],
}

# State sections always included regardless of domain match.
# Time is universally useful (scheduling, greetings, context).
# Session (tracked_apps, foreground_app) is needed by nearly every
# domain that interacts with applications — media, system, browser.
# Without it, the LLM guesses app names → entity resolution fails.
ALWAYS_INCLUDE = ["time", "system.session"]

# ─────────────────────────────────────────────────────────────
# Domain keyword detection (confidence-gated)
# ─────────────────────────────────────────────────────────────
# Two tiers of keywords per domain:
#
# - unique_keywords: High signal. Domain-specific. One match = confident.
#   e.g. "brightness" unambiguously means system.hardware.
#
# - ambiguous_keywords: Low signal. Shared across domains.
#   Requires ≥2 matches from the SAME domain to trigger.
#   e.g. "open" alone could be system, fs, or browser.
#
# This prevents misclassification like:
#   "open research paper" → system (because "open")
#   "create summary"      → fs     (because "create")
#   "search memory usage" → browser (because "search")
#
# Rule: False negatives (full state) are SAFER than false positives
# (wrong filtering). When in doubt, return full state.

DOMAIN_KEYWORDS: Dict[str, Dict[str, List[str]]] = {
    "system": {
        "unique": [
            "brightness", "volume", "mute", "unmute", "nightlight",
            "night light", "battery",
        ],
        "ambiguous": [
            "open", "close", "launch", "app", "application", "focus",
            "cpu", "memory", "disk",
        ],
    },
    "fs": {
        "unique": [
            "folder", "directory",
        ],
        "ambiguous": [
            "file", "create", "move", "copy", "rename",
            "path", "desktop", "documents", "downloads",
        ],
    },
    "media": {
        "unique": [
            "spotify", "music", "song", "track", "audio",
        ],
        "ambiguous": [
            "play", "pause", "stop", "resume",
            "next", "skip", "previous", "media",
        ],
    },
    "browser": {
        "unique": [
            "chrome", "firefox", "edge", "browser", "tab",
            "url", "website",
        ],
        "ambiguous": [
            "search", "browse", "web",
        ],
    },
}

# Minimum ambiguous keyword matches to trigger a domain
_AMBIGUOUS_THRESHOLD = 2


class FilteredWorldStateProvider(WorldStateProvider):
    """Query-scoped WorldState view projection.

    Projects only the state sections relevant to the detected skill domains.
    The full WorldState remains unmodified internally — this only controls
    what the LLM sees.

    Domain resolution uses UNION of:
    - Candidate skill domains (from SkillDiscovery)
    - Keyword-detected domains (confidence-gated)

    This ensures cross-domain queries include all relevant state sections.
    """

    def __init__(
        self,
        domain_state_map: Optional[Dict[str, List[str]]] = None,
        domain_keywords: Optional[Dict[str, Dict[str, List[str]]]] = None,
        always_include: Optional[List[str]] = None,
    ):
        self._domain_state_map = domain_state_map or DEFAULT_DOMAIN_STATE_MAP
        self._domain_keywords = domain_keywords or DOMAIN_KEYWORDS
        self._always_include = always_include or ALWAYS_INCLUDE

    def build_schema(
        self,
        snapshot: WorldSnapshot,
        query: Optional[str] = None,
        candidate_skills: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        """Project a query-scoped view of WorldState for the LLM.

        Domain resolution:
        1. Extract domains from candidate_skills (if provided)
        2. Detect domains from query keywords (confidence-gated)
        3. Union both sets
        4. If empty → full state (safe default)
        """
        full_state = snapshot.state.model_dump()

        # Determine relevant domains (union)
        domains = self._resolve_domains(query, candidate_skills)

        if not domains:
            # No domain signal — return full state (safe default)
            logger.debug("FilteredWSP: no domain signal, returning full state")
            return full_state

        # Collect relevant state sections
        relevant_paths = set(self._always_include)
        for domain in domains:
            paths = self._domain_state_map.get(domain, [])
            relevant_paths.update(paths)

        # Project the view
        view = self._project(full_state, relevant_paths)

        logger.info(
            "FilteredWSP: domains=%s, sections=%s (full=%d keys, view=%d keys)",
            sorted(domains),
            sorted(relevant_paths),
            self._count_leaf_keys(full_state),
            self._count_leaf_keys(view),
        )

        return view

    def _resolve_domains(
        self,
        query: Optional[str],
        candidate_skills: Optional[Set[str]],
    ) -> Set[str]:
        """Determine relevant domains via UNION of skills and keywords.

        Returns: domains_from_skills ∪ domains_from_keywords

        Keyword detection is confidence-gated:
        - Domain-unique keywords: 1 match sufficient
        - Ambiguous keywords: ≥2 matches required from same domain
        """
        domains: Set[str] = set()

        # Source 1: candidate skill names (e.g. "system.set_volume" → "system")
        if candidate_skills:
            for skill_name in candidate_skills:
                domain = skill_name.split(".")[0]
                if domain in self._domain_state_map:
                    domains.add(domain)

        # Source 2: keyword detection (ALWAYS runs — union, not fallback)
        if query:
            keyword_domains = self._detect_keyword_domains(query)
            domains.update(keyword_domains)

        return domains

    def _detect_keyword_domains(self, query: str) -> Set[str]:
        """Confidence-gated keyword domain detection.

        A domain is detected if:
        - ≥1 unique keyword matches, OR
        - ≥AMBIGUOUS_THRESHOLD ambiguous keywords match

        This prevents false positives from generic words like
        "open", "create", "search".
        """
        query_lower = query.lower()
        detected: Set[str] = set()

        for domain, tiers in self._domain_keywords.items():
            unique_kws = tiers.get("unique", [])
            ambiguous_kws = tiers.get("ambiguous", [])

            # Check unique keywords first (high confidence, 1 match enough)
            has_unique = any(kw in query_lower for kw in unique_kws)
            if has_unique:
                detected.add(domain)
                continue

            # Check ambiguous keywords (need ≥ threshold matches)
            ambiguous_count = sum(
                1 for kw in ambiguous_kws if kw in query_lower
            )
            if ambiguous_count >= _AMBIGUOUS_THRESHOLD:
                detected.add(domain)

        return detected

    def _project(
        self,
        full_state: Dict[str, Any],
        relevant_paths: Set[str],
    ) -> Dict[str, Any]:
        """Extract only the relevant sections from the full state dict.

        Paths are dot-separated keys into the nested state dict.
        e.g. "system.hardware" → full_state["system"]["hardware"]

        A path like "system.hardware" includes the ENTIRE hardware subtree.
        A top-level path like "time" includes the entire time dict.
        """
        view: Dict[str, Any] = {}

        for path in relevant_paths:
            parts = path.split(".")
            # Navigate to the value in full_state
            source = full_state
            valid = True
            for part in parts:
                if isinstance(source, dict) and part in source:
                    source = source[part]
                else:
                    valid = False
                    break

            if not valid or source is None:
                continue

            # Place the value at the same path in the view
            target = view
            for part in parts[:-1]:
                if part not in target:
                    target[part] = {}
                target = target[part]
            target[parts[-1]] = source

        return view

    @staticmethod
    def _count_leaf_keys(d: Any, _depth: int = 0) -> int:
        """Count leaf keys in a nested dict (for logging only)."""
        if not isinstance(d, dict):
            return 1 if d is not None else 0
        return sum(
            FilteredWorldStateProvider._count_leaf_keys(v, _depth + 1)
            for v in d.values()
        )
