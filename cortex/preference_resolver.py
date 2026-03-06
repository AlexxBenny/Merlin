# cortex/preference_resolver.py

"""
PreferenceResolver — Semantic parameter resolution via episodic memory.

Sits after ParameterResolver in the pipeline. Detects parameter values
that reference user preferences (e.g., "preferred", "favorite", "default",
"my usual") and resolves them through MemoryStore.

Design rules:
- Never mutates the original plan — produces a new one
- Unresolvable preferences pass through unchanged (graceful degradation)
- Resolution is bounded — single memory query per preference token
- No LLM calls — pure retrieval
- Forward-compatible — unknown preference keys are ignored

Pipeline position:
    Compiler → ParameterResolver → PreferenceResolver → Executor

Example:
    Input:  {"level": "my preferred level"}
    Lookup: MemoryStore.retrieve("preferred_volume")
    Output: {"level": 35}
"""

from __future__ import annotations

import copy
import logging
import re
from typing import Any, Dict, List, Optional

from ir.mission import MissionPlan, MissionNode

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Preference detection patterns
# ─────────────────────────────────────────────────────────────

# Regex patterns that indicate a preference reference in a parameter value.
# These are checked against string-valued inputs only.
_PREFERENCE_PATTERNS = [
    re.compile(r"\b(?:my\s+)?preferred?\b", re.IGNORECASE),
    re.compile(r"\b(?:my\s+)?favou?rite?\b", re.IGNORECASE),
    re.compile(r"\b(?:my\s+)?(?:usual|default|normal)\b", re.IGNORECASE),
    re.compile(r"\b(?:my\s+)?(?:custom|saved)\b", re.IGNORECASE),
]


# ─────────────────────────────────────────────────────────────
# Memory protocol (minimal interface — avoids importing MemoryStore)
# ─────────────────────────────────────────────────────────────

class PreferenceMemory:
    """Protocol-like interface for preference retrieval.

    Priority order:
    1. Session cache (fastest)
    2. UserKnowledgeStore (persistent structured memory)
    3. Episode metadata scan (legacy fallback)
    """

    def __init__(self, memory_store=None, user_knowledge=None):
        """
        Args:
            memory_store: MemoryStore instance for episode fallback.
            user_knowledge: UserKnowledgeStore instance (primary).
        """
        self._store = memory_store
        self._user_knowledge = user_knowledge
        self._cache: Dict[str, Any] = {}

    def lookup(self, preference_key: str) -> Optional[Any]:
        """Look up a preference value by key.

        Returns the resolved value, or None if not found.

        Search order:
        1. Session cache
        2. UserKnowledgeStore (if available)
        3. Episode metadata (legacy fallback)
        """
        # 1. Check session cache
        if preference_key in self._cache:
            return self._cache[preference_key]

        # 2. Query UserKnowledgeStore (primary source)
        if self._user_knowledge is not None:
            value = self._user_knowledge.get_preference(preference_key)
            if value is not None:
                self._cache[preference_key] = value
                logger.info(
                    "[PREFERENCE] Resolved '%s' → %r from UserKnowledgeStore",
                    preference_key, value,
                )
                return value

        # 3. Legacy fallback: scan episode metadata
        if self._store is None:
            return None

        try:
            episodes = self._store.retrieve_relevant(
                query=preference_key, top_k=3,
            )
        except Exception:
            logger.debug(
                "PreferenceMemory: memory lookup failed for '%s'",
                preference_key,
            )
            return None

        for episode in episodes:
            meta = episode.get("metadata", {})
            prefs = meta.get("preferences", {})
            if preference_key in prefs:
                value = prefs[preference_key]
                self._cache[preference_key] = value
                logger.info(
                    "[PREFERENCE] Resolved '%s' → %r from episode memory",
                    preference_key, value,
                )
                return value

        return None

    def store(self, preference_key: str, value: Any) -> None:
        """Cache a preference value for this session.

        This does NOT persist to MemoryStore — persistence is handled
        by the orchestrator when storing mission outcomes.
        """
        self._cache[preference_key] = value
        logger.info(
            "[PREFERENCE] Cached '%s' → %r", preference_key, value,
        )


# ─────────────────────────────────────────────────────────────
# Resolver
# ─────────────────────────────────────────────────────────────

class PreferenceResolver:
    """Resolve preference-referencing parameters via episodic memory.

    Usage:
        resolver = PreferenceResolver(preference_memory)
        resolved_plan = resolver.resolve_plan(plan)
    """

    def __init__(self, memory: Optional[PreferenceMemory] = None):
        self._memory = memory or PreferenceMemory()

    def resolve_plan(self, plan: MissionPlan) -> MissionPlan:
        """Produce a new MissionPlan with preference values resolved.

        Never mutates the original plan.
        Unresolvable preferences pass through unchanged.
        """
        any_resolved = False
        resolved_nodes: List[MissionNode] = []

        for node in plan.nodes:
            resolved_inputs, changed = self._resolve_node(node)
            any_resolved = any_resolved or changed

            new_node = MissionNode(
                id=node.id,
                skill=node.skill,
                inputs=resolved_inputs,
                outputs=node.outputs,
                depends_on=list(node.depends_on),
                mode=node.mode,
            )
            resolved_nodes.append(new_node)

        if not any_resolved:
            return plan  # No preferences detected — return original

        return MissionPlan(
            id=plan.id,
            nodes=resolved_nodes,
            metadata=plan.metadata if plan.metadata else {"ir_version": "1.0"},
        )

    def _resolve_node(
        self, node: MissionNode,
    ) -> tuple[Dict[str, Any], bool]:
        """Resolve preference references in a single node's inputs.

        Returns (resolved_inputs, any_changed).
        """
        resolved: Dict[str, Any] = {}
        any_changed = False

        for key, value in node.inputs.items():
            resolved_value = self._try_resolve(key, value, node.skill)
            if resolved_value is not value:
                any_changed = True
            resolved[key] = resolved_value

        return resolved, any_changed

    def _try_resolve(
        self, param_key: str, value: Any, skill_name: str,
    ) -> Any:
        """Attempt to resolve a single parameter value.

        Returns the original value if:
        - Value is not a string
        - Value does not match any preference pattern
        - Memory lookup returns None
        """
        if not isinstance(value, str):
            return value

        # Check if value contains a preference reference
        if not any(p.search(value) for p in _PREFERENCE_PATTERNS):
            return value

        # Build preference key from skill name + parameter name
        # e.g., skill=system.set_volume, param=level → preferred_volume
        preference_key = self._derive_preference_key(
            param_key, skill_name,
        )
        if not preference_key:
            return value

        logger.info(
            "[PREFERENCE] Detected preference reference: skill=%s "
            "param=%s value=%r → key=%s",
            skill_name, param_key, value, preference_key,
        )

        resolved = self._memory.lookup(preference_key)
        if resolved is not None:
            logger.info(
                "[PREFERENCE] Resolved %s.%s: %r → %r",
                skill_name, param_key, value, resolved,
            )
            return resolved

        logger.info(
            "[PREFERENCE] No stored preference for '%s' — "
            "passing through original value",
            preference_key,
        )
        return value

    @staticmethod
    def _derive_preference_key(param_key: str, skill_name: str) -> str:
        """Derive a preference key from skill + parameter context.

        Examples:
            ("level", "system.set_volume")   → "preferred_volume"
            ("app_name", "system.open_app")  → "preferred_app"
            ("path", "fs.create_folder")     → "preferred_path"
            ("editor", "system.open_app")    → "preferred_editor"

        Returns empty string if no meaningful key can be derived.
        """
        # Extract domain hint from skill name (e.g., "set_volume" → "volume")
        if "." in skill_name:
            action_part = skill_name.split(".")[-1]
        else:
            action_part = skill_name

        # Strip common prefixes
        for prefix in ("set_", "get_", "toggle_", "adjust_"):
            if action_part.startswith(prefix):
                action_part = action_part[len(prefix):]
                break

        # Build key: "preferred_" + domain hint
        if action_part:
            return f"preferred_{action_part}"

        # Fallback: use param_key itself
        if param_key:
            return f"preferred_{param_key}"

        return ""
