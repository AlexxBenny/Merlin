# world/resolver.py

"""
WorldResolver — deterministic reference resolution.

Resolves referential language ("the second video", "that folder",
"that app", "the last window") against MissionOutcome.visible_lists,
ConversationFrame context, and WorldState.system.session.

Hard invariants:
- Resolver runs BEFORE cortex.compile()
- Resolver NEVER mutates MissionPlan
- Resolver NEVER calls LLM
- Resolution is deterministic: same inputs → same outputs
- Resolver NEVER accesses infrastructure — only WorldState

Resolution priority order (enforced programmatically):
1. Explicit ordinal ("the second app")
2. Explicit domain entity from visible_lists
3. Active entity from ConversationFrame
4. Foreground session from WorldState
"""

import re
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field


class ReferenceResolutionError(Exception):
    pass


@dataclass(frozen=True)
class ResolvedReference:
    """Result of resolving referential language.

    Attached to query context, NOT used for text replacement
    (unless resolution is deterministic and unambiguous).
    """
    ordinal: Optional[str] = None
    index: Optional[int] = None
    list_key: Optional[str] = None
    resolved_value: Optional[Any] = None
    entity_hint: Optional[str] = None
    resolution_source: Optional[str] = None  # "ordinal" | "entity" | "active" | "session"


@dataclass
class QueryContext:
    """Annotations attached to a query before compilation.

    This is the resolver's output. It annotates, never mutates.
    """
    original_text: str
    resolved_references: List[ResolvedReference] = field(default_factory=list)
    has_referential_language: bool = False

    @property
    def is_resolved(self) -> bool:
        return len(self.resolved_references) > 0


class WorldResolver:
    """
    Deterministic resolver for referential language.
    No LLM. No guessing.

    Resolves against:
    - MissionOutcome.visible_lists (ordinals, entity lists)
    - ConversationFrame.active_entity (context)
    - WorldState.system.session (foreground app, open apps)

    Priority is enforced programmatically (not by pattern match order).
    """

    ORDINAL_MAP = {
        "first": 0,
        "second": 1,
        "third": 2,
        "fourth": 3,
        "fifth": 4,
        "sixth": 5,
        "seventh": 6,
        "eighth": 7,
        "ninth": 8,
        "tenth": 9,
    }

    # ── Referential patterns ──
    # General patterns
    _REFERENTIAL_PATTERNS = [
        re.compile(r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b", re.IGNORECASE),
        re.compile(r"\b(that|the|this)\s+(one|file|folder|video|result|item|link|page)\b", re.IGNORECASE),
        re.compile(r"\b(it|them|those|these)\b", re.IGNORECASE),
    ]

    # System domain referential patterns
    _SYSTEM_PATTERNS = [
        re.compile(r"\b(that|the|this)\s+(app|window|process|application|program)\b", re.IGNORECASE),
        re.compile(r"\b(last|previous|current|active)\s+(app|window|application|program)\b", re.IGNORECASE),
    ]

    # Entity type → list of domain nouns
    _ENTITY_PATTERNS = re.compile(
        r"\b(that|the|this)\s+(one|file|folder|video|result|item|link|page)\b",
        re.IGNORECASE,
    )

    _SYSTEM_ENTITY_PATTERN = re.compile(
        r"\b(that|the|this|last|previous|current|active)\s+(app|window|process|application|program)\b",
        re.IGNORECASE,
    )

    @classmethod
    def detect_referential_language(cls, text: str) -> bool:
        """Check whether user text contains referential language."""
        if any(p.search(text) for p in cls._REFERENTIAL_PATTERNS):
            return True
        if any(p.search(text) for p in cls._SYSTEM_PATTERNS):
            return True
        return False

    @classmethod
    def resolve(
        cls,
        user_text: str,
        visible_lists: Dict[str, list],
        active_entity: Optional[str] = None,
        system_session: Optional[Dict[str, Any]] = None,
        entity_registry: Optional[Dict[str, Any]] = None,
    ) -> QueryContext:
        """
        Attempt to resolve referential language in user text.

        Args:
            user_text: raw user input
            visible_lists: from last MissionOutcome (or empty)
            active_entity: from ConversationFrame (e.g. "folder 'hello'")
            system_session: from WorldState.system.session (dict with
                            foreground_app, foreground_window, open_apps)
            entity_registry: from ConversationFrame.entity_registry
                             (Dict[str, EntityRecord] — typed entities that
                             survive domain switches)

        Returns:
            QueryContext with resolved references (may be empty).
            NEVER raises — fails cleanly.

        Resolution priority (enforced explicitly):
            1. Ordinal ("the second app") → visible_lists
            1.5 Ordinal → entity_registry (fallback if visible_lists miss)
            2. Domain entity ("that folder") → active_entity
            3. System entity ("that app") → system_session
            4. Foreground fallback → system_session.foreground_app
        """
        ctx = QueryContext(
            original_text=user_text,
            has_referential_language=cls.detect_referential_language(user_text),
        )

        if not ctx.has_referential_language:
            return ctx

        # ── Priority 1: Ordinal resolution ──
        cls._resolve_ordinals(user_text, visible_lists, ctx)
        if ctx.resolved_references:
            return ctx

        # ── Priority 1.5: Ordinal from entity_registry (survives domain switches) ──
        if entity_registry:
            cls._resolve_from_entity_registry(user_text, entity_registry, ctx)
            if ctx.resolved_references:
                return ctx

        # ── Priority 2: Domain entity resolution ("that folder") ──
        if active_entity:
            cls._resolve_entity_reference(user_text, active_entity, ctx)
            if ctx.resolved_references:
                return ctx

        # ── Priority 3: System entity resolution ("that app") ──
        if system_session:
            cls._resolve_system_reference(user_text, system_session, ctx)
            if ctx.resolved_references:
                return ctx

        # ── Priority 4: Foreground fallback (pronouns like "it") ──
        if system_session and not ctx.resolved_references:
            cls._resolve_foreground_fallback(user_text, system_session, ctx)

        return ctx

    @classmethod
    def _resolve_ordinals(
        cls,
        text: str,
        visible_lists: Dict[str, list],
        ctx: QueryContext,
    ) -> None:
        """Resolve ordinal references ("second video") against visible_lists."""
        ordinal_pattern = re.compile(
            r"\b(" + "|".join(cls.ORDINAL_MAP.keys()) + r")\b",
            re.IGNORECASE,
        )

        for match in ordinal_pattern.finditer(text):
            ordinal = match.group(1).lower()
            index = cls.ORDINAL_MAP[ordinal]

            # Try each visible list — take the first that has enough items
            for list_key, items in visible_lists.items():
                if index < len(items):
                    ctx.resolved_references.append(ResolvedReference(
                        ordinal=ordinal,
                        index=index,
                        list_key=list_key,
                        resolved_value=items[index],
                        resolution_source="ordinal",
                    ))
                    break  # Only resolve first matching list

    @classmethod
    def _resolve_from_entity_registry(
        cls,
        text: str,
        entity_registry: Dict[str, Any],
        ctx: QueryContext,
    ) -> None:
        """Resolve ordinals against entity_registry list entries.

        This is the fallback path when visible_lists (from the last outcome)
        don't contain a matching list — e.g. because a domain switch occurred.
        The entity_registry preserves list entities across domain switches.

        Only attempts ordinal resolution. Entity records whose value is a
        list are treated as implicit visible_lists.
        """
        ordinal_pattern = re.compile(
            r"\b(" + "|".join(cls.ORDINAL_MAP.keys()) + r")\b",
            re.IGNORECASE,
        )

        for match in ordinal_pattern.finditer(text):
            ordinal = match.group(1).lower()
            index = cls.ORDINAL_MAP[ordinal]

            # Scan entity_registry for list-type entries
            for key, record in entity_registry.items():
                # EntityRecord objects have .value and .type
                value = getattr(record, "value", record) if hasattr(record, "value") else record
                if isinstance(value, list) and index < len(value):
                    ctx.resolved_references.append(ResolvedReference(
                        ordinal=ordinal,
                        index=index,
                        list_key=f"entity_registry.{key}",
                        resolved_value=value[index],
                        resolution_source="entity_registry",
                    ))
                    break  # Only resolve first matching list

    @classmethod
    def _resolve_entity_reference(
        cls,
        text: str,
        active_entity: str,
        ctx: QueryContext,
    ) -> None:
        """Resolve 'that X' / 'the X' references against active_entity."""
        if cls._ENTITY_PATTERNS.search(text):
            ctx.resolved_references.append(ResolvedReference(
                entity_hint=active_entity,
                resolution_source="active",
            ))

    @classmethod
    def _resolve_system_reference(
        cls,
        text: str,
        system_session: Dict[str, Any],
        ctx: QueryContext,
    ) -> None:
        """
        Resolve system entity references against WorldState.system.session.

        Handles: "that app", "the window", "last app", "current application"
        Resolves to: foreground_app, or most recent from open_apps.
        """
        match = cls._SYSTEM_ENTITY_PATTERN.search(text)
        if not match:
            return

        modifier = match.group(1).lower()
        entity_type = match.group(2).lower()

        # "current" / "active" / "this" → foreground app
        if modifier in {"current", "active", "this"}:
            fg = system_session.get("foreground_app")
            if fg:
                ctx.resolved_references.append(ResolvedReference(
                    entity_hint=f"app '{fg}'",
                    resolved_value=fg,
                    resolution_source="session",
                ))
            return

        # "that" / "the" → foreground app (most likely referent)
        if modifier in {"that", "the"}:
            fg = system_session.get("foreground_app")
            if fg:
                ctx.resolved_references.append(ResolvedReference(
                    entity_hint=f"app '{fg}'",
                    resolved_value=fg,
                    resolution_source="session",
                ))
            return

        # "last" / "previous" → second-to-last in open_apps (or foreground)
        if modifier in {"last", "previous"}:
            open_apps = system_session.get("open_apps", [])
            if len(open_apps) >= 2:
                # Last opened before current foreground
                prev = open_apps[-2]
                ctx.resolved_references.append(ResolvedReference(
                    entity_hint=f"app '{prev}'",
                    resolved_value=prev,
                    resolution_source="session",
                ))
            elif open_apps:
                ctx.resolved_references.append(ResolvedReference(
                    entity_hint=f"app '{open_apps[-1]}'",
                    resolved_value=open_apps[-1],
                    resolution_source="session",
                ))
            return

    @classmethod
    def _resolve_foreground_fallback(
        cls,
        text: str,
        system_session: Dict[str, Any],
        ctx: QueryContext,
    ) -> None:
        """
        Fallback: resolve bare pronouns ("it", "close it") to foreground app.
        Only fires if nothing else resolved.
        """
        pronoun_pattern = re.compile(r"\b(it)\b", re.IGNORECASE)
        if pronoun_pattern.search(text):
            fg = system_session.get("foreground_app")
            if fg:
                ctx.resolved_references.append(ResolvedReference(
                    entity_hint=f"app '{fg}'",
                    resolved_value=fg,
                    resolution_source="session",
                ))

    @classmethod
    def resolve_ordinal_from_lists(
        cls,
        ordinal: str,
        visible_lists: Dict[str, list],
    ) -> Tuple[str, Any]:
        """
        Direct ordinal resolution for programmatic use.

        Returns (list_key, resolved_item).
        Raises ReferenceResolutionError on failure.
        """
        ordinal_lower = ordinal.lower()
        if ordinal_lower not in cls.ORDINAL_MAP:
            raise ReferenceResolutionError(
                f"Unsupported ordinal '{ordinal}'"
            )

        index = cls.ORDINAL_MAP[ordinal_lower]

        if not visible_lists:
            raise ReferenceResolutionError(
                "No visible lists available for resolution"
            )

        for list_key, items in visible_lists.items():
            if index < len(items):
                return list_key, items[index]

        raise ReferenceResolutionError(
            f"Ordinal '{ordinal}' out of range for all lists"
        )
