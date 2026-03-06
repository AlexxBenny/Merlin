# memory/user_knowledge.py

"""
UserKnowledgeStore — Structured semantic memory for user knowledge.

Stores five knowledge domains:
    preferences: typed values the user prefers (volume=80, theme="dark")
    facts:       user-declared facts (name="Alex", timezone="IST")
    traits:      inferred behavioral characteristics (technical_level="advanced")
    policies:    conditional rules (activity=movie → set_volume=90)
    relationships: entity associations (laptop="Lenovo LOQ")

Design rules:
- Every entry is versioned: value, confidence, timestamp, source, history
- Preferences have minimal schema enforcement (type validation)
- Canonical key normalization prevents fragmented storage
- History is bounded (max 20 entries per key)
- Policies use structured condition/action dicts with subset matching
- JSON persistence with atomic writes (temp file → rename)
- No LLM. Pure data layer.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Canonical Key Normalization
# ─────────────────────────────────────────────────────────────

# Maps alias → canonical key. Prevents fragmented storage.
# e.g., "sound", "audio level", "speaker volume" all → "volume"
_CANONICAL_KEYS: Dict[str, str] = {
    # Volume aliases
    "sound": "volume",
    "audio": "volume",
    "audio level": "volume",
    "audio_level": "volume",
    "speaker volume": "volume",
    "speaker_volume": "volume",
    "sound level": "volume",
    "sound_level": "volume",
    # Brightness aliases
    "screen brightness": "brightness",
    "screen_brightness": "brightness",
    "display brightness": "brightness",
    "display_brightness": "brightness",
    # Theme aliases
    "color scheme": "theme",
    "color_scheme": "theme",
    "colour scheme": "theme",
    "appearance": "theme",
}


def normalize_key(key: str) -> str:
    """Normalize a preference/fact key to its canonical form.

    Steps:
    1. Lowercase + strip
    2. Replace spaces with underscores
    3. Look up alias table
    4. Strip common prefixes (preferred_, my_, default_)
    """
    k = key.lower().strip()
    # Check raw alias (with spaces)
    if k in _CANONICAL_KEYS:
        return _CANONICAL_KEYS[k]
    # Normalize separators
    k = k.replace(" ", "_")
    if k in _CANONICAL_KEYS:
        return _CANONICAL_KEYS[k]
    # Strip common prefixes
    for prefix in ("preferred_", "my_", "default_", "favorite_", "favourite_"):
        if k.startswith(prefix):
            k = k[len(prefix):]
            break
    # Final alias check after prefix strip
    if k in _CANONICAL_KEYS:
        return _CANONICAL_KEYS[k]
    return k


# ─────────────────────────────────────────────────────────────
# Schema Enforcement (minimal — type validation only)
# ─────────────────────────────────────────────────────────────

# Maps canonical key → expected Python type.
# Values are cast to this type on store. Invalid casts are rejected.
PREFERENCE_SCHEMA: Dict[str, type] = {
    "volume": int,
    "brightness": int,
    "theme": str,
    "font_size": int,
    "language": str,
}


def validate_and_cast(key: str, value: Any) -> Any:
    """Validate and cast a value to its schema type.

    Returns the cast value, or the original value if no schema exists.
    Raises ValueError if cast fails for a schema-defined key.
    """
    if key not in PREFERENCE_SCHEMA:
        return value  # No schema — accept as-is
    expected_type = PREFERENCE_SCHEMA[key]
    if isinstance(value, expected_type):
        return value
    try:
        return expected_type(value)
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"Cannot store '{key}': expected {expected_type.__name__}, "
            f"got {type(value).__name__} ({value!r})"
        ) from e


# ─────────────────────────────────────────────────────────────
# Data Model
# ─────────────────────────────────────────────────────────────

MAX_HISTORY = 20  # Maximum previous values retained per key


@dataclass
class KnowledgeEntry:
    """A versioned knowledge entry with history tracking."""
    value: Any
    confidence: float = 1.0
    source: str = "user"           # "user", "inferred", "system"
    updated_at: str = ""
    value_type: str = ""           # type(value).__name__
    history: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        if not self.updated_at:
            self.updated_at = datetime.now(timezone.utc).isoformat()
        if not self.value_type:
            self.value_type = type(self.value).__name__

    def update(self, new_value: Any, source: str = "user") -> None:
        """Update value, pushing current to history (bounded)."""
        # Push current value to history
        self.history.append({
            "value": self.value,
            "confidence": self.confidence,
            "source": self.source,
            "updated_at": self.updated_at,
        })
        # Enforce history limit
        if len(self.history) > MAX_HISTORY:
            self.history = self.history[-MAX_HISTORY:]
        # Set new value
        self.value = new_value
        self.confidence = 1.0
        self.source = source
        self.updated_at = datetime.now(timezone.utc).isoformat()
        self.value_type = type(new_value).__name__

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "KnowledgeEntry":
        return cls(
            value=d["value"],
            confidence=d.get("confidence", 1.0),
            source=d.get("source", "user"),
            updated_at=d.get("updated_at", ""),
            value_type=d.get("value_type", ""),
            history=d.get("history", []),
        )


@dataclass
class Policy:
    """A conditional rule with structured condition/action matching.

    condition: Dict[str, Any] — fields to match against context.
    action: Dict[str, Any] — parameter overrides to apply.

    Matching uses SUBSET logic: policy matches if
    every key in condition exists in context with a matching value.
    """
    condition: Dict[str, Any]
    action: Dict[str, Any]
    priority: int = 0
    created_at: str = ""
    label: str = ""
    id: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
        if not self.id:
            self.id = str(uuid.uuid4())

    def matches(self, context: Dict[str, Any]) -> bool:
        """Subset match: condition ⊆ context."""
        for key, expected in self.condition.items():
            actual = context.get(key)
            if actual is None:
                return False
            # Case-insensitive string comparison
            if isinstance(expected, str) and isinstance(actual, str):
                if expected.lower() != actual.lower():
                    return False
            elif actual != expected:
                return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Policy":
        return cls(
            condition=d["condition"],
            action=d["action"],
            priority=d.get("priority", 0),
            created_at=d.get("created_at", ""),
            label=d.get("label", ""),
            id=d.get("id", ""),
        )


# ─────────────────────────────────────────────────────────────
# UserKnowledgeStore
# ─────────────────────────────────────────────────────────────

class UserKnowledgeStore:
    """Structured user knowledge with persistence.

    Five domains:
        preferences — typed values (volume=80)
        facts       — declared information (name="Alex")
        traits      — inferred characteristics (technical_level="advanced")
        policies    — conditional rules ({activity: movie} → {set_volume: 90})
        relationships — entity associations (laptop="Lenovo LOQ")
    """

    def __init__(self, persist_path: Optional[str] = None):
        self._preferences: Dict[str, KnowledgeEntry] = {}
        self._facts: Dict[str, KnowledgeEntry] = {}
        self._traits: Dict[str, KnowledgeEntry] = {}
        self._policies: List[Policy] = []
        self._relationships: Dict[str, KnowledgeEntry] = {}
        self._persist_path = persist_path
        if persist_path:
            self._load()

    # ── Preferences ──

    def set_preference(self, key: str, value: Any, source: str = "user") -> None:
        """Store a preference with type validation and history."""
        canonical = normalize_key(key)
        value = validate_and_cast(canonical, value)
        if canonical in self._preferences:
            self._preferences[canonical].update(value, source=source)
        else:
            self._preferences[canonical] = KnowledgeEntry(
                value=value, source=source,
            )
        logger.info(
            "[KNOWLEDGE] Stored preference: %s = %r (source=%s)",
            canonical, value, source,
        )
        self._save()

    def get_preference(self, key: str) -> Optional[Any]:
        """Get a preference value by key (canonical normalized)."""
        canonical = normalize_key(key)
        entry = self._preferences.get(canonical)
        return entry.value if entry else None

    def get_preference_entry(self, key: str) -> Optional[KnowledgeEntry]:
        """Get the full KnowledgeEntry for a preference."""
        canonical = normalize_key(key)
        return self._preferences.get(canonical)

    # ── Facts ──

    def set_fact(self, key: str, value: Any, source: str = "user") -> None:
        """Store a fact about the user."""
        canonical = normalize_key(key)
        if canonical in self._facts:
            self._facts[canonical].update(value, source=source)
        else:
            self._facts[canonical] = KnowledgeEntry(
                value=value, source=source,
            )
        logger.info("[KNOWLEDGE] Stored fact: %s = %r", canonical, value)
        self._save()

    def get_fact(self, key: str) -> Optional[Any]:
        """Get a fact value by key."""
        canonical = normalize_key(key)
        entry = self._facts.get(canonical)
        return entry.value if entry else None

    # ── Traits ──

    def set_trait(self, key: str, value: Any, source: str = "inferred") -> None:
        """Store an inferred trait."""
        canonical = normalize_key(key)
        if canonical in self._traits:
            self._traits[canonical].update(value, source=source)
        else:
            self._traits[canonical] = KnowledgeEntry(
                value=value, source=source,
            )
        logger.info("[KNOWLEDGE] Stored trait: %s = %r", canonical, value)
        self._save()

    # ── Relationships ──

    def set_relationship(self, key: str, value: Any, source: str = "user") -> None:
        """Store an entity relationship."""
        canonical = normalize_key(key)
        if canonical in self._relationships:
            self._relationships[canonical].update(value, source=source)
        else:
            self._relationships[canonical] = KnowledgeEntry(
                value=value, source=source,
            )
        logger.info("[KNOWLEDGE] Stored relationship: %s = %r", canonical, value)
        self._save()

    # ── Policies ──

    def add_policy(
        self,
        condition: Dict[str, Any],
        action: Dict[str, Any],
        priority: int = 0,
        label: str = "",
    ) -> str:
        """Add a conditional policy rule. Returns the policy id."""
        policy = Policy(
            condition=condition,
            action=action,
            priority=priority,
            label=label,
        )
        self._policies.append(policy)
        logger.info(
            "[KNOWLEDGE] Stored policy %s: %s → %s",
            policy.id, condition, action,
        )
        self._save()
        return policy.id

    def get_matching_policies(
        self, context: Dict[str, Any],
    ) -> List[Policy]:
        """Return all policies whose conditions are a subset of context.

        Results sorted by priority (highest first).
        """
        matched = [p for p in self._policies if p.matches(context)]
        matched.sort(key=lambda p: p.priority, reverse=True)
        return matched

    # ── Unified Query ──

    def query(self, key: str) -> Optional[Any]:
        """Search across all domains: preferences → facts → traits → relationships.

        Returns the first matching value, or None.
        """
        canonical = normalize_key(key)
        for domain in (
            self._preferences,
            self._facts,
            self._traits,
            self._relationships,
        ):
            entry = domain.get(canonical)
            if entry is not None:
                return entry.value
        return None

    # ── Memory Context Retrieval (LLM prompt injection seam) ──

    def retrieve_memory_context(
        self, query: str, max_entries: int = 50,
    ) -> str:
        """Build a compact text summary of stored user knowledge.

        Returns structured text for injection into LLM prompts.

        This is the RETRIEVAL SEAM. Current implementation returns
        all entries (sufficient for < ~100 entries). Future phases:
          Phase 2: key-match filtering based on query subjects
          Phase 3: embedding-based semantic retrieval (RAG)

        The coordinator never changes — only this method evolves.

        Args:
            query: The user's query (for future relevance filtering).
            max_entries: Maximum entries to include in the context.
        """
        lines: list[str] = []
        for key, entry in self._facts.items():
            lines.append(f"  fact: {key} = {entry.value}")
        for key, entry in self._preferences.items():
            lines.append(f"  preference: {key} = {entry.value}")
        for key, entry in self._traits.items():
            lines.append(f"  trait: {key} = {entry.value}")
        for key, entry in self._relationships.items():
            lines.append(f"  relationship: {key} = {entry.value}")
        if self._policies:
            for p in self._policies[:5]:
                label = p.label or str(p.condition)
                lines.append(f"  policy: {label} → {p.action}")
            if len(self._policies) > 5:
                lines.append(
                    f"  ... and {len(self._policies) - 5} more policies"
                )
        # Bound output
        if len(lines) > max_entries:
            total = len(lines)
            lines = lines[:max_entries]
            lines.append(f"  ... ({total} total entries, truncated)")
        return "\n".join(lines) if lines else "  (no stored knowledge)"

    # ── Persistence (atomic writes) ──

    def _save(self) -> None:
        """Persist store to JSON. Atomic: write temp → rename."""
        if not self._persist_path:
            return
        try:
            data = {
                "preferences": {
                    k: v.to_dict() for k, v in self._preferences.items()
                },
                "facts": {
                    k: v.to_dict() for k, v in self._facts.items()
                },
                "traits": {
                    k: v.to_dict() for k, v in self._traits.items()
                },
                "relationships": {
                    k: v.to_dict() for k, v in self._relationships.items()
                },
                "policies": [p.to_dict() for p in self._policies],
            }
            # Ensure directory exists
            dir_path = os.path.dirname(self._persist_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            # Atomic write: temp file → rename
            fd, tmp_path = tempfile.mkstemp(
                dir=dir_path or ".",
                suffix=".tmp",
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                # Atomic rename (on Windows, need to remove target first)
                if os.path.exists(self._persist_path):
                    os.replace(tmp_path, self._persist_path)
                else:
                    os.rename(tmp_path, self._persist_path)
            except Exception:
                # Clean up temp file on failure
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                raise
        except Exception as e:
            logger.warning("[KNOWLEDGE] Failed to persist: %s", e)

    def _load(self) -> None:
        """Load store from JSON file."""
        if not self._persist_path or not os.path.exists(self._persist_path):
            return
        try:
            with open(self._persist_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Preferences
            for k, v in data.get("preferences", {}).items():
                self._preferences[k] = KnowledgeEntry.from_dict(v)
            # Facts
            for k, v in data.get("facts", {}).items():
                self._facts[k] = KnowledgeEntry.from_dict(v)
            # Traits
            for k, v in data.get("traits", {}).items():
                self._traits[k] = KnowledgeEntry.from_dict(v)
            # Relationships
            for k, v in data.get("relationships", {}).items():
                self._relationships[k] = KnowledgeEntry.from_dict(v)
            # Policies
            for p in data.get("policies", []):
                self._policies.append(Policy.from_dict(p))
            logger.info(
                "[KNOWLEDGE] Loaded: %d preferences, %d facts, "
                "%d traits, %d policies, %d relationships",
                len(self._preferences), len(self._facts),
                len(self._traits), len(self._policies),
                len(self._relationships),
            )
        except Exception as e:
            logger.warning("[KNOWLEDGE] Failed to load %s: %s",
                           self._persist_path, e)
