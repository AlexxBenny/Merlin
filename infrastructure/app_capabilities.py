# infrastructure/app_capabilities.py

"""
AppCapabilityRegistry — Static application capability metadata.

Capabilities describe what an application TYPE supports,
not what any particular instance is doing.

Design rules:
- Capabilities are static per app type (not per session)
- Loaded from config/app_capabilities.yaml at startup
- Queried by the compiler (to prevent nonsense plans)
  and by the supervisor (to validate guard applicability)
- Unknown apps get _default capabilities (conservative: all False)
- Never imported by skills — skills don't decide, they execute

This is infrastructure, not cognition.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# AppCapabilities — per-app-type capability descriptor
# ─────────────────────────────────────────────────────────────

class AppCapabilities(BaseModel):
    """Static capability flags for an application type.

    These describe what the app supports in general,
    not what a specific instance is currently doing.
    """
    model_config = ConfigDict(extra="forbid")

    supports_typing: bool = False
    supports_copy: bool = False
    supports_save: bool = False


# ─────────────────────────────────────────────────────────────
# Default capabilities (conservative — all False)
# ─────────────────────────────────────────────────────────────

_DEFAULT_CAPABILITIES = AppCapabilities(
    supports_typing=False,
    supports_copy=False,
    supports_save=False,
)


# ─────────────────────────────────────────────────────────────
# AppCapabilityRegistry
# ─────────────────────────────────────────────────────────────

class AppCapabilityRegistry:
    """Lookup table for application capabilities.

    Loaded from config at startup. Immutable after construction.
    Query by app name (case-insensitive, strip .exe suffix).

    Usage:
        registry = AppCapabilityRegistry.from_yaml("config/app_capabilities.yaml")
        caps = registry.get("notepad")
        if caps.supports_typing:
            ...
    """

    def __init__(self, entries: Dict[str, AppCapabilities]):
        # Normalize keys to lowercase, strip .exe
        self._entries: Dict[str, AppCapabilities] = {}
        default = _DEFAULT_CAPABILITIES

        for key, caps in entries.items():
            if key == "_default":
                default = caps
            else:
                normalized = self._normalize_name(key)
                self._entries[normalized] = caps

        self._default = default
        logger.info(
            "AppCapabilityRegistry: %d app types loaded, default=%s",
            len(self._entries), self._default,
        )

    @classmethod
    def from_yaml(cls, path: str) -> "AppCapabilityRegistry":
        """Load capability registry from a YAML file.

        Expected format:
            notepad:
              supports_typing: true
              supports_copy: true
              supports_save: true
            _default:
              supports_typing: false
              supports_copy: false
              supports_save: false
        """
        yaml_path = Path(path)
        if not yaml_path.exists():
            logger.warning(
                "App capabilities config not found: %s — using empty registry",
                yaml_path,
            )
            return cls({})

        with open(yaml_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        entries: Dict[str, AppCapabilities] = {}
        for key, value in raw.items():
            if not isinstance(value, dict):
                logger.warning(
                    "Skipping malformed capability entry '%s': expected dict, got %s",
                    key, type(value).__name__,
                )
                continue
            try:
                entries[key] = AppCapabilities(**value)
            except Exception as e:
                logger.warning(
                    "Failed to parse capabilities for '%s': %s", key, e,
                )

        return cls(entries)

    @classmethod
    def from_dict(cls, data: Dict[str, Dict[str, Any]]) -> "AppCapabilityRegistry":
        """Build from a plain dict (e.g., inline config or tests)."""
        entries: Dict[str, AppCapabilities] = {}
        for key, value in data.items():
            if isinstance(value, dict):
                try:
                    entries[key] = AppCapabilities(**value)
                except Exception as e:
                    logger.warning(
                        "Failed to parse capabilities for '%s': %s", key, e,
                    )
            elif isinstance(value, AppCapabilities):
                entries[key] = value
        return cls(entries)

    def get(self, app_name: str) -> AppCapabilities:
        """Look up capabilities for an app. Returns default if unknown."""
        normalized = self._normalize_name(app_name)
        return self._entries.get(normalized, self._default)

    def supports(self, app_name: str, capability: str) -> bool:
        """Check if an app supports a specific capability by name."""
        caps = self.get(app_name)
        return getattr(caps, capability, False)

    @property
    def known_apps(self) -> set:
        """Return the set of app names with explicit capability entries."""
        return set(self._entries.keys())

    @staticmethod
    def _normalize_name(name: str) -> str:
        """Normalize app name: lowercase, strip .exe suffix."""
        n = name.lower().strip()
        if n.endswith(".exe"):
            n = n[:-4]
        return n
