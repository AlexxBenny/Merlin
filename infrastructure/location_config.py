# infrastructure/location_config.py

"""
LocationConfig — Infrastructure-only path anchor resolution.

Resolves anchor names → absolute filesystem Paths.
Loaded from config/paths.yaml. Injected via constructor.

INVARIANTS:
- No language parsing. No text matching. No inference.
- No component below MissionCortex sees raw user text.
- WORKSPACE is always dynamic (from cwd), never in YAML.
- This code is infrastructure, not intelligence.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


logger = logging.getLogger(__name__)


class LocationConfig:
    """
    Infrastructure-only anchor resolution.

    Resolves symbolic anchor names (e.g., "DESKTOP", "DRIVE_D")
    to absolute filesystem Paths. No language. No inference.

    Constructed from config/paths.yaml. Injected — not a singleton.
    """

    def __init__(self, anchors: Dict[str, Path], cwd: Path):
        """
        Args:
            anchors: Mapping of anchor name → resolved absolute Path.
            cwd: Current working directory (backs the WORKSPACE anchor).
        """
        self._anchors = anchors
        self._cwd = cwd

    # ─────────────────────────────────────────────────────────
    # Factory
    # ─────────────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, config_path: Path, cwd: Optional[Path] = None) -> "LocationConfig":
        """
        Load anchors from a YAML config file.

        Expected YAML structure:

            anchors:
              DESKTOP:
                path: "${USERPROFILE}/Desktop"
              DOCUMENTS:
                path: "${USERPROFILE}/Documents"

            drives:
              enabled: ["C", "D", "E"]

        Args:
            config_path: Path to the YAML config file.
            cwd: Current working directory. Defaults to Path.cwd().
        """
        if cwd is None:
            cwd = Path.cwd()

        if not config_path.exists():
            logger.warning("LocationConfig: %s not found, using defaults", config_path)
            return cls._default(cwd)

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
        except Exception as e:
            logger.error("LocationConfig: Failed to load %s: %s", config_path, e)
            return cls._default(cwd)

        anchors = cls._parse_anchors(raw)
        config = cls(anchors=anchors, cwd=cwd)
        config._validate()

        logger.info(
            "LocationConfig: Loaded %d anchors from %s",
            len(anchors),
            config_path,
        )

        return config

    @classmethod
    def _default(cls, cwd: Path) -> "LocationConfig":
        """Minimal fallback defaults."""
        home = Path.home()
        return cls(
            anchors={
                "DESKTOP": home / "Desktop",
                "DOCUMENTS": home / "Documents",
                "DOWNLOADS": home / "Downloads",
            },
            cwd=cwd,
        )

    # ─────────────────────────────────────────────────────────
    # Parsing
    # ─────────────────────────────────────────────────────────

    @classmethod
    def _parse_anchors(cls, raw: Dict[str, Any]) -> Dict[str, Path]:
        """Parse YAML into anchor name → Path mapping."""
        anchors: Dict[str, Path] = {}

        # Named anchors
        for name, data in raw.get("anchors", {}).items():
            if name == "WORKSPACE":
                logger.warning(
                    "LocationConfig: WORKSPACE must not be in YAML — skipped"
                )
                continue

            path_template = data.get("path", "")
            resolved = cls._resolve_template(path_template)
            anchors[name] = Path(resolved)

        # Drive anchors (auto-generated)
        drives = raw.get("drives", {})
        for letter in drives.get("enabled", []):
            anchor_name = f"DRIVE_{letter.upper()}"
            anchors[anchor_name] = Path(f"{letter.upper()}:/")

        return anchors

    @staticmethod
    def _resolve_template(template: str) -> str:
        """
        Resolve environment variable templates in path strings.

        Supports ${VAR} syntax. Unresolved vars are left as-is.
        """
        import re

        def _replace(match: object) -> str:
            var_name = match.group(1)
            return os.environ.get(var_name, match.group(0))

        return re.sub(r"\$\{(\w+)\}", _replace, template)

    # ─────────────────────────────────────────────────────────
    # Validation
    # ─────────────────────────────────────────────────────────

    def _validate(self) -> None:
        """Validate config on load. Log warnings for non-existent paths."""
        for name, path in self._anchors.items():
            if not path.exists():
                logger.warning(
                    "LocationConfig: Anchor '%s' path does not exist: %s",
                    name,
                    path,
                )

    # ─────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────

    def resolve(self, anchor: str) -> Path:
        """
        Resolve anchor name to absolute Path.

        Args:
            anchor: Anchor name (e.g., "DESKTOP", "DRIVE_D", "WORKSPACE")

        Returns:
            Absolute Path

        Raises:
            KeyError: If anchor is unknown
        """
        if anchor == "WORKSPACE":
            return self._cwd

        if anchor not in self._anchors:
            raise KeyError(f"Unknown anchor: '{anchor}'")

        return self._anchors[anchor]

    def all_anchor_names(self) -> List[str]:
        """All anchor names including WORKSPACE. Used by MissionCortex prompt."""
        return sorted(self._anchors.keys()) + ["WORKSPACE"]

    def all_anchors(self) -> Dict[str, Path]:
        """All anchor name → Path mappings including WORKSPACE."""
        result = dict(self._anchors)
        result["WORKSPACE"] = self._cwd
        return result
