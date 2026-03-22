# world/file_ref.py

"""
FileRef — Structured file reference for discovery skills.

Replaces raw path leakage with a typed abstraction that:
1. Stores identity (anchor + relative_path) without coupling to filesystem
2. Resolves to absolute path at execution time via LocationConfig
3. Uses the SAME resolution pattern as all fs skills:
   location_config.resolve(anchor) / relative_path

Lifecycle:
- Created by search_file / list_directory at execution time
- Stored in ExecutionState.file_refs[ref_id] for the mission duration
- Piped between nodes via OutputReference (existing IR mechanism)
- Resolved to absolute path only when needed (inside skill.execute)
"""

import logging
from pathlib import Path

from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)


def _generate_ref_id() -> str:
    """Generate a unique file reference ID."""
    import time
    import hashlib
    raw = f"fref_{time.time_ns()}"
    return f"fref_{hashlib.sha256(raw.encode()).hexdigest()[:12]}"


class FileRef(BaseModel):
    """Structured file reference — identity without path coupling.

    Stored in ExecutionState.file_refs[ref_id].
    Resolved to absolute path via location_config.resolve(anchor) / relative_path
    (same pattern as read_file, write_file, create_folder).
    """
    model_config = ConfigDict(extra="forbid")

    ref_id: str              # unique ref (e.g., "fref_a1b2c3d4e5f6")
    name: str                # "Alex_Benny.pdf"
    anchor: str              # "DESKTOP"
    relative_path: str       # "projects/Alex_Benny.pdf"
    size_bytes: int = 0
    confidence: float = 1.0  # 0.0-1.0 (search match quality)
    is_directory: bool = False

    def resolve(self, location_config) -> Path:
        """Resolve to absolute path. Same pattern as all fs skills.

        Uses location_config.resolve(anchor) / relative_path — the exact
        same call used by ReadFileSkill, WriteFileSkill, CreateFolderSkill.
        """
        base = location_config.resolve(self.anchor)
        return base / self.relative_path

    def to_output_dict(self) -> dict:
        """Serialize for skill output (consumed by OutputReference)."""
        return {
            "ref_id": self.ref_id,
            "name": self.name,
            "anchor": self.anchor,
            "relative_path": self.relative_path,
            "size_bytes": self.size_bytes,
            "confidence": self.confidence,
            "is_directory": self.is_directory,
        }
