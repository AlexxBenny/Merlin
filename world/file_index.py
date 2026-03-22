# world/file_index.py

"""
FileIndex — Lazy-built file index across all configured anchors.

key: filename_lower → value: List[FileRef]
Built on first search. NOT TTL-invalidated — validate on selection.

Invalidation strategy:
1. Index build: one-time walk on first search, bounded max_depth=5
2. Incremental updates: file_written/folder_created → add_ref() (no rebuild)
3. No TTL: stale data doesn't matter because:
4. Validate-on-selection: when a FileRef is chosen, the existing
   FILE_EXISTS StepGuard validates the path BEFORE execution.
   - Path exists → proceed
   - Path gone → guard fails → _attempt_repair() → re-search
5. Full rebuild: only on explicit invalidate() call (rare)

This avoids the TTL problem: we don't care if the index has stale entries,
because every selected ref is validated before use. External deletes/moves
are caught by the guard, not the index.
"""

import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from world.file_ref import FileRef, _generate_ref_id

logger = logging.getLogger(__name__)

# Bounds to prevent runaway walks
MAX_DEPTH = 5
MAX_FILES = 50000  # max files to index across all anchors
MAX_RESULTS = 50   # max search results returned

# Anchors indexed first (user content — most likely search targets)
_PRIORITY_ANCHORS = {
    "DESKTOP", "DOCUMENTS", "DOWNLOADS", "HOME",
    "MUSIC", "PICTURES", "VIDEOS", "WORKSPACE",
}

# Directories to skip during walks (heavy system/tool dirs)
_SKIP_DIRS = {
    # Windows system
    "windows", "$recycle.bin", "system volume information",
    "program files", "program files (x86)", "programdata",
    "recovery", "perflogs",
    # Dev tools / package managers
    "node_modules", ".git", "__pycache__", ".venv", "venv",
    ".tox", ".mypy_cache", ".pytest_cache", "dist", "build",
    ".cargo", ".rustup", ".gradle",
    # App data (huge, rarely contains user files)
    "appdata",
}


class FileIndex:
    """Lazy-built file index with incremental updates.

    Uses the same location_config.resolve(anchor) pattern as all fs skills.
    """

    def __init__(self):
        self._index: Dict[str, List[FileRef]] = defaultdict(list)
        self._built: bool = False
        self._total_files: int = 0

    def build(self, location_config, max_depth: int = MAX_DEPTH) -> None:
        """Build index by walking all configured anchors.

        Bounded: max_depth levels, max MAX_FILES total entries.
        Uses location_config.all_anchors() for discovery.
        Priority: user content anchors indexed first, drives last.
        """
        self._index.clear()
        self._total_files = 0

        try:
            anchors = location_config.all_anchors()
        except AttributeError:
            logger.warning("LocationConfig has no all_anchors() — index empty")
            self._built = True
            return

        # Priority sort: user dirs first, drives last
        sorted_anchors = sorted(
            anchors.items(),
            key=lambda kv: (0 if kv[0] in _PRIORITY_ANCHORS else 1, kv[0]),
        )

        for anchor_name, anchor_path in sorted_anchors:
            base = Path(anchor_path)
            if not base.exists() or not base.is_dir():
                continue

            self._walk_directory(base, anchor_name, "", max_depth, 0)

            if self._total_files >= MAX_FILES:
                logger.warning(
                    "FileIndex hit MAX_FILES (%d) — stopping early",
                    MAX_FILES,
                )
                break

        self._built = True
        logger.info(
            "FileIndex built: %d files indexed across %d anchors",
            self._total_files, len(anchors),
        )

    def _walk_directory(
        self,
        directory: Path,
        anchor: str,
        relative_prefix: str,
        max_depth: int,
        current_depth: int,
    ) -> None:
        """Walk a directory tree, bounded by depth and total file count."""
        if current_depth >= max_depth or self._total_files >= MAX_FILES:
            return

        try:
            entries = sorted(directory.iterdir())
        except (PermissionError, OSError) as e:
            logger.debug("Cannot read %s: %s", directory, e)
            return

        for entry in entries:
            if self._total_files >= MAX_FILES:
                return

            rel_path = (
                f"{relative_prefix}/{entry.name}" if relative_prefix
                else entry.name
            )

            if entry.is_file():
                try:
                    size = entry.stat().st_size
                except OSError:
                    size = 0

                ref = FileRef(
                    ref_id=_generate_ref_id(),
                    name=entry.name,
                    anchor=anchor,
                    relative_path=rel_path,
                    size_bytes=size,
                    confidence=1.0,
                    is_directory=False,
                )
                self._index[entry.name.lower()].append(ref)
                self._total_files += 1

            elif entry.is_dir():
                # Skip hidden directories
                if entry.name.startswith("."):
                    continue
                # Skip known heavy system/tool directories
                if entry.name.lower() in _SKIP_DIRS:
                    continue
                # Recurse
                self._walk_directory(
                    entry, anchor, rel_path, max_depth, current_depth + 1,
                )

    def search(
        self,
        query: str,
        location_config=None,
        max_results: int = MAX_RESULTS,
    ) -> List[FileRef]:
        """Search for files matching query. Lazy-builds index on first call.

        Matching: case-insensitive substring on filename.
        Ranked by confidence: exact match > starts-with > substring.
        """
        if not self._built and location_config is not None:
            self.build(location_config)

        if not self._built:
            return []

        query_lower = query.lower().strip()
        if not query_lower:
            return []

        matches: List[tuple] = []  # (confidence, FileRef)

        for filename_lower, refs in self._index.items():
            if query_lower in filename_lower:
                # Score: exact > starts-with > substring
                if filename_lower == query_lower:
                    score = 1.0
                elif filename_lower.startswith(query_lower):
                    score = 0.8
                else:
                    score = 0.5

                for ref in refs:
                    scored_ref = ref.model_copy(update={"confidence": score})
                    matches.append((score, scored_ref))

        # Sort by confidence descending, then by name
        matches.sort(key=lambda x: (-x[0], x[1].name))
        return [ref for _, ref in matches[:max_results]]

    def add_ref(self, ref: FileRef) -> None:
        """Incremental update — add a new FileRef without rebuilding."""
        self._index[ref.name.lower()].append(ref)
        self._total_files += 1

    def remove_ref(self, ref_id: str) -> None:
        """Remove a FileRef by ID."""
        for key, refs in self._index.items():
            self._index[key] = [r for r in refs if r.ref_id != ref_id]

    def invalidate(self) -> None:
        """Mark index for full rebuild on next search."""
        self._built = False
        self._index.clear()
        self._total_files = 0

    @property
    def is_built(self) -> bool:
        return self._built

    @property
    def total_files(self) -> int:
        return self._total_files
