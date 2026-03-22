# skills/fs/list_directory.py

"""
ListDirectorySkill — List contents of a directory at an anchor location.

Outputs directory entries as FileRef objects (structured, not raw paths).

Effect model:
  requires=[]           — no preconditions
  produces=["file_reference"]  — reveals file identities
  effect_type="reveal"  — discovery, not creation
"""

from pathlib import Path
from typing import Any, Dict, Optional

from skills.skill_result import SkillResult
from skills.base import Skill
from skills.contract import SkillContract, FailurePolicy
from ir.mission import ExecutionMode
from world.timeline import WorldTimeline
from world.snapshot import WorldSnapshot
from world.file_ref import FileRef, _generate_ref_id
from infrastructure.location_config import LocationConfig

import logging

logger = logging.getLogger(__name__)


class ListDirectorySkill(Skill):
    """List contents of a directory at an anchor location.

    Returns sorted list of entries as FileRef objects.
    Immediate children only (no recursion).
    """

    contract = SkillContract(
        name="fs.list_directory",
        action="list_directory",
        target_type="directory",
        description="List directory contents",
        narration_template="listing {path}",
        intent_verbs=["list", "show", "browse", "ls"],
        intent_keywords=["directory", "folder", "files", "contents"],
        verb_specificity="generic",
        domain="fs",
        resource_cost="low",
        inputs={},
        optional_inputs={"path": "file_path_input", "anchor": "anchor_name"},
        outputs={"contents": "directory_contents"},
        allowed_modes={ExecutionMode.foreground},
        failure_policy={ExecutionMode.foreground: FailurePolicy.FAIL},
        emits_events=["directory_listed"],
        mutates_world=False,
        output_style="rich",
        requires=[],
        produces=["file_reference"],
        effect_type="reveal",
    )

    def __init__(self, location_config: LocationConfig):
        self._location_config = location_config

    def execute(
        self,
        inputs: Dict[str, Any],
        world: WorldTimeline,
        snapshot: Optional[WorldSnapshot] = None,
        context=None,
    ) -> SkillResult:
        rel_path = inputs.get("path", "")
        anchor = inputs.get("anchor", "WORKSPACE")

        logger.info(
            "[TRACE] ListDirectorySkill.execute: path=%r, anchor=%r",
            rel_path, anchor,
        )

        # Resolve anchor → absolute base path (same as all fs skills)
        base = self._location_config.resolve(anchor)

        if rel_path:
            target = base / rel_path
        else:
            target = base

        if not target.exists():
            raise FileNotFoundError(
                f"Directory not found: {target} "
                f"(anchor={anchor}, path={rel_path})"
            )

        if not target.is_dir():
            raise NotADirectoryError(
                f"Not a directory: {target}"
            )

        # List immediate children, sorted by name
        entries = []
        try:
            for entry in sorted(target.iterdir(), key=lambda e: e.name.lower()):
                # Skip hidden files
                if entry.name.startswith("."):
                    continue

                try:
                    stat = entry.stat()
                    size = stat.st_size if entry.is_file() else 0
                except OSError:
                    size = 0

                entry_rel_path = (
                    f"{rel_path}/{entry.name}" if rel_path
                    else entry.name
                )

                ref = FileRef(
                    ref_id=_generate_ref_id(),
                    name=entry.name,
                    anchor=anchor,
                    relative_path=entry_rel_path,
                    size_bytes=size,
                    confidence=1.0,
                    is_directory=entry.is_dir(),
                )
                entries.append(ref.to_output_dict())

        except PermissionError:
            raise PermissionError(
                f"Cannot read directory: {target}"
            )

        # Emit event
        world.emit("skill.fs", "directory_listed", {
            "path": str(target),
            "anchor": anchor,
            "entry_count": len(entries),
        })

        return SkillResult(
            outputs={"contents": entries},
            metadata={
                "domain": "fs",
                "entity": f"directory '{target.name}'",
            },
        )
