from pathlib import Path
from typing import Any, Dict

from skills.skill_result import SkillResult

from skills.base import Skill
from skills.contract import SkillContract, FailurePolicy
from ir.mission import ExecutionMode
from world.timeline import WorldTimeline
from infrastructure.location_config import LocationConfig


class CreateFolderSkill(Skill):
    """
    Create a directory at an anchor-resolved location.

    Inputs:
        name:   Folder name to create (e.g., "myProject")
        anchor: Location anchor (e.g., "DESKTOP", "DRIVE_D", "WORKSPACE")
        parent: Optional relative path under anchor (e.g., "alex/projects")

    Resolution (inside execute, using LocationConfig):
        base = location_config.resolve(anchor)   # e.g., D:/
        path = base / parent / name               # e.g., D:/alex/projects/myProject

    No raw paths from LLM. No path guessing. No intelligence here.
    """

    contract = SkillContract(
        name="fs.create_folder",
        action="create_folder",
        target_type="folder",
        description="Create a folder",
        narration_template="create folder {name}",
        intent_verbs=["create", "make", "new"],
        intent_keywords=["folder", "directory", "dir"],
        verb_specificity="generic",
        domain="fs",
        inputs={
            "name": "folder_name",
        },
        optional_inputs={
            "anchor": "anchor_name",
            "parent": "relative_path",
        },
        outputs={"created": "filesystem_path"},
        allowed_modes={ExecutionMode.foreground, ExecutionMode.side_effect},
        failure_policy={
            ExecutionMode.foreground: FailurePolicy.FAIL,
            ExecutionMode.side_effect: FailurePolicy.IGNORE,
        },
        emits_events=["folder_created"],
        mutates_world=True,
        output_style="terse",
    )

    def __init__(self, location_config: LocationConfig):
        self._location_config = location_config

    def execute(self, inputs: Dict[str, Any], world: WorldTimeline, snapshot=None) -> SkillResult:
        import logging
        _logger = logging.getLogger(__name__)

        name = inputs["name"]
        anchor = inputs.get("anchor", "WORKSPACE")
        parent = inputs.get("parent", "")

        _logger.info(
            "[TRACE] CreateFolderSkill.execute: name=%r, anchor=%r, parent=%r",
            name, anchor, parent,
        )

        # Resolve anchor → absolute base path
        base = self._location_config.resolve(anchor)

        _logger.info(
            "[TRACE] CreateFolderSkill: base=%s",
            base,
        )

        # Build full path: base / parent / name
        if parent:
            path = base / parent / name
        else:
            path = base / name

        _logger.info(
            "[TRACE] CreateFolderSkill: final path=%s (will mkdir)",
            path,
        )

        path.mkdir(parents=True, exist_ok=True)

        world.emit("skill.fs", "folder_created", {
            "path": str(path),
            "anchor": anchor,
            "name": name,
        })

        return SkillResult(
            outputs={"created": str(path)},
            metadata={"entity": f"folder '{name}'", "domain": "fs"},
        )
