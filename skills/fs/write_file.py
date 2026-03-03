from pathlib import Path
from typing import Any, Dict

from skills.skill_result import SkillResult
from skills.base import Skill
from skills.contract import SkillContract, FailurePolicy
from ir.mission import ExecutionMode
from world.timeline import WorldTimeline
from infrastructure.location_config import LocationConfig


class WriteFileSkill(Skill):
    """
    Write text content to an anchor-resolved file path.

    Inputs:
        path:    File name or relative path (e.g., "poem.txt", "notes/ideas.md")
        content: Text content to write (string, may come from $ref)
        anchor:  Location anchor (e.g., "DESKTOP", "DRIVE_D", "WORKSPACE")

    Resolution (using LocationConfig):
        base = location_config.resolve(anchor)
        full = base / path

    Creates parent directories automatically.
    Overwrites existing files (idempotent).
    """

    contract = SkillContract(
        name="fs.write_file",
        action="write_file",
        target_type="file",
        description="Write text content to a file",
        narration_template="write to {path}",
        intent_verbs=["write", "save", "store", "create"],
        intent_keywords=["file", "text", "content", "document"],
        verb_specificity="generic",
        domain="fs",
        inputs={
            "path": "file_path_input",
            "content": "file_content",
        },
        optional_inputs={
            "anchor": "anchor_name",
        },
        outputs={"written": "filesystem_path"},
        allowed_modes={ExecutionMode.foreground, ExecutionMode.side_effect},
        failure_policy={
            ExecutionMode.foreground: FailurePolicy.FAIL,
            ExecutionMode.side_effect: FailurePolicy.IGNORE,
        },
        emits_events=["file_written"],
        mutates_world=True,
    )

    def __init__(self, location_config: LocationConfig):
        self._location_config = location_config

    def execute(self, inputs: Dict[str, Any], world: WorldTimeline, snapshot=None) -> SkillResult:
        import logging
        _logger = logging.getLogger(__name__)

        file_path = inputs["path"]
        content = inputs["content"]
        anchor = inputs.get("anchor", "WORKSPACE")

        _logger.info(
            "[TRACE] WriteFileSkill.execute: path=%r, anchor=%r, content_len=%d",
            file_path, anchor, len(content),
        )

        # Resolve anchor → absolute base path
        base = self._location_config.resolve(anchor)
        full_path = base / file_path

        _logger.info(
            "[TRACE] WriteFileSkill: full_path=%s",
            full_path,
        )

        # Create parent directories
        full_path.parent.mkdir(parents=True, exist_ok=True)

        # Write content
        full_path.write_text(content, encoding="utf-8")

        world.emit("skill.fs", "file_written", {
            "path": str(full_path),
            "anchor": anchor,
            "size_bytes": len(content.encode("utf-8")),
        })

        return SkillResult(
            outputs={"written": str(full_path)},
            metadata={"domain": "fs"},
        )
