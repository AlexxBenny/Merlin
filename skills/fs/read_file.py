from pathlib import Path
from typing import Any, Dict

from skills.skill_result import SkillResult
from skills.base import Skill
from skills.contract import SkillContract, FailurePolicy
from ir.mission import ExecutionMode
from world.timeline import WorldTimeline
from infrastructure.location_config import LocationConfig


class ReadFileSkill(Skill):
    """
    Read text content from an anchor-resolved file path.

    Inputs:
        path:   File name or relative path (e.g., "poem.txt", "notes/ideas.md")
        anchor: Location anchor (e.g., "DESKTOP", "DRIVE_D", "WORKSPACE")

    Resolution (using LocationConfig):
        base = location_config.resolve(anchor)
        full = base / path

    Returns file content as string output, referenceable by downstream nodes.
    """

    contract = SkillContract(
        name="fs.read_file",
        action="read_file",
        target_type="file",
        description="Read text content from a file",
        narration_template="read {path}",
        intent_verbs=["read", "open", "load", "get", "show"],
        intent_keywords=["file", "text", "content", "document"],
        verb_specificity="generic",
        domain="fs",
        inputs={
            "path": "file_path_input",
        },
        optional_inputs={
            "anchor": "anchor_name",
        },
        outputs={"content": "file_content"},
        allowed_modes={ExecutionMode.foreground},
        failure_policy={
            ExecutionMode.foreground: FailurePolicy.FAIL,
        },
        emits_events=["file_read"],
        mutates_world=False,
        output_style="terse",
    )

    def __init__(self, location_config: LocationConfig):
        self._location_config = location_config

    def execute(self, inputs: Dict[str, Any], world: WorldTimeline, snapshot=None) -> SkillResult:
        import logging
        _logger = logging.getLogger(__name__)

        file_path = inputs["path"]
        anchor = inputs.get("anchor", "WORKSPACE")

        _logger.info(
            "[TRACE] ReadFileSkill.execute: path=%r, anchor=%r",
            file_path, anchor,
        )

        # Resolve anchor → absolute base path
        base = self._location_config.resolve(anchor)
        full_path = base / file_path

        _logger.info(
            "[TRACE] ReadFileSkill: full_path=%s",
            full_path,
        )

        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {full_path}")

        if not full_path.is_file():
            raise ValueError(f"Not a file: {full_path}")

        content = full_path.read_text(encoding="utf-8")

        world.emit("skill.fs", "file_read", {
            "path": str(full_path),
            "anchor": anchor,
            "size_bytes": len(content.encode("utf-8")),
        })

        return SkillResult(
            outputs={"content": content},
            metadata={"domain": "fs"},
        )
