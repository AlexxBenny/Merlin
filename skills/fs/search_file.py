# skills/fs/search_file.py

"""
SearchFileSkill — Search for files by name/pattern across configured anchors.

Uses FileIndex for O(1) lookup instead of brute-force directory walk.
Outputs List[FileRef] — structured references, not raw paths.

Effect model:
  requires=[]           — no preconditions (can always search)
  produces=["file_reference"]  — reveals file identity
  effect_type="reveal"  — discovery, not creation
"""

from typing import Any, Dict, Optional

from skills.skill_result import SkillResult
from skills.base import Skill
from skills.contract import SkillContract, FailurePolicy
from ir.mission import ExecutionMode
from world.timeline import WorldTimeline
from world.snapshot import WorldSnapshot
from world.file_ref import FileRef
from world.file_index import FileIndex
from infrastructure.location_config import LocationConfig

import logging

logger = logging.getLogger(__name__)


class SearchFileSkill(Skill):
    """Search for files by name/pattern across configured anchors.

    Uses FileIndex for fast lookup. Outputs List[FileRef].
    Path resolution deferred to downstream skills at execution time.
    """

    contract = SkillContract(
        name="fs.search_file",
        action="search_file",
        target_type="file",
        description="Search for files by name",
        narration_template="searching for {query}",
        intent_verbs=["find", "search", "locate", "where"],
        intent_keywords=["file", "files", "document", "pdf", "attachment"],
        verb_specificity="generic",
        domain="fs",
        resource_cost="medium",
        inputs={"query": "file_search_query"},
        optional_inputs={"anchor": "anchor_name"},
        outputs={"matches": "file_ref_list"},
        allowed_modes={ExecutionMode.foreground},
        failure_policy={ExecutionMode.foreground: FailurePolicy.FAIL},
        emits_events=["file_search_completed"],
        mutates_world=False,
        output_style="rich",
        requires=[],
        produces=["file_reference"],
        effect_type="reveal",
    )

    def __init__(
        self,
        location_config: LocationConfig,
        file_index: Optional[FileIndex] = None,
    ):
        self._location_config = location_config
        self._file_index = file_index or FileIndex()

    def execute(
        self,
        inputs: Dict[str, Any],
        world: WorldTimeline,
        snapshot: Optional[WorldSnapshot] = None,
        context=None,
    ) -> SkillResult:
        query = inputs["query"]
        anchor = inputs.get("anchor")

        logger.info(
            "[TRACE] SearchFileSkill.execute: query=%r, anchor=%r",
            query, anchor,
        )

        # Search via FileIndex (lazy-builds on first call)
        matches = self._file_index.search(
            query,
            location_config=self._location_config,
        )

        # Filter by anchor if specified
        if anchor:
            matches = [m for m in matches if m.anchor == anchor]

        # Emit event
        world.emit("skill.fs", "file_search_completed", {
            "query": query,
            "match_count": len(matches),
            "anchor_filter": anchor,
        })

        # Return FileRef dicts (consumed via OutputReference downstream)
        match_dicts = [m.to_output_dict() for m in matches]

        if not matches:
            return SkillResult(
                outputs={"matches": []},
                metadata={
                    "domain": "fs",
                    "entity": f"search for '{query}'",
                    "response_template": f"No files found matching '{query}'.",
                },
            )

        # ── Ambiguity detection ──
        # When all matches share the same filename, the user intended
        # ONE file but multiple copies exist in different directories.
        # Signal ambiguity so the recovery loop asks the user.
        names = [m.name for m in matches]
        if len(matches) > 1 and len(set(names)) == 1:
            options = [
                f"  {i+1}. {m.relative_path} ({m.anchor})"
                for i, m in enumerate(matches[:5])
            ]
            question = (
                f"I found {len(matches)} files named '{matches[0].name}':\n"
                + "\n".join(options)
                + "\nWhich one did you mean?"
            )
            return SkillResult(
                outputs={"matches": match_dicts},
                status="no_op",
                metadata={
                    "domain": "fs",
                    "entity": f"search for '{query}'",
                    "reason": "ambiguous_input",
                    "message": question,
                    "options": options,
                },
            )

        return SkillResult(
            outputs={"matches": match_dicts},
            metadata={
                "domain": "fs",
                "entity": f"search for '{query}'",
            },
        )
