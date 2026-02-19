# cortex/fallback.py

"""
FallbackCompiler — Deterministic template-based compilation. No LLM.

Scope:
- Single-node skill patterns ONLY.
- No DAG inference.
- No dependency construction.
- No multi-step reasoning.

Behavior:
- "Fast reflex but IR-compliant."
- Produces MissionPlan that passes IR validation + executor contract.
- Indistinguishable from LLM-generated plan downstream.

Used when:
- LLM is unavailable (ConnectionError).
- User query is simple enough for keyword matching.
"""

import hashlib
import json
from typing import Any, Dict, Optional, Union

from ir.mission import IR_VERSION, ExecutionMode, MissionNode, MissionPlan
from execution.registry import SkillRegistry
from infrastructure.location_config import LocationConfig
from errors import FailureIR


class FallbackCompiler:
    """Deterministic single-node compiler. No LLM.

    Matches user query keywords against skill contracts.
    Produces a valid MissionPlan for simple commands,
    or FailureIR for queries too complex for keyword matching.
    """

    def __init__(
        self,
        registry: SkillRegistry,
        location_config: Optional["LocationConfig"] = None,
    ):
        self.registry = registry
        self._location_config = location_config

    def compile(
        self,
        user_query: str,
        world_state_schema: Optional[Dict[str, Any]] = None,
    ) -> Union[MissionPlan, FailureIR]:
        """Attempt deterministic compilation from user query.

        Returns MissionPlan | FailureIR. Never raises. Never returns None.
        """
        query_lower = user_query.strip().lower()

        # Try each registered skill for a keyword match
        for skill_name in self.registry.all_names():
            skill = self.registry.get(skill_name)
            match_result = self._try_match(query_lower, skill)
            if match_result is not None:
                node, inputs = match_result
                return self._build_plan(
                    node_id="fallback_n1",
                    skill_name=skill.name,
                    inputs=inputs,
                    user_query=user_query,
                    world_state_schema=world_state_schema or {},
                )

        # No skill matched — too complex for fallback
        return FailureIR(
            error_type="llm_unavailable",
            error_message=(
                "LLM is unavailable and query is too complex "
                "for deterministic fallback compilation."
            ),
            user_query=user_query,
            internal_error=True,
        )

    def _try_match(self, query_lower: str, skill) -> Optional[tuple]:
        """Try to match a query to a skill using keyword heuristics.

        Returns (skill, inputs_dict) if matched, None otherwise.

        This is deliberately conservative — only matches patterns
        where we can confidently extract the required inputs.
        Ambiguity → None → FailureIR.
        """
        contract = skill.contract

        # ── fs.create_folder ──
        if contract.name == "fs.create_folder":
            return self._match_create_folder(query_lower)

        # Future skills will have matchers added here.
        # No generic matching — each skill must have explicit rules.
        return None

    def _match_create_folder(self, query: str) -> Optional[tuple]:
        """Match 'create a folder named X' patterns.

        Supported patterns:
        - "create a folder named hello"
        - "create folder hello on desktop"
        - "make a folder called test"
        - "new folder myproject"

        Returns (skill, inputs) or None.
        """
        keywords = ["create", "make", "new"]
        folder_indicators = ["folder", "directory", "dir"]

        has_keyword = any(k in query for k in keywords)
        has_folder = any(f in query for f in folder_indicators)

        if not (has_keyword and has_folder):
            return None

        # Extract folder name — word after "named", "called", or last word
        name = self._extract_after(query, ["named", "called"])
        if not name:
            # Try last meaningful word after folder indicator
            name = self._extract_last_noun(query, folder_indicators)

        if not name:
            return None  # Can't determine name → bail

        # Extract anchor if mentioned — derive from LocationConfig if available
        anchor = "DESKTOP"  # default
        if self._location_config is not None:
            anchor_keywords = {
                name.lower(): name
                for name in self._location_config.all_anchor_names()
            }
        else:
            anchor_keywords = {
                "desktop": "DESKTOP",
                "documents": "DOCUMENTS",
                "downloads": "DOWNLOADS",
                "workspace": "WORKSPACE",
            }
        for kw, anc in anchor_keywords.items():
            if kw in query:
                anchor = anc
                break

        inputs = {
            "name": name,
            "anchor": anchor,
        }

        return ("fs.create_folder", inputs)

    @staticmethod
    def _extract_after(query: str, markers: list) -> Optional[str]:
        """Extract the first word after any of the given markers."""
        words = query.split()
        for marker in markers:
            if marker in words:
                idx = words.index(marker)
                if idx + 1 < len(words):
                    # Take the next word, strip punctuation
                    candidate = words[idx + 1].strip(".,!?\"'")
                    if candidate and candidate not in ("a", "an", "the", "on", "in", "at"):
                        return candidate
        return None

    @staticmethod
    def _extract_last_noun(query: str, skip_words: list) -> Optional[str]:
        """Extract the last meaningful word after any skip_word."""
        words = query.split()
        # Find last occurrence of any skip word
        last_skip_idx = -1
        for sw in skip_words:
            if sw in words:
                idx = words.index(sw)
                last_skip_idx = max(last_skip_idx, idx)

        if last_skip_idx == -1:
            return None

        # Get remaining words after the skip word
        remaining = words[last_skip_idx + 1:]
        # Filter out prepositions and articles
        noise = {"a", "an", "the", "on", "in", "at", "to", "for", "with"}
        meaningful = [w.strip(".,!?\"'") for w in remaining if w.lower() not in noise]

        if meaningful:
            return meaningful[0]
        return None

    @staticmethod
    def _build_plan(
        node_id: str,
        skill_name: str,
        inputs: Dict[str, Any],
        user_query: str,
        world_state_schema: Dict[str, Any],
    ) -> MissionPlan:
        """Build an IR-valid single-node MissionPlan.

        Passes validate_mission_plan() and executor contract enforcement.
        Indistinguishable from LLM-generated plan downstream.
        """
        # Deterministic ID — same as MissionCortex
        normalized = json.dumps(
            {"query": user_query.strip().lower(),
             "world": world_state_schema},
            sort_keys=True,
            separators=(",", ":"),
        )
        mission_id = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]

        return MissionPlan(
            id=mission_id,
            nodes=[
                MissionNode(
                    id=node_id,
                    skill=skill_name,
                    inputs=inputs,
                    mode=ExecutionMode.foreground,
                ),
            ],
            metadata={
                "ir_version": IR_VERSION,
                "source": "fallback_compiler",
            },
        )
