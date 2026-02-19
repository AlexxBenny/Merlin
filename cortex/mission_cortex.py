# cortex/mission_cortex.py

import hashlib
import json
import logging
from typing import Any, Dict, List, Optional, Union

from cortex.json_extraction import extract_json_block
from cortex.normalizer import normalize_plan, validate_anchors
from cortex.semantic_types import SEMANTIC_TYPES
from errors import (
    CompilationError,
    FailureIR,
    LLMUnavailableError,
    MalformedPlanError,
    ParseError,
)
from cortex.fallback import FallbackCompiler
from cortex.context_provider import ContextProvider, SimpleContextProvider
from cortex.skill_discovery import SkillDiscovery, AllSkillsDiscovery

from ir.mission import (
    IR_VERSION,
    ConditionExpr,
    ExecutionMode,
    MissionNode,
    MissionPlan,
    OutputReference,
    OutputSpec,
)
from cortex.validators import validate_mission_plan, MissionValidationError
from execution.registry import SkillRegistry
from infrastructure.location_config import LocationConfig
from conversation.frame import ConversationFrame
from conversation.outcome import MissionOutcome

logger = logging.getLogger(__name__)


class MissionCompilationError(Exception):
    """Raised when the LLM fails to produce a valid MissionPlan."""


class MissionCortex:
    """
    MissionCortex is a COMPILER.

    It transforms a user query into a fully explicit MissionPlan (DAG)
    in exactly one LLM call.

    No retries.
    No replanning.
    No execution.
    """

    def __init__(
        self,
        llm_client,
        registry: SkillRegistry,
        location_config: Optional[LocationConfig] = None,
        context_provider: Optional[ContextProvider] = None,
        skill_discovery: Optional[SkillDiscovery] = None,
    ):
        self.llm = llm_client
        self.registry = registry
        self._location_config = location_config
        self.context_provider = context_provider or SimpleContextProvider()
        self.skill_discovery = skill_discovery or AllSkillsDiscovery()

    # ── Maximum intent units injected into compile prompt ──
    MAX_INTENT_INJECTION: int = 8

    def compile(
        self,
        user_query: str,
        world_state_schema: Dict[str, Any],
        conversation: Optional[ConversationFrame] = None,
        intent_checklist: Optional[List[Dict[str, str]]] = None,
    ) -> Union[MissionPlan, FailureIR]:
        """
        Compile user query into a MissionPlan.

        Returns MissionPlan | FailureIR. Never raises. Never returns None.

        Args:
            intent_checklist: Optional list of intent units (Tier 2+).
                Injected into the prompt so the LLM covers all intents.
                Capped at MAX_INTENT_INJECTION entries.

        Retry discipline:
        - Only ParseError triggers retry (JSON extraction/decode failed).
        - Exactly one retry. No loops.
        - Second attempt uses lower temperature + stricter JSON instruction.
        - Explicit flag prevents recursive retry.
        - MalformedPlanError, LLMUnavailableError: NO retry.
        """
        result = self._compile_once(
            user_query=user_query,
            world_state_schema=world_state_schema,
            conversation=conversation,
            temperature=None,  # default
            intent_checklist=intent_checklist,
        )

        # Only retry on parse_error
        if isinstance(result, FailureIR) and result.error_type == "parse_error":
            # One retry — lower temperature, tighter instruction
            result = self._compile_once(
                user_query=user_query,
                world_state_schema=world_state_schema,
                conversation=conversation,
                temperature=0.1,
                strict_json=True,
                _retry_attempted=True,
                intent_checklist=intent_checklist,
            )

        # Fallback path: LLM unavailable → try deterministic compiler
        if isinstance(result, FailureIR) and result.error_type == "llm_unavailable":
            fallback = FallbackCompiler(self.registry, self._location_config)
            fallback_result = fallback.compile(
                user_query=user_query,
                world_state_schema=world_state_schema,
            )
            if isinstance(fallback_result, MissionPlan):
                # Validate fallback plan — same gate as LLM path
                try:
                    skill_manifest = self._build_skill_manifest()
                    available_skills = set(skill_manifest.keys())
                    validate_mission_plan(
                        fallback_result, available_skills,
                        registry=self.registry,
                    )
                    return fallback_result
                except MissionValidationError:
                    pass  # Fallback plan invalid — fall through to original FailureIR

        return result

    def decompose_intents(self, user_query: str) -> Optional[List[Dict[str, str]]]:
        """Pre-compile intent decomposition. Tier 2+ only.

        Extracts atomic intent units from a multi-intent query.
        Returns None on failure (graceful degradation to Tier 1).

        Each intent unit:
        {
            "verb": "create",
            "object": "folder docs",
            "modifiers": "on desktop",
            "domain_hint": "fs"
        }

        Token cap: returns at most MAX_INTENT_INJECTION entries.
        Verification must operate on the SAME set returned here.
        """
        if self.llm is None:
            return None

        # Collect available domains from registry
        domains = set()
        for name in self.registry.all_names():
            skill = self.registry.get(name)
            d = skill.contract.domain or name.split(".")[0]
            domains.add(d)
        domain_list = sorted(domains)

        prompt = f"""Extract every distinct action the user wants performed.
For each action, identify the most likely skill domain from: {json.dumps(domain_list)}

Output a JSON array. Each element:
{{"verb": "action_word", "object": "what_to_act_on", "modifiers": "additional_context", "domain_hint": "domain"}}

Rules:
- One entry per distinct action. Do NOT merge actions.
- "verb" = the action word (create, open, play, set, etc.)
- "object" = what the action targets (folder docs, spotify, music, brightness)
- "modifiers" = qualifiers (on desktop, to 80, in background) — empty string if none
- "domain_hint" = best matching domain from the list above
- Output ONLY the JSON array. No explanation. No markdown.

User query:
\"\"\"{user_query}\"\"\"
""".strip()

        try:
            raw_response = self.llm.complete(prompt, temperature=0.1)
            logger.info(
                "[DECOMPOSE] Raw decomposition for '%s': %s",
                user_query[:80], raw_response[:1000],
            )

            clean_json = extract_json_block(raw_response)
            intents = json.loads(clean_json)

            if not isinstance(intents, list):
                logger.warning(
                    "[DECOMPOSE] Expected list, got %s — degrading to Tier 1",
                    type(intents).__name__,
                )
                return None

            # Validate structure: each entry must have verb + object
            valid_intents = []
            for intent in intents:
                if isinstance(intent, dict) and "verb" in intent and "object" in intent:
                    valid_intents.append({
                        "verb": str(intent.get("verb", "")),
                        "object": str(intent.get("object", "")),
                        "modifiers": str(intent.get("modifiers", "")),
                        "domain_hint": str(intent.get("domain_hint", "")),
                    })

            if not valid_intents:
                logger.warning(
                    "[DECOMPOSE] No valid intents extracted — degrading to Tier 1"
                )
                return None

            # Token cap: truncate to MAX_INTENT_INJECTION
            if len(valid_intents) > self.MAX_INTENT_INJECTION:
                logger.info(
                    "[DECOMPOSE] Truncating %d intents to %d (token cap)",
                    len(valid_intents), self.MAX_INTENT_INJECTION,
                )
                valid_intents = valid_intents[:self.MAX_INTENT_INJECTION]

            logger.info(
                "[DECOMPOSE] Extracted %d intent units for '%s'",
                len(valid_intents), user_query[:80],
            )
            return valid_intents

        except (ConnectionError, RuntimeError) as e:
            logger.warning(
                "[DECOMPOSE] LLM call failed: %s — degrading to Tier 1", e
            )
            return None
        except (ParseError, json.JSONDecodeError) as e:
            logger.warning(
                "[DECOMPOSE] Parse failed: %s — degrading to Tier 1", e
            )
            return None
        except Exception as e:
            logger.warning(
                "[DECOMPOSE] Unexpected error: %s — degrading to Tier 1", e
            )
            return None

    def _compile_once(
        self,
        user_query: str,
        world_state_schema: Dict[str, Any],
        conversation: Optional[ConversationFrame] = None,
        temperature: Optional[float] = None,
        strict_json: bool = False,
        _retry_attempted: bool = False,
        intent_checklist: Optional[List[Dict[str, str]]] = None,
    ) -> Union[MissionPlan, FailureIR]:
        """
        Single compilation attempt. Never raises. Never returns None.

        Args:
            temperature: Per-call temperature override.
            strict_json: If True, prepend a strict JSON-only instruction.
            _retry_attempted: Explicit flag — if True, this IS the retry.
                              ParseError on retry → final FailureIR (no further retry).
            intent_checklist: Optional intent units for Tier 2+ injection.
        """

        # Guard: LLM not available → immediate fallback path
        if self.llm is None:
            return FailureIR(
                error_type="llm_unavailable",
                error_message="No LLM client configured",
                user_query=user_query,
                internal_error=True,
            )

        skill_manifest = self.skill_discovery.find_candidates(
            query=user_query, registry=self.registry,
        )

        context_section = self.context_provider.build_context(
            query=user_query,
            conversation=conversation,
            world_state=world_state_schema,
        )

        # Build intent checklist section for Tier 2+ prompts
        intent_section = self._build_intent_checklist_section(intent_checklist)

        prompt = self._build_prompt(
            user_query=user_query,
            skill_manifest=skill_manifest,
            world_state_schema=world_state_schema,
            context_section=context_section,
            intent_checklist_section=intent_section,
        )

        if strict_json:
            prompt = (
                "CRITICAL: You must respond with ONLY a valid JSON object. "
                "No markdown, no commentary, no explanation. "
                "Start with { and end with }.\n\n"
                + prompt
            )

        # ── LLM call ──
        try:
            raw_response = self.llm.complete(prompt, temperature=temperature)
            logger.info(
                "[TRACE] LLM raw response (first 2000 chars) for '%s':\n%s",
                user_query[:80],
                raw_response[:2000],
            )
        except (ConnectionError, RuntimeError) as e:
            return FailureIR(
                error_type="llm_unavailable",
                error_message=str(e),
                user_query=user_query,
                internal_error=True,
            )

        # ── JSON extraction + parse ──
        try:
            clean_json = extract_json_block(raw_response)
            payload = json.loads(clean_json)
            logger.info(
                "[TRACE] Parsed JSON payload for '%s':\n%s",
                user_query[:80],
                json.dumps(payload, indent=2)[:3000],
            )
        except ParseError as e:
            return FailureIR(
                error_type="parse_error",
                error_message=str(e),
                user_query=user_query,
                retryable=not _retry_attempted,
                internal_error=True,
            )
        except json.JSONDecodeError as e:
            return FailureIR(
                error_type="parse_error",
                error_message=f"Invalid JSON after extraction: {e}",
                user_query=user_query,
                retryable=not _retry_attempted,
                internal_error=True,
            )

        # ── Schema-level normalization (LLM → IR boundary) ──
        try:
            payload = normalize_plan(payload)
            logger.info(
                "[TRACE] After normalization: %s",
                json.dumps(payload, indent=2)[:3000],
            )
        except TypeError as e:
            return FailureIR(
                error_type="malformed_plan",
                error_message=f"Plan normalization failed: {e}",
                user_query=user_query,
            )

        # ── Anchor validation (structural guard) ──
        try:
            valid_anchors = set()
            if self._location_config is not None:
                valid_anchors = set(
                    self._location_config.all_anchor_names()
                )
            logger.info(
                "[TRACE] Valid anchors: %s",
                sorted(valid_anchors),
            )
            validate_anchors(payload, valid_anchors)
            logger.info("[TRACE] Anchor validation passed")
        except TypeError as e:
            return FailureIR(
                error_type="malformed_plan",
                error_message=f"Anchor validation failed: {e}",
                user_query=user_query,
            )

        # ── Plan construction ──
        try:
            mission = self._parse_mission_plan(
                payload,
                user_query=user_query,
                world_state_schema=world_state_schema,
            )
        except (MissionCompilationError, KeyError, TypeError, ValueError) as e:
            return FailureIR(
                error_type="malformed_plan",
                error_message=f"Plan construction failed: {e}",
                user_query=user_query,
            )

        # ── Validation gate ──
        try:
            available_skills = set(skill_manifest.keys())
            validate_mission_plan(
                mission, available_skills, registry=self.registry,
            )
        except MissionValidationError as e:
            return FailureIR(
                error_type="malformed_plan",
                error_message=f"Plan validation failed: {e}",
                user_query=user_query,
            )

        return mission

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_skill_manifest(self) -> Dict[str, Any]:
        """
        Expose declarative skill metadata with semantic types to the LLM.

        Separates required and optional inputs explicitly so the LLM
        understands which parameters are mandatory vs have defaults.

        Semantic types come from SkillContract.inputs/outputs, which
        are documented centrally in cortex/semantic_types.py.
        """
        manifest = {}

        for name in self.registry.all_names():
            skill = self.registry.get(name)
            entry = {
                "description": skill.contract.description,
                "inputs": {
                    k: v for k, v in skill.contract.inputs.items()
                },
                "output_keys": sorted(skill.contract.outputs.keys()),
                "allowed_modes": sorted(
                    m.value for m in skill.contract.allowed_modes
                ),
            }

            # Only include optional_inputs if the skill has any
            if skill.contract.optional_inputs:
                entry["optional_inputs"] = {
                    k: v for k, v in skill.contract.optional_inputs.items()
                }

            manifest[skill.name] = entry

        return manifest

    def _build_semantic_type_docs(
        self, skill_manifest: Dict[str, Any],
    ) -> str:
        """
        Auto-generate semantic type documentation from the registry.

        Only documents types actually used by available skills.
        This ensures zero prompt bloat and zero drift from implementation.
        """
        used_types: set = set()
        for info in skill_manifest.values():
            used_types.update(info["inputs"].values())
            # Include optional input types too
            if "optional_inputs" in info:
                used_types.update(info["optional_inputs"].values())

        # Filter to input-visible types only (output types are not LLM-facing)
        lines = []
        for t in sorted(used_types):
            entry = SEMANTIC_TYPES.get(t)
            if entry and entry.direction in ("input", "both"):
                lines.append(f"- {t}: {entry.description}")

        if not lines:
            return ""
        return "Semantic Input Types:\n" + "\n".join(lines)

    def _build_intent_checklist_section(
        self,
        intent_checklist: Optional[List[Dict[str, str]]],
    ) -> str:
        """Format intent units into a checklist for prompt injection.

        Returns empty string if no checklist provided.
        """
        if not intent_checklist:
            return ""

        lines = ["INTENT CHECKLIST — your plan MUST cover ALL of these:"]
        for i, intent in enumerate(intent_checklist, 1):
            domain = intent.get("domain_hint", "")
            verb = intent.get("verb", "")
            obj = intent.get("object", "")
            mods = intent.get("modifiers", "")
            entry = f"{i}. [{domain}] {verb} {obj}"
            if mods:
                entry += f" {mods}"
            lines.append(entry)

        lines.append("")
        lines.append(
            "Each intent above MUST map to at least one node in your plan. "
            "Missing intents will cause plan rejection."
        )
        return "\n".join(lines)

    def _build_prompt(
        self,
        user_query: str,
        skill_manifest: Dict[str, Any],
        world_state_schema: Dict[str, Any],
        context_section: str = "",
        intent_checklist_section: str = "",
    ) -> str:
        """
        Construct a STRICT prompt.

        The LLM is instructed to emit JSON only, using IR v1 wire format.
        $ref is the EXTERNAL wire format — the parser strips it.

        Prompt structure:
        1. Role + rules
        2. World state
        3. Available skills (with semantic types)
        4. Semantic type documentation (auto-generated)
        5. Anchor vocabulary
        6. JSON schema
        7. Few-shot examples
        8. Self-check instruction
        9. User request
        """

        # Build anchor vocabulary section if LocationConfig is available
        anchor_section = ""
        if self._location_config is not None:
            anchor_names = self._location_config.all_anchor_names()
            anchor_section = f"""
Available Location Anchors:
{json.dumps(anchor_names)}

When a skill input has semantic type "anchor_name", you MUST choose from this list.
Default anchor: "WORKSPACE" (the user's current working directory).
Do NOT emit raw filesystem paths. Use anchor names.
"""

        # Auto-generate semantic type docs from registry
        type_docs = self._build_semantic_type_docs(skill_manifest)

        return f"""
You are a deterministic mission compiler.

Your task:
Convert the user request into a STATIC Mission DAG.

Rules (MANDATORY):
- Output MUST be valid JSON.
- Output MUST conform to the schema exactly.
- Do NOT invent skills.
- Do NOT omit dependencies.
- Do NOT infer implicit context.
- Skill names MUST use the format: domain.action or domain.action.variant
- If the request cannot be compiled, output an error object.

REJECTION RULES — If you emit ANY of the following, the ENTIRE response will be rejected:
- An "id" field. IDs are assigned by the compiler, not by you.
- A "condition_on" field. Conditional execution is compiler-managed.
- Fields not present in the schema below.
- Skills not listed in Available Skills.
- Empty inputs when the skill requires parameters.
- depends_on entries that reference non-existent step indices.

Execution Modes:
- foreground: blocks mission completion, failure fails mission
- background: non-blocking, failure logged only
- side_effect: non-blocking, failure ignored

World State (read-only, symbolic):
{json.dumps(world_state_schema, indent=2)}

Available Skills:
{json.dumps(skill_manifest, indent=2)}

{type_docs}
{anchor_section}
{context_section}
{intent_checklist_section}
MissionPlan JSON Schema:
{{
  "nodes": [
    {{
      "skill": "domain.action[.variant]",
      "inputs": {{
        "key": "literal_value",
        "key2": {{"$ref": {{"node": 0, "output": "namespaced.output.name"}}}}
      }},
      "outputs": {{
        "key": {{"name": "namespaced.output.name", "type": "semantic_type"}}
      }},
      "depends_on": [0],
      "mode": "foreground | background | side_effect"
    }}
  ]
}}

IMPORTANT:
- Nodes are ordered. The first node is index 0, the second is index 1, etc.
- depends_on uses integer step indices referencing earlier nodes in the array.
- To reference another node's output, use: {{"$ref": {{"node": 0, "output": "output_name"}}}}
- Do NOT use $node.output string format. Use the $ref object.
- Output names must be namespaced (e.g. "research.findings.v1")

INVALID — This will be rejected:
{{"id": "1", "skill": "system.media_play", "inputs": {{}}, "depends_on": [], "mode": "foreground", "condition_on": {{"source": "1", "equals": true}}}}
Reasons: contains "id" field, contains "condition_on" field, empty inputs for a skill that may require them

Examples:

User: "create a folder called docs on desktop"
{{
  "nodes": [
    {{"skill": "fs.create_folder", "inputs": {{"name": "docs", "anchor": "DESKTOP"}}, "outputs": {{}}, "depends_on": [], "mode": "foreground"}}
  ]
}}

User: "create folders X and Y"
{{
  "nodes": [
    {{"skill": "fs.create_folder", "inputs": {{"name": "X", "anchor": "WORKSPACE"}}, "outputs": {{}}, "depends_on": [], "mode": "foreground"}},
    {{"skill": "fs.create_folder", "inputs": {{"name": "Y", "anchor": "WORKSPACE"}}, "outputs": {{}}, "depends_on": [], "mode": "foreground"}}
  ]
}}

User: "create folder A, inside A create folder B, inside B create folder C"
{{
  "nodes": [
    {{"skill": "fs.create_folder", "inputs": {{"name": "A", "anchor": "WORKSPACE"}}, "outputs": {{}}, "depends_on": [], "mode": "foreground"}},
    {{"skill": "fs.create_folder", "inputs": {{"name": "B", "anchor": "WORKSPACE", "parent": "A"}}, "outputs": {{}}, "depends_on": [0], "mode": "foreground"}},
    {{"skill": "fs.create_folder", "inputs": {{"name": "C", "anchor": "WORKSPACE", "parent": "A/B"}}, "outputs": {{}}, "depends_on": [1], "mode": "foreground"}}
  ]
}}

User: "open notepad and create folder logs on desktop"
{{
  "nodes": [
    {{"skill": "system.open_app", "inputs": {{"app_name": "notepad"}}, "outputs": {{}}, "depends_on": [], "mode": "foreground"}},
    {{"skill": "fs.create_folder", "inputs": {{"name": "logs", "anchor": "DESKTOP"}}, "outputs": {{}}, "depends_on": [], "mode": "foreground"}}
  ]
}}

User: "unmute, set brightness to 10, and play music"
{{
  "nodes": [
    {{"skill": "system.unmute", "inputs": {{}}, "outputs": {{}}, "depends_on": [], "mode": "foreground"}},
    {{"skill": "system.set_brightness", "inputs": {{"level": 10}}, "outputs": {{}}, "depends_on": [], "mode": "foreground"}},
    {{"skill": "system.media_play", "inputs": {{}}, "outputs": {{}}, "depends_on": [], "mode": "foreground"}}
  ]
}}

Before producing JSON, verify:
1. All anchor values are symbolic names from the Available Location Anchors list.
2. No raw filesystem paths appear anywhere in inputs.
3. Every depends_on entry is a valid integer index of an earlier node.
4. Nested operations use the "parent" field, not a raw path in "anchor".
5. No "id" field appears anywhere. No "condition_on" field appears anywhere.

User Request:
\"\"\"{user_query}\"\"\"
""".strip()

    def _generate_mission_id(
        self,
        user_query: str,
        world_state_schema: Dict[str, Any],
    ) -> str:
        """
        Deterministic mission ID. No clocks. No randomness.
        Derived from normalized inputs for reproducibility, replay, and audit.
        """
        normalized = json.dumps(
            {"query": user_query.strip().lower(),
             "world": world_state_schema},
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]

    def _parse_mission_plan(
        self,
        payload: Dict[str, Any],
        user_query: str,
        world_state_schema: Dict[str, Any],
    ) -> MissionPlan:
        """
        Parse LLM JSON output into IR v1 MissionPlan.

        CRITICAL: This is the $ref stripping boundary.
        $ref must NOT survive past this function.

        ID ASSIGNMENT: The LLM does NOT generate IDs.
        IDs are assigned deterministically here: node_0, node_1, node_2, ...
        depends_on from LLM uses integer indices, converted to ID strings.

        CONDITION SYNTHESIS: condition_on is NOT emitted by the LLM.
        It remains None unless compiler-side detection adds it.
        """
        if "nodes" not in payload:
            raise MissionCompilationError("Missing 'nodes' in MissionPlan")

        raw_nodes = payload["nodes"]
        node_count = len(raw_nodes)

        # Build index → ID mapping for deterministic assignment
        id_map = {i: f"node_{i}" for i in range(node_count)}

        nodes: List[MissionNode] = []

        for idx, raw in enumerate(raw_nodes):
            try:
                # Deterministic ID assignment — compiler owns this
                node_id = id_map[idx]

                # Warn if LLM still emitted an id field (prompt violation)
                if "id" in raw:
                    logger.warning(
                        "[PROMPT_VIOLATION] LLM emitted 'id' field '%s' "
                        "for node %d — ignoring, using '%s'",
                        raw["id"], idx, node_id,
                    )

                # Warn if LLM still emitted condition_on (prompt violation)
                if raw.get("condition_on") is not None:
                    logger.warning(
                        "[PROMPT_VIOLATION] LLM emitted 'condition_on' "
                        "for node %d — ignoring, condition_on is "
                        "compiler-managed",
                        idx,
                    )

                logger.info(
                    "[TRACE] Parsing node '%s' (index %d): skill=%s, "
                    "inputs=%r, depends_on=%r, mode=%r",
                    node_id, idx,
                    raw.get("skill"),
                    raw.get("inputs"),
                    raw.get("depends_on"),
                    raw.get("mode"),
                )

                # Parse inputs — strip $ref at this boundary
                # Also convert index-based $ref.node to ID-based
                inputs = self._parse_inputs(raw.get("inputs", {}), id_map)

                # Parse outputs — convert to OutputSpec
                outputs = self._parse_outputs(raw.get("outputs", {}))

                # Convert depends_on from indices to IDs
                raw_deps = raw.get("depends_on", [])
                depends_on = self._resolve_depends_on(
                    raw_deps, id_map, node_id, node_count
                )

                # condition_on is compiler-managed, not LLM-generated
                # Stays None unless compiler-side detection adds it
                condition_on = None

                node = MissionNode(
                    id=node_id,
                    skill=raw["skill"],
                    inputs=inputs,
                    outputs=outputs,
                    depends_on=depends_on,
                    mode=ExecutionMode(raw.get("mode", "foreground")),
                    condition_on=condition_on,
                )

            except MissionCompilationError:
                raise
            except Exception as e:
                raise MissionCompilationError(
                    f"Invalid MissionNode structure: {e}"
                )

            nodes.append(node)
            logger.info(
                "[TRACE] Constructed MissionNode '%s': skill=%s, "
                "resolved_inputs=%r, depends_on=%r",
                node.id, node.skill, node.inputs, node.depends_on,
            )

        # ── Canonicalization invariant ──
        # IDs MUST be node_0 .. node_{N-1}, no gaps, no reuse.
        # This is the deterministic contract downstream depends on.
        expected_ids = [f"node_{i}" for i in range(len(nodes))]
        actual_ids = [n.id for n in nodes]
        if actual_ids != expected_ids:
            raise MissionCompilationError(
                f"ID canonicalization violated: expected {expected_ids}, "
                f"got {actual_ids}"
            )

        mission_id = self._generate_mission_id(
            user_query, world_state_schema
        )

        return MissionPlan(
            id=mission_id,
            nodes=nodes,
            metadata={"ir_version": IR_VERSION},
        )

    def _parse_inputs(
        self,
        raw_inputs: Dict[str, Any],
        id_map: Dict[int, str] | None = None,
    ) -> Dict[str, Any]:
        """
        Parse input values from LLM output.
        Strips $ref wrapper and converts to OutputReference.
        $ref must NOT survive past this function.

        If id_map is provided, converts index-based $ref.node
        (integers) to ID-based (node_0, node_1, ...).
        """
        parsed = {}
        for key, value in raw_inputs.items():
            if isinstance(value, dict) and "$ref" in value:
                ref = value["$ref"]
                raw_node_ref = ref["node"]

                # Convert index → ID if applicable
                if id_map is not None and isinstance(raw_node_ref, int):
                    if raw_node_ref in id_map:
                        resolved_ref = id_map[raw_node_ref]
                    else:
                        raise MissionCompilationError(
                            f"$ref.node index {raw_node_ref} out of "
                            f"range (max {max(id_map.keys())})"
                        )
                else:
                    resolved_ref = str(raw_node_ref)

                parsed[key] = OutputReference(
                    node=resolved_ref,
                    output=ref["output"],
                )
            else:
                parsed[key] = value
        return parsed

    def _resolve_depends_on(
        self,
        raw_deps: list,
        id_map: Dict[int, str],
        current_node_id: str,
        node_count: int,
    ) -> List[str]:
        """
        Convert depends_on from index-based (LLM output) to ID-based.

        Handles:
        - Integer indices (new schema): 0 → "node_0"
        - String indices (backward compat): "0" → "node_0"
        - Already-resolved IDs (backward compat): "node_0" → "node_0"

        Validation happens here to ensure all references are valid.
        """
        resolved = []
        for dep in raw_deps:
            if isinstance(dep, int):
                # New schema: integer index
                if dep < 0 or dep >= node_count:
                    raise MissionCompilationError(
                        f"Node '{current_node_id}' depends_on index "
                        f"{dep} out of range (0..{node_count - 1})"
                    )
                resolved.append(id_map[dep])
            elif isinstance(dep, str):
                # Backward compat: string index ("0") or ID ("node_0")
                if dep.isdigit():
                    idx = int(dep)
                    if idx < 0 or idx >= node_count:
                        raise MissionCompilationError(
                            f"Node '{current_node_id}' depends_on "
                            f"index '{dep}' out of range"
                        )
                    resolved.append(id_map[idx])
                elif dep.startswith("node_"):
                    # Already an ID
                    resolved.append(dep)
                else:
                    # LLM emitted some other string — treat as legacy
                    logger.warning(
                        "[DEPENDS_ON] Node '%s' has non-index "
                        "dependency '%s' — passing through",
                        current_node_id, dep,
                    )
                    resolved.append(dep)
            else:
                raise MissionCompilationError(
                    f"Node '{current_node_id}' depends_on contains "
                    f"invalid type: {type(dep).__name__}"
                )
        return resolved

    def _parse_outputs(
        self, raw_outputs: Dict[str, Any]
    ) -> Dict[str, OutputSpec]:
        """
        Parse output declarations from LLM output.
        Converts raw dicts to OutputSpec objects.
        """
        parsed = {}
        for key, value in raw_outputs.items():
            if isinstance(value, dict):
                parsed[key] = OutputSpec(
                    name=value.get("name", key),
                    type=value.get("type", "unknown"),
                )
            else:
                # LLM may emit simple string — treat as name
                parsed[key] = OutputSpec(name=str(value), type="unknown")
        return parsed

    def _parse_condition(
        self, raw_condition: Any
    ) -> ConditionExpr | None:
        """
        Parse condition_on from LLM output.
        Converts raw dict to ConditionExpr.
        """
        if raw_condition is None:
            return None
        if isinstance(raw_condition, dict):
            return ConditionExpr(
                source=raw_condition["source"],
                equals=raw_condition["equals"],
            )
        raise MissionCompilationError(
            f"Invalid condition_on format: {raw_condition}"
        )

    # ------------------------------------------------------------------
    # Context injection (summarized, never raw)
    # ------------------------------------------------------------------

    MAX_CONTEXT_TURNS: int = 5

    def _build_context_section(
        self,
        conversation: Optional[ConversationFrame],
    ) -> str:
        """
        Build a context section for the LLM prompt.

        Rules:
        - Raw lists NEVER enter the prompt
        - Only the LAST outcome is injected (not all)
        - Prompt growth is bounded by MAX_CONTEXT_TURNS
        - Omitted cleanly when no history exists
        """
        if conversation is None:
            return ""

        parts: List[str] = []

        # Active focus (from last outcome)
        if conversation.active_domain or conversation.active_entity:
            focus = []
            if conversation.active_domain:
                focus.append(f"domain: {conversation.active_domain}")
            if conversation.active_entity:
                focus.append(f"entity: {conversation.active_entity}")
            parts.append(f"Current Focus: {', '.join(focus)}")

        # Recent conversation turns (summarized, bounded)
        recent = conversation.history[-self.MAX_CONTEXT_TURNS:]
        if recent:
            turn_lines = []
            for turn in recent:
                prefix = "User" if turn.role == "user" else "Assistant"
                # Truncate long texts to prevent prompt explosion
                text = turn.text[:200]
                if len(turn.text) > 200:
                    text += "..."
                turn_lines.append(f"  {prefix}: {text}")
            parts.append("Recent Conversation:\n" + "\n".join(turn_lines))

        # Last outcome visible lists (summarized, NEVER raw)
        if conversation.outcomes:
            last_outcome = conversation.outcomes[-1]
            if hasattr(last_outcome, 'visible_lists') and last_outcome.visible_lists:
                summary = self.summarize_visible_lists(last_outcome)
                if summary:
                    parts.append(f"Previous Results:\n{summary}")

        # Resolved references from WorldResolver
        if hasattr(conversation, 'unresolved_references') and conversation.unresolved_references:
            refs = conversation.unresolved_references
            if "resolved" in refs:
                ref_lines = []
                for r in refs["resolved"]:
                    if r.get("ordinal") and r.get("value"):
                        ref_lines.append(f'  "{r["ordinal"]}" → {r["value"]}')
                    elif r.get("entity_hint"):
                        ref_lines.append(f'  reference → {r["entity_hint"]}')
                if ref_lines:
                    parts.append("Resolved References:\n" + "\n".join(ref_lines))
            elif refs.get("unresolved"):
                parts.append(
                    "Note: User uses referential language but "
                    "no resolution was possible."
                )

        if not parts:
            return ""

        return "Conversation Context:\n" + "\n\n".join(parts)

    @staticmethod
    def summarize_visible_lists(outcome: MissionOutcome) -> str:
        """
        Summarize visible_lists from an outcome for prompt injection.

        NEVER injects raw list contents. Produces count + sample:
          'search_results: 5 items (first: "Python tutorial")'

        This enables the LLM to know WHAT was produced,
        not every item in the list.
        """
        lines: List[str] = []
        for name, items in outcome.visible_lists.items():
            count = len(items)
            if count == 0:
                lines.append(f"  {name}: empty list")
                continue

            # Build a safe sample from the first item
            first = items[0]
            if isinstance(first, dict):
                # Use first key-value pair as representative
                for k, v in first.items():
                    sample = f'{k}="{v}"'
                    break
                else:
                    sample = str(first)[:50]
            else:
                sample = str(first)[:50]

            lines.append(f"  {name}: {count} items (first: {sample})")

        return "\n".join(lines)

