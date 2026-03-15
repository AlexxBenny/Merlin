# cortex/normalizer.py

"""
LLM Output Normalizer — The Trust Boundary.

This module sits between JSON extraction and IR model construction.
Its ONLY job: transform untrusted LLM dict shapes into IR-compatible shapes.

Design rules:
- No invention: never adds fields the LLM didn't emit.
- No reasoning: doesn't interpret semantic meaning.
- No suppression: unknown keys pass through to Pydantic (which rejects via extra="forbid").
- Hierarchy flattening: LLM decomposition may use hierarchical representation
  (e.g. parent + path). Execution skills require flattened paths.
  The normalizer bridges these two representations.
- 'id' is compiler-assigned — if LLM emits it, it passes through for warning/ignore.
- 'condition_on' is compiler-managed — if LLM emits it, it passes through for warning/ignore.
- Required field (skill) is NOT defaulted. None → Pydantic rejection.
- Coercion is strict: only safe, unambiguous type conversions.
- This layer handles NULL and SHAPE problems. Validity is Pydantic's job.

Pipeline position:
    LLM JSON → json_extraction → **normalizer** → _parse_mission_plan → validators → executor
"""

import logging
from pathlib import PurePosixPath
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Plan-level normalization
# ─────────────────────────────────────────────────────────────

def normalize_plan(payload: dict) -> dict:
    """Normalize a raw LLM plan payload into an IR-compatible shape.

    Handles:
    - "nodes": null → []
    - "nodes": {} → TypeError (unrecoverable — LLM emitted an object, not a list)
    - Missing "nodes" → []

    Returns a dict with guaranteed-list "nodes", each node normalized.
    Raises TypeError for unrecoverable shape violations.
    """
    nodes_raw = payload.get("nodes")

    if nodes_raw is None:
        nodes_raw = []

    if not isinstance(nodes_raw, list):
        raise TypeError(
            f"'nodes' must be a list, got {type(nodes_raw).__name__}"
        )

    return {"nodes": [normalize_node(n) for n in nodes_raw]}


# ─────────────────────────────────────────────────────────────
# Node-level normalization
# ─────────────────────────────────────────────────────────────

def normalize_node(raw: dict) -> dict:
    """Normalize a single node dict.

    The LLM no longer emits 'id' or 'condition_on' fields.
    IDs are assigned by the compiler in _parse_mission_plan().
    condition_on is compiler-managed.

    If \'id\' is present (legacy LLM output or prompt violation),
    it passes through — the parser will log a warning and ignore it.

    Optional fields are coerced:
    - inputs/outputs: null → {}
    - depends_on: null → [], str → [str], int → [int]
    - mode: null → "foreground"
    - condition_on: null passes through (Optional field in IR)

    Unknown keys are preserved — Pydantic's extra="forbid" will reject them.
    """
    normalized = {
        "skill":        raw.get("skill"),
        "inputs":       _coerce_dict(raw.get("inputs"), "inputs"),
        "outputs":      _coerce_dict(raw.get("outputs"), "outputs"),
        "depends_on":   _coerce_list(raw.get("depends_on"), "depends_on"),
        "mode":         raw.get("mode") or "foreground",
    }

    # Pass through id if LLM emitted it (parser will warn and ignore)
    if "id" in raw:
        normalized["id"] = raw["id"]

    # Pass through condition_on if LLM emitted it (parser will warn and ignore)
    if "condition_on" in raw:
        normalized["condition_on"] = raw["condition_on"]

    # Preserve unknown keys — let Pydantic enforce extra="forbid"
    known_keys = {"id", "skill", "inputs", "outputs", "depends_on", "mode", "condition_on"}
    for key in raw:
        if key not in known_keys:
            normalized[key] = raw[key]

    # Flatten hierarchical path inputs (parent + path → path)
    normalized["inputs"] = _flatten_parent_into_path(
        normalized.get("skill") or "", normalized["inputs"],
    )

    return normalized


# ─────────────────────────────────────────────────────────────
# Type coercion helpers (strict, not open-ended)
# ─────────────────────────────────────────────────────────────

def _coerce_list(value: Any, field_name: str) -> List:
    """Coerce a value into a list.

    Rules:
    - None → []
    - list → pass-through
    - str → [str] (LLM may emit "node_id" instead of ["node_id"])
    - int → [int] (LLM may emit 0 instead of [0] for index-based depends_on)
    - Anything else → TypeError (unrecoverable)
    """
    if value is None:
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, str):
        logger.debug(
            "Normalizer: coerced string '%s' to list for field '%s'",
            value, field_name,
        )
        return [value]

    if isinstance(value, int) and not isinstance(value, bool):
        logger.debug(
            "Normalizer: coerced int %d to list for field '%s'",
            value, field_name,
        )
        return [value]

    raise TypeError(
        f"'{field_name}' must be a list or null, got {type(value).__name__}"
    )


def _coerce_dict(value: Any, field_name: str) -> Dict:
    """Coerce a value into a dict.

    Rules:
    - None → {}
    - dict → pass-through
    - Anything else → TypeError (unrecoverable)
    """
    if value is None:
        return {}

    if isinstance(value, dict):
        return value

    raise TypeError(
        f"'{field_name}' must be a dict or null, got {type(value).__name__}"
    )


# ─────────────────────────────────────────────────────────────
# Hierarchy flattening (decomposition → execution bridge)
# ─────────────────────────────────────────────────────────────

def _flatten_parent_into_path(skill: str, inputs: dict) -> dict:
    """Fold hierarchical `parent` into `path` for file-operation skills.

    LLM decomposition may produce hierarchical representation:
        write_file(path="test.py", parent="alex")

    Execution skills require flattened paths:
        write_file(path="alex/test.py")

    Trigger condition (structural, not skill-name-based):
        Both 'parent' and 'path' are present in inputs.

    This naturally excludes fs.create_folder (uses 'name', not 'path').

    If 'parent' exists WITHOUT 'path', it is left untouched —
    the validator will reject it as an unexpected input.
    """
    parent = inputs.get("parent")
    path = inputs.get("path")

    if not parent or not path:
        return inputs

    # Path-safe join via PurePosixPath (preserves forward slashes)
    folded = str(PurePosixPath(parent) / path)

    logger.info(
        "[NORMALIZER] Folded parent=%r into path=%r for %s",
        parent, folded, skill,
    )

    inputs["path"] = folded
    inputs.pop("parent")

    return inputs


# ─────────────────────────────────────────────────────────────
# Anchor validation (structural, not semantic)
# ─────────────────────────────────────────────────────────────

def validate_anchors(payload: dict, valid_anchors: set) -> None:
    """Validate that all anchor values in node inputs are symbolic names.

    This is a structural guard — it checks that anchors are from the
    allowed set. It does NOT attempt to rewrite, decompose, or infer
    intent from invalid anchors.

    Raises TypeError for invalid anchors (caught as malformed_plan
    upstream).
    """
    if not valid_anchors:
        return

    for node in payload.get("nodes", []):
        inputs = node.get("inputs", {})
        anchor = inputs.get("anchor")
        if anchor is not None and anchor not in valid_anchors:
            node_id = node.get("id", "?")
            raise TypeError(
                f"Node '{node_id}': invalid anchor '{anchor}'. "
                f"Must be one of: {sorted(valid_anchors)}. "
                f"Raw filesystem paths are not allowed as anchors."
            )
