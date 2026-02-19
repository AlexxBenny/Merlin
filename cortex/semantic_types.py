# cortex/semantic_types.py

"""
Semantic Type Registry — Single source of truth for input/output type documentation.

Every semantic type declared in a SkillContract.inputs or SkillContract.outputs
MUST have an entry here. The prompt builder generates type documentation
dynamically from this registry, documenting only types actually used by
available skills.

INVARIANTS:
- This is the ONLY place type documentation lives.
- Adding a new type here auto-documents it in the LLM prompt.
- A skill declaring a type NOT in this registry triggers a loud failure.
- direction constrains where the type may appear:
    "input"  → may only appear in SkillContract.inputs
    "output" → may only appear in SkillContract.outputs
    "both"   → may appear in either
"""

from typing import Dict


class SemanticType:
    """A registered semantic type with documentation and direction."""

    __slots__ = ("description", "direction")

    def __init__(self, description: str, direction: str = "both"):
        if direction not in ("input", "output", "both"):
            raise ValueError(
                f"direction must be 'input', 'output', or 'both', got '{direction}'"
            )
        self.description = description
        self.direction = direction

    def __repr__(self) -> str:
        return f"SemanticType({self.description!r}, direction={self.direction!r})"


# ─────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────

SEMANTIC_TYPES: Dict[str, SemanticType] = {

    # ── Location / filesystem ──
    "anchor_name": SemanticType(
        description=(
            "A symbolic location anchor. MUST be from the Available Location "
            "Anchors list. Default: \"WORKSPACE\". "
            "Do NOT use raw filesystem paths as anchors."
        ),
        direction="input",
    ),
    "relative_path": SemanticType(
        description=(
            "A path relative to the anchor (e.g., \"projects/myapp\", \"X\"). "
            "Use this for nesting: to place something inside a previously "
            "created folder, set parent to that folder's relative path."
        ),
        direction="input",
    ),
    "folder_name": SemanticType(
        description=(
            "The name of the folder (e.g., \"myProject\"). "
            "Just the name — no path separators."
        ),
        direction="input",
    ),
    "filesystem_path": SemanticType(
        description="An absolute filesystem path.",
        direction="output",
    ),

    # ── Application ──
    "application_name": SemanticType(
        description=(
            "Name or command of an application (e.g., \"notepad\", \"chrome\")."
        ),
        direction="both",
    ),
    "cli_arguments": SemanticType(
        description="Optional command-line arguments.",
        direction="input",
    ),
    "process_id": SemanticType(
        description="An OS process identifier.",
        direction="output",
    ),

    # ── Volume / brightness ──
    "volume_percentage": SemanticType(
        description="Volume level as an integer 0–100.",
        direction="input",
    ),
    "actual_volume": SemanticType(
        description="The actual volume level after adjustment.",
        direction="output",
    ),
    "brightness_percentage": SemanticType(
        description="Brightness level as an integer 0–100.",
        direction="input",
    ),
    "actual_brightness": SemanticType(
        description="The actual brightness level after adjustment.",
        direction="output",
    ),

    # ── Toggle states ──
    "mute_state": SemanticType(
        description="Whether the system is muted (true/false).",
        direction="output",
    ),
    "nightlight_state": SemanticType(
        description="Whether night light is enabled (true/false).",
        direction="output",
    ),

    # ── Media ──
    "media_key_sent": SemanticType(
        description="Whether the media key was sent successfully.",
        direction="output",
    ),
    "whether_playback_state_was_changed": SemanticType(
        description="Whether the playback state actually changed.",
        direction="output",
    ),

    # ── Lists ──
    "application_list": SemanticType(
        description="List of running applications.",
        direction="output",
    ),
}


# ─────────────────────────────────────────────────────────────
# Assertion helper
# ─────────────────────────────────────────────────────────────

def assert_types_registered(
    skill_name: str,
    declared_inputs: Dict[str, str],
    declared_outputs: Dict[str, str],
) -> None:
    """Fail loudly if a skill declares a semantic type not in the registry.

    Also validates direction constraints:
    - input-only types must not appear in outputs
    - output-only types must not appear in inputs

    Called at skill registration time, not at runtime.

    Raises:
        ValueError: If an unregistered or misused type is found.
    """
    for key, stype in declared_inputs.items():
        if stype not in SEMANTIC_TYPES:
            raise ValueError(
                f"Skill '{skill_name}' declares input '{key}' with "
                f"unregistered semantic type '{stype}'. "
                f"Register it in cortex/semantic_types.py first."
            )
        entry = SEMANTIC_TYPES[stype]
        if entry.direction == "output":
            raise ValueError(
                f"Skill '{skill_name}' uses output-only type '{stype}' "
                f"as input '{key}'. Check direction in SEMANTIC_TYPES."
            )

    for key, stype in declared_outputs.items():
        if stype not in SEMANTIC_TYPES:
            raise ValueError(
                f"Skill '{skill_name}' declares output '{key}' with "
                f"unregistered semantic type '{stype}'. "
                f"Register it in cortex/semantic_types.py first."
            )
        entry = SEMANTIC_TYPES[stype]
        if entry.direction == "input":
            raise ValueError(
                f"Skill '{skill_name}' uses input-only type '{stype}' "
                f"as output '{key}'. Check direction in SEMANTIC_TYPES."
            )
