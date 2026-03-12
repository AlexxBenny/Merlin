# cortex/semantic_types.py

"""
Semantic Type Registry — Typed coercion, validation, and documentation.

Every semantic type declared in a SkillContract.inputs or SkillContract.outputs
MUST have an entry here. This registry serves THREE purposes:

1. LLM prompt documentation — type descriptions ground the compiler
2. Registration-time validation — direction constraints enforced
3. Runtime parameter resolution — coerce, validate, clamp inputs

INVARIANTS:
- This is the ONLY place type definitions live.
- Adding a new type auto-documents it in the LLM prompt AND enables coercion.
- A skill declaring a type NOT in this registry triggers a loud failure.
- direction constrains where the type may appear:
    "input"  → may only appear in SkillContract.inputs
    "output" → may only appear in SkillContract.outputs
    "both"   → may appear in either

Phase 9A: Types are now coercion-capable.
  - resolve() performs alias lookup → type coercion → range enforcement
  - strict=True → reject out-of-range values
  - strict=False → clamp to bounds (default)
  - Output-only types skip resolution (no user input to coerce)
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


class SemanticType:
    """A registered semantic type with coercion, validation, and documentation.

    Resolution priority (for input types):
    1. Alias lookup (case-insensitive string match)
    2. Type coercion (python_type or custom coerce_fn)
    3. Range enforcement (clamp or reject based on strict flag)
    """

    __slots__ = (
        "description", "direction", "python_type", "coerce_fn",
        "range_min", "range_max", "aliases", "strict",
    )

    def __init__(
        self,
        description: str,
        direction: str = "both",
        python_type: type = str,
        coerce_fn: Optional[Callable[[Any], Any]] = None,
        range_min: Optional[float] = None,
        range_max: Optional[float] = None,
        aliases: Optional[Dict[str, Any]] = None,
        strict: bool = False,
    ):
        if direction not in ("input", "output", "both"):
            raise ValueError(
                f"direction must be 'input', 'output', or 'both', got '{direction}'"
            )
        self.description = description
        self.direction = direction
        self.python_type = python_type
        self.coerce_fn = coerce_fn
        self.range_min = range_min
        self.range_max = range_max
        self.aliases = {k.lower(): v for k, v in (aliases or {}).items()}
        self.strict = strict

    def resolve(self, raw_value: Any) -> Any:
        """Coerce, validate, and clamp a raw input value.

        Returns the resolved value.
        Raises ValueError on unresolvable input (with clear message).
        """
        value = raw_value

        # 1. Alias lookup (case-insensitive)
        if isinstance(value, str) and value.lower() in self.aliases:
            value = self.aliases[value.lower()]

        # 2. Type coercion
        if self.coerce_fn:
            try:
                value = self.coerce_fn(value)
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"Cannot coerce {raw_value!r} using custom coercion: {e}"
                ) from e
        elif self.python_type is not str:
            # Don't coerce if already correct type
            if not isinstance(value, self.python_type):
                try:
                    value = self.python_type(value)
                except (ValueError, TypeError) as e:
                    raise ValueError(
                        f"Cannot coerce {raw_value!r} to "
                        f"{self.python_type.__name__}: {e}"
                    ) from e

        # 3. Range enforcement
        if self.range_min is not None or self.range_max is not None:
            try:
                numeric = float(value)
            except (ValueError, TypeError):
                pass  # Non-numeric values skip range check
            else:
                if self.strict:
                    if self.range_min is not None and numeric < self.range_min:
                        raise ValueError(
                            f"Value {raw_value!r} ({numeric}) is below "
                            f"minimum {self.range_min}"
                        )
                    if self.range_max is not None and numeric > self.range_max:
                        raise ValueError(
                            f"Value {raw_value!r} ({numeric}) exceeds "
                            f"maximum {self.range_max}"
                        )
                else:
                    # Lenient: clamp silently
                    if self.range_min is not None and numeric < self.range_min:
                        value = self.python_type(self.range_min)
                    if self.range_max is not None and numeric > self.range_max:
                        value = self.python_type(self.range_max)

        return value

    @property
    def is_resolvable(self) -> bool:
        """Whether this type has resolution logic (not passthrough)."""
        return bool(
            self.aliases
            or self.python_type is not str
            or self.coerce_fn
            or self.range_min is not None
            or self.range_max is not None
        )

    def __repr__(self) -> str:
        parts = [f"description={self.description!r}", f"direction={self.direction!r}"]
        if self.python_type is not str:
            parts.append(f"python_type={self.python_type.__name__}")
        if self.aliases:
            parts.append(f"aliases={len(self.aliases)}")
        if self.range_min is not None or self.range_max is not None:
            parts.append(f"range=[{self.range_min}, {self.range_max}]")
        if self.strict:
            parts.append("strict=True")
        return f"SemanticType({', '.join(parts)})"


# ─────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────
#
# Scaling rule: define coercion per TYPE, not per skill.
# 40 skills using volume_percentage all inherit the same aliases.

# ── Shared alias sets (DRY) ──
_PERCENTAGE_ALIASES = {
    "full": 100, "max": 100, "maximum": 100,
    "half": 50, "mid": 50, "middle": 50,
    "low": 20, "min": 0, "minimum": 0,
    "off": 0, "zero": 0,
}

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
        python_type=int,
        range_min=0, range_max=100,
        aliases={
            **_PERCENTAGE_ALIASES,
            "mute": 0, "quiet": 15, "loud": 85,
        },
    ),
    "actual_volume": SemanticType(
        description="The actual volume level after adjustment.",
        direction="output",
    ),
    "brightness_percentage": SemanticType(
        description="Brightness level as an integer 0–100.",
        direction="input",
        python_type=int,
        range_min=0, range_max=100,
        aliases={
            **_PERCENTAGE_ALIASES,
            "dim": 20, "bright": 80,
        },
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

    # ── Reusable output type for query skills ──
    # All human-readable string outputs use this ONE type.
    # Output types are semantic categories, not per-field labels.
    # Scales: 500 query skills, 1 output type.
    "info_string": SemanticType(
        description="A human-readable informational string.",
        direction="output",
    ),

    # ── Content generation ──
    "text_prompt": SemanticType(
        description=(
            "A natural language prompt or topic for content generation "
            "(e.g., \"a short poem about AI\", \"summary of India\")."
        ),
        direction="input",
    ),
    "generated_text": SemanticType(
        description="LLM-generated text content (story, summary, poem, etc.).",
        direction="output",
    ),

    # ── File system ──
    "file_path_input": SemanticType(
        description=(
            "A file name or relative path for file operations "
            "(e.g., \"poem.txt\", \"notes/ideas.md\")."
        ),
        direction="input",
    ),
    "file_content": SemanticType(
        description="Text content of a file (read or to be written).",
        direction="both",
    ),

    # ── Scheduler domain ──
    # Domain entity: Job
    # Canonical types: job_identifier (input), job_list (output)
    # Do NOT add job_summary / job_details / job_info — use info_string
    # for human-readable outputs instead.
    "job_list": SemanticType(
        description="List of scheduled jobs with metadata.",
        direction="output",
    ),
    "job_identifier": SemanticType(
        description=(
            "Short ID of a scheduled job (e.g., \"J-3\", \"3\", or \"job 3\"). "
            "Normalized internally."
        ),
        direction="input",
    ),
    "filter_value": SemanticType(
        description=(
            "A filter/qualifier for list queries "
            "(e.g., \"active\", \"completed\", \"failed\", \"all\")."
        ),
        direction="input",
    ),

    # ── Memory domain ──
    # Types for UserKnowledgeStore skill wrappers.
    # Values are user-defined: passthrough semantics, no coercion.
    "preference_key": SemanticType(
        description=(
            "A canonical preference key (e.g., \"volume\", \"brightness\", "
            "\"theme\"). Normalized to lowercase snake_case at storage time."
        ),
        direction="both",
    ),
    "any": SemanticType(
        description=(
            "An untyped passthrough value. Use only when the value schema "
            "is user-defined and cannot statically be typed (e.g., "
            "preference values, raw outputs)."
        ),
        direction="both",
    ),
    "boolean": SemanticType(
        description="A true/false boolean value.",
        direction="output",  # never a user input param
        python_type=bool,
        coerce_fn=lambda v: bool(v) if not isinstance(v, str)
                   else v.lower() not in ("false", "0", "no", ""),
    ),
    "fact_key": SemanticType(
        description=(
            "A canonical fact key (e.g., \"name\", \"location\", \"age\"). "
            "Used to label persistent user facts in memory."
        ),
        direction="both",
    ),
    "policy_condition": SemanticType(
        description=(
            "A dict describing when a policy applies "
            "(e.g., {\"activity\": \"movie\"})."
        ),
        direction="input",
        python_type=dict,
    ),
    "policy_action": SemanticType(
        description=(
            "A dict describing what to do when the policy matches "
            "(e.g., {\"set_volume\": 90})."
        ),
        direction="input",
        python_type=dict,
    ),
    "policy_label": SemanticType(
        description=(
            "A short human-readable label for a policy "
            "(e.g., \"movie mode\", \"meeting mode\")."
        ),
        direction="input",
    ),
    "policy_id": SemanticType(
        description="UUID string identifying a stored policy.",
        direction="output",
    ),

    # ── Browser domain ──
    "browser_task_description": SemanticType(
        description=(
            "A natural language description of a browser task to perform "
            "(e.g., \"search gaming laptops on amazon\", "
            "\"go to youtube.com and play lofi music\")."
        ),
        direction="input",
    ),
    "url_string": SemanticType(
        description="A URL string (e.g., \"https://example.com\").",
        direction="both",
    ),
    "step_limit": SemanticType(
        description="Maximum number of browser automation steps (default: 20).",
        direction="input",
        python_type=int,
        range_min=1, range_max=50,
    ),
    "entity_index": SemanticType(
        description="Browser DOM entity index to interact with.",
        direction="input",
        python_type=int,
    ),
    "scroll_direction": SemanticType(
        description="Scroll direction: 'up' or 'down'.",
        direction="input",
    ),
    "fill_text": SemanticType(
        description="Text to type into a browser input field.",
        direction="input",
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
