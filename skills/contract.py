from enum import Enum
from typing import Dict, Literal, Set, List

from pydantic import BaseModel, ConfigDict, Field

from ir.mission import ExecutionMode


class FailurePolicy(str, Enum):
    """
    What happens when a skill fails under a given execution mode.
    The executor reads this — it does NOT hardcode mode→failure mapping.
    """
    FAIL = "fail"           # Propagate failure (mission fails)
    CONTINUE = "continue"   # Log and continue (mission proceeds)
    IGNORE = "ignore"       # Silently ignore


class SkillContract(BaseModel):
    """
    Frozen guarantees a skill declares.

    This is a schema + validator + enforcement rule.
    NOT a new subsystem.

    The executor reads this contract at runtime to enforce:
    - allowed execution modes
    - failure semantics per mode
    - event emission guarantees
    - world mutation permission
    - focus and conflict constraints
    """
    model_config = ConfigDict(extra="forbid")

    name: str                                       # domain.action[.variant]

    # ── description (governed field — human-action phrase) ──
    # This field serves THREE consumers simultaneously:
    #   1. LLM skill manifest (compiler grounding)
    #   2. Intent decomposition prompts
    #   3. NarrationPolicy hot-path narration
    # Style contract:
    #   - Imperative human-action phrase: "Open an application"
    #   - Start with verb, ≤ 6 words
    #   - No parentheses, no commas, no technical adjectives
    #   - Not documentation — this is UX-visible
    description: str = ""

    # ── Capability metadata (for intent matching) ──
    action: str = ""                                # Canonical action: "set_volume", "open_app"
    target_type: str = ""                           # What this skill acts on: "volume", "app"

    inputs: Dict[str, str]                          # key → semantic type (REQUIRED)
    optional_inputs: Dict[str, str] = {}            # key → semantic type (OPTIONAL, have defaults)
    outputs: Dict[str, str]                         # key → semantic type

    allowed_modes: Set[ExecutionMode]                # Which modes this skill permits
    failure_policy: Dict[ExecutionMode, FailurePolicy]  # Mode → what to do on failure

    emits_events: List[str] = []                    # Event types this skill may emit
    mutates_world: bool = False                     # Whether skill may emit world events

    idempotent: bool = False                        # Safe to retry without side effects

    # ── Domain & scheduling metadata ──
    domain: str = ""                                # "fs", "system", "browser", "media"
    requires_focus: bool = False                    # Needs foreground window control
    resource_cost: str = "low"                      # "low" | "medium" | "high"
    conflicts_with: List[str] = Field(default_factory=list)  # Skill names that cannot run in parallel

    # ── Intent matching metadata (Phase 10) ──
    # Used by IntentIndex for scored clause matching — NOT regex.
    # Contract describes capability. Matcher interprets intent.
    #
    # intent_verbs: verbs the user might use to invoke this skill.
    #   e.g. media_play → ["play", "start", "resume"]
    # intent_keywords: nouns/synonyms the user might reference.
    #   e.g. set_volume → ["volume", "sound", "audio"]
    # intent_priority: tie-breaker when scores are equal (higher wins).
    #   e.g. mute=2, set_volume=1 → "mute" preferred over "set volume to 0"
    # verb_specificity:
    #   "specific" — verb alone IS the complete intent (mute, play, pause).
    #     No keyword reinforcement needed. Reflex executes on verb match.
    #   "generic" — verb describes an action CLASS (create, set, list, open).
    #     Requires keyword reinforcement to disambiguate intent.
    #     Verb-only matches are rejected → escalated to MISSION.
    intent_verbs: List[str] = Field(default_factory=list)
    intent_keywords: List[str] = Field(default_factory=list)
    intent_priority: int = 1
    verb_specificity: Literal["generic", "specific"] = "specific"

    # ── Narration metadata (Phase 8) ──
    # Controls how NarrationPolicy announces this skill during execution.
    # narration_visibility:
    #   "foreground" — included in pre-narration (default)
    #   "background" — omitted from narration but logged
    #   "silent"     — completely invisible to narration
    # narration_template:
    #   Optional sentence fragment with {placeholder} interpolation.
    #   e.g. "set brightness to {level}%"
    #   Empty = auto-generate from description field.
    #   Use only when: parameterization needed, or shorter phrasing desired.
    narration_visibility: str = "foreground"
    narration_template: str = ""

    # ── Data freshness policy ──
    # Declares whether this skill reads from the frozen WorldSnapshot
    # or from a live authoritative source at execution time.
    #
    # "snapshot" — Reads snapshot.state.* (default).
    #   Suitable for: actuation guards, decision inputs, state checks.
    #   Data is as fresh as the last event source poll.
    #
    # "live" — Reads the authoritative source directly (OS, stdlib, etc.).
    #   Required for: ephemeral telemetry queries (time, battery, CPU).
    #   The snapshot parameter is still passed but MUST NOT be the data
    #   source for the skill's primary output.
    #
    # CONSTRAINT: Skills with data_freshness="live" must NOT be used as
    # branching inputs (condition_on) in mission plans. Their outputs
    # are non-deterministic across replays.
    data_freshness: Literal["snapshot", "live"] = "snapshot"

    # ── Output formatting style (Phase 14) ──
    # Controls how the reflex path formats skill output for the user.
    # This is a PRESENTATION concern, not a capability concern.
    #
    # "terse"     — one-word/Done/changed flag (mutating skills: play, mute)
    #               Handler: deterministic formatting in _format_reflex_response
    #
    # "templated" — skill provides response_template in SkillResult.metadata
    #               Handler: template.format(**outputs) → deterministic text
    #               Examples: get_time → "It's {time}, {day}, {date}."
    #
    # "rich"      — structured/list data that needs LLM narration
    #               Handler: routed through ReportBuilder.build_from_skill_result()
    #               Examples: list_jobs, list_apps (output is list of dicts)
    output_style: Literal["terse", "templated", "rich"] = "terse"

