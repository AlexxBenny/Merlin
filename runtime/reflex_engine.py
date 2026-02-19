# runtime/reflex_engine.py

"""
ReflexEngine — Deterministic, zero-LLM reaction engine.

Two modes of operation:
1. Event reflexes: registered rules that fire on world events
2. Command reflexes: template-matched user commands that bypass cortex

Design rules:
- Zero LLM. Zero reasoning. Zero planning.
- Pattern → parameters → skill. That's it.
- If a template doesn't match, escalate to MISSION.
- Must NEVER crash the runtime.
- Command reflexes route through MissionExecutor for contract enforcement.
"""

import re
import time
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from world.timeline import WorldEvent, WorldTimeline
from world.state import WorldState
from world.snapshot import WorldSnapshot
from execution.registry import SkillRegistry
from ir.mission import IR_VERSION, MissionPlan, MissionNode

if TYPE_CHECKING:
    from execution.executor import MissionExecutor


logger = logging.getLogger(__name__)

ReflexRule = Callable[[WorldSnapshot, WorldEvent], None]


@dataclass(frozen=True)
class ReflexTemplate:
    """
    A parameterized command pattern that maps directly to a skill.

    pattern: compiled regex with named groups
    skill: registered skill name
    param_map: regex group name → skill parameter name
    """
    pattern: re.Pattern
    skill: str
    param_map: Dict[str, str]


@dataclass(frozen=True)
class ReflexMatch:
    """Result of a successful template match."""
    skill: str
    params: Dict[str, Any]


@dataclass(frozen=True)
class ReflexResult:
    """Structured result of reflex execution.

    Reflex is authoritative:
    - matched=True, success=True  → report success
    - matched=True, success=False → report failure (NEVER escalate)
    - matched=False               → escalate to MISSION
    """
    matched: bool
    success: bool
    outputs: Dict[str, Any] = None  # type: ignore[assignment]
    metadata: Dict[str, Any] = None  # type: ignore[assignment]
    error: Optional[str] = None

    def __post_init__(self):
        if self.outputs is None:
            object.__setattr__(self, 'outputs', {})
        if self.metadata is None:
            object.__setattr__(self, 'metadata', {})


class ReflexEngine:
    """
    Deterministic, zero-LLM reaction engine.

    Handles:
    1. World event reflexes (registered rules)
    2. User command reflexes (template matching)

    Command reflexes route through MissionExecutor for contract enforcement.
    This ensures world mutations, event emissions, and skill contracts are
    always enforced — never bypassed.
    """

    def __init__(
        self,
        timeline: WorldTimeline,
        registry: Optional[SkillRegistry] = None,
        executor: Optional["MissionExecutor"] = None,
    ):
        self.timeline = timeline
        self.registry = registry
        self.executor = executor
        self._rules: List[ReflexRule] = []
        self._templates: List[ReflexTemplate] = []

    # ─────────────────────────────────────────────────────────
    # Registration
    # ─────────────────────────────────────────────────────────

    def register(self, rule: ReflexRule):
        """Register a world-event reflex rule."""
        self._rules.append(rule)

    def register_template(self, template: ReflexTemplate):
        """Register a command reflex template."""
        self._templates.append(template)

    @classmethod
    def load_templates(cls, config_entries: list[dict]) -> List[ReflexTemplate]:
        """
        Build ReflexTemplate objects from config/routing.yaml entries.

        Expected format:
        - pattern: "set brightness to (?P<level>\\d+)"
          skill: "system.set_brightness"
          param_map: {level: "level"}
        """
        templates = []
        for entry in config_entries:
            try:
                compiled = re.compile(entry["pattern"], re.IGNORECASE)
                templates.append(ReflexTemplate(
                    pattern=compiled,
                    skill=entry["skill"],
                    param_map=entry.get("param_map", {}),
                ))
            except re.error as e:
                logger.warning(
                    "Invalid reflex template pattern '%s': %s",
                    entry.get("pattern", "?"), e
                )
        return templates

    # ─────────────────────────────────────────────────────────
    # World event handling
    # ─────────────────────────────────────────────────────────

    def on_event(self, event: WorldEvent):
        """
        React immediately to a world event.
        Must NEVER crash.
        """

        # Build snapshot at reaction time
        events = self.timeline.all_events()
        state = WorldState.from_events(events)
        snapshot = WorldSnapshot.build(
            state, events[-10:] if events else []
        )

        for rule in self._rules:
            try:
                rule(snapshot, event)
            except Exception:
                # Reflexes must NEVER break runtime
                continue

    # ─────────────────────────────────────────────────────────
    # User command template matching
    # ─────────────────────────────────────────────────────────

    def try_match(self, text: str) -> Optional[ReflexMatch]:
        """
        Try to match user text against registered reflex templates.

        Returns ReflexMatch if a template matches, None otherwise.
        First match wins (templates are checked in registration order).
        """
        text_lower = text.lower().strip()

        for template in self._templates:
            match = template.pattern.search(text_lower)
            if match:
                # Extract parameters using named groups
                raw_params = match.groupdict()

                # Map regex group names → skill parameter names
                skill_params = {}
                for group_name, param_name in template.param_map.items():
                    if group_name in raw_params:
                        skill_params[param_name] = raw_params[group_name]

                return ReflexMatch(
                    skill=template.skill,
                    params=skill_params,
                )

        return None

    def execute_reflex(
        self,
        reflex_match: ReflexMatch,
        snapshot: Optional[WorldSnapshot] = None,
    ) -> ReflexResult:
        """
        Execute a matched reflex command through the MissionExecutor.

        Constructs a single-node MissionPlan and routes it through
        executor.run() — gaining contract enforcement, event emission
        validation, and world mutation tracking.

        Args:
            reflex_match: Matched template with skill name and params.
            snapshot: Upstream WorldSnapshot — forwarded to executor
                      so skills can read state without rebuilding.

        Returns ReflexResult — always structured, never None.
        Must NEVER crash.
        """
        if not self.executor:
            return ReflexResult(
                matched=True, success=False,
                error="ReflexEngine has no executor configured",
            )

        try:
            # Verify skill exists before building plan
            if self.registry:
                try:
                    self.registry.get(reflex_match.skill)
                except KeyError:
                    return ReflexResult(
                        matched=True, success=False,
                        error=f"Skill '{reflex_match.skill}' not found in registry",
                    )

            # Build single-node MissionPlan
            reflex_node_id = "reflex_0"
            plan = MissionPlan(
                id=f"reflex_{int(time.time())}",
                nodes=[
                    MissionNode(
                        id=reflex_node_id,
                        skill=reflex_match.skill,
                        inputs=reflex_match.params,
                    )
                ],
                metadata={"ir_version": IR_VERSION},
            )

            # Route through executor — full contract enforcement
            # Forward snapshot so skills can read upstream state
            exec_result = self.executor.run(plan, world_snapshot=snapshot)
            outputs = exec_result.results.get(reflex_node_id, {})

            if reflex_node_id in exec_result.failed:
                return ReflexResult(
                    matched=True, success=False,
                    error=f"Skill '{reflex_match.skill}' execution failed",
                )

            return ReflexResult(
                matched=True, success=True,
                outputs=outputs,
                metadata=exec_result.metadata.get(reflex_node_id, {}),
            )

        except Exception as e:
            logger.warning(
                "Reflex execution failed for '%s'",
                reflex_match.skill,
                exc_info=True,
            )
            return ReflexResult(
                matched=True, success=False,
                error=str(e),
            )
