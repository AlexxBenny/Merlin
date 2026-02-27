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
    from cortex.intent_engine import IntentMatcher


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
        intent_matcher: Optional["IntentMatcher"] = None,
    ):
        self.timeline = timeline
        self.registry = registry
        self.executor = executor
        self._rules: List[ReflexRule] = []
        self._templates: List[ReflexTemplate] = []
        self._intent_matcher = intent_matcher
        self._last_multi_matches: Optional[List[ReflexMatch]] = None  # cache for route→handler

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
        Try to match user text against skills.

        Priority: IntentMatcher (scored) → regex templates (legacy fallback).
        First match wins.
        """
        from perception.normalize import normalize_for_matching
        text_normalized = normalize_for_matching(text)

        logger.debug(
            "Reflex try_match: raw='%s' normalized='%s'",
            text[:50], text_normalized[:50],
        )

        # ── Primary: IntentMatcher (Phase 10) ──
        if self._intent_matcher:
            intent_match = self._intent_matcher.match_clause(text_normalized)
            if intent_match:
                # Defense-in-depth: verify required params are present.
                # If extraction failed (e.g., escalation preposition
                # detected), reject the match → falls to MISSION.
                contract = self._intent_matcher._index.contracts[
                    intent_match.skill_name
                ]
                for key in contract.inputs:
                    if key not in intent_match.params:
                        logger.debug(
                            "Reflex: '%s' matched %s but missing required "
                            "param '%s' → reject",
                            text[:50], intent_match.skill_name, key,
                        )
                        return None

                logger.debug(
                    "Reflex match: HIT via IntentMatcher skill=%s score=%.1f",
                    intent_match.skill_name, intent_match.score,
                )
                return ReflexMatch(
                    skill=intent_match.skill_name,
                    params=intent_match.params,
                )

        # ── Fallback: regex templates ──
        for template in self._templates:
            match = template.pattern.search(text_normalized)
            if match:
                raw_params = match.groupdict()
                skill_params = {}
                for group_name, param_name in template.param_map.items():
                    if group_name in raw_params:
                        skill_params[param_name] = raw_params[group_name]

                logger.debug(
                    "Reflex match: HIT via regex skill=%s (checked %d templates)",
                    template.skill, len(self._templates),
                )
                return ReflexMatch(
                    skill=template.skill,
                    params=skill_params,
                )

        logger.debug(
            "Reflex match: MISS (intent=%s, templates=%d)",
            bool(self._intent_matcher), len(self._templates),
        )
        return None

    # ─────────────────────────────────────────────────────────
    # Multi-intent conjunction matching (Phase 9B + 10.1)
    # ─────────────────────────────────────────────────────────

    # Conjunction patterns — split on standardized separators
    _CLAUSE_SPLITTERS = re.compile(
        r"\s+(?:and|then|also|plus|,|;)\s+",
        re.IGNORECASE,
    )

    @staticmethod
    def _normalize_separators(text: str) -> str:
        """Pad separators with spaces so split regex works uniformly.

        Converts: 'volume 5, brightness 65' → 'volume 5 , brightness 65'
        Converts: 'a; b; c' → 'a ; b ; c'
        Converts: 'a & b'   → 'a and b'
        """
        text = text.replace(',', ' , ')
        text = text.replace(';', ' ; ')
        text = text.replace(' & ', ' and ')
        # Collapse multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def try_match_multi(self, text: str) -> Optional[List[ReflexMatch]]:
        """Try to match multiple intents in a conjunction query.

        Process:
        1. Normalize separators (pad commas, semicolons)
        2. Split on conjunction words
        3. Match each clause
        4. ALL must match → return list, else None

        Results are cached in _last_multi_matches for handler consumption.
        """
        # Clear cache from previous call
        self._last_multi_matches = None

        from perception.normalize import normalize_for_matching
        text_normalized = normalize_for_matching(text)

        # Pre-normalize separators
        text_normalized = self._normalize_separators(text_normalized)

        # Split into clauses
        clauses = self._CLAUSE_SPLITTERS.split(text_normalized)
        clauses = [c.strip() for c in clauses if c.strip()]

        # Must have 2+ clauses to be multi-reflex
        if len(clauses) < 2:
            return None

        # Verb-or-keyword clause awareness:
        # Each clause must contain at least one recognized verb or keyword
        # to be a valid split point. This prevents noun conjunctions
        # ("alex and ai") from being treated as clause separators.
        if self._intent_matcher:
            from cortex.synonyms import expand_token
            idx = self._intent_matcher._index
            for clause in clauses:
                tokens = clause.lower().split()
                has_signal = any(
                    expand_token(t) in idx.verb_index
                    or expand_token(t) in idx.keyword_index
                    for t in tokens
                )
                if not has_signal:
                    logger.debug(
                        "Multi-reflex: clause '%s' has no verb/keyword "
                        "signal → reject split",
                        clause,
                    )
                    return None

        logger.debug(
            "Multi-reflex: %d clauses from '%s': %r",
            len(clauses), text_normalized[:60], clauses,
        )

        matches: List[ReflexMatch] = []
        for clause in clauses:
            match = self._match_clause(clause)
            if match is None:
                logger.debug(
                    "Multi-reflex: clause '%s' unmatched → escalate to LLM",
                    clause,
                )
                return None
            matches.append(match)

        logger.info(
            "Multi-reflex: ALL %d clauses matched → %s",
            len(matches),
            [m.skill for m in matches],
        )

        # Cache for handler consumption (avoid duplicate work)
        self._last_multi_matches = matches
        return matches

    def _match_clause(self, clause: str) -> Optional[ReflexMatch]:
        """Match a single clause (no normalization — already done).

        Priority: IntentMatcher → regex fallback.
        Defense-in-depth: reject matches with missing required params.
        """
        # Primary: scored intent matching
        if self._intent_matcher:
            intent_match = self._intent_matcher.match_clause(clause)
            if intent_match:
                # Defense-in-depth: verify required params are present.
                # If extraction returned empty for a required input
                # (e.g., escalation preposition detected), reject the match.
                contract = self._intent_matcher._index.contracts[
                    intent_match.skill_name
                ]
                for key in contract.inputs:
                    if key not in intent_match.params:
                        logger.debug(
                            "Multi-reflex: '%s' matched %s but missing "
                            "required param '%s' → reject",
                            clause, intent_match.skill_name, key,
                        )
                        return None
                return ReflexMatch(
                    skill=intent_match.skill_name,
                    params=intent_match.params,
                )

        # Fallback: regex templates
        for template in self._templates:
            match = template.pattern.search(clause)
            if match:
                raw_params = match.groupdict()
                skill_params = {}
                for group_name, param_name in template.param_map.items():
                    if group_name in raw_params:
                        skill_params[param_name] = raw_params[group_name]
                return ReflexMatch(skill=template.skill, params=skill_params)
        return None

    def execute_multi_reflex(
        self,
        matches: List[ReflexMatch],
        snapshot: Optional[WorldSnapshot] = None,
    ) -> MissionPlan:
        """Build a multi-node MissionPlan from matched reflex clauses.

        Returns the plan — caller routes through orchestrator for
        narration + reporting + lifecycle management.

        All nodes are independent (depends_on=[]) since multi-reflex
        commands are structurally parallel atomic operations.
        """
        nodes = []
        for i, match in enumerate(matches):
            nodes.append(
                MissionNode(
                    id=f"reflex_{i}",
                    skill=match.skill,
                    inputs=match.params,
                )
            )

        return MissionPlan(
            id=f"multi_reflex_{int(time.time())}",
            nodes=nodes,
            metadata={"ir_version": IR_VERSION},
        )


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
