# reporting/narration.py

"""
NarrationPolicy — Layer-aware execution narration.

Deterministic. No LLM. Hot path only.

Design principles:
- Metadata-driven: reads SkillContract fields (description, domain,
  narration_visibility, narration_template) — no hardcoded skill→phrase maps
- Layer-aware: narrates at DAG layer boundaries, not per-node
- Compression: groups parallel nodes into domain phrases
- Restraint: single-node missions are silent; pre-narration suppresses
  redundant layer narration

Scaling rules:
- New skills auto-narrate via their contract.description field
- Skills opt out with narration_visibility="silent"
- Fine-tune with narration_template without touching this module
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ir.mission import MissionPlan, MissionNode
    from skills.registry import SkillRegistry


logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Domain humanization
# ─────────────────────────────────────────────────────────────

_DOMAIN_LABELS: Dict[str, str] = {
    "system": "system settings",
    "fs": "file operations",
    "media": "media",
    "browser": "browser",
}


def _humanize_domain(domain: str) -> str:
    """Convert domain key to human phrase."""
    return _DOMAIN_LABELS.get(domain, domain)


def _natural_join(items: List[str]) -> str:
    """'a, b and c' style join."""
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


# ─────────────────────────────────────────────────────────────
# NarrationPolicy
# ─────────────────────────────────────────────────────────────

class NarrationPolicy:
    """Deterministic layer-aware narration. No LLM. Hot path only.

    Usage:
        policy = NarrationPolicy(config)
        text = policy.narrate_pre_execution(plan, node_index, registry)
        if text:
            output_channel.send(text)
    """

    def __init__(
        self,
        single_node_silent: bool = True,
        compression_threshold: int = 3,
        heartbeat_threshold_seconds: float = 3.0,
    ):
        self._single_node_silent = single_node_silent
        self._compression_threshold = compression_threshold
        self._heartbeat_threshold = heartbeat_threshold_seconds
        self._heartbeat_fired = False  # Reset per mission

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "NarrationPolicy":
        """Build from narration config section."""
        return cls(
            single_node_silent=config.get("single_node_silent", True),
            compression_threshold=config.get("compression_threshold", 3),
            heartbeat_threshold_seconds=config.get(
                "heartbeat_threshold_seconds", 3.0
            ),
        )

    # ─────────────────────────────────────────────────────────
    # Pre-execution narration
    # ─────────────────────────────────────────────────────────

    def narrate_pre_execution(
        self,
        plan: "MissionPlan",
        node_index: Dict[str, "MissionNode"],
        registry: "SkillRegistry",
    ) -> Optional[str]:
        """Compressed intent before execution. Returns None for silent.

        Rules:
        - 1 node → silent (reflex-level, no overhead)
        - 2–3 foreground nodes → compressed intent
        - 4+ foreground nodes → phase summary by domain
        """
        self._heartbeat_fired = False  # Reset for new mission

        visible = self._visible_nodes(plan, registry)
        count = len(visible)

        if count == 0:
            logger.debug("[NARR] No visible nodes — silent")
            return None

        if self._single_node_silent and count <= 1:
            logger.debug("[NARR] Single node — silent")
            return None

        if count <= self._compression_threshold:
            text = self._compress_intent(visible, registry)
            logger.debug("[NARR] Compressed (%d nodes): %s", count, text)
            return text

        text = self._phase_summary(visible, registry)
        logger.debug("[NARR] Phase summary (%d nodes): %s", count, text)
        return text

    # ─────────────────────────────────────────────────────────
    # Layer-level narration (for large missions only)
    # ─────────────────────────────────────────────────────────

    def narrate_layer_start(
        self,
        layer: List[str],
        node_index: Dict[str, "MissionNode"],
        registry: "SkillRegistry",
        layer_idx: int,
        total_layers: int,
        pre_narration_fired: bool,
    ) -> Optional[str]:
        """Per-layer narration for large missions. None = silent.

        Suppressed when:
        - Only 1 layer total (pre-narration covered it)
        - Pre-narration already summarized ≤ compression_threshold nodes
        - All nodes in layer are background/silent
        """
        # If pre-narration already covered a small mission, stay silent
        if pre_narration_fired and total_layers <= 2:
            logger.debug(
                "[NARR] Layer %d suppressed — pre-narration covered intent",
                layer_idx,
            )
            return None

        # Collect visible domains in this layer
        domains = set()
        for nid in layer:
            node = node_index.get(nid)
            if not node:
                continue
            try:
                skill = registry.get(node.skill)
                if skill.contract.narration_visibility == "silent":
                    continue
                domains.add(skill.contract.domain or "general")
            except (KeyError, AttributeError):
                domains.add("general")

        if not domains:
            logger.debug("[NARR] Layer %d all silent nodes — skipped", layer_idx)
            return None

        human_domains = [_humanize_domain(d) for d in sorted(domains)]
        text = f"Working on {_natural_join(human_domains)}..."
        logger.debug("[NARR] Layer %d: %s", layer_idx, text)
        return text

    # ─────────────────────────────────────────────────────────
    # Heartbeat (one-shot)
    # ─────────────────────────────────────────────────────────

    def narrate_heartbeat(self, elapsed_seconds: float) -> Optional[str]:
        """One-shot heartbeat if execution exceeds threshold.

        Fires at most once per mission. Returns None after first fire.
        """
        if self._heartbeat_fired:
            return None
        if elapsed_seconds < self._heartbeat_threshold:
            return None

        self._heartbeat_fired = True
        logger.debug(
            "[NARR] Heartbeat after %.1fs", elapsed_seconds,
        )
        return "Still working on that..."

    # ─────────────────────────────────────────────────────────
    # Internals
    # ─────────────────────────────────────────────────────────

    def _visible_nodes(
        self,
        plan: "MissionPlan",
        registry: "SkillRegistry",
    ) -> List["MissionNode"]:
        """Filter to foreground-visible nodes only."""
        visible = []
        for node in plan.nodes:
            try:
                skill = registry.get(node.skill)
                if skill.contract.narration_visibility != "silent":
                    visible.append(node)
            except (KeyError, AttributeError):
                # Unknown skill → include (safe default)
                visible.append(node)
        return visible

    def _compress_intent(
        self,
        nodes: List["MissionNode"],
        registry: "SkillRegistry",
    ) -> str:
        """'I'll adjust your settings and pause the music.'"""
        verbs = []
        for node in nodes:
            verb = self._get_verb(node, registry)
            verbs.append(verb)

        return f"I'll {_natural_join(verbs)}."

    def _phase_summary(
        self,
        nodes: List["MissionNode"],
        registry: "SkillRegistry",
    ) -> str:
        """'I'll handle your system settings and media.'"""
        domains = set()
        for node in nodes:
            try:
                skill = registry.get(node.skill)
                domains.add(skill.contract.domain or "general")
            except (KeyError, AttributeError):
                domains.add("general")

        human_domains = [_humanize_domain(d) for d in sorted(domains)]
        return f"I'll handle your {_natural_join(human_domains)}."

    @staticmethod
    def _get_verb(node: "MissionNode", registry: "SkillRegistry") -> str:
        """Extract narration phrase from skill contract metadata.

        Priority:
        1. contract.narration_template (parameterized override)
        2. contract.description (human-action phrase — governed field)
        3. contract.action with underscores → spaces (fallback)
        """
        try:
            skill = registry.get(node.skill)
            contract = skill.contract

            if contract.narration_template:
                phrase = contract.narration_template
                # Interpolate input values if template has {placeholders}
                if "{" in phrase:
                    # OutputReference values aren't resolved at narration
                    # time — fall back to description to avoid leaking
                    # raw repr into user-facing speech.
                    from ir.mission import OutputReference
                    if any(isinstance(v, OutputReference)
                           for v in node.inputs.values()):
                        if contract.description:
                            desc = contract.description
                            return desc[0].lower() + desc[1:] if desc else desc
                    else:
                        try:
                            phrase = phrase.format_map(node.inputs)
                        except (KeyError, IndexError):
                            pass
                return phrase

            if contract.description:
                # Use description as human verb — lowercase first char
                desc = contract.description
                return desc[0].lower() + desc[1:] if desc else desc

            if contract.action:
                return contract.action.replace("_", " ")

        except (KeyError, AttributeError):
            pass

        # Ultimate fallback — never crashes
        skill_name = node.skill.split(".")[-1].replace("_", " ")
        return f"handle {skill_name}"
