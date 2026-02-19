# cortex/context_provider.py

"""
ContextProvider — Abstraction seam for how context enters the LLM prompt.

Implementations:
  SimpleContextProvider      — Deterministic, unbounded (original behavior).
  RetrievalContextProvider   — Token-budgeted, priority-ordered, memory-aware.

This seam prevents cortex from coupling to conversation internals.
Swapping the provider requires ZERO cortex changes.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from conversation.frame import ConversationFrame, ConversationTurn
from conversation.outcome import MissionOutcome

if TYPE_CHECKING:
    from memory.store import MemoryStore

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Token estimation
# ─────────────────────────────────────────────────────────────

def _estimate_tokens(text: str) -> int:
    """Approximate token count.  1 token ≈ 4 characters.

    Consistent with Phase 3A metrics.  Good enough for relative
    budgeting — exact counts depend on the tokenizer.
    """
    return max(len(text) // 4, 1) if text else 0


# ─────────────────────────────────────────────────────────────
# ABC
# ─────────────────────────────────────────────────────────────

class ContextProvider(ABC):
    """How context enters the LLM prompt.

    The single abstraction point between conversation state
    and the cortex prompt builder. Cortex depends ONLY on this.
    """

    @abstractmethod
    def build_context(
        self,
        query: str,
        conversation: Optional[ConversationFrame],
        world_state: Dict[str, Any],
    ) -> str:
        """Return a formatted context string for the LLM prompt.

        Must be bounded in size. Must never inject raw lists.
        """
        ...


# ─────────────────────────────────────────────────────────────
# SimpleContextProvider (original, unbounded)
# ─────────────────────────────────────────────────────────────

class SimpleContextProvider(ContextProvider):
    """Deterministic context from conversation frame.

    Extracts: active focus, recent turns, last outcome summary,
    resolved references. No retrieval, no embeddings.

    This is the CURRENT behavior, extracted behind the interface.
    """

    MAX_CONTEXT_TURNS: int = 5

    def build_context(
        self,
        query: str,
        conversation: Optional[ConversationFrame],
        world_state: Dict[str, Any],
    ) -> str:
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
                text = turn.text[:200]
                if len(turn.text) > 200:
                    text += "..."
                turn_lines.append(f"  {prefix}: {text}")
            parts.append("Recent Conversation:\n" + "\n".join(turn_lines))

        # Last outcome visible lists (summarized, NEVER raw)
        if conversation.outcomes:
            last_outcome = conversation.outcomes[-1]
            if hasattr(last_outcome, 'visible_lists') and last_outcome.visible_lists:
                summary = self._summarize_visible_lists(last_outcome)
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

        # Entity registry summary (bounded, never raw)
        if conversation.entity_registry:
            entity_lines = []
            for key, record in list(conversation.entity_registry.items())[:10]:
                if isinstance(record.value, list):
                    entity_lines.append(
                        f"  {key} ({record.type}): {len(record.value)} items"
                    )
                else:
                    entity_lines.append(
                        f"  {key} ({record.type}): {str(record.value)[:100]}"
                    )
            parts.append("Known Entities:\n" + "\n".join(entity_lines))

        # Active goals
        active_goals = conversation.get_active_goals()
        if active_goals:
            goal_lines = [f"  - {g.description}" for g in active_goals[:5]]
            parts.append("Active Goals:\n" + "\n".join(goal_lines))

        if not parts:
            return ""

        return "Conversation Context:\n" + "\n\n".join(parts)


    @staticmethod
    def _summarize_visible_lists(outcome: MissionOutcome) -> str:
        """Summarize visible_lists for prompt injection.

        NEVER injects raw list contents. Produces count + sample.
        """
        lines: List[str] = []
        for name, items in outcome.visible_lists.items():
            count = len(items)
            if count == 0:
                lines.append(f"  {name}: empty list")
                continue

            first = items[0]
            if isinstance(first, dict):
                for k, v in first.items():
                    sample = f'{k}="{v}"'
                    break
                else:
                    sample = str(first)[:50]
            else:
                sample = str(first)[:50]

            lines.append(f"  {name}: {count} items (first: {sample})")

        return "\n".join(lines) if lines else ""


# ─────────────────────────────────────────────────────────────
# Turn compaction strategy (modular — seam for future semantic)
# ─────────────────────────────────────────────────────────────

# A TurnCompactor transforms a list of turns into a budget-bounded
# string.  Phase 3B uses deterministic truncation.  Phase 3C+ can
# swap in LLM-backed semantic summarization via the same interface.
TurnCompactor = Callable[[List[ConversationTurn], int], str]


def truncation_compactor(turns: List[ConversationTurn], char_budget: int) -> str:
    """Deterministic turn compactor: recent turns full, older compressed.

    Strategy:
    - Last 2 turns: full text up to 200 chars
    - Older turns: first 50 chars only
    - Drop oldest turns first when budget exceeded
    """
    if not turns:
        return ""

    lines: List[str] = []
    used = 0

    # Process newest first, reverse at the end
    for i, turn in enumerate(reversed(turns)):
        prefix = "User" if turn.role == "user" else "Assistant"

        if i < 2:
            # Recent: full text, capped at 200 chars
            text = turn.text[:200]
            if len(turn.text) > 200:
                text += "..."
        else:
            # Older: compressed
            text = turn.text[:50]
            if len(turn.text) > 50:
                text += "..."

        line = f"  {prefix}: {text}"
        line_len = len(line)

        if used + line_len > char_budget:
            # Budget exhausted — stop adding older turns
            break

        lines.append(line)
        used += line_len

    lines.reverse()
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Query heuristics for dynamic budget allocation
# ─────────────────────────────────────────────────────────────

# Temporal keywords that signal memory relevance
_MEMORY_SIGNALS = [
    "yesterday", "earlier", "last time", "before", "previously",
    "remember", "again", "continue", "resume",
]

# Goal-reference keywords
_GOAL_SIGNALS = [
    "goal", "continue", "finish", "complete", "progress",
    "started", "working on", "the plan", "next step",
]

# Short follow-up detection: queries ≤ N words with no explicit domain
_SHORT_FOLLOWUP_THRESHOLD = 4


def _detect_query_signals(query: str) -> Dict[str, float]:
    """Deterministic relevance heuristics.

    Returns weight multipliers for each context section.
    Multipliers > 1.0 boost a section, < 1.0 suppress it.
    Default is 1.0 (neutral).

    Rules:
    - Temporal keywords → boost memory, suppress turns
    - Goal keywords → boost goals
    - Short follow-up → boost turns, suppress memory
    """
    q = query.lower()
    weights: Dict[str, float] = {
        "goals": 1.0,
        "entities": 1.0,
        "memory": 1.0,
        "turns": 1.0,
        "refs": 1.0,
    }

    has_memory_signal = any(kw in q for kw in _MEMORY_SIGNALS)
    has_goal_signal = any(kw in q for kw in _GOAL_SIGNALS)
    is_short_followup = len(query.split()) <= _SHORT_FOLLOWUP_THRESHOLD

    if has_goal_signal:
        weights["goals"] = 2.5    # 15% → ~37%
        weights["turns"] = 0.5    # 30% → ~15%

    if has_memory_signal:
        weights["memory"] = 2.0   # 30% → ~60%
        weights["turns"] = 0.5    # 30% → ~15%

    if is_short_followup and not has_memory_signal and not has_goal_signal:
        weights["turns"] = 2.0    # 30% → ~60%
        weights["memory"] = 0.5   # 30% → ~15%

    return weights


# ─────────────────────────────────────────────────────────────
# RetrievalContextProvider
# ─────────────────────────────────────────────────────────────

# Default base percentages (before dynamic adjustment)
_DEFAULT_BUDGETS = {
    "goals": 0.15,
    "entities": 0.15,
    "memory": 0.30,
    "turns": 0.30,
    "refs": 0.10,
}


class RetrievalContextProvider(ContextProvider):
    """Token-budgeted, priority-ordered, memory-aware context provider.

    Guarantees:
    - Hard token cap (default 800) NEVER exceeded.
    - Single-pass downward cascade — no upward loops.
    - Total output is mathematically bounded by assertion.

    Budget allocation is DYNAMIC:
    - Base percentages are adjusted by deterministic query heuristics.
    - Unused budget cascades downward through the priority order.

    Priority order (highest → lowest):
        goals → entities → memory → turns → refs

    Turn compaction is MODULAR:
    - Default: deterministic truncation (truncation_compactor)
    - Future: swap in semantic summarizer via constructor injection
    """

    # Priority order matches budget cascade direction
    _SECTION_ORDER = ["goals", "entities", "memory", "turns", "refs"]

    def __init__(
        self,
        memory: Optional[MemoryStore] = None,
        token_budget: int = 800,
        base_budgets: Optional[Dict[str, float]] = None,
        turn_compactor: Optional[TurnCompactor] = None,
        memory_top_k: int = 3,
    ):
        self._memory = memory
        self._token_budget = token_budget
        self._base_budgets = base_budgets or dict(_DEFAULT_BUDGETS)
        self._turn_compactor = turn_compactor or truncation_compactor
        self._memory_top_k = memory_top_k

        # Validate base budgets sum to ~1.0
        total = sum(self._base_budgets.values())
        assert 0.99 <= total <= 1.01, (
            f"Base budgets must sum to 1.0, got {total}"
        )

    def build_context(
        self,
        query: str,
        conversation: Optional[ConversationFrame],
        world_state: Dict[str, Any],
    ) -> str:
        """Build token-budgeted context string.

        Invariant: output token count ≤ self._token_budget.
        """
        if conversation is None and self._memory is None:
            return ""

        # ── Step 1: Compute dynamic budgets ──
        char_budget = self._token_budget * 4  # tokens → chars
        weights = _detect_query_signals(query)
        section_char_budgets = self._allocate_budgets(weights, char_budget)

        # ── Step 2: Render each section within its budget ──
        sections: Dict[str, str] = {}

        sections["goals"] = self._render_goals(
            conversation, section_char_budgets["goals"],
        )
        sections["entities"] = self._render_entities(
            conversation, section_char_budgets["entities"],
        )
        sections["memory"] = self._render_memory(
            query, section_char_budgets["memory"],
        )
        sections["turns"] = self._render_turns(
            conversation, section_char_budgets["turns"],
        )
        sections["refs"] = self._render_refs(
            conversation, section_char_budgets["refs"],
        )

        # ── Step 3: Cascade unused budget downward ──
        # Single-pass: for each section in priority order, if it used
        # less than its budget, the surplus flows to the next section.
        # Re-render the next section with the expanded budget.
        surplus = 0
        final_parts: List[str] = []

        for section_name in self._SECTION_ORDER:
            allocated = section_char_budgets[section_name] + surplus
            rendered = sections[section_name]
            used = len(rendered)

            if used < allocated and surplus > 0:
                # Re-render with expanded budget (only if surplus exists)
                rendered = self._render_section(
                    section_name, query, conversation, allocated,
                )
                used = len(rendered)

            surplus = max(0, allocated - used)

            if rendered.strip():
                final_parts.append(rendered)

        if not final_parts:
            return ""

        result = "Conversation Context:\n" + "\n\n".join(final_parts)

        # ── Step 4: Hard cap enforcement (mathematical guarantee) ──
        result_tokens = _estimate_tokens(result)
        if result_tokens > self._token_budget:
            # Truncate from the end — preserves highest-priority content
            max_chars = self._token_budget * 4
            result = result[:max_chars]
            logger.warning(
                "RetrievalCP: hard-capped output from %d to %d tokens",
                result_tokens, _estimate_tokens(result),
            )

        assert _estimate_tokens(result) <= self._token_budget + 1, (
            f"INVARIANT VIOLATED: context {_estimate_tokens(result)} tokens "
            f"exceeds budget {self._token_budget}"
        )

        return result

    # ── Budget allocation ──

    def _allocate_budgets(
        self,
        weights: Dict[str, float],
        total_chars: int,
    ) -> Dict[str, int]:
        """Allocate character budgets per section using weighted base percentages.

        Process:
        1. Multiply base percentages by query-derived weights
        2. Normalize so they sum to 1.0
        3. Distribute total_chars proportionally

        Guarantee: sum(budgets) == total_chars
        """
        raw: Dict[str, float] = {}
        for section in self._SECTION_ORDER:
            raw[section] = self._base_budgets[section] * weights.get(section, 1.0)

        total_raw = sum(raw.values())
        if total_raw == 0:
            # Degenerate case — equal distribution
            per_section = total_chars // len(self._SECTION_ORDER)
            return {s: per_section for s in self._SECTION_ORDER}

        budgets: Dict[str, int] = {}
        allocated = 0
        sections = list(self._SECTION_ORDER)
        for i, section in enumerate(sections):
            if i == len(sections) - 1:
                # Last section gets remainder — guarantees exact sum
                budgets[section] = total_chars - allocated
            else:
                portion = int(total_chars * raw[section] / total_raw)
                budgets[section] = portion
                allocated += portion

        return budgets

    # ── Section renderers ──

    def _render_section(
        self,
        name: str,
        query: str,
        conversation: Optional[ConversationFrame],
        char_budget: int,
    ) -> str:
        """Re-render a section with updated budget (for cascade)."""
        if name == "goals":
            return self._render_goals(conversation, char_budget)
        elif name == "entities":
            return self._render_entities(conversation, char_budget)
        elif name == "memory":
            return self._render_memory(query, char_budget)
        elif name == "turns":
            return self._render_turns(conversation, char_budget)
        elif name == "refs":
            return self._render_refs(conversation, char_budget)
        return ""

    @staticmethod
    def _render_goals(
        conversation: Optional[ConversationFrame],
        char_budget: int,
    ) -> str:
        """Render active goals within budget."""
        if not conversation:
            return ""

        active = conversation.get_active_goals()
        if not active:
            return ""

        lines: List[str] = []
        header = "Active Goals:"
        used = len(header)

        for goal in active[:5]:
            line = f"  - {goal.description[:100]}"
            if used + len(line) + 1 > char_budget:
                break
            lines.append(line)
            used += len(line) + 1  # +1 for newline

        if not lines:
            return ""
        return header + "\n" + "\n".join(lines)

    @staticmethod
    def _render_entities(
        conversation: Optional[ConversationFrame],
        char_budget: int,
    ) -> str:
        """Render entity registry within budget, sorted by recency."""
        if not conversation or not conversation.entity_registry:
            return ""

        # Sort by created_at (newest first)
        sorted_entities = sorted(
            conversation.entity_registry.items(),
            key=lambda kv: kv[1].created_at,
            reverse=True,
        )

        header = "Known Entities:"
        used = len(header)
        lines: List[str] = []

        for key, record in sorted_entities:
            if isinstance(record.value, list):
                line = f"  {key} ({record.type}): {len(record.value)} items"
            else:
                line = f"  {key} ({record.type}): {str(record.value)[:80]}"

            if used + len(line) + 1 > char_budget:
                break
            lines.append(line)
            used += len(line) + 1

        if not lines:
            return ""
        return header + "\n" + "\n".join(lines)

    def _render_memory(self, query: str, char_budget: int) -> str:
        """Render memory episodes within budget."""
        if not self._memory:
            return ""

        episodes = self._memory.retrieve_relevant(
            query=query, top_k=self._memory_top_k,
        )
        if not episodes:
            return ""

        header = "Relevant Memory:"
        used = len(header)
        lines: List[str] = []

        for ep in episodes:
            q = ep.get("query", "?")[:60]
            outcome = ep.get("outcome_summary", "")[:80]
            line = f"  [{q}] → {outcome}"

            if used + len(line) + 1 > char_budget:
                break
            lines.append(line)
            used += len(line) + 1

        if not lines:
            return ""
        return header + "\n" + "\n".join(lines)

    def _render_turns(
        self,
        conversation: Optional[ConversationFrame],
        char_budget: int,
    ) -> str:
        """Render recent turns using the pluggable compactor."""
        if not conversation or not conversation.history:
            return ""

        header = "Recent Conversation:"
        remaining = char_budget - len(header) - 1
        if remaining <= 0:
            return ""

        compacted = self._turn_compactor(conversation.history, remaining)
        if not compacted.strip():
            return ""

        return header + "\n" + compacted

    @staticmethod
    def _render_refs(
        conversation: Optional[ConversationFrame],
        char_budget: int,
    ) -> str:
        """Render resolved references within budget."""
        if not conversation:
            return ""

        if not hasattr(conversation, 'unresolved_references'):
            return ""
        if not conversation.unresolved_references:
            return ""

        refs = conversation.unresolved_references
        if "resolved" not in refs:
            if refs.get("unresolved"):
                note = "Note: User uses referential language but no resolution was possible."
                return note if len(note) <= char_budget else ""
            return ""

        header = "Resolved References:"
        used = len(header)
        lines: List[str] = []

        for r in refs["resolved"]:
            if r.get("ordinal") and r.get("value"):
                line = f'  "{r["ordinal"]}" → {r["value"]}'
            elif r.get("entity_hint"):
                line = f'  reference → {r["entity_hint"]}'
            else:
                continue

            if used + len(line) + 1 > char_budget:
                break
            lines.append(line)
            used += len(line) + 1

        if not lines:
            return ""
        return header + "\n" + "\n".join(lines)
