# metrics/measure_phase3b_context_reduction.py

"""
Phase 3B Metrics: Context Token Reduction Measurement

Measures:
- Token count from SimpleContextProvider vs RetrievalContextProvider
- Token budget adherence (hard cap never exceeded)
- Dynamic allocation effectiveness (query heuristics)
- Scaling behavior across conversation lengths

Output:
- metrics/phase3b_context.jsonl  — per-scenario results
- metrics/phase3b_summary.json  — aggregated summary
"""

import json
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from conversation.frame import ConversationFrame
from cortex.context_provider import (
    SimpleContextProvider,
    RetrievalContextProvider,
    _estimate_tokens,
)
from memory.store import ListMemoryStore

# ─────────────────────────────────────────────────────────────
# Scenario definitions
# ─────────────────────────────────────────────────────────────

def make_memory(n: int) -> ListMemoryStore:
    store = ListMemoryStore()
    for i in range(n):
        store.store_episode(
            mission_id=f"m{i}",
            query=f"User asked to perform action {i} on the system",
            outcome_summary=f"Completed action {i}: created folder, opened app, adjusted volume",
        )
    return store


def make_frame(
    n_turns: int,
    n_entities: int = 0,
    n_goals: int = 0,
    turn_length: int = 100,
) -> ConversationFrame:
    frame = ConversationFrame()
    for i in range(n_turns):
        role = "user" if i % 2 == 0 else "assistant"
        text = f"Turn {i}: " + "x" * turn_length
        frame.append_turn(role, text)
    for i in range(n_entities):
        frame.register_entity(
            key=f"entity_{i}",
            value=f"value for entity {i} with some extra detail text",
            entity_type="test",
            source_mission=f"mission_{i}",
        )
    for i in range(n_goals):
        frame.add_goal(f"goal_{i}", f"Complete the project phase {i}")
    return frame


SCENARIOS = [
    # (name, query, n_turns, n_entities, n_goals, n_episodes, turn_length)
    # ── Baseline: minimal
    ("minimal_empty", "hello", 0, 0, 0, 0, 50),
    ("minimal_1turn", "set volume to 50", 1, 0, 0, 0, 50),

    # ── Short conversation
    ("short_5turns", "play some music", 5, 2, 1, 0, 80),
    ("short_5turns_memory", "play some music", 5, 2, 1, 5, 80),

    # ── Medium conversation
    ("medium_20turns", "open chrome and search for AI", 20, 5, 2, 10, 100),
    ("medium_20turns_long", "open chrome and search for AI", 20, 5, 2, 10, 300),

    # ── Heavy conversation
    ("heavy_50turns", "create folder structure", 50, 10, 3, 20, 150),
    ("heavy_100turns", "set brightness to max", 100, 20, 5, 50, 200),

    # ── Extreme: stress test
    ("extreme_200turns", "unmute the system", 200, 50, 5, 50, 300),

    # ── Dynamic allocation: goal signal
    ("goal_signal", "continue working on the plan", 20, 5, 3, 10, 100),

    # ── Dynamic allocation: memory signal
    ("memory_signal", "do what we did yesterday", 20, 5, 2, 10, 100),

    # ── Dynamic allocation: short followup
    ("short_followup", "yes please", 20, 5, 2, 10, 100),

    # ── Dynamic allocation: neutral
    ("neutral_query", "set the volume to fifty percent", 20, 5, 2, 10, 100),
]


def run_measurements():
    results = []
    simple = SimpleContextProvider()

    for name, query, n_turns, n_entities, n_goals, n_episodes, turn_len in SCENARIOS:
        frame = make_frame(n_turns, n_entities, n_goals, turn_len)
        memory = make_memory(n_episodes) if n_episodes > 0 else None
        retrieval = RetrievalContextProvider(
            memory=memory,
            token_budget=800,
        )

        # ── SimpleContextProvider ──
        simple_result = simple.build_context(query, frame, {})
        simple_tokens = _estimate_tokens(simple_result)

        # ── RetrievalContextProvider ──
        retrieval_result = retrieval.build_context(query, frame, {})
        retrieval_tokens = _estimate_tokens(retrieval_result)

        # ── Budget check ──
        budget_adhered = retrieval_tokens <= 800

        # ── Reduction ──
        if simple_tokens > 0:
            reduction_pct = round(
                (1 - retrieval_tokens / simple_tokens) * 100, 1,
            )
        else:
            reduction_pct = 0.0

        record = {
            "scenario": name,
            "query": query,
            "n_turns": n_turns,
            "n_entities": n_entities,
            "n_goals": n_goals,
            "n_episodes": n_episodes,
            "simple_tokens": simple_tokens,
            "retrieval_tokens": retrieval_tokens,
            "reduction_pct": reduction_pct,
            "budget_adhered": budget_adhered,
            "budget": 800,
        }
        results.append(record)

        status = "✅" if budget_adhered else "❌"
        print(
            f"  {status} {name:25s} | "
            f"Simple: {simple_tokens:5d} | "
            f"Retrieval: {retrieval_tokens:5d} | "
            f"Reduction: {reduction_pct:6.1f}%"
        )

    return results


def write_results(results):
    os.makedirs("metrics", exist_ok=True)

    # JSONL
    jsonl_path = "metrics/phase3b_context.jsonl"
    with open(jsonl_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # Summary
    total = len(results)
    budget_pass = sum(1 for r in results if r["budget_adhered"])
    reductions = [r["reduction_pct"] for r in results if r["simple_tokens"] > 0]
    bounded_scenarios = [r for r in results if r["simple_tokens"] > 800]
    max_retrieval = max(r["retrieval_tokens"] for r in results)

    summary = {
        "phase": "3B",
        "component": "RetrievalContextProvider",
        "total_scenarios": total,
        "budget_adherence": f"{budget_pass}/{total}",
        "budget_adherence_pct": round(budget_pass / total * 100, 1),
        "avg_reduction_pct": round(sum(reductions) / len(reductions), 1) if reductions else 0,
        "min_reduction_pct": min(reductions) if reductions else 0,
        "max_reduction_pct": max(reductions) if reductions else 0,
        "max_retrieval_tokens": max_retrieval,
        "token_budget": 800,
        "hard_cap_guaranteed": all(r["budget_adhered"] for r in results),
        "scenarios_where_simple_exceeded_budget": len(bounded_scenarios),
        "worst_simple_tokens": max(r["simple_tokens"] for r in results) if results else 0,
    }

    summary_path = "metrics/phase3b_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    print("=" * 70)
    print("Phase 3B: Context Token Reduction Measurement")
    print("=" * 70)
    print()
    print("Comparing: SimpleContextProvider vs RetrievalContextProvider")
    print(f"Budget: 800 tokens")
    print()

    results = run_measurements()

    print()
    print("-" * 70)
    summary = write_results(results)

    print()
    print("Summary:")
    print(f"  Budget adherence: {summary['budget_adherence']} ({summary['budget_adherence_pct']}%)")
    print(f"  Hard cap guaranteed: {summary['hard_cap_guaranteed']}")
    print(f"  Avg reduction: {summary['avg_reduction_pct']}%")
    print(f"  Max retrieval tokens: {summary['max_retrieval_tokens']} / {summary['token_budget']}")
    print(f"  Worst Simple tokens: {summary['worst_simple_tokens']}")
    print(f"  Scenarios where Simple > budget: {summary['scenarios_where_simple_exceeded_budget']}")
    print()
    print(f"Results: metrics/phase3b_context.jsonl")
    print(f"Summary: metrics/phase3b_summary.json")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
