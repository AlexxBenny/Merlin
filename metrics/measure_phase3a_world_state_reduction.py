# metrics/measure_phase3a_world_state_reduction.py

"""
Phase 3A Metrics: World State Token Reduction.

Measures the effect of FilteredWorldStateProvider on prompt size.
Compares full state (SimpleWorldStateProvider) vs filtered state
for a battery of representative queries.

Metrics per query:
  - full_state_tokens: estimated token count from full WorldState
  - filtered_state_tokens: estimated token count from filtered view
  - reduction_pct: percentage reduction
  - domains_detected: which domains were detected

Output:
  - metrics/phase3a_world_state.jsonl  (per-query)
  - metrics/phase3a_summary.json       (aggregate)

Usage:
    python -m metrics.measure_phase3a_world_state_reduction

No LLM required — pure deterministic measurement.
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from world.state import (
    WorldState, SystemState, HardwareState, SessionState,
    ResourceState, MediaState, TimeState,
)
from world.snapshot import WorldSnapshot
from cortex.world_state_provider import SimpleWorldStateProvider
from cortex.filtered_world_state_provider import FilteredWorldStateProvider

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("phase3a_metrics")

METRICS_DIR = Path(__file__).parent


# ─────────────────────────────────────────────────────────────
# Token estimation
# ─────────────────────────────────────────────────────────────

def estimate_tokens(d: dict) -> int:
    """Rough token estimate from JSON serialization.

    Rule of thumb: 1 token ≈ 4 characters in JSON.
    This is a consistent approximation, not exact.
    Good enough for relative comparison.
    """
    serialized = json.dumps(d, indent=2)
    return len(serialized) // 4


def count_leaf_keys(d, _depth=0) -> int:
    """Count non-None leaf values in nested dict."""
    if not isinstance(d, dict):
        return 1 if d is not None else 0
    return sum(count_leaf_keys(v, _depth + 1) for v in d.values())


# ─────────────────────────────────────────────────────────────
# Test fixture: fully populated WorldState
# ─────────────────────────────────────────────────────────────

def build_full_world_state() -> WorldState:
    """Realistic WorldState with every section populated."""
    return WorldState(
        active_app="Chrome",
        active_window="Google - Generative AI Research - Chrome",
        cwd="C:\\Users\\alex\\Projects\\MERLIN",
        media=MediaState(
            platform="Spotify",
            title="Bohemian Rhapsody",
            artist="Queen",
            is_playing=True,
            is_ad=False,
        ),
        system=SystemState(
            resources=ResourceState(
                cpu_percent=42.3,
                cpu_status="normal",
                memory_percent=67.8,
                memory_status="normal",
                disk_percent=55.0,
            ),
            hardware=HardwareState(
                battery_percent=72.0,
                battery_charging=False,
                battery_status="normal",
                brightness_percent=65,
                volume_percent=40,
                muted=False,
                nightlight_enabled=True,
            ),
            session=SessionState(
                foreground_app="Chrome",
                foreground_window="Google - Generative AI Research - Chrome",
                idle_seconds=5.0,
                open_apps=["Chrome", "VSCode", "Spotify", "Terminal", "Explorer"],
            ),
        ),
        time=TimeState(
            hour=14,
            minute=30,
            day_of_week="Tuesday",
            date="2026-02-18",
        ),
        last_user_focus="browser_tab_3",
    )


# ─────────────────────────────────────────────────────────────
# Test queries (same as compiler baseline + drift-risk queries)
# ─────────────────────────────────────────────────────────────

QUERIES = [
    # Simple single-domain
    {"query": "play music", "category": "simple", "expected_domains": ["media"]},
    {"query": "set brightness to 50", "category": "simple", "expected_domains": ["system"]},
    {"query": "unmute", "category": "simple", "expected_domains": ["system"]},
    {"query": "pause the music", "category": "simple", "expected_domains": ["media"]},
    {"query": "create folder hello on desktop", "category": "simple", "expected_domains": ["fs"]},

    # Multi-domain (union)
    {"query": "unmute and play music", "category": "multi_domain", "expected_domains": ["system", "media"]},
    {"query": "set brightness to 80 and unmute the volume", "category": "multi_domain", "expected_domains": ["system"]},
    {"query": "open chrome and create folder test", "category": "multi_domain", "expected_domains": ["browser", "fs"]},
    {"query": "mute the volume, then play music", "category": "multi_domain", "expected_domains": ["system", "media"]},

    # Cross-domain complex
    {"query": "open chrome, search agentic AI, create folder, play music", "category": "cross_domain", "expected_domains": ["browser", "fs", "media"]},

    # Drift risk (ambiguous keywords — should NOT filter)
    {"query": "open research paper", "category": "drift_risk", "expected_domains": []},
    {"query": "create summary", "category": "drift_risk", "expected_domains": []},
    {"query": "search memory usage", "category": "drift_risk", "expected_domains": []},
    {"query": "tell me a joke", "category": "drift_risk", "expected_domains": []},
    {"query": "play something interesting", "category": "drift_risk", "expected_domains": []},

    # Edge cases (unique keywords)
    {"query": "set brightness to maximum", "category": "edge_case", "expected_domains": ["system"]},
    {"query": "open spotify", "category": "edge_case", "expected_domains": ["media"]},
    {"query": "create a folder named test", "category": "edge_case", "expected_domains": ["fs"]},
]


def main():
    state = build_full_world_state()
    snapshot = WorldSnapshot.build(state=state, recent_events=[])

    simple_provider = SimpleWorldStateProvider()
    filtered_provider = FilteredWorldStateProvider()

    full_state = simple_provider.build_schema(snapshot)
    full_tokens = estimate_tokens(full_state)
    full_keys = count_leaf_keys(full_state)

    logger.info("=" * 60)
    logger.info("Phase 3A: World State Token Reduction Measurement")
    logger.info("=" * 60)
    logger.info("Full state: %d tokens (est), %d leaf keys", full_tokens, full_keys)
    logger.info("Queries: %d", len(QUERIES))
    logger.info("=" * 60)

    results = []
    total_reduction = 0
    filtered_count = 0
    full_fallback_count = 0
    correct_domain_count = 0

    for i, q in enumerate(QUERIES):
        query = q["query"]
        category = q["category"]

        filtered_state = filtered_provider.build_schema(snapshot, query=query)
        filtered_tokens = estimate_tokens(filtered_state)
        filtered_keys = count_leaf_keys(filtered_state)

        is_full = filtered_state == full_state
        reduction_pct = round(100 * (1 - filtered_tokens / full_tokens), 1) if not is_full else 0.0

        # Detect which domains were resolved
        domains = filtered_provider._resolve_domains(query, None)

        record = {
            "query": query,
            "category": category,
            "full_state_tokens": full_tokens,
            "filtered_state_tokens": filtered_tokens,
            "reduction_pct": reduction_pct,
            "full_state_keys": full_keys,
            "filtered_keys": filtered_keys,
            "is_full_state_fallback": is_full,
            "domains_detected": sorted(domains),
            "expected_domains": sorted(q["expected_domains"]),
            "domain_match": sorted(domains) == sorted(q["expected_domains"]),
        }

        results.append(record)

        if is_full:
            full_fallback_count += 1
            logger.info(
                "  [%d/%d] %-55s FULL (fallback) %d tokens",
                i + 1, len(QUERIES), query, filtered_tokens,
            )
        else:
            filtered_count += 1
            total_reduction += reduction_pct
            logger.info(
                "  [%d/%d] %-55s %d → %d tokens (%.0f%% reduction) domains=%s",
                i + 1, len(QUERIES), query,
                full_tokens, filtered_tokens, reduction_pct,
                sorted(domains),
            )

        if record["domain_match"]:
            correct_domain_count += 1

    # ── Write per-query results ──
    jsonl_path = METRICS_DIR / "phase3a_world_state.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # ── Aggregate summary ──
    avg_reduction = round(total_reduction / filtered_count, 1) if filtered_count else 0
    reductions = [r["reduction_pct"] for r in results if not r["is_full_state_fallback"]]
    drift_risk_correct = sum(
        1 for r in results
        if r["category"] == "drift_risk" and r["is_full_state_fallback"]
    )
    drift_risk_total = sum(1 for r in results if r["category"] == "drift_risk")

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "phase": "3A",
        "label": "world_state_token_reduction",
        "full_state_tokens": full_tokens,
        "full_state_keys": full_keys,
        "total_queries": len(QUERIES),
        "filtered_queries": filtered_count,
        "full_fallback_queries": full_fallback_count,
        "avg_token_reduction_pct": avg_reduction,
        "min_token_reduction_pct": round(min(reductions), 1) if reductions else 0,
        "max_token_reduction_pct": round(max(reductions), 1) if reductions else 0,
        "domain_detection_accuracy_pct": round(100 * correct_domain_count / len(QUERIES), 1),
        "drift_risk_correct_pct": round(100 * drift_risk_correct / drift_risk_total, 1) if drift_risk_total else 0,
        "confidence_gate": {
            "ambiguous_threshold": 2,
            "design": "unique_keywords=1_match, ambiguous_keywords=gte2_matches",
        },
        "notes": "No LLM. Pure deterministic measurement of FilteredWorldStateProvider.",
    }

    summary_path = METRICS_DIR / "phase3a_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # ── Console summary ──
    logger.info("=" * 60)
    logger.info("PHASE 3A RESULTS")
    logger.info("=" * 60)
    logger.info("Full state: %d tokens | %d leaf keys", full_tokens, full_keys)
    logger.info("Filtered queries: %d/%d", filtered_count, len(QUERIES))
    logger.info("Full fallback queries: %d/%d", full_fallback_count, len(QUERIES))
    logger.info("Avg token reduction: %.1f%%", avg_reduction)
    if reductions:
        logger.info("Min/Max reduction: %.1f%% / %.1f%%", min(reductions), max(reductions))
    logger.info("Domain detection accuracy: %.1f%%", summary["domain_detection_accuracy_pct"])
    logger.info("Drift risk protection: %d/%d correct (%.1f%%)",
                drift_risk_correct, drift_risk_total, summary["drift_risk_correct_pct"])
    logger.info("=" * 60)
    logger.info("Saved: %s", jsonl_path)
    logger.info("Saved: %s", summary_path)


if __name__ == "__main__":
    main()
