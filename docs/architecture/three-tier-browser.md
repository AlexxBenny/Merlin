# Three-Tier Browser Execution

MERLIN's browser interaction follows a strict capability hierarchy that minimizes LLM usage and maximizes determinism.

## Architecture

```
User: "play the howard stark video"
    │
    ▼
  Compiler
    │
    ▼
  Entity Resolution (BEFORE execution gate)
    │
    ├── Found clearly? → Tier 1 (deterministic click)
    ├── Ambiguous?     → Ask user for clarification
    ├── Not found?     → Recovery recompile (1 attempt)
    └── Recompile fails? → Tier 3 (autonomous, last resort)
```

## Tier 1 — Deterministic Skills

No semantic reasoning required. Direct skill execution.

| Skill | Example Input | Action |
|-------|--------------|--------|
| `browser.scroll` | "scroll down" | `controller.scroll_page("down")` |
| `browser.click` | "click entity 3" | `controller.click(backend_node_id)` |
| `browser.navigate` | "open youtube.com" | `controller.navigate("https://youtube.com")` |
| `browser.fill` | "type ironman in search" | `controller.fill(entity, "ironman")` |
| `browser.go_back` | "go back" | `controller.go_back()` |
| `browser.go_forward` | "go forward" | `controller.go_forward()` |

**Scroll normalization**: Direction synonyms are normalized: "above"/"top"/"upward" → "up", everything else → "down".

## Tier 2 — Entity Resolution

Requires page understanding but not planning. Resolves text references to specific DOM elements.

### Resolution Flow

```
entity_ref = "howard stark"
    │
    ▼
EntityResolver._resolve_browser_entity()
    │
    ├── Tokenize query: ["howard", "stark"]
    ├── Tokenize each entity text
    ├── Compute cosine similarity per entity
    │
    │   Entity                Score
    │   Howard Stark Speech   0.94  ← winner
    │   Tony Stark Lab        0.52
    │   Ironman Scene         0.07
    │
    ├── Score > 0.55 and clear winner?
    │     → RESOLVED: entity_index=2, _resolved_entity_text="Howard Stark Speech"
    │
    ├── Score < 0.55?
    │     → NOT_FOUND: trigger recovery recompile
    │
    └── Second score > 0.8 × top?
          → AMBIGUOUS: ask user which one
```

### Cosine Similarity Scoring

Pure token-based cosine similarity. No TF-IDF (unstable with small entity sets).

```
score = dot(query_vec, entity_vec) / (|query_vec| × |entity_vec|)
```

Tokens are lowercased with stopwords removed ("the", "a", "and", "in", "on", "to", "of", "for", "is").

### Thresholds

| Condition | Threshold | Result |
|-----------|-----------|--------|
| Top score < 0.55 | NOT_FOUND | Recovery recompile |
| Second > 0.8 × top | AMBIGUOUS | Ask user |
| Clear winner | RESOLVED | Deterministic click |

## Tier 3 — Autonomous Agent

For complex tasks requiring multi-step exploration on unknown page structures.

**Trigger**: Only when Tier 1/2 cannot handle the task, or after failed recovery recompile.

Uses `browser.autonomous_task` which delegates to the `browser-use` agent for LLM-driven page interaction.

**Examples**: "find the cheapest MacBook and compare prices", "fill out this form with my details"

## Index Drift Protection

Between entity resolution and skill execution, the DOM can mutate. MERLIN protects against this:

1. **Fresh snapshot**: `browser.click` uses `get_snapshot(cached=False)` — always reads live DOM
2. **Text verification**: Resolver passes `_resolved_entity_text`. Skill verifies entity at resolved index still matches
3. **Text-based fallback**: If index drifted, searches all entities by text to find the correct one

```
Resolver: entity_index=1, _resolved_entity_text="Howard Stark Speech"
    │
    ▼
Skill: get_snapshot(cached=False)  # Fresh DOM
    │
    ├── Entity at index 1 = "Howard Stark Speech"? → ✓ Click it
    │
    └── Entity at index 1 = "New Element"?
          → Search all entities for "Howard Stark Speech"
          → Found at index 4 → Click that instead
```

## Entity Source of Truth

**Single source**: `WorldState.browser.top_entities`

- Populated by `BrowserSource` event polling → `WorldEvent` → `WorldState.from_events()`
- Up to 10 entities stored as `{index, type, text}`
- Compiler receives entities in prompt context
- EntityResolver reads from world state, NOT from browser session

## Recovery Pipeline

```
NOT_FOUND entity
    │
    ▼
Recovery recompile (1 attempt max)
    → Compiler receives: execution_failures = ["entity 'X' not found on page"]
    → Generates alternative plan (e.g., search instead of click)
    │
    ├── New plan works? → Execute
    └── Still fails? → Ask user or autonomous fallback
```

## Browser Health Check

Uses CDP-level signals (not Playwright internals) to verify browser liveness:

```python
is_alive = browser._playwright_browser.is_connected()
         and browser._cdp_session is not None
         and browser._cdp_session._connection
         and not browser._cdp_session._connection._closed
```

This prevents ghost sessions where Playwright reports the browser as alive but CDP commands fail.
