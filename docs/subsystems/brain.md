# Brain — Nervous System Core

**Location**: `brain/`

The brainstem of MERLIN. Makes constant-time routing decisions. Almost never changes.

## Components

### BrainCore (`brain/core.py`)

Routes incoming percepts to the correct cognitive path.

```python
class BrainCore:
    def route(percept: Percept) -> CognitiveRoute
```

**Routes**:
| Route | When | Handler |
|-------|------|---------|
| `REFLEX` | Simple unambiguous command | ReflexEngine (no LLM) |
| `MULTI_REFLEX` | Multiple deterministic commands in one utterance | ReflexEngine (multi-clause fast path) |
| `MISSION` | Any query requiring planning/reasoning/coordination | Full cognitive pipeline |
| `REFUSE` | Safety gate / disallowed intent | Immediate refusal path |

**Percept model**:
```python
@dataclass
class Percept:
    modality: str  # "text", "speech", "vision"
    payload: str
    confidence: float
    timestamp: float
```

### EscalationPolicy (`brain/escalation_policy.py`)

Classifies query complexity into tiers for resource allocation.

**Tiers**:
| Tier | Characteristics | Example |
|------|---------------|---------|
| `SIMPLE` | Single intent, no reasoning | "mute", "open chrome" |
| `MODERATE` | Multi-intent or needs context | "pause music and open youtube" |
| `COMPLEX` | Requires planning/reasoning | "find cheapest MacBook, save specs to file" |

**Decisions**: `PROCEED`, `CLARIFY`, `REJECT`

`HeuristicTierClassifier` — rule-based classifier (no LLM). Examines query length, conjunction count, and domain spans.

### StructuralClassifier (`brain/structural_classifier.py`)

Detects speech acts and intent structure.

**Speech Act Types**: `SpeechActType` enum:
- `COMMAND` — "open youtube"
- `QUESTION` — "what's the battery level?"
- `STATEMENT` — "my favorite color is blue"
- `CONTINUATION` — "the ironman one"
- `CONFIRMATION` — "yes", "do it"
- `NEGATION` — "no", "cancel"

### OrdinalResolver (`brain/ordinal_resolver.py`)

Resolves ordinal references in conversation context.

| Input | Resolution |
|-------|-----------|
| "the first one" | Index 0 from last entity list |
| "the second result" | Index 1 from last entity list |
| "not that, the other one" | Next candidate |

Uses `ConversationFrame` entity history for reference resolution.

## Design Rules

- **No reasoning** — only pattern matching and constant-time classification
- **No state** — reads world snapshot but does not mutate
- **No LLM calls** — everything is deterministic
- **Almost frozen** — this layer should rarely change
