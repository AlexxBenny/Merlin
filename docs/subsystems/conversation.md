# Conversation System

**Location**: `conversation/`

Turn-by-turn conversation tracking with entity history for reference resolution.

## Components

### ConversationFrame (`conversation/frame.py`)

Maintains conversation history and context:

- **Turn history**: List of `(role, content)` tuples
- **Entity tracking**: Last-mentioned entities for ordinal resolution
- **Context window**: Manages how much history is sent to LLMs
- **Topic tracking**: Current conversation topic for continuations

**Methods**:
| Method | Purpose |
|--------|---------|
| `append_turn(role, content)` | Add a turn to history |
| `get_context(max_turns)` | Get recent turns for LLM prompt |
| `track_entities(entities)` | Store entities for ordinal resolution |
| `get_last_entities()` | Retrieve for "the first one" resolution |

### Outcome (`conversation/outcome.py`)

Structured result of a mission execution:

| Field | Description |
|-------|-----------|
| `status` | `success`, `partial`, `failed` |
| `results` | Dict of node_id → skill outputs |
| `errors` | List of error descriptions |
| `narration` | Human-readable summary |
