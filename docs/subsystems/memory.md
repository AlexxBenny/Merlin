# Memory System

**Location**: `memory/`

Persistent user knowledge. Injected as first-class context into LLM prompts.

## Components

### UserKnowledgeStore (`memory/user_knowledge.py`)

Structured user knowledge with typed categories:

| Category | Purpose | Example |
|----------|---------|---------|
| `preferences` | User likes/dislikes | `{"key": "music_genre", "value": "jazz"}` |
| `facts` | Known information | `{"key": "name", "value": "Alex"}` |
| `traits` | Personality characteristics | `{"key": "communication_style", "value": "brief"}` |
| `policies` | Behavioral rules | `{"key": "always_confirm_delete", "value": true}` |
| `relationships` | People/entity connections | `{"key": "sister", "value": "Sarah"}` |

**Persistence**: JSON file at `state/user_knowledge.json`

**Integration points**:
- **Coordinator prompt**: Injected as structured context for reasoning
- **PreferenceResolver**: Preferences applied to skill parameters
- **Proactive policy eval**: Policies evaluated by RuntimeEventLoop

### MemoryStore (`memory/store.py`)

Simple key-value persistent storage backend.

## Memory Skills

Skills that allow the user to teach MERLIN:

| Skill | Usage |
|-------|-------|
| `memory.set_preference` | "I prefer dark mode" |
| `memory.set_fact` | "My name is Alex" |
| `memory.add_policy` | "Always ask before deleting files" |
| `memory.get_preference` | "What's my preferred music genre?" |

## Data Flow

```
User: "I prefer jazz music"
    │
    ▼
Compiler → memory.set_preference(key="music_genre", value="jazz")
    │
    ▼
Skill executes → UserKnowledgeStore.set_preference()
    │
    ▼
Persisted to state/user_knowledge.json
    │
    ▼
Next request: Coordinator prompt includes "Preferences: music_genre=jazz"
    │
    ▼
PreferenceResolver applies preference to relevant skill params
```
