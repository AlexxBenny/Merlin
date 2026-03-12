# Memory Skills

**Location**: `skills/memory/`

4 skills for teaching MERLIN about the user.

## Skills

### `memory.set_preference` (`memory_skills.py`)

Store a user preference.

| Input | Description | Example |
|-------|-----------|---------|
| `key` | Preference name | "music_genre" |
| `value` | Preference value | "jazz" |

### `memory.get_preference` (`memory_skills.py`)

Retrieve a stored preference.

| Input | Description |
|-------|-----------|
| `key` | Preference name to look up |

### `memory.set_fact` (`memory_skills.py`)

Store a fact about the user.

| Input | Description | Example |
|-------|-----------|---------|
| `key` | Fact name | "name" |
| `value` | Fact value | "Alex" |

### `memory.add_policy` (`memory_skills.py`)

Add a behavioral policy.

| Input | Description | Example |
|-------|-----------|---------|
| `key` | Policy name | "always_confirm_delete" |
| `value` | Policy rule | true |

## How Memory Works

```
"I prefer dark mode"
  → Compiler → memory.set_preference(key="theme", value="dark mode")
  → UserKnowledgeStore persists to state/user_knowledge.json
  → Future requests: Coordinator sees "Preferences: theme=dark mode"
  → PreferenceResolver applies preference to relevant skill parameters
```

All 4 skills are implemented in a single file (`memory_skills.py`) using the shared `UserKnowledgeStore`.
