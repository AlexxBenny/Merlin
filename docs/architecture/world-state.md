# World State System

The single source of truth for all environmental state in MERLIN.

**Location**: `world/`

## Core Principle

> World state is **derived, never mutated directly**. The only mutable structure is the append-only `WorldTimeline`. Everything else is computed from it.

## Components

### WorldTimeline (`world/timeline.py`)

Append-only event log. Thread-safe via `_lock`.

```python
class WorldTimeline:
    def emit(source, event_type, payload)  # Append event
    def all_events() -> List[WorldEvent]   # Read all events
    def events_since(timestamp) -> List    # Read recent events
```

### WorldEvent (`world/timeline.py`)

```python
class WorldEvent(BaseModel):
    timestamp: float    # Unix timestamp
    source: str         # "skill.browser", "system", "time", etc.
    type: str           # Event type identifier
    payload: Dict       # Event-specific data
```

### WorldState (`world/state.py`)

Pure reducer: `events → state`. Stateless function — same events always produce same state.

```python
state = WorldState.from_events(timeline.all_events())
```

**State domains**:

| Domain | Model | Key Fields |
|--------|-------|-----------|
| `browser` | `BrowserWorldState` | `active`, `url`, `title`, `entity_count`, `top_entities` |
| `media` | `MediaState` | `platform`, `title`, `artist`, `is_playing`, `is_ad` |
| `system` | `SystemState` | `hardware` (battery, volume, brightness), `resources` (CPU, mem), `session` (apps, idle) |
| `time` | `TimeState` | `hour`, `minute`, `day_of_week`, `date` |
| `visible_lists` | `Dict[str, VisibleList]` | List items rendered to user |

### WorldSnapshot (`world/snapshot.py`)

Frozen snapshot passed to skills. Lightweight wrapper around WorldState.

### WorldResolver (`world/resolver.py`)

Generates a JSON schema of current world state for LLM prompt injection. Uses `FilteredWorldStateProvider` to minimize token usage.

## Event Sources (`runtime/sources/`)

Each source polls a system subsystem and emits events on state change.

### SystemSource (`runtime/sources/system.py`)

| Event Type | Trigger | Payload |
|-----------|---------|---------|
| `system_state_snapshot` | Bootstrap | CPU, memory, battery, volume, brightness, muted |
| `cpu_high` / `cpu_normal` | CPU > 80% or recovers | `value` |
| `memory_pressure` / `memory_normal` | Memory > 80% | `value` |
| `battery_low` / `battery_critical` | < 20% / < 10% | `value` |
| `battery_charging` | Charger connected | `value` |
| `brightness_changed` | User changes brightness | `value` |
| `volume_changed` / `mute_changed` | Volume/mute change | `value` / `muted` |
| `app_focused` | Foreground app changes | `app`, `window` |
| `idle_detected` / `idle_ended` | User idle > threshold | `seconds` |
| `app_launched` / `app_closed` | App lifecycle | `app`, `pid` |

### MediaSource (`runtime/sources/media.py`)

| Event Type | Trigger |
|-----------|---------|
| `media_state_snapshot` | Bootstrap |
| `media_started` / `media_paused` / `media_stopped` | Playback state change |
| `track_changed` | New track |
| `ad_detected` | Ad playing |

### TimeSource (`runtime/sources/time.py`)

| Event Type | Trigger |
|-----------|---------|
| `time_tick` | Bootstrap |
| `hour_changed` | New hour |
| `date_changed` | New day |

### BrowserSource (`runtime/sources/browser.py`)

| Event Type | Trigger |
|-----------|---------|
| `browser_state_snapshot` | Bootstrap (browser alive) |
| `browser_page_changed` | URL, title, or tab count changes |
| `browser_entities_refreshed` | Entity count changes (scroll/dynamic load) |
| `browser_disconnected` | Browser connection lost |

**Key**: `top_entities` (up to 10) propagate through events → WorldState → compiler prompt, forming the single source of truth for browser entities.

## Data Flow

```
Event Sources (poll every 2-30s)
    │
    ▼
WorldTimeline (append-only log)
    │
    ▼
WorldState.from_events() (pure reducer)
    │
    ├──▶ WorldResolver → LLM prompt context
    ├──▶ CognitiveCoordinator → world-aware reasoning
    ├──▶ MissionCortex → entity-aware compilation
    └──▶ EntityResolver → browser entity matching
```
