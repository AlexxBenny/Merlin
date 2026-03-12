# Runtime System

**Location**: `runtime/`

The operational backbone. Manages the event loop, reflex matching, scheduling, and task persistence.

## Components

### RuntimeEventLoop (`runtime/event_loop.py`)

Central polling loop. Runs on a background thread.

**Lifecycle**:
1. **Bootstrap**: All event sources emit initial state → WorldTimeline
2. **Poll loop** (every iteration):
   - Poll each event source
   - Emit new events to WorldTimeline
   - Tick scheduler (check for due jobs)
   - Process completion queue

### ReflexEngine (`runtime/reflex_engine.py`)

Sub-100ms deterministic skill matching. Bypasses the entire LLM pipeline.

**Matching strategy**:
1. Parse input for verbs and keywords
2. Match against skill contracts (intent_verbs, intent_keywords)
3. Extract parameters from input text
4. If confidence > threshold → execute directly

**Example**: "mute" → `system.mute` skill, no LLM call.

### TickSchedulerManager (`runtime/tick_scheduler.py`)

Persistent job scheduler. Executes delayed and recurring tasks.

**Task types**:
| Type | Behavior |
|------|---------|
| `once` | Execute at `schedule_at` timestamp, then complete |
| `recurring` | Execute every `interval_seconds`, up to `max_repeats` |

**Features**:
- Condition evaluation per tick
- Concurrent execution limit
- Retry with backoff (up to `max_retries`)
- Survives process restart (JSON persistence)

### TaskStore / JsonTaskStore (`runtime/task_store.py`, `runtime/json_task_store.py`)

Job persistence layer.

**JsonTaskStore**: File-based persistence in `state/jobs/`. Each job is a JSON file.

**Operations**: `save()`, `load()`, `delete()`, `list_all()`, `list_by_status()`

### TemporalResolver (`runtime/temporal_resolver.py`)

Natural language → absolute timestamps.

| Input | Output |
|-------|--------|
| "in 5 minutes" | `now + 300s` |
| "tomorrow at 3pm" | `next_day_at_15:00` UTC |
| "every 30 minutes" | `interval_seconds=1800` |
| "at 10:30" | `today_or_next_at_10:30` UTC |

### AttentionManager (`runtime/attention.py`)

Mission lifecycle tracking and concurrent mission limits.

**State machine**: `IDLE` → `ACTIVE` → `COMPLETED` / `FAILED`

Tracks `MissionState` for each active mission. Prevents mission overlap.

### CompletionQueue (`runtime/completion_event.py`)

Async completion events for background missions. Allows the event loop to process completed background tasks.

## Event Sources (`runtime/sources/`)

| Source | File | Poll Interval | Purpose |
|--------|------|--------------|---------|
| `SystemSource` | `system.py` | 30s | CPU, memory, battery, volume, brightness, apps, idle |
| `MediaSource` | `media.py` | 2s | Now-playing, track changes, ads |
| `TimeSource` | `time.py` | 60s | Time ticks, hour/date changes |
| `BrowserSource` | `browser.py` | 2s | Page state, entities, connection |

Each source implements `EventSource` base class with `bootstrap()` and `poll()` methods.
