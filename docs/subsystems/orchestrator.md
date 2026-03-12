# Orchestrator

**Location**: `orchestrator/`

The control loop between cortex and executor. Handles the messy reality of entity resolution, failure recovery, and replanning.

## MissionOrchestrator (`orchestrator/mission_orchestrator.py`)

### Pipeline

```
Compiled MissionPlan
    │
    ▼
1. ParameterResolver.resolve_plan()      # Semantic types → concrete values
    │
    ▼
2. PreferenceResolver.resolve_plan()     # User preferences → skill params
    │
    ▼
3. EntityResolver.resolve_plan(world_snapshot)
    │                                     # App entities: app_target → app_id
    │                                     # Browser entities: entity_ref → entity_index
    │
    ├── Violations found?
    │     ├── App NOT_FOUND/AMBIGUOUS → ask user (clarification)
    │     └── Browser NOT_FOUND → recovery recompile
    │
    ▼
4. MissionExecutor.execute(plan)         # DAG walk
    │
    ▼
5. Results → ReportBuilder              # Narrate to user
```

### Recovery Recompile

When a browser entity is not found, the orchestrator triggers a recompile instead of immediate escalation:

1. `EntityResolutionError` caught with `not_found_browser` violations
2. Separate browser NOT_FOUND from app NOT_FOUND
3. Build `execution_failures` context for compiler
4. `cortex.compile()` with failure context → new plan (e.g., search instead of click)
5. Limited to **1 recompile attempt** per entity failure
6. If recompile also fails → ask user or fall back to autonomous

### World Snapshot Wiring

All three call sites pass `world_snapshot` to `EntityResolver.resolve_plan()`:
- `MissionOrchestrator.resolve_plan()` (mission path)
- `merlin.py` reflex path
- Recovery recompile path

The world snapshot is built from `WorldState.from_events(timeline.all_events())` for fresh browser entity data.

### Error Handling

| Error Type | Action |
|-----------|--------|
| `ParameterError` | Report to user |
| `EntityResolutionError` (app) | Ask user for clarification |
| `EntityResolutionError` (browser) | Recovery recompile |
| Execution failure | MetaCognition analysis → partial/failed report |
