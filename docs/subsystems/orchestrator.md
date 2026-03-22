# Orchestrator

**Location**: `orchestrator/`

The control loop between cortex and executor. Handles entity resolution, inline recovery, and failure escalation.

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
    │                                     # File entities: bare filename → resolved path + anchor
    │
    ├── Violations found?
    │     ├── App NOT_FOUND/AMBIGUOUS → ask user (clarification)
    │     ├── Browser NOT_FOUND → recovery recompile
    │     └── File NOT_FOUND/AMBIGUOUS → structured options
    │
    ▼
4. ExecutionSupervisor.run(plan, snapshot, decision_engine=decision_engine)
    │                                     # Passes DecisionEngine for inline recovery
    │
    ▼
5. Results → ReportBuilder              # Narrate to user
```

### Inline Recovery (at execution time)

When a node fails during execution, the supervisor can perform inline recovery before escalating:

1. `MetaCognition.classify()` → `FailureVerdict`
2. `DecisionEngine.decide(verdict, snapshot)` → `ActionDecision | EscalationDecision`
3. If `ActionDecision` with safe effect_type:
   - Build ephemeral `MissionNode` from decision
   - Execute through `executor.execute_node()` (same 10-step pipeline)
   - If succeeds → retry original node (preconditions re-evaluated)
4. Bounded: `MAX_INLINE_RECOVERY = 2` per node, deduped, safety-gated

### Recovery Recompile (for entity failures)

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
| `EntityResolutionError` (file) | Structured options for user |
| Execution failure (inline recoverable) | DecisionEngine → inline recovery → retry |
| Execution failure (not recoverable) | MetaCognition analysis → partial/failed report |
