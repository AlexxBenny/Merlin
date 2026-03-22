# Cognitive Pipeline

The full lifecycle of a user request through MERLIN.

## Entry Point

Every interaction starts in `merlin.py` — the Conductor. It:
1. Receives a `Percept` from perception/or input adapters
2. Routes through the cognitive pipeline via `handle_percept()`

## Path 1: Reflex (< 100ms)

```
Percept → ReflexEngine.match(text)
  → Pattern + verb match against SkillRegistry
  → If match: extract params → ParameterResolver → EntityResolver → execute skill directly
  → If no match: fall through to cognitive path
```

**When**: Simple, unambiguous commands: "mute", "scroll down", "pause music", "volume 50"

**Key components**:
- `ReflexEngine` (`runtime/reflex_engine.py`) — deterministic pattern matching
- `ParameterResolver` — resolves semantic types (e.g., `volume_level` → integer)
- `EntityResolver` — resolves app/browser entities

## Path 2: Cognitive (MISSION)

```
Percept
  → BrainCore.route(percept)
    → CognitiveRoute: REFLEX | MULTI_REFLEX | MISSION | REFUSE
  
  → EscalationPolicy.classify(tier)
    → CognitiveTier: SIMPLE | MODERATE | COMPLEX
  
  → CognitiveCoordinator.reason(query, world_state, conversation)
    → CoordinatorMode: DIRECT_ANSWER | SKILL_PLAN | REASONED_PLAN
```

### DIRECT_ANSWER
Coordinator answers directly (no skills needed). Examples: "what time is it?", "how's the battery?"

### SKILL_PLAN / REASONED_PLAN
Falls through to compilation:

```
  → MissionCortex.compile(query, context)
    → IntentEngine.discover_skills(query)        # Phase 1: candidate skills
    → ScoredDiscovery.score_candidates()          # TF-IDF ranking
    → _build_session_context()                    # World state + entities + templates
    → LLM call → JSON plan                       # Single LLM compilation
    → Validators.validate(plan)                   # Cycle check, contract enforcement
    → MissionPlan (IR v1)                         # Frozen DAG
  
  → MissionOrchestrator.resolve_and_execute(plan)
    → ParameterResolver.resolve_plan()            # Semantic types
    → PreferenceResolver.resolve_plan()           # User preferences
    → EntityResolver.resolve_plan(world_snapshot)  # App + browser entities
    → MissionExecutor.execute(plan)               # DAG walk
      → For each node (respecting depends_on):
        → ExecutionSupervisor.pre_step()          # Pre-validation
        → Skill.execute(inputs, world, snapshot)  # Actual work
        → ExecutionSupervisor.post_step()         # Post-validation
        → Emit WorldEvents                        # State updates
  
  → ReportBuilder.build(results)                  # LLM narration
  → OutputChannel.send(narration)                 # Console + TTS
```

## Path 3: Pending Mission (Clarification)

When the system needs more information:

```
  → EscalationPolicy → CLARIFY
  → PendingMission created (stores original percept + snapshot)
  → User responds
  → Original percept + response merged
  → Re-enters pipeline
```

## Path 4: Scheduled Execution

For time-deferred commands ("remind me in 5 minutes"):

```
  → MissionCortex decomposes → executable/scheduled/informational/vague intents
  → TemporalResolver → absolute timestamps
  → Scheduled clauses route to TickSchedulerManager via merlin.py
  → TickSchedulerManager executes at scheduled time
  → Job stored in JsonTaskStore (survives restart)
```

## Error Recovery

```
Skill execution fails
  → MetaCognitionEngine.analyze_outcome()
  → ExecutionSupervisor determines RepairAction:
    → CONTINUE: skip and proceed
    → RETRY: retry with backoff (within declared limits)
    → FAIL: report failure
  
Entity NOT_FOUND (browser):
  → Recovery recompile: cortex receives execution_failures context
  → New plan generated (e.g., search instead of click)
  → Limited to 1 recompile attempt
  → If still fails: ask user or autonomous fallback
```

## Parallelism

```
MissionPlan with independent branches:
  Node A (no deps) ─┐
  Node B (no deps) ─┤→ Execute in parallel
  Node C (dep: A)  ─┘→ Wait for A, then execute
```

- `ExecutionMode.foreground` — blocks mission completion
- `ExecutionMode.background` — non-blocking, failure logged only
- `ExecutionMode.side_effect` — non-blocking, failure ignored
