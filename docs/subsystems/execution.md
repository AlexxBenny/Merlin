# Execution Engine

**Location**: `execution/`

The spinal cord. Executes Mission DAGs deterministically.

## Components

### MissionExecutor (`execution/executor.py`)

DAG walker. Core execution loop:

1. Topological sort of nodes by `depends_on`
2. Independent nodes execute in parallel (up to `max_workers`)
3. Dependent nodes wait for upstream completion
4. Output references resolved at runtime
5. Conditions evaluated at scheduling time

**Execution modes**:
| Mode | Blocking | On Failure |
|------|----------|-----------|
| `foreground` | Yes | Fails mission |
| `background` | No | Logged only |
| `side_effect` | No | Ignored |

### ExecutionSupervisor (`execution/supervisor.py`)

Step-level guard with pre/post validation.

**Pre-step**: Validate inputs, check world state preconditions.

**Post-step**: Validate outputs, detect anomalies.

**Repair actions**:
| Action | When |
|--------|------|
| `CONTINUE` | Non-critical failure, skip and proceed |
| `RETRY` | Transient failure, retry with backoff |
| `FAIL` | Critical failure, stop execution |

### MetaCognitionEngine (`execution/metacognition.py`)

Outcome analysis after execution:
- Success: all foreground nodes completed
- Partial: some nodes failed, some succeeded
- Failure: critical node(s) failed

Generates structured `OutcomeAnalysis` for the ReportBuilder.

### SkillRegistry (`execution/registry.py`)

O(1) skill lookup by name. Enforces:
- Idempotent registration (duplicate registrations silently skip, preserving first instance)
- Action namespace uniqueness (44 skills, 44 unique actions)
- Contract validation at registration
- Domain-based grouping

### SkillContext (`execution/skill_context.py`)

Frozen per-mission context passed to skills that opt-in via `context` parameter:
- `user`: Typed `UserProfile` (name, email, timezone) from `UserKnowledgeStore`
- `time`: Current `datetime` at mission start

Built fresh per-mission by `MissionOrchestrator.run()`. Backward-compatible via `inspect.signature` check — existing skills without `context` parameter continue to work unchanged.

### SchedulerBridge (`execution/scheduler.py`)

Submits persistent jobs to `TickSchedulerManager` after successful mission completion.

**Invariants**:
- Jobs submitted only if immediate mission COMPLETED
- Jobs must be fully grounded (no unresolved references)
- No cross-domain OutputReference allowed

## Design Rules

- **No replanning** — executor executes, never re-compiles
- **No semantic interpretation** — treat inputs/outputs as opaque data
- **No implicit context** — all data flows through explicit `OutputReference`
- **Parallelism is executor-level** — never cognitive
