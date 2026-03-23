# Execution Engine

**Location**: `execution/`

The spinal cord. Executes Mission DAGs deterministically, with inline recovery via DecisionEngine.

## Components

### MissionExecutor (`execution/executor.py`)

DAG walker. Core execution loop:

1. Topological sort of nodes by `depends_on`
2. Independent nodes execute in parallel (up to `max_workers`)
3. Dependent nodes wait for upstream completion
4. Output references resolved at runtime
5. Conditions evaluated at scheduling time
6. **10-step contract enforcement pipeline**: dependency check → condition eval → input resolution → mode validation → input key validation → timeline tracking → context injection → failure policy → output/event validation → semantic status derivation

**Execution modes**:
| Mode | Blocking | On Failure |
|------|----------|-----------| 
| `foreground` | Yes | Fails mission |
| `background` | No | Logged only |
| `side_effect` | No | Ignored |

### ExecutionSupervisor (`execution/supervisor.py`)

Step-level guard with pre/post validation and **inline recovery**.

**Per-node execution** (`_execute_guarded_node`):
1. Evaluate preconditions (StepGuard)
2. Assumption gate (if CognitiveContext active)
3. Execute via `executor.execute_node()`
4. Evaluate postconditions
5. On failure: classify via MetaCognition → **attempt inline recovery** → repair + retry
6. Uncertainty update

**Inline recovery** (`_attempt_inline_recovery`):
- Calls `DecisionEngine.decide()` at point of failure
- Builds ephemeral `MissionNode` from `ActionDecision`
- Routes through `executor.execute_node()` (same 10-step pipeline)
- If recovery succeeds → retry original node (re-evaluates ALL preconditions)
- **Bounded**: `MAX_INLINE_RECOVERY = 2` per node
- **Deduped**: same `(skill, inputs)` never tried twice via `inline_recovery_seen`
- **Safety-gated**: only `effect_type ∈ {"reveal", "create", "maintain"}`
- **No DAG mutation**: recovery node is ephemeral, not added to plan

**Repair actions** (guard-based, pre-inline-recovery):
| Action | When |
|--------|------|
| `CONTINUE` | Non-critical failure, skip and proceed |
| `RETRY` | Transient failure, retry with backoff |
| `FAIL` | Critical failure, stop execution |

### DecisionEngine (`execution/metacognition.py`)

Effect-driven adaptive recovery. 2-axis classification (cause × scope).

**7-step recovery loop**:
1. `classify_complexity()` — `FailureCause` × `FailureScope`
2. `_check_requires()` — skill contract preconditions
3. `_find_creators()` — skills that produce missing state
4. `_find_revealers()` — skills that discover data (effect_type="reveal")
5. `_find_enablers()` — skills that enable missing capabilities
6. `_score_normalized()` — expected-value scoring with 2-step lookahead
7. Return `ActionDecision | AmbiguityDecision | EscalationDecision`

**Three tiers**:
| Tier | Latency | When |
|------|---------|------|
| 1. Heuristic | ~0ms | Known pattern via contract chain |
| 2. LLM single | ~200ms | Unknown, 1-node fix via `recovery_reasoner` |
| 3. Cortex recompile | 2-5s | Multi-step (existing!) |

### MetaCognitionEngine (`execution/metacognition.py`)

Failure classification and outcome analysis:
- `classify()` — categorizes node failures into `FailureVerdict`
- `should_abort()` — checks if plan is fundamentally broken
- `OutcomeAnalyzer` — classifies execution outcomes (BENIGN, SOFT_FAILURE, HARD_FAILURE)

### CognitiveContext (`execution/cognitive_context.py`)

Single source of truth for all decision-making. Built once per mission.

| Component | Purpose |
|-----------|---------|
| `GoalState` | Original query, required/achieved/pending outcomes |
| `ExecutionState` | Step count, recovery depth, uncertainty, attempt history, causal trace |
| `DecisionSnapshot` | Frozen per-step immutable view (Contract 2) |
| `Assumption` | Guard-mapped, invertible assumption model |
| `Commitment` | Chosen alternative tracking with reconsideration |
| `DecisionRecord` | Causal graph node with parent_ids + caused_by |
| `SimulatedState` | Zero-cost projected state for lookahead |

**Budgets**: `MAX_TOTAL_STEPS=20`, `MAX_RECOVERY_DEPTH=5`, `MAX_DYNAMIC_QUEUE=10`

### SkillRegistry (`execution/registry.py`)

O(1) skill lookup by name. Enforces:
- Idempotent registration (duplicate registrations silently skip)
- Action namespace uniqueness (48 skills, 48 unique actions)
- Contract validation at registration
- Domain-based grouping

### SkillContext (`execution/skill_context.py`)

Frozen per-mission context passed to skills that opt-in via `context` parameter:
- `user`: Typed `UserProfile` (name, email, timezone) from `UserKnowledgeStore`
- `time`: Current `datetime` at mission start

### SchedulerBridge (`execution/scheduler.py`)

Submits persistent jobs to `TickSchedulerManager` after successful mission completion.

## Design Rules

- **Inline recovery is bounded** — max 2 per node, safety-gated, deduped
- **No DAG mutation** — recovery nodes are ephemeral
- **Same pipeline** — recovery routes through `executor.execute_node()`
- **No semantic interpretation** — treat inputs/outputs as opaque data
- **No implicit context** — all data flows through explicit `OutputReference`
- **Parallelism is executor-level** — never cognitive
