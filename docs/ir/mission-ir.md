# Mission IR — Intermediate Representation v1

**Location**: `ir/mission.py`

**Status**: FROZEN. No field removals, no semantic reinterpretation.

## MissionPlan

The top-level plan structure:

```python
class MissionPlan(BaseModel):
    id: str                         # Globally unique per request
    nodes: List[MissionNode]        # Closed set (no dynamic insertion)
    metadata: Dict[str, Any]        # Must contain ir_version="1.0"
```

## MissionNode

Atomic skill invocation:

```python
class MissionNode(BaseModel):
    id: str                         # Unique within plan
    skill: str                      # domain.action[.variant]
    inputs: Dict[str, InputValue]   # Skill inputs
    outputs: Dict[str, OutputSpec]  # Declared outputs
    depends_on: List[str]           # Node IDs this depends on
    mode: ExecutionMode             # foreground | background | side_effect
    condition_on: ConditionExpr     # Optional skip-gate
```

**Skill name format**: `domain.action[.variant]` — lowercase alphanumeric + underscores.

**Input constraints**: No `$`-prefixed strings (use `OutputReference` instead).

## OutputReference

Typed inter-node data pipe:

```python
class OutputReference(IRReference):
    node: str                       # Source node ID
    output: str                     # Output field name
    index: Optional[int]            # List element access (0-based)
    field: Optional[str]            # Dict field access (one level)
```

**Resolution order**: `results[node][output]` → `[index]` → `[field]`

**Constraints**: No nested chains. No computed indices. No slices.

## ExecutionMode

```python
class ExecutionMode(str, Enum):
    foreground = "foreground"       # Blocks mission, failure fails mission
    background = "background"       # Non-blocking, failure logged
    side_effect = "side_effect"     # Non-blocking, failure ignored
```

## ConditionExpr

Skip-gate evaluated once at scheduling time:

```python
class ConditionExpr(BaseModel):
    source: str                     # Node ID or world.* namespace
    equals: Any                     # Expected value
```

If condition fails → node is SKIPPED (no output, no failure, marked SKIPPED).

## OutputSpec

```python
class OutputSpec(BaseModel):
    name: str                       # e.g. "research.findings.v1"
    type: str                       # Semantic descriptor (not executable)
```

## Version Policy

- **IR v1**: Current, frozen
- Future versions: IR v2, v3, etc. (additive only)
- `metadata.ir_version` is validated on every plan
