# Skills — Overview

**Location**: `skills/`

## Design

Every capability in MERLIN is a **Skill** — a deterministic, isolated, testable unit of work.

### Skill Base Class (`skills/base.py`)

```python
class Skill:
    contract: SkillContract    # Static metadata
    def execute(inputs, world, snapshot) -> SkillResult
```

Skills:
- Are **stateless** (or locally stateful only)
- Are **independently testable**
- Have **no knowledge** of other skills
- **Cannot** call other skills or modify the DAG
- **Cannot** reason about intent

### SkillContract (`skills/contract.py`)

Static metadata that describes a skill:

| Field | Type | Purpose |
|-------|------|---------|
| `name` | str | Unique skill ID (`domain.action`) |
| `action` | str | Verb for intent matching |
| `target_type` | str | What the skill operates on |
| `description` | str | Human-readable description |
| `narration_template` | str | Template for ReportBuilder |
| `intent_verbs` | List[str] | Verbs for IntentEngine matching |
| `intent_keywords` | List[str] | Keywords for discovery |
| `verb_specificity` | str | "specific" or "generic" |
| `domain` | str | Skill domain (system, browser, fs, etc.) |
| `requires_focus` | bool | Needs foreground focus? |
| `inputs` | Dict[str, str] | Input name → semantic type |
| `outputs` | Dict[str, str] | Output name → semantic type |
| `allowed_modes` | Set[ExecutionMode] | foreground, background, side_effect |
| `failure_policy` | Dict[ExecutionMode, FailurePolicy] | FAIL, RETRY, IGNORE |
| `emits_events` | List[str] | WorldTimeline event types emitted |
| `mutates_world` | bool | Modifies external state? |
| `output_style` | str | "terse", "normal", "verbose" |

### SkillResult (`skills/skill_result.py`)

```python
@dataclass
class SkillResult:
    outputs: Dict[str, Any]      # Named outputs matching contract
    metadata: Dict[str, Any]     # Domain, entity, timing info
```

## Registration

Skills are registered in `main.py` via `SkillRegistry.register()`. The registry enforces:
- Unique action namespace (34 skills, 34 unique actions)
- Contract validation at registration time

## Skill Inventory (34 skills)

| Domain | Skills |
|--------|--------|
| `system` | 20 — media, volume, brightness, apps, jobs, time, battery |
| `browser` | 7 — click, fill, scroll, navigate, go_back, go_forward, autonomous_task |
| `fs` | 3 — read_file, write_file, create_folder |
| `memory` | 4 — get_preference, set_preference, set_fact, add_policy |
| `reasoning` | 1 — generate_text |

See domain-specific docs for details on each skill.
