# Skills — Overview

**Location**: `skills/`

## Design

Every capability in MERLIN is a **Skill** — a deterministic, isolated, testable unit of work.

### Skill Base Class (`skills/base.py`)

```python
class Skill:
    contract: SkillContract    # Static metadata
    def execute(inputs, world, snapshot=None, context=None) -> SkillResult
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
| `requires` | List[str] | Preconditions (state that must exist before execution) |
| `produces` | List[str] | Postconditions (state created/maintained on success) |
| `effect_type` | str | How skill relates to state: "create", "maintain", "destroy", "reveal" |

### SkillResult (`skills/skill_result.py`)

```python
@dataclass
class SkillResult:
    outputs: Dict[str, Any]      # Named outputs matching contract
    metadata: Dict[str, Any]     # Domain, entity, timing info
```

## Registration

Skills are registered in `main.py` via `SkillRegistry.register()`. The registry enforces:
- Idempotent registration (duplicates silently skip, first instance preserved)
- Unique action namespace (48 skills, 48 unique actions)
- Contract validation at registration time

Skills can opt into receiving `SkillContext` (frozen per-mission `user` + `time`) by accepting a `context` parameter. Backward-compatible: existing skills without `context` continue to work.

## Skill Inventory (48 skills)

| Domain | Skills |
|--------|--------|
| `system` | 19 — media, volume, brightness, apps, jobs, time, battery |
| `browser` | 12 — click, fill, scroll, navigate, go_back, go_forward, autonomous_task, + 5 more |
| `email` | 5 — read_inbox, draft_message, modify_draft, send_message, search_email |
| `fs` | 5 — read_file, write_file, create_folder, search_file, list_directory |
| `memory` | 4 — get_preference, set_preference, set_fact, add_policy |
| `whatsapp` | 2 — send_message, send_file |
| `reasoning` | 1 — generate_text |

See domain-specific docs for details on each skill.
