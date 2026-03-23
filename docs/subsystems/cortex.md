# Cortex — Mission Compiler & Reasoning

**Location**: `cortex/`

The intelligence center. Transforms user intent into executable plans. The only layer that uses LLMs for reasoning.

## Components

### MissionCortex (`cortex/mission_cortex.py`)

The plan compiler. Single LLM call per compilation.

**Pipeline**:
1. `IntentEngine.discover_skills()` — candidate skill list
2. `ScoredDiscovery.score_candidates()` — TF-IDF ranking
3. `_build_session_context()` — world state + browser entities + action templates
4. LLM call → JSON plan
5. `Validators.validate()` — cycle check, contract enforcement
6. Result: `MissionPlan` (IR v1)

**Action templates** (injected into compiler prompt):
```
browser.click(entity_index=N)        → Tier 1 deterministic
browser.click(entity_ref="text")     → Tier 2 entity resolution
browser.autonomous_task(task="...")   → Tier 3 autonomous (last resort)
browser.scroll(direction="up|down")
browser.navigate(url="...")
browser.fill(entity_index=N, text="...")
```

### CognitiveCoordinator (`cortex/cognitive_coordinator.py`)

Intermediate reasoning layer. Decides whether skills are needed.

**Modes**:
| Mode | Description | Skills? |
|------|------------|---------|
| `DIRECT_ANSWER` | Answer from world state + knowledge | No |
| `SKILL_PLAN` | Pass to compiler | Yes |
| `REASONED_PLAN` | Computed variables + rewritten query | Yes |

**Constraint**: Maximum 1 LLM call per invocation. No loops. No retries.

**Inputs**: Query, world state schema, conversation history, user memory.

### IntentEngine (`cortex/intent_engine.py`)

Verb/keyword-based skill discovery. Maintains an inverted index:
- Verb entries mapped per `intent_verbs` in each SkillContract
- Keyword entries mapped per `intent_keywords` (including plural normalization)
- 48 skills indexed across 7 domains

### ScoredDiscovery (`cortex/scored_discovery.py`)

Phase 1 filtering with intent scoring. Scores candidates by:
- Verb match weight (+2.0 per match)
- Keyword overlap (+1.0 per match)
- Simple plural normalization (`emails` → also matches `email`)
- Domain expansion for chaining safety (bounded sibling inclusion)

### EntityResolver (`cortex/entity_resolver.py`)

Three-phase entity resolution:

**Phase 9C — App entities**:
- Registry lookup → name index → alias table → fuzzy match
- Resolution: `app_target` → `app_id`

**Phase 9D — Browser entities**:
- Cosine similarity on tokenized entity text
- Resolution: `entity_ref` → `entity_index` + `_resolved_entity_text`
- Thresholds: < 0.55 = NOT_FOUND, second > 0.8 × top = AMBIGUOUS

**Phase 9E — File entities**:
- `FileIndex.search()` for bare filename lookup
- Resolution: `file_path_input` → `relative_path` + `anchor` + `_resolved_file_ref_id`
- 0 matches → `not_found_file` violation
- 1 match → replace path + set anchor
- N matches → `ambiguous_file` violation with structured options
- Skips explicit paths (containing `/` or `\`) and `IRReference` values

### ParameterResolver (`cortex/parameter_resolver.py`)

Resolves semantic types in skill inputs against the `SemanticTypes` registry.

### PreferenceResolver (`cortex/preference_resolver.py`)

Injects user preferences into skill parameters. Uses `UserKnowledgeStore` preferences.

### Validators (`cortex/validators.py`)

Plan validation:
- Cycle detection in dependency graph
- Contract enforcement (required inputs present)
- Skill existence check
- Output reference validity

### ContextProvider (`cortex/context_provider.py`)

Builds conversation context for LLM prompts. Manages context window size.

### SemanticTypes (`cortex/semantic_types.py`)

Type system for skill inputs:
- `volume_level`, `brightness_level` — numeric ranges
- `entity_index` — browser entity display index
- `entity_ref` — text-based entity reference
- `app_target` — application name
- `scroll_direction` — "up" or "down"
- `file_path`, `folder_path` — filesystem paths
- And many more

### Normalizer (`cortex/normalizer.py`)

Query normalization: lowercase, whitespace cleanup, abbreviation expansion.

### FallbackHandler (`cortex/fallback.py`)

Graceful degradation when compilation fails. Generates helpful error messages.

### Synonyms (`cortex/synonyms.py`)

Action synonym mapping for skill discovery flexibility.
