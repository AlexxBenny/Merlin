# MERLIN — Architecture

> **Status**: Living document. Updated March 2026.
>
> This document defines the architectural laws of the system. If a future change violates any rule here, the change is invalid.

---

## 1. Core Design Philosophy

### 1.1 Intelligence Is Narrow, Execution Is Broad

Intelligence exists only to transform user intent into structure. Execution carries out that structure deterministically. Intelligence is frozen early; execution is infinitely extensible.

### 1.2 Structure Replaces Interpretation

Long queries are not "understood continuously". They are **compiled once** into a static structure (Mission DAG). After compilation, no interpretation occurs.

### 1.3 Infrastructure Is Not Intelligence

Any component required for all tasks is infrastructure, not cognition. Placing infrastructure inside cognition is a critical architectural error.

---

## 2. Cognitive Layers

The system has exactly **four cognitive layers**. Adding more or merging layers is forbidden.

| Layer | Name | Module |
|-------|------|--------|
| 1 | Perception | `perception/` |
| 2 | Nervous System Core | `brain/` |
| 3 | Mission Cortex | `cortex/` |
| 4 | Skill Layer | `skills/` |

Execution, infrastructure, world state, and runtime exist **outside cognition**.

---

## 3. Layer Responsibilities

### 3.1 Perception Layer (`perception/`)

Converts external signals into `Percept` objects.

- **No** reasoning, state, routing, or execution
- Components: `PerceptionOrchestrator`, `TextPerception`, `SpeechPerception`
- STT engines: Whisper, Mock
- Audio: `AudioRecorder` with voice activity detection

### 3.2 Nervous System Core (`brain/`)

Decides what kind of cognition is required. Constant-time routing.

| Component | Purpose |
|-----------|---------|
| `BrainCore` | Route percepts → cognitive path (`REFLEX`, `DIRECT`, `MISSION`) |
| `EscalationPolicy` | Tier classification (`HeuristicTierClassifier`) |
| `StructuralClassifier` | Speech act detection, intent classification |
| `OrdinalResolver` | Resolve "first", "second", "the other one" |

**Forbidden**: Reasoning, planning, skill awareness, environment access.

### 3.3 Mission Cortex (`cortex/`)

Transforms user intent into a static Mission DAG.

| Component | Purpose |
|-----------|---------|
| `MissionCortex` | LLM-based plan compilation with action templates |
| `CognitiveCoordinator` | Intermediate reasoning, direct answers, computed variables |
| `IntentEngine` | Verb/keyword-based skill discovery |
| `ScoredDiscovery` | TF-IDF scored skill matching with plural normalization |
| `EntityResolver` | App + browser + file entity resolution (9C/9D/9E) |
| `ParameterResolver` | Semantic type resolution for skill inputs |
| `PreferenceResolver` | User preference injection into skill parameters |
| `Normalizer` | Query normalization and canonicalization |
| `Validators` | Plan validation, cycle detection, contract enforcement |
| `ContextProvider` | Conversation context injection for LLM prompts |
| `SemanticTypes` | Type system for skill inputs (`entity_index`, `entity_ref`, etc.) |
| `FallbackHandler` | Graceful degradation when compilation fails |

**Cognitive Coordinator modes**:
- `DIRECT_ANSWER` — Final user-facing text (no skills needed)
- `SKILL_PLAN` — Pass-through to Mission Cortex
- `REASONED_PLAN` — Computed variables + rewritten query for compilation

**Forbidden**: Execution, path resolution, session management, skill logic.

### 3.4 Skill Layer (`skills/`)

Deterministic, testable capabilities. 46 registered skills across 6 domains:

| Domain | Count | Examples |
|--------|-------|---------|
| `system.*` | 19 | media control, volume, brightness, apps, jobs, time, battery |
| `browser.*` | 12 | click, fill, scroll, navigate, go_back, go_forward, autonomous_task |
| `email.*` | 5 | read_inbox, draft_message, modify_draft, send_message, search_email |
| `memory.*` | 4 | get_preference, set_preference, set_fact, add_policy |
| `fs.*` | 5 | read_file, write_file, create_folder, search_file, list_directory |
| `reasoning.*` | 1 | generate_text |

Each skill has a `SkillContract` defining: name, inputs, outputs, semantic types, execution mode, failure policy, domain, narration template, `requires` (preconditions), `produces` (postconditions), and `effect_type` (create/maintain/destroy/reveal).

**Forbidden**: Calling other skills, modifying the DAG, reasoning about intent.

---

## 4. Execution & Infrastructure (Non-Cognitive)

### 4.1 Executor (`execution/`)

| Component | Purpose |
|-----------|---------|
| `MissionExecutor` | DAG walker — 10-step contract enforcement, runs skills, manages parallelism |
| `ExecutionSupervisor` | Step-level guard: pre/post validation, assumption gate, **inline recovery** |
| `DecisionEngine` | Effect-driven recovery: contract chain → lookahead → expected-value scoring |
| `MetaCognitionEngine` | 2-axis failure classification (cause × scope) + outcome analysis |
| `CognitiveContext` | Single shared context: `GoalState`, `ExecutionState`, `DecisionSnapshot` |
| `SkillRegistry` | O(1) skill lookup, idempotent registration, action namespace audit |
| `SkillContext` | Frozen per-mission context (`user` profile + `time`) passed to skills |
| `SchedulerBridge` | Persistent job submission after mission success |

**Key files**: `executor.py`, `supervisor.py`, `metacognition.py`, `cognitive_context.py`, `registry.py`, `skill_context.py`, `scheduler.py`

**Forbidden**: Replanning, semantic interpretation, implicit context passing.

### 4.2 Orchestrator (`orchestrator/`)

`MissionOrchestrator` — The control loop between cortex and executor:

- Receives compiled plan from cortex
- Runs entity resolution (app + browser entities)
- Executes via `MissionExecutor`
- Handles recovery recompile on entity NOT_FOUND
- Manages the replanning pipeline for execution failures

### 4.3 Infrastructure Services (`infrastructure/`)

| Component | Purpose |
|-----------|---------|
| `BrowserUseAdapter` | Playwright/CDP browser lifecycle management |
| `BrowserUseController` | DOM interaction: snapshot, click, fill, scroll, navigate |
| `BrowserController` | Abstract interface — `DOMEntity`, `PageSnapshot` |
| `BrowserSafety` | URL allowlist, download blocking |
| `SystemController` | OS interaction: volume, brightness, media, apps, nightlight |
| `ApplicationRegistry` | App discovery, name resolution, alias mapping |
| `AppDiscovery` | Windows app enumeration (Start Menu, PATH, UWP) |
| `AppCapabilities` | Per-app media capability detection |
| `Session` | Browser session management with popup watchdog |
| `LocationConfig` | User-configurable paths and download locations |
| `Observer` | File system observer for download detection |
| `VoiceFactory` | TTS voice selection and configuration |

**Rules**: Skills may request infrastructure. Infrastructure never calls cognition. Cortex is blind to infrastructure.

---

## 5. World State System (`world/`)

The single source of truth for all environmental state.

| Component | Purpose |
|-----------|---------|
| `WorldTimeline` | Append-only event log — the ONLY mutable world structure |
| `WorldEvent` | Timestamped event: `{timestamp, source, type, payload}` |
| `WorldState` | Pure reducer: `events → state`. Derived, never mutated directly |
| `WorldSnapshot` | Frozen state snapshot passed to skills and coordinator |
| `WorldResolver` | Schema generation for LLM prompt injection |

**Event sources** (`runtime/sources/`):
| Source | Events |
|--------|--------|
| `SystemSource` | CPU, memory, battery, brightness, volume, idle, app lifecycle |
| `MediaSource` | Now-playing, track change, ad detection |
| `TimeSource` | Time ticks, hour changes, date changes |
| `BrowserSource` | Page changes, entity refresh, connect/disconnect |

**State domains**: `BrowserWorldState`, `MediaState`, `SystemState`, `TimeState`, `AppState`.

---

## 6. Runtime System (`runtime/`)

| Component | Purpose |
|-----------|---------|
| `RuntimeEventLoop` | Central polling loop — bootstraps world, polls sources, ticks scheduler |
| `ReflexEngine` | Sub-100ms deterministic skill matching (bypasses LLM) |
| `TickSchedulerManager` | Persistent recurring/delayed job execution |
| `TaskStore` / `JsonTaskStore` | Job persistence with JSON file backend |
| `TemporalResolver` | Natural language → absolute timestamps ("tomorrow at 3pm") |
| `AttentionManager` | Mission lifecycle tracking, concurrent mission limits |
| `CompletionQueue` | Async completion events for background missions |

---

## 7. Memory System (`memory/`)

| Component | Purpose |
|-----------|---------|
| `UserKnowledgeStore` | Structured user knowledge: preferences, facts, traits, policies, relationships |
| `MemoryStore` | Key-value persistent storage backend |

Memory is injected as first-class context into the coordinator's LLM prompt. Memory skills (`memory.set_preference`, `memory.set_fact`, `memory.add_policy`, `memory.get_preference`) allow the user to teach MERLIN.

**Memory Injection Pipeline**: `UserKnowledgeStore.get_user_profile()` extracts identity-relevant facts via an allow-list filter (prevents sensitive data exposure). `format_profile_for_prompt()` sanitizes and sorts the profile for deterministic LLM prompt injection. A typed `UserProfile` dataclass feeds into `SkillContext` for per-mission identity propagation to skills.

---

## 8. Conversation System (`conversation/`)

| Component | Purpose |
|-----------|---------|
| `ConversationFrame` | Turn-by-turn history with entity tracking |
| `Outcome` | Structured result of a mission (success/failure/partial) |

---

## 9. Reporting System (`reporting/`)

| Component | Purpose |
|-----------|---------|
| `ReportBuilder` | LLM-based natural language narration of skill results |
| `NarrationPolicy` | Controls verbosity and narration style |
| `NotificationPolicy` | Decides what to speak vs. what to show |
| `OutputChannel` | Console + TTS output multiplexing |
| `TTSEngine` | Abstract TTS interface |
| TTS engines: `pyttsx3` (offline), `SilentTTS` (testing) |

---

## 10. Model Layer (`models/`)

| Component | Purpose |
|-----------|---------|
| `LLMClient` | Abstract interface for all LLM calls |
| `ModelRouter` | Role-based model selection from `config/models.yaml` |
| `OpenRouterClient` | OpenRouter API (GPT, Claude, Gemini, etc.) |
| `GeminiClient` | Google Gemini API direct access |
| `OllamaClient` | Local Ollama models |
| `HuggingFaceClient` | Hugging Face Inference API |
| `KeyPool` | API key rotation and rate limit management |

---

## 11. IR Layer (`ir/`)

`MissionPlan` — The frozen intermediate representation:

| Component | Purpose |
|-----------|---------|
| `MissionPlan` | Immutable DAG: `{id, nodes, metadata}` |
| `MissionNode` | Atomic skill invocation: `{id, skill, inputs, outputs, depends_on, mode}` |
| `OutputReference` | Typed inter-node data pipe: `{node, output, index?, field?}` |
| `ExecutionMode` | `foreground`, `background`, `side_effect` |
| `ConditionExpr` | Skip-gate evaluated at scheduling time |

**IR Version**: `1.0` (frozen). No field removals, no semantic reinterpretation.

---

## 12. Three-Tier Browser Execution

Browser interaction follows a strict capability hierarchy:

| Tier | Name | When | Example |
|------|------|------|---------|
| 1 | Deterministic | No reasoning needed | `scroll down`, `click entity 3`, `navigate to youtube.com` |
| 2 | Entity Resolution | Page understanding, no planning | `play the howard stark video` → cosine match → `click(entity_index=2)` |
| 3 | Autonomous Agent | Multi-step exploration | `find cheapest MacBook and compare prices` |

**Entity Resolution**: Pure cosine similarity on tokenized text. Thresholds: `< 0.55` = NOT_FOUND, `second > 0.8 × top` = AMBIGUOUS.

**Failure cascade**: Entity found → deterministic action → Ambiguous → ask user → NOT_FOUND → recovery recompile → recompile fails → autonomous (last resort).

**Index drift protection**: Resolver passes `_resolved_entity_text` alongside `entity_index`. Skill gets fresh DOM snapshot and verifies text match before clicking. Falls back to text search on drift.

---

## 13. Context Model

Three legal forms of context:

| Form | Properties |
|------|-----------|
| **World State** | Passive facts, no triggers, no inference |
| **Node Outputs** | Explicit, typed, directed (via `OutputReference`) |
| **Skill-local Memory** | Private, non-shared, replaceable |

There is **no global mutable context**. If context is implicit, it is a bug.

---

## 14. Execution Domains

| Domain | Lifecycle | Handler |
|--------|-----------|---------|
| **Immediate Mission** | Dies with request | `MissionExecutor` (DAG walker) |
| **Persistent Job** | Survives beyond request | `TickSchedulerManager` |

A single query may produce units in both domains. Domain is a property of executable units, not queries.

**Persistent invariants**: Fully grounded (no pronoun ambiguity), no cross-domain `OutputReference`, submission requires immediate mission completion, temporal values must be absolute.

---

## 15. Failure Semantics

Failures are **loud**, **explicit**, and **bounded**.

- **Tier 1 — Inline recovery**: DecisionEngine at point of failure, bounded (max 2), deduped, safety-gated (reveal/create/maintain only), same executor pipeline
- **Tier 2 — Post-execution**: DecisionEngine on accumulated verdicts, causal graph linked
- **Tier 3 — Cortex recompile**: Limited to 1 attempt. Compiler receives `execution_failures` context
- **Produces**: Partial results, clear error attribution, deterministic report
- **Forbidden**: Silent retries beyond declared limits, hidden fallbacks, DAG mutation during recovery

---

## 16. Frontend System (`interface/`, `ui/`)

The frontend is a **separate-process** architecture. The UI never touches MERLIN internals.

### 16.1 Interface Layer (`interface/`)

| Component | Process | Purpose |
|-----------|---------|---------|
| `MerlinBridge` | MERLIN (daemon thread) | Export state → JSON files, poll command queue |
| `LogBufferHandler` | MERLIN (root logger) | Ring-buffer capture of all log records |
| `api_server.py` | **Separate** (uvicorn) | FastAPI REST + WebSocket + SSE on port 8420 |
| `config_schema.py` | API server | Pydantic validation for config editing |

**IPC**: Filesystem-based. Bridge writes `state/api/*.json` (atomic tmp→rename). API server reads them. Commands flow via `state/api/command_queue/` with exactly-once semantics.

**Endpoints**: 15+ REST endpoints under `/api/v1/`, two WebSocket channels (`/ws/logs`, `/ws/events`), SSE streaming for chat.

### 16.2 Dashboard (`ui/dashboard/`)

React + TypeScript + Vite + Tailwind CSS v4. Dark theme with cyan accent.

8 pages: Overview (gauges), Chat (SSE streaming), Scheduler (pause/resume), Memory (5 domains), Logs (WebSocket live), Config (inline editing), Missions (DAG inspector), World State (tree view).

### 16.3 Desktop Widget (`ui/widget/`)

PySide6 floating orb. Click to expand into chat panel. Polls `/api/v1/health` every 5 seconds — grey when disconnected, cyan when connected.

### 16.4 Activation

```bash
python main.py --ui
```

Startup order: MERLIN core → bridge thread → API subprocess → widget subprocess.
Shutdown order (reverse): widget → API → bridge → core.

**Forbidden**: UI importing core modules. Bridge accessing MERLIN from the API process. Shared memory between processes.

---

## 17. Repository Structure

```
MERLIN/
├── ARCHITECTURE.md           # This document
├── README.md                 # Project overview
├── docs/                     # Detailed documentation
├── main.py                   # Entry point + dependency wiring
├── merlin.py                 # The Conductor — owns all components, routes percepts
│
├── brain/                    # Layer 2: Nervous System Core
│   ├── core.py               # BrainCore — percept routing
│   ├── escalation_policy.py  # Tier classification
│   ├── structural_classifier.py  # Speech act + intent detection
│   └── ordinal_resolver.py   # Ordinal reference resolution
│
├── cortex/                   # Layer 3: Mission Cortex
│   ├── mission_cortex.py     # LLM plan compiler
│   ├── cognitive_coordinator.py  # Intermediate reasoning
│   ├── intent_engine.py      # Skill discovery
│   ├── entity_resolver.py    # App + browser entity resolution
│   ├── parameter_resolver.py # Semantic type resolution
│   ├── preference_resolver.py # User preference injection
│   ├── validators.py         # Plan validation
│   ├── semantic_types.py     # Type system
│   ├── context_provider.py   # LLM context injection
│   ├── scored_discovery.py   # TF-IDF skill scoring
│   ├── normalizer.py         # Query normalization
│   ├── fallback.py           # Graceful degradation
│   └── synonyms.py           # Action synonym mapping
│
├── execution/                # DAG execution engine
│   ├── executor.py           # Mission executor (10-step contract enforcement)
│   ├── supervisor.py         # Step-level guard + inline recovery loop
│   ├── metacognition.py      # DecisionEngine + MetaCognition (2-axis classification)
│   ├── cognitive_context.py  # CognitiveContext, ExecutionState, DecisionSnapshot
│   ├── skill_context.py      # Frozen per-mission SkillContext + UserProfile
│   ├── registry.py           # Skill registry
│   └── scheduler.py          # Scheduler bridge
│
├── orchestrator/             # Control loop
│   └── mission_orchestrator.py  # Plan → resolve → execute → report
│
├── ir/                       # Intermediate Representation
│   └── mission.py            # MissionPlan, MissionNode, OutputReference
│
├── world/                    # World state system
│   ├── timeline.py           # Append-only event log
│   ├── state.py              # Pure reducer: events → state
│   ├── snapshot.py           # Frozen state snapshot
│   └── resolver.py           # Schema generation
│
├── runtime/                  # Event loop + scheduling
│   ├── event_loop.py         # Central polling loop
│   ├── reflex_engine.py      # Deterministic fast-path
│   ├── tick_scheduler.py     # Persistent job scheduler
│   ├── task_store.py         # Job persistence
│   ├── json_task_store.py    # JSON file backend
│   ├── temporal_resolver.py  # Natural language → timestamps
│   ├── attention.py          # Mission lifecycle tracking
│   ├── completion_event.py   # Async completion events
│   └── sources/              # Event sources (system, media, time, browser)
│
├── memory/                   # User knowledge
│   ├── user_knowledge.py     # Structured knowledge store
│   └── store.py              # Key-value backend
│
├── infrastructure/           # OS + browser interaction
│   ├── browser_use_adapter.py    # Playwright/CDP lifecycle
│   ├── browser_use_controller.py # DOM interaction
│   ├── browser_controller.py     # Abstract interface
│   ├── browser_safety.py         # URL safety
│   ├── system_controller.py      # OS control
│   ├── application_registry.py   # App discovery + resolution
│   ├── app_discovery.py          # Windows app enumeration
│   ├── app_capabilities.py       # Media capability detection
│   ├── session.py                # Browser session management
│   └── location_config.py        # Path configuration
│
├── skills/                   # Layer 4: Capabilities
│   ├── base.py               # Skill base class
│   ├── contract.py           # SkillContract definition
│   ├── browser/              # 12 browser skills
│   ├── system/               # 19 system skills
│   ├── email/                # 5 email skills
│   ├── fs/                   # 3 file system skills
│   ├── memory/               # 4 memory skills
│   └── reasoning/            # 1 reasoning skill
│
├── perception/               # Layer 1: Input
│   ├── perception_orchestrator.py # Multi-modal orchestration
│   ├── text.py               # Text input
│   ├── speech.py             # Speech input
│   └── engines/              # STT engines (Whisper, Mock)
│
├── reporting/                # Output synthesis
│   ├── report_builder.py     # LLM narration
│   ├── narration.py          # Narration policy
│   ├── notification_policy.py # Speak vs. show
│   ├── output.py             # Console + TTS multiplexing
│   └── engines/              # TTS engines (pyttsx3, Silent)
│
├── models/                   # LLM adapters
│   ├── router.py             # Role-based model selection
│   ├── base.py               # Abstract LLM client
│   ├── openrouter_client.py  # OpenRouter API
│   ├── gemini_client.py      # Google Gemini
│   ├── ollama_client.py      # Local Ollama
│   ├── huggingface_client.py # Hugging Face
│   └── key_pool.py           # API key rotation
│
├── conversation/             # Turn tracking
│   ├── frame.py              # ConversationFrame
│   └── outcome.py            # Mission outcome
│
├── config/                   # Configuration
│   ├── models.yaml           # LLM model assignments
│   ├── skills.yaml           # Skill registry metadata
│   ├── routing.yaml          # Cognitive routing rules
│   ├── execution.yaml        # Execution limits + policies
│   ├── paths.yaml            # OS path aliases
│   ├── browser.yaml          # Browser configuration
│   ├── email.yaml            # Email provider configuration
│   ├── app_aliases.yaml      # Application name aliases
│   └── app_capabilities.yaml # Per-app media capabilities
│
├── interface/                # API boundary layer
│   ├── bridge.py             # IPC bridge (daemon thread inside MERLIN)
│   ├── api_server.py         # FastAPI server (separate process)
│   ├── log_buffer.py         # Ring-buffer log handler
│   └── config_schema.py      # Pydantic config validation
│
├── ui/                       # Frontend UIs
│   ├── dashboard/            # React + Vite + Tailwind dashboard
│   │   └── src/              # TypeScript source (8 pages + layout + API client)
│   └── widget/               # PySide6 desktop widget
│       └── widget.py         # Floating orb + chat panel
│
├── state/                    # Persistent state
│   ├── user_knowledge.json   # Stored user knowledge
│   ├── jobs/                 # Persisted scheduled jobs
│   └── api/                  # IPC bridge state (auto-generated)
│       ├── system.json       # System metrics
│       ├── jobs.json         # Scheduler jobs export
│       ├── memory.json       # Knowledge store export
│       ├── world.json        # World state snapshot
│       ├── missions.json     # Mission history
│       ├── logs.json         # Log buffer export
│       ├── command_queue/    # Pending commands
│       ├── responses/        # Command responses
│       └── chat_sessions/    # Session-scoped chat history
│
├── metrics/                  # Performance measurement
│   ├── collect_compiler_baseline.py
│   └── measure_phase3*.py
│
└── tests/                    # Test suite (1736 tests)