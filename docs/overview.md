# MERLIN — System Overview

MERLIN is a cognitive operating system that transforms natural language into deterministic execution. It functions as a personal AI assistant that can control your computer, browse the web, manage files, remember your preferences, and schedule tasks.

## Component Map

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           merlin.py (Conductor)                        │
│  Owns all components. Routes percepts. Manages lifecycle. Never reasons│
├──────────┬──────────┬──────────┬──────────┬──────────┬──────────────────┤
│          │          │          │          │          │                  │
│  BRAIN   │  CORTEX  │EXECUTION │ RUNTIME  │  WORLD   │    INFRA        │
│          │          │          │          │          │                  │
│ Percept  │ Compile  │ Execute  │ Poll     │ Track    │ Control         │
│ routing  │ plans    │ DAGs     │ events   │ state    │ hardware        │
│          │          │          │          │          │                  │
│ brain/   │ cortex/  │execution/│ runtime/ │ world/   │infrastructure/  │
└──────────┴──────────┴──────────┴──────────┴──────────┴──────────────────┘
       │          │          │          │          │            │
       ▼          ▼          ▼          ▼          ▼            ▼
  ┌─────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌──────┐ ┌──────────┐
  │Perception│ │Skills  │ │Orchestr│ │Memory  │ │Config│ │Reporting │
  │          │ │        │ │        │ │        │ │      │ │          │
  │perception│ │skills/ │ │orchstr/│ │memory/ │ │config│ │reporting/│
  └─────────┘ └────────┘ └────────┘ └────────┘ └──────┘ └──────────┘
```

## Request Lifecycle (Happy Path)

```
User speaks/types
  → Perception (text/speech → Percept)
  → BrainCore (route: REFLEX | DIRECT | MISSION)
  
  REFLEX path:
    → ReflexEngine (deterministic match, <100ms)
    → Direct skill execution
  
  MISSION path:
    → CognitiveCoordinator (reason → DIRECT_ANSWER | SKILL_PLAN)
    → MissionCortex (compile → MissionPlan IR v1)
    → MissionOrchestrator (resolve entities → execute → handle failures)
    → MissionExecutor (walk DAG → run skills)
    → ReportBuilder (narrate results)
    → OutputChannel (console + TTS)
```

## Key Design Constraints

| Rule | Enforcement |
|------|-------------|
| No global mutable context | WorldTimeline is append-only; WorldState is derived |
| Skills are isolated | No inter-skill calls; no DAG modification |
| One LLM call per compilation | MissionCortex compiles once; no loops |
| Infrastructure ≠ Intelligence | Cortex cannot access OS, browser, or filesystem directly |
| Failures are bounded and explicit | Inline recovery max 2 per node; Tier 3 limited to 1 recompile |

## Documentation Index

| Document | Contents |
|----------|----------|
| [Cognitive Pipeline](architecture/cognitive-pipeline.md) | Full request lifecycle with all paths |
| [World State](architecture/world-state.md) | Timeline, events, state, snapshots |
| [Three-Tier Browser](architecture/three-tier-browser.md) | Deterministic → entity → autonomous |
| [Brain](subsystems/brain.md) | Percept routing, escalation, classification |
| [Cortex](subsystems/cortex.md) | Compiler, coordinator, resolvers |
| [Execution](subsystems/execution.md) | Executor, supervisor, DecisionEngine, CognitiveContext, inline recovery |
| [Orchestrator](subsystems/orchestrator.md) | Recovery, replanning, entity resolution |
| [Runtime](subsystems/runtime.md) | Event loop, reflex, scheduler, task store |
| [Memory](subsystems/memory.md) | User knowledge, preferences, facts |
| [Infrastructure](subsystems/infrastructure.md) | Browser, system, apps, sessions |
| [Perception](subsystems/perception.md) | STT, audio, text input |
| [Reporting](subsystems/reporting.md) | Narration, TTS, output channel |
| [Models](subsystems/models.md) | LLM clients, router, key pool |
| [Conversation](subsystems/conversation.md) | Conversation frame, outcomes |
| [Skills Overview](skills/overview.md) | Contracts, base class, registration |
| [Browser Skills](skills/browser-skills.md) | All 12 browser skills |
| [System Skills](skills/system-skills.md) | All 19 system skills |
| [File Skills](skills/file-skills.md) | read, write, create_folder, search_file, list_directory |
| [Memory Skills](skills/memory-skills.md) | Preference, fact, policy management |
| [Reasoning Skills](skills/reasoning-skills.md) | generate_text |
| [Environment](configuration/environment.md) | .env variables, API keys |
| [Config Files](configuration/config-files.md) | YAML configuration reference |
| [Mission IR](ir/mission-ir.md) | IR v1 specification |
| [Releasing](releasing.md) | End-to-end GitHub tag, release, and PyPI publishing flow |
| [Autonomous Readiness Analysis](analysis-report.md) | End-to-end readiness report, scenario simulation, and gap analysis |
| [Email Skills Analysis](email-analysis-report.md) | End-to-end production behavior review for email skills and integration readiness |
