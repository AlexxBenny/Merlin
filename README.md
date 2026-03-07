# 🧙‍♂️ MERLIN

**MERLIN** is an advanced, multi-modal cognitive agent architecture built for reliability, determinism, and deep system integration. 

Unlike traditional LLM wrappers that rely on chaotic ReAct loops and prompt-injection-heavy execution, MERLIN uses a strict **four-layer cognitive architecture** to separate *intelligence* (reasoning, comprehension) from *physiology* (execution, state mutation). 

> **Core Philosophy:** *Intelligence is narrow, execution is broad. Structure replaces interpretation.*

---

## 🧠 The Architecture

MERLIN is divided into distinct execution layers, preventing hallucinations and enforcing strict operational contracts.

### 1. Perception Layer (The Senses)
Manages concurrent, multi-modal inputs. It handles text inputs and voice recognition simultaneously with explicit cancellation semantics (e.g., typing text immediately aborts an ongoing voice recording).
- Handles text through CLI prompts (`TextPerception`)
- Handles voice via Speech-to-Text inference (`SpeechPerception`)
- Emits structured `Percept` objects into the system.

### 2. The Nervous System (BrainCore & Reflexes)
The routing authority and fast-path execution loop. MERLIN doesn't invoke an LLM for everything.
- **BrainCore Circuit Breaker:** Analyzes the structural features of an input. Simple commands bypass heavy cognitive processing entirely.
- **Reflex Engine:** A zero-LLM, deterministic reaction layer. If you say "mute volume", MERLIN matches the intent and instantly invokes the skill, taking milliseconds instead of seconds.
- **Always-On Event Loop:** Listens to background events, dispatches scheduled background jobs, and triggers proactive reflexes.

### 3. Mission Cortex (The Planner)
When complex reasoning is required, the `MissionCortex` acts as an LLM-powered compiler.
- Translates natural language intent (e.g., "Find the latest report, summarize it, and email it to my boss") into a deterministic **Mission Plan (Directed Acyclic Graph)**.
- Enforces rigid Intermediate Representation (IR) validation. The plan is verified for correct skill arguments, routing, and dependencies *before* a single action is taken.

### 4. Skill & Execution Layer (The Physiology)
The non-cognitive layer that mutates the world.
- **MissionExecutor:** Executes the compiled DAG (Mission Plan). Enables concurrent execution of parallel nodes (within the same dependency layer).
- **Enforced Execution Contracts:** Skills are rigorously defined. An LLM cannot make up arguments or manipulate the system in undefined ways.
- **World State Timeline:** Every action, input, and response is tracked as an append-only event stream in the `WorldTimeline`, allowing MERLIN to maintain perfect contextual awareness of its environment. 

---

## 🛠️ Superpowers & Capabilities

MERLIN comes fully equipped to be a JARVIS-level assistant, with deep operating system integrations and advanced context management.

* **Advanced OS Manipulation:** Can natively manipulate your OS via the `SystemController`. Open/close/focus applications, control media playback, intercept hardware stats (battery, system status), and adjust display settings (brightness, volume, night light).
* **Semantic User Memory (`UserKnowledgeStore`):** A versioned, deterministic memory system that tracks user **facts**, **preferences**, **traits**, and **policies**. If you establish a policy (e.g., "When a movie starts playing, set volume to 90% and dim brightness"), MERLIN enforces it deterministically without depending on an LLM to remember.
* **Proactive Attention Management:** MERLIN doesn't just respond when spoken to. It runs scheduled tasks, evaluates completion queues, and uses an `AttentionManager` to decide whether to *interrupt* you immediately, *queue* a notification for later, or *suppress* it entirely based on priority.
* **File System & Browser Control:** Native capabilities to search, read, write to the file system, and scrape/control the web browser.
* **Granular LLM Routing:** Configure different LLM providers (e.g., OpenRouter, Gemini, local models) for different cognitive tasks based on speed, cost, and intelligence requirements via `models.yaml` and `routing.yaml`.

---

## 📂 Codebase Topography

- `main.py` & `merlin.py`: The entry points and central orchestration loops.
- `brain/`: The routing authority, deciding when to think vs. when to react.
- `cortex/`: The compiler turning user text into actionable Mission DAGs.
- `execution/`: The engine running the Mission Plans and enforcing Skill Contracts.
- `infrastructure/`: Native OS adapters like the `SystemController` for Windows integration.
- `memory/`: The pure-data storage system for recording user preferences and traits.
- `perception/`: Input handling (Speech & Text concurrent tracking).
- `reporting/`: Proactive intelligence formatting, deciding how and when to talk back.
- `runtime/`: The always-on heartbeat (`event_loop.py`), reflex matching, and job scheduler.
- `skills/`: The vast registry of executable actions (browser, fs, system, reasoning).

---

## ⚡ Getting Started
*Configuration, setup instructions, and deployment guides will be placed here.*
