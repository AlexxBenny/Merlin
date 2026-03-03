Purpose

This document defines the immutable architectural laws of the system.

The goal is to build a scalable, deterministic cognitive operating system capable of handling:

very long, complex, multi-domain user queries

thousands of skills

parallel, background, and dependent tasks

without cognitive drift, rewrites, or entropy

If a future change violates any rule here, the change is invalid, regardless of functionality.

1. Core Design Philosophy
1.1 Intelligence is narrow, execution is broad

Intelligence exists only to transform user intent into structure.

Execution exists to carry out that structure deterministically.

Intelligence must be frozen early; execution must be infinitely extensible.

1.2 Structure replaces interpretation

Long queries are not “understood continuously”.

They are compiled once into a static structure (Mission DAG).

After compilation, no interpretation occurs.

1.3 Infrastructure is not intelligence

Any component required for all tasks is infrastructure, not cognition.

Examples:

filesystem paths

browser sessions

downloads

retries

OS interaction

Placing infrastructure inside cognition is a critical architectural error.

2. Cognitive Layers (Immutable)

The system has exactly four cognitive layers.

Adding more or merging layers is forbidden.

1. Perception Layer
2. Nervous System Core
3. Mission Cortex
4. Skill Layer


Execution and infrastructure exist outside cognition.

3. Layer Responsibilities (Strict)
3.1 Perception Layer

Responsibility

Convert external signals into Percept objects

Properties

No reasoning

No state

No routing

No execution

Examples

text input

speech transcription

vision input

3.2 Nervous System Core (Brainstem)

Responsibility

Decide what kind of cognition is required

Allowed

Constant-time routing

Minimal pattern checks

Forbidden

Reasoning

Planning

Skill awareness

Environment access

Invariant

This layer should almost never change.

3.3 Mission Cortex

Responsibility

Transform user intent into a static Mission DAG

Allowed

LLM usage

Reasoning

Decomposition

Validation

Forbidden

Execution

Path resolution

Session management

Skill logic

Retrying or replanning

Output

A fully specified Mission DAG

Explicit dependencies

Explicit conditionals

Explicit data flow

Once emitted, the DAG is final.

3.3a Cognitive Coordinator (within Cortex Layer)

Responsibility

Compute intermediate values and direct answers using world state + LLM reasoning.

Allowed

Reasoning over world state

Arithmetic, date math, factual recall

Producing direct answers (no skills needed)

Computing variables that feed into DAG compilation

Forbidden

Execution

OS mutation

Skill invocation

Replanning (Phase 1)

Looping (Phase 1)

Output

DIRECT_ANSWER: final user-facing text (no skills)

SKILL_PLAN: pass-through to Mission Cortex (unchanged)

REASONED_PLAN: computed variables + rewritten query for compilation

Constraint: Maximum 1 LLM call per invocation. No loops. No retries.

Evolution seams:

Phase 2 — evaluate() for post-execution bounded replan (MAX_REPLAN=1)

Phase 3 — plan_iterative() for GoalGraph decomposition

3.4 Skill Layer

Responsibility

Provide deterministic, testable capabilities

Properties

Stateless or locally stateful

Replaceable

Independently testable

No knowledge of other skills

Forbidden

Calling other skills

Modifying the DAG

Reasoning about intent

Skills execute; they do not decide.

4. Execution & Infrastructure (Non-Cognitive)

Execution and infrastructure are physiology, not intelligence.

They are required for every task and therefore must not influence cognition.

4.1 Executor

Responsibilities

Enforce DAG order

Run skills

Manage parallelism

Propagate explicit outputs

Forbidden

Replanning

Semantic interpretation

Implicit context passing

4.2 Infrastructure Services

Examples:

Path resolution

Browser session pooling

Download management

OS abstraction

Resource limits

Rules

Skills may request infrastructure

Infrastructure never calls cognition

Cortex is blind to infrastructure

5. Context Model (Extremely Strict)

There are only three legal forms of context.

World State

Passive facts

No triggers

No inference

Node Outputs

Explicit

Typed

Directed

Skill-local Memory

Private

Non-shared

Replaceable

There is no global mutable context.

If context is implicit, it is a bug.

6. Scalability Laws
6.1 Skill Scalability

Adding a skill must not require modifying existing skills.

Skill count must not affect cognition complexity.

Skill lookup must be O(1).

6.2 Query Scalability

Long queries increase DAG size, not reasoning depth.

The system must perform one LLM planning call per request.

Execution time scales with actions, not words.

6.3 Parallelism

Independent DAG branches execute in parallel.

Background nodes do not block foreground completion.

Parallelism is executor-level, never cognitive.

7. Configuration Rules
Centralized

Models

Skill registry

Path aliases

OS/environment facts

Execution limits

Decentralized

Skill behavior

Domain logic

Execution strategy per task

Configuration is global. Behavior is local.

8. Failure Semantics

Failures must be:

Loud

Explicit

Final

Forbidden

Silent retries beyond declared limits

Automatic replanning

Hidden fallbacks

Failure produces:

Partial results

Clear error attribution

Deterministic report

9. Explainability Requirement

The system must be explainable by inspecting:

Mission DAG

Execution logs

Node outputs

If an explanation requires:

“the model decided…”

The architecture has been violated.

10. Non-Negotiable Rule

If a component is required for every task, it is not part of intelligence.

Violating this rule recreates AURA-style collapse.

11. Execution Domains (Two, Strict)

The system has exactly two execution domains.

Immediate Mission

Executes synchronously with user request.

Has no persistent identity.

Dies with request lifecycle.

Handled by Executor (DAG walker).

Uses IR v1 MissionPlan.

Persistent Job

Has persistent identity (TaskStore).

Survives beyond request lifecycle.

Executed by SchedulerManager.

Uses TaskSchedule for temporal metadata.

Must be fully self-contained.

These domains are orthogonal.

A single user query may produce units in both domains.

Domain is a property of executable units, not queries.

12. Persistent Execution Invariants (Frozen)

12.1 Persistent units must be fully grounded

No pronoun ambiguity survives decomposition.

All referents are resolved to concrete skill parameters.

"mute it" → "set_volume(level=0, target=YouTube)" before submission.

12.2 No cross-domain OutputReference

Persistent units may reference world state (always live).

Persistent units may NOT reference outputs of immediate missions.

If cross-domain data flow detected → UNSUPPORTED.

12.3 Persistent submission requires immediate completion

Persistent units are submitted to scheduler ONLY if

the immediate MissionPlan status is COMPLETED.

Partial success does not qualify.

12.4 Scheduler condition semantics

Recurring tasks evaluate condition_on each tick.

If condition false → skip execution for that cycle.

Task remains active. No implicit cancellation.

Only explicit user cancellation or max_repeats exhaustion terminates a task.

12.5 Temporal values must be absolute

delay_seconds is relative to submission time.

schedule_at is UTC epoch timestamp.

No relative-to-previous-unit offsets.

Scheduler operates without knowledge of unit ordering context.

Repository Structure (Enforces the Architecture)

This structure is designed to make architectural violations uncomfortable.

jarvis/
│
├── ARCHITECTURE.md        # this document (frozen)
│
├── config/                # CENTRALIZED configuration ONLY
│   ├── models.yaml
│   ├── skills.yaml
│   ├── paths.yaml
│   └── execution.yaml
│
├── perception/            # Layer 1
│   ├── __init__.py
│   ├── text.py
│   ├── speech.py          # later
│   └── vision.py          # later
│
├── brain/                 # Layer 2 (almost frozen)
│   ├── __init__.py
│   └── core.py
│
├── cortex/                # Layer 3 (LLM reasoning)
│   ├── __init__.py
│   ├── mission_schema.py  # DAG schema (frozen early)
│   ├── mission_cortex.py
│   └── validators.py
│
├── infrastructure/        # PHYSIOLOGY (not cognition)
│   ├── __init__.py
│   ├── paths.py
│   ├── browser_sessions.py
│   ├── downloads.py
│   └── filesystem.py
│
├── skills/                # Layer 4 (capabilities)
│   ├── __init__.py
│   ├── base.py
│   ├── fs/
│   ├── browser/
│   ├── research/
│   ├── judgment/
│   ├── presentation/
│   └── system/
│
├── execution/             # Spinal cord
│   ├── __init__.py
│   ├── registry.py
│   ├── executor.py
│   └── scheduler.py
│
├── reporting/             # Final synthesis
│   └── report_builder.py
│
├── models/                # LLM adapters only
│   └── ollama_client.py
│
├── jarvis.py              # Orchestrator
└── main.py                # Entry point