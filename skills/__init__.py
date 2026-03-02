# skills/__init__.py

"""
MERLIN Skills Package — Architectural Rules

RULE 1: Snapshot vs. Live Data
    - The WorldSnapshot is for deterministic reasoning and actuation guards.
    - Skills that QUERY ephemeral telemetry (time, battery, CPU, etc.)
      must read the authoritative source LIVE at execution time.
    - Do NOT return ephemeral sensor data from snapshot.state.*.
    - Declare data_freshness="live" in the SkillContract for such skills.

RULE 2: Branching Constraint
    - Skills with data_freshness="live" produce non-deterministic outputs
      across replays. Their outputs must NOT be used as branching inputs
      (condition_on) in mission plans.
    - This constraint is safe today (no conditional execution support).
    - If conditional execution is added, enforce this deterministically.

RULE 3: Uniform Interface
    - All skills accept (inputs, world, snapshot) — never special-case.
    - Live-read skills receive snapshot but do not use it for primary output.
    - Actuation skills may still use snapshot for idempotency guards
      (e.g., "already playing" check) — this is correct.

See: skills/contract.py SkillContract.data_freshness for enforcement.
"""
