# skills/system/list_jobs.py

"""
ListJobsSkill — Query pending and active scheduled jobs.

Returns structured job list for downstream composition
via OutputReference(index, field).

READS task_store — never mutates.
data_freshness="live" — task states are runtime-only.

Design: single output key "jobs" (job_list).
No summary string — ReportBuilder generates the user-facing text
from the structured data. Single source of truth.
"""

import logging
from typing import Any, Dict, Optional

from skills.skill_result import SkillResult
from skills.base import Skill
from skills.contract import SkillContract, FailurePolicy
from ir.mission import ExecutionMode
from world.timeline import WorldTimeline
from world.snapshot import WorldSnapshot

logger = logging.getLogger(__name__)


class ListJobsSkill(Skill):
    """
    List pending and active scheduled jobs.

    Zero inputs. Returns structured list for IR composition.
    Each job dict contains: short_id, query, status, scheduled_at.
    Downstream nodes can reference individual fields via
    OutputReference(index=N, field="short_id").
    """

    contract = SkillContract(
        name="system.list_jobs",
        action="list_jobs",
        target_type="job",
        description="List scheduled jobs by status",
        intent_verbs=["list", "show", "check", "any", "what"],
        intent_keywords=[
            "jobs", "tasks", "pending", "scheduled",
            "reminders", "queued", "completed", "failed",
        ],
        verb_specificity="generic",
        domain="system",
        requires_focus=False,
        resource_cost="low",
        inputs={},
        optional_inputs={
            "status": "filter_value",  # active|completed|failed|cancelled|all
        },
        outputs={"jobs": "job_list"},
        allowed_modes={ExecutionMode.foreground},
        failure_policy={
            ExecutionMode.foreground: FailurePolicy.FAIL,
        },
        emits_events=[],
        mutates_world=False,
        idempotent=True,
        data_freshness="live",
        output_style="rich",
    )

    # ── Status filter mapping ──
    # Maps user-facing status names to TaskStatus enum values.
    # "active" is the default: shows pending + running.
    _STATUS_MAP = {
        "active":    ["PENDING", "RUNNING"],
        "pending":   ["PENDING"],
        "running":   ["RUNNING"],
        "completed": ["COMPLETED"],
        "failed":    ["FAILED"],
        "cancelled": ["CANCELLED"],
        "all":       ["PENDING", "RUNNING", "COMPLETED", "FAILED", "CANCELLED"],
    }

    def __init__(self, task_store):
        self._store = task_store

    def execute(
        self,
        inputs: Dict[str, Any],
        world: WorldTimeline,
        snapshot: Optional[WorldSnapshot] = None,
    ) -> SkillResult:
        from runtime.task_store import TaskStatus

        # Resolve status filter (default: active = pending + running)
        status_filter = inputs.get("status", "active").lower().strip()
        status_names = self._STATUS_MAP.get(status_filter, ["PENDING", "RUNNING"])

        all_tasks = []
        for name in status_names:
            ts = TaskStatus(name.lower())
            all_tasks.extend(self._store.list_by_status(ts))

        job_list = []
        for t in all_tasks:
            entry = {
                "short_id": t.short_id,
                "query": t.query,
                "status": t.status.value if hasattr(t.status, 'value') else str(t.status),
                "type": t.type.value if hasattr(t.type, 'value') else str(t.type),
                "created_at": getattr(t, 'created_at', None),
                "next_run": t.next_run,
                "scheduled_at": t.next_run,  # alias for backward compat
            }
            # Include completed_at for finished jobs
            completed_at = getattr(t, 'completed_at', None)
            if completed_at:
                entry["completed_at"] = completed_at
            job_list.append(entry)

        return SkillResult(
            outputs={"jobs": job_list},
            metadata={
                "domain": "system",
                "entity": "scheduled jobs",
                "status_filter": status_filter,
            },
        )
