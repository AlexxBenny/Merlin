# skills/system/cancel_job.py

"""
CancelJobSkill — Cancel a scheduled job by short ID.

Receives job_id (e.g. "J-3") as input.
Supports IR composition: can receive job_id via OutputReference
from a prior list_jobs node.

Uses task_store.cancel() which handles:
    - Status transition: PENDING → CANCELLED
    - Persistence flush
    - Returns False if not found or not cancellable

mutates_world=True — cancellation is a side effect.
idempotent=True — cancelling already-cancelled is a no-op.
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


class CancelJobSkill(Skill):
    """
    Cancel a scheduled job by its human-friendly short ID.

    Input: job_id (e.g. "J-3")
    Output: cancelled (confirmation string)

    Can be composed with list_jobs via OutputReference:
        OutputReference(node="list_node", output="jobs", index=N, field="short_id")
    """

    contract = SkillContract(
        name="system.cancel_job",
        action="cancel_job",
        target_type="job",
        description="Cancel a scheduled job",
        intent_verbs=["cancel", "remove", "delete", "stop"],
        intent_keywords=[
            "job", "task", "reminder", "scheduled",
        ],
        verb_specificity="generic",
        domain="system",
        requires_focus=False,
        resource_cost="low",
        inputs={"job_id": "job_identifier"},
        outputs={"cancelled": "info_string"},
        allowed_modes={ExecutionMode.foreground},
        failure_policy={
            ExecutionMode.foreground: FailurePolicy.FAIL,
        },
        emits_events=["job_cancelled"],
        mutates_world=True,
        idempotent=True,
        data_freshness="live",
        output_style="rich",
    )

    def __init__(self, task_store):
        self._store = task_store

    def execute(
        self,
        inputs: Dict[str, Any],
        world: WorldTimeline,
        snapshot: Optional[WorldSnapshot] = None,
    ) -> SkillResult:
        from runtime.task_store import TaskStatus

        import re

        job_id = inputs["job_id"]

        # Normalize: accept many formats the coordinator might emit:
        #   "J-3", "j-3", "3", "job 3", "#3", "job J-3"
        if isinstance(job_id, str):
            job_id = job_id.strip()
            # Try extracting J-N pattern first
            m = re.search(r'[Jj]-(\d+)', job_id)
            if m:
                job_id = f"J-{m.group(1)}"
            else:
                # Extract any bare number (from "job 3", "#3", "3")
                m = re.search(r'#?(\d+)', job_id)
                if m:
                    job_id = f"J-{m.group(1)}"
                else:
                    job_id = job_id.upper()

        # Look up by short_id
        task = self._store.get_by_short_id(job_id)

        if task is None:
            return SkillResult(
                outputs={"cancelled": f"No job found with ID {job_id}."},
                metadata={
                    "domain": "system",
                    "reason": "not_found",
                },
            )

        if task.status != TaskStatus.PENDING:
            return SkillResult(
                outputs={
                    "cancelled": (
                        f"Job {job_id} is {task.status.value} "
                        f"and cannot be cancelled."
                    ),
                },
                metadata={
                    "domain": "system",
                    "reason": "not_cancellable",
                },
            )

        success = self._store.cancel(task.id)

        if success:
            # Emit event for timeline tracking
            world.emit(
                source=self.contract.name,
                event_type="job_cancelled",
                payload={
                    "task_id": task.id,
                    "short_id": task.short_id,
                    "query": task.query,
                },
            )
            return SkillResult(
                outputs={
                    "cancelled": f"Cancelled {job_id}: {task.query[:50]}",
                },
                metadata={
                    "domain": "system",
                    "entity": f"job {job_id}",
                },
            )
        else:
            return SkillResult(
                outputs={
                    "cancelled": f"Could not cancel {job_id} — may already be completed.",
                },
                metadata={
                    "domain": "system",
                    "reason": "cancel_failed",
                },
            )
