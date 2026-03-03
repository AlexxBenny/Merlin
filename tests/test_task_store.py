# tests/test_task_store.py

"""
Tests for runtime/task_store.py — TaskStore protocol and InMemoryTaskStore.

Validates:
- Task creation and retrieval
- Status transitions
- Cancel semantics (only PENDING tasks)
- List by status filtering
- completed_at timestamp set on terminal states
- TaskType does NOT contain IMMEDIATE
- TaskSchedule epoch semantics
"""

import time

import pytest

from runtime.task_store import (
    InMemoryTaskStore,
    Task,
    TaskSchedule,
    TaskStatus,
    TaskType,
)


@pytest.fixture
def store():
    return InMemoryTaskStore()


def _make_task(
    task_id="t1",
    task_type=TaskType.DELAYED,
    query="pause after 10 seconds",
    **kwargs,
):
    return Task(
        id=task_id,
        type=task_type,
        query=query,
        **kwargs,
    )


# ─────────────────────────────────────────────────────────────
# TaskType invariants
# ─────────────────────────────────────────────────────────────

class TestTaskType:
    def test_no_immediate_in_task_type(self):
        """IMMEDIATE must not exist — immediate missions never enter TaskStore."""
        values = {t.value for t in TaskType}
        assert "immediate" not in values

    def test_all_persistent_types_exist(self):
        assert TaskType.DELAYED.value == "delayed"
        assert TaskType.SCHEDULED.value == "scheduled"
        assert TaskType.RECURRING.value == "recurring"
        assert TaskType.TRIGGERED.value == "triggered"


# ─────────────────────────────────────────────────────────────
# CRUD operations
# ─────────────────────────────────────────────────────────────

class TestInMemoryTaskStore:
    def test_create_and_get(self, store):
        task = _make_task()
        task_id = store.create(task)
        assert task_id == "t1"
        retrieved = store.get("t1")
        assert retrieved is not None
        assert retrieved.query == "pause after 10 seconds"
        assert retrieved.status == TaskStatus.PENDING

    def test_get_nonexistent(self, store):
        assert store.get("nonexistent") is None

    def test_task_count(self, store):
        assert store.task_count == 0
        store.create(_make_task("t1"))
        assert store.task_count == 1
        store.create(_make_task("t2"))
        assert store.task_count == 2

    def test_list_by_status(self, store):
        store.create(_make_task("t1"))
        store.create(_make_task("t2"))
        store.update_status("t1", TaskStatus.RUNNING)
        pending = store.list_by_status(TaskStatus.PENDING)
        running = store.list_by_status(TaskStatus.RUNNING)
        assert len(pending) == 1
        assert pending[0].id == "t2"
        assert len(running) == 1
        assert running[0].id == "t1"


# ─────────────────────────────────────────────────────────────
# Status transitions
# ─────────────────────────────────────────────────────────────

class TestStatusTransitions:
    def test_pending_to_running(self, store):
        store.create(_make_task())
        store.update_status("t1", TaskStatus.RUNNING)
        assert store.get("t1").status == TaskStatus.RUNNING

    def test_running_to_completed_sets_timestamp(self, store):
        store.create(_make_task())
        store.update_status("t1", TaskStatus.RUNNING)
        store.update_status("t1", TaskStatus.COMPLETED)
        task = store.get("t1")
        assert task.status == TaskStatus.COMPLETED
        assert task.completed_at is not None

    def test_running_to_failed_sets_error(self, store):
        store.create(_make_task())
        store.update_status("t1", TaskStatus.RUNNING)
        store.update_status("t1", TaskStatus.FAILED, error="timeout")
        task = store.get("t1")
        assert task.status == TaskStatus.FAILED
        assert task.error == "timeout"
        assert task.completed_at is not None

    def test_update_nonexistent_is_noop(self, store):
        """Updating a nonexistent task should not raise."""
        store.update_status("ghost", TaskStatus.RUNNING)  # No error


# ─────────────────────────────────────────────────────────────
# Cancel semantics
# ─────────────────────────────────────────────────────────────

class TestCancelSemantics:
    def test_cancel_pending(self, store):
        store.create(_make_task())
        assert store.cancel("t1") is True
        assert store.get("t1").status == TaskStatus.CANCELLED

    def test_cancel_running_fails(self, store):
        """Cannot cancel a running task."""
        store.create(_make_task())
        store.update_status("t1", TaskStatus.RUNNING)
        assert store.cancel("t1") is False
        assert store.get("t1").status == TaskStatus.RUNNING

    def test_cancel_completed_fails(self, store):
        store.create(_make_task())
        store.update_status("t1", TaskStatus.COMPLETED)
        assert store.cancel("t1") is False

    def test_cancel_nonexistent(self, store):
        assert store.cancel("ghost") is False


# ─────────────────────────────────────────────────────────────
# TaskSchedule
# ─────────────────────────────────────────────────────────────

class TestTaskSchedule:
    def test_delay_seconds(self):
        sched = TaskSchedule(delay_seconds=10)
        assert sched.delay_seconds == 10
        assert sched.schedule_at is None

    def test_schedule_at_epoch(self):
        """schedule_at must be epoch float, not string."""
        epoch = time.time() + 3600
        sched = TaskSchedule(schedule_at=epoch)
        assert isinstance(sched.schedule_at, float)

    def test_recurring(self):
        sched = TaskSchedule(repeat_interval=3600, max_repeats=24)
        assert sched.repeat_interval == 3600
        assert sched.max_repeats == 24

    def test_default_max_repeats_is_one(self):
        """Bounded iteration: default is 1 repeat."""
        sched = TaskSchedule()
        assert sched.max_repeats == 1


# ─────────────────────────────────────────────────────────────
# Task model
# ─────────────────────────────────────────────────────────────

class TestTaskModel:
    def test_default_status_is_pending(self):
        task = _make_task()
        assert task.status == TaskStatus.PENDING

    def test_created_at_is_populated(self):
        task = _make_task()
        assert task.created_at > 0

    def test_mission_data_is_dict_not_ir(self):
        """mission_data must be JSON-safe dict, not IR model."""
        task = _make_task(mission_data={"nodes": [{"skill": "test"}]})
        assert isinstance(task.mission_data, dict)

    def test_no_extra_fields_allowed(self):
        """Pydantic extra='forbid' must reject unknown fields."""
        with pytest.raises(Exception):
            Task(
                id="t1",
                type=TaskType.DELAYED,
                query="test",
                unknown_field="bad",
            )
