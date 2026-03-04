# tests/test_tick_scheduler.py

"""
Tests for runtime/tick_scheduler.py — TickSchedulerManager.

Validates:
- Submit computes next_run from schedule
- tick() returns due tasks and transitions to RUNNING
- Concurrency cap slicing (due[:available_slots], NOT blocking all)
- Priority-based dispatch ordering
- on_completion success → COMPLETED
- on_completion failure → retry with backoff (PENDING, not FAILED)
- on_completion failure after max_retries → FAILED (terminal)
- Recurring respawn with aligned next_run
- Boot recovery: RUNNING reset, overdue handling, missed_policy
- Recurring fast-forward (execute once, not N times)
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
from runtime.tick_scheduler import TickSchedulerManager


@pytest.fixture
def store():
    return InMemoryTaskStore()


@pytest.fixture
def scheduler(store):
    return TickSchedulerManager(
        store=store,
        max_concurrent_jobs=2,
        missed_job_staleness_seconds=300.0,
        default_max_retries=3,
    )


def _delayed_task(
    task_id="t1",
    delay=10,
    query="pause after 10 seconds",
    **kwargs,
):
    return Task(
        id=task_id,
        type=TaskType.DELAYED,
        query=query,
        schedule=TaskSchedule(delay_seconds=delay),
        **kwargs,
    )


def _scheduled_task(
    task_id="t1",
    at=None,
    query="mute at 3pm",
    **kwargs,
):
    return Task(
        id=task_id,
        type=TaskType.SCHEDULED,
        query=query,
        schedule=TaskSchedule(schedule_at=at or (time.time() + 3600)),
        **kwargs,
    )


def _recurring_task(
    task_id="t1",
    interval=60,
    max_repeats=10,
    query="check battery every minute",
    **kwargs,
):
    return Task(
        id=task_id,
        type=TaskType.RECURRING,
        query=query,
        schedule=TaskSchedule(
            repeat_interval=interval,
            max_repeats=max_repeats,
        ),
        **kwargs,
    )


# ─────────────────────────────────────────────────────────────
# Submit
# ─────────────────────────────────────────────────────────────

class TestSubmit:
    def test_submit_returns_task_id(self, scheduler):
        task = _delayed_task()
        result = scheduler.submit(task)
        assert result == "t1"

    def test_submit_computes_next_run_from_delay(self, scheduler, store):
        task = _delayed_task(delay=10)
        before = time.time()
        scheduler.submit(task)
        after = time.time()
        stored = store.get("t1")
        assert stored.next_run is not None
        assert before + 10 <= stored.next_run <= after + 10

    def test_submit_computes_next_run_from_schedule_at(self, scheduler, store):
        target = time.time() + 7200  # 2 hours from now
        task = _scheduled_task(at=target)
        scheduler.submit(task)
        assert store.get("t1").next_run == target

    def test_submit_preserves_existing_next_run(self, scheduler, store):
        task = _delayed_task()
        task.next_run = 12345.0
        scheduler.submit(task)
        assert store.get("t1").next_run == 12345.0

    def test_submit_sets_default_max_retries(self, scheduler, store):
        task = _delayed_task()
        scheduler.submit(task)
        assert store.get("t1").max_retries == 3  # From config

    def test_pending_count(self, scheduler):
        scheduler.submit(_delayed_task("a"))
        scheduler.submit(_delayed_task("b"))
        assert scheduler.pending_count() == 2


# ─────────────────────────────────────────────────────────────
# Tick — basic dispatch
# ─────────────────────────────────────────────────────────────

class TestTick:
    def test_returns_due_tasks(self, scheduler, store):
        task = _delayed_task()
        task.next_run = time.time() - 1  # Already due
        scheduler.submit(task)
        dispatched = scheduler.tick()
        assert len(dispatched) == 1
        assert dispatched[0].id == "t1"

    def test_transitions_to_running(self, scheduler, store):
        task = _delayed_task()
        task.next_run = time.time() - 1
        scheduler.submit(task)
        scheduler.tick()
        assert store.get("t1").status == TaskStatus.RUNNING

    def test_increments_attempts(self, scheduler, store):
        task = _delayed_task()
        task.next_run = time.time() - 1
        scheduler.submit(task)
        scheduler.tick()
        assert store.get("t1").attempts == 1

    def test_does_not_return_future_tasks(self, scheduler):
        task = _delayed_task()
        task.next_run = time.time() + 3600  # 1 hour from now
        scheduler.submit(task)
        dispatched = scheduler.tick()
        assert len(dispatched) == 0

    def test_does_not_return_running_tasks(self, scheduler, store):
        task = _delayed_task()
        task.next_run = time.time() - 1
        scheduler.submit(task)
        scheduler.tick()  # Now RUNNING
        dispatched = scheduler.tick()  # Should not return again
        assert len(dispatched) == 0


# ─────────────────────────────────────────────────────────────
# Concurrency cap
# ─────────────────────────────────────────────────────────────

class TestConcurrencyCap:
    def test_respects_cap(self, scheduler, store):
        """With cap=2, only 2 of 3 due tasks are dispatched."""
        for i in range(3):
            t = _delayed_task(f"t{i}")
            t.next_run = time.time() - 1
            scheduler.submit(t)

        dispatched = scheduler.tick()
        assert len(dispatched) == 2  # Cap is 2

    def test_slices_not_blocks(self, scheduler, store):
        """The 3rd task is deferred to next tick, not lost forever."""
        for i in range(3):
            t = _delayed_task(f"t{i}")
            t.next_run = time.time() - 1
            scheduler.submit(t)

        first = scheduler.tick()
        assert len(first) == 2

        # Complete the first two
        for task in first:
            scheduler.on_completion(task.id, success=True)

        # Third task should now be dispatchable
        second = scheduler.tick()
        assert len(second) == 1

    def test_no_dispatch_when_full(self, scheduler, store):
        """No dispatch when max_concurrent already running."""
        for i in range(2):
            t = _delayed_task(f"running_{i}")
            t.next_run = time.time() - 1
            scheduler.submit(t)
        scheduler.tick()  # Both now RUNNING

        # New due task
        t = _delayed_task("new")
        t.next_run = time.time() - 1
        scheduler.submit(t)

        dispatched = scheduler.tick()
        assert len(dispatched) == 0  # Full, deferred


# ─────────────────────────────────────────────────────────────
# Priority ordering
# ─────────────────────────────────────────────────────────────

class TestPriorityOrdering:
    def test_high_priority_first(self, scheduler, store):
        """High priority tasks dispatched before normal/low."""
        low = _delayed_task("low", priority="low")
        low.next_run = time.time() - 10
        high = _delayed_task("high", priority="high")
        high.next_run = time.time() - 5  # Due later but higher priority

        scheduler.submit(low)
        scheduler.submit(high)

        dispatched = scheduler.tick()
        assert dispatched[0].id == "high"


# ─────────────────────────────────────────────────────────────
# Completion — success
# ─────────────────────────────────────────────────────────────

class TestCompletionSuccess:
    def test_marks_completed(self, scheduler, store):
        task = _delayed_task()
        task.next_run = time.time() - 1
        scheduler.submit(task)
        scheduler.tick()
        scheduler.on_completion("t1", success=True)
        assert store.get("t1").status == TaskStatus.COMPLETED

    def test_completion_sets_timestamp(self, scheduler, store):
        task = _delayed_task()
        task.next_run = time.time() - 1
        scheduler.submit(task)
        scheduler.tick()
        scheduler.on_completion("t1", success=True)
        assert store.get("t1").completed_at is not None


# ─────────────────────────────────────────────────────────────
# Completion — failure with retry
# ─────────────────────────────────────────────────────────────

class TestRetry:
    def test_failure_retries_to_pending(self, scheduler, store):
        """First failure: transitions back to PENDING, not FAILED."""
        task = _delayed_task()
        task.next_run = time.time() - 1
        scheduler.submit(task)
        scheduler.tick()  # attempt 1

        scheduler.on_completion("t1", success=False, error="timeout")
        t = store.get("t1")
        assert t.status == TaskStatus.PENDING
        assert t.error == "timeout"

    def test_retry_has_backoff_delay(self, scheduler, store):
        """Retry sets future next_run (not immediate)."""
        task = _delayed_task()
        task.next_run = time.time() - 1
        scheduler.submit(task)
        scheduler.tick()

        before = time.time()
        scheduler.on_completion("t1", success=False, error="fail")
        t = store.get("t1")
        # Backoff base=2s, first retry = 2s
        assert t.next_run >= before + 1.5  # At least some backoff

    def test_retry_not_dispatched_same_tick(self, scheduler, store):
        """Retry task has future next_run — won't fire on immediate tick."""
        task = _delayed_task()
        task.next_run = time.time() - 1
        scheduler.submit(task)
        scheduler.tick()
        scheduler.on_completion("t1", success=False, error="fail")

        # Immediate tick should NOT dispatch the retry
        dispatched = scheduler.tick()
        assert len(dispatched) == 0

    def test_terminal_failure_after_max_retries(self, scheduler, store):
        """After max_retries exhausted: FAILED (terminal)."""
        task = _delayed_task()
        task.next_run = time.time() - 1
        scheduler.submit(task)

        # Exhaust all retries
        for i in range(3):
            scheduler.tick()
            scheduler.on_completion("t1", success=False, error=f"fail_{i}")
            t = store.get("t1")
            if t.status == TaskStatus.PENDING:
                # Set next_run to now for immediate retry
                t.next_run = time.time() - 1
                store.update_task(t)

        final = store.get("t1")
        assert final.status == TaskStatus.FAILED


# ─────────────────────────────────────────────────────────────
# Recurring respawn
# ─────────────────────────────────────────────────────────────

class TestRecurring:
    def test_respawn_on_success(self, scheduler, store):
        """Successful recurring task creates a new PENDING task."""
        task = _recurring_task(interval=60, max_repeats=5)
        task.next_run = time.time() - 1
        scheduler.submit(task)
        scheduler.tick()
        scheduler.on_completion("t1", success=True)

        # Should have the original (COMPLETED) + new task (PENDING)
        all_tasks = store.get_all()
        pending = [t for t in all_tasks if t.status == TaskStatus.PENDING]
        assert len(pending) == 1
        assert pending[0].type == TaskType.RECURRING

    def test_no_respawn_at_max_repeats(self, scheduler, store):
        """No respawn when max_repeats reached."""
        task = _recurring_task(interval=60, max_repeats=1)
        task.next_run = time.time() - 1
        scheduler.submit(task)
        scheduler.tick()
        scheduler.on_completion("t1", success=True)

        pending = store.list_by_status(TaskStatus.PENDING)
        assert len(pending) == 0


# ─────────────────────────────────────────────────────────────
# Cancel
# ─────────────────────────────────────────────────────────────

class TestCancel:
    def test_cancel_pending(self, scheduler, store):
        task = _delayed_task()
        task.next_run = time.time() + 3600
        scheduler.submit(task)
        assert scheduler.cancel("t1") is True
        assert store.get("t1").status == TaskStatus.CANCELLED

    def test_cancel_nonexistent(self, scheduler):
        assert scheduler.cancel("ghost") is False


# ─────────────────────────────────────────────────────────────
# Boot recovery
# ─────────────────────────────────────────────────────────────

class TestBootRecovery:
    def test_resets_running_to_pending(self, scheduler, store):
        """Tasks that were RUNNING on crash → reset to PENDING."""
        task = _delayed_task()
        task.next_run = time.time() - 1
        scheduler.submit(task)
        store.update_status("t1", TaskStatus.RUNNING)

        scheduler.recover()
        assert store.get("t1").status == TaskStatus.PENDING

    def test_recurring_fast_forward(self, scheduler, store):
        """Recurring job overdue by 3 hours → execute once, align future."""
        task = _recurring_task(interval=60)
        # Was due 3 hours ago (180 intervals missed)
        task.next_run = time.time() - (3 * 3600)
        scheduler.submit(task)

        scheduler.recover()

        t = store.get("t1")
        assert t.status == TaskStatus.PENDING
        # next_run should be in the near future, not 180 executions ago
        assert t.next_run > time.time() - 60  # Within 1 interval of now

    def test_skip_policy_cancels_stale(self, store):
        """Stale overdue task with skip policy → CANCELLED."""
        sched = TickSchedulerManager(
            store=store,
            missed_job_staleness_seconds=60,
        )
        task = _delayed_task()
        task.schedule = TaskSchedule(delay_seconds=10, missed_policy="skip")
        task.next_run = time.time() - 120  # 120s overdue, threshold is 60
        sched.submit(task)

        sched.recover()
        assert store.get("t1").status == TaskStatus.CANCELLED

    def test_execute_policy_keeps_pending(self, store):
        """Stale overdue task with execute policy → stays PENDING."""
        sched = TickSchedulerManager(
            store=store,
            missed_job_staleness_seconds=60,
        )
        task = _delayed_task()
        task.schedule = TaskSchedule(delay_seconds=10, missed_policy="execute")
        task.next_run = time.time() - 120
        sched.submit(task)

        sched.recover()
        assert store.get("t1").status == TaskStatus.PENDING


# ─────────────────────────────────────────────────────────────
# Fast-forward math
# ─────────────────────────────────────────────────────────────

class TestFastForward:
    def test_aligns_to_interval(self):
        """Fast-forward computes aligned future, not last + interval."""
        now = 1000.0
        last = 100.0  # 900s ago
        interval = 60  # Every 60s

        result = TickSchedulerManager._fast_forward_recurring(last, interval, now)
        # Should be in the future: last + (ceil(900/60)) * 60 = 100 + 960 = 1060
        assert result > now
        assert result <= now + interval

    def test_not_in_past(self):
        """Fast-forward result is always in the future."""
        now = 5000.0
        last = 100.0
        interval = 60

        result = TickSchedulerManager._fast_forward_recurring(last, interval, now)
        assert result > now
