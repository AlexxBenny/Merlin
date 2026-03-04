# tests/test_job_skills.py

"""
Tests for Phase 11 job query skills and Phase 10 isolation changes.

Matrix:
    A. ListJobsSkill (5 tests)
       - Empty store
       - Populated with pending/running jobs
       - Output structure matches SkillResult contract
       - Output includes all required fields
       - Only pending+running returned (not completed/cancelled)

    B. CancelJobSkill (7 tests)
       - Cancel by short_id "J-3"
       - Cancel by lowercase "j-3"
       - Cancel by bare number "3"
       - Cancel by "job 3" format
       - Cancel by "#3" format
       - Not found
       - Already completed (not cancellable)

    C. _build_job_summary (4 tests)
       - Extracts text from reasoning node
       - Falls back to long string output
       - Falls back to deferred query
       - Handles missing results attribute

    D. _drain_completions routing (4 tests)
       - INTERRUPT → delivered immediately
       - QUEUE → enqueued on AttentionManager
       - SUPPRESS → skipped
       - Failed event → error text

    E. ConsoleOutputChannel thread safety (1 test)
       - Concurrent sends don't crash
"""

import threading
import time

import pytest
from unittest.mock import MagicMock, patch

from skills.skill_result import SkillResult
from runtime.task_store import Task, TaskSchedule, TaskStatus, TaskType


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────


def _make_store(tasks=None):
    """Create a mock task_store with controllable list_by_status."""
    store = MagicMock()
    all_tasks = tasks or []

    def list_by_status(status):
        return [t for t in all_tasks if t.status == status]

    store.list_by_status.side_effect = list_by_status

    def get_by_short_id(short_id):
        for t in all_tasks:
            if t.short_id == short_id:
                return t
        return None

    store.get_by_short_id.side_effect = get_by_short_id
    store.cancel.return_value = True
    store.get_all.return_value = all_tasks
    return store


def _make_task(
    task_id="t1",
    short_id="J-1",
    query="remind to drink water",
    status=TaskStatus.PENDING,
    task_type=TaskType.DELAYED,
    next_run=None,
    created_at=None,
):
    return Task(
        id=task_id,
        type=task_type,
        status=status,
        query=query,
        short_id=short_id,
        next_run=next_run or time.time() + 60,
        created_at=created_at or time.time(),
    )


def _make_timeline():
    """Mock WorldTimeline."""
    timeline = MagicMock()
    return timeline


# ──────────────────────────────────────────────────────────────
# A. ListJobsSkill
# ──────────────────────────────────────────────────────────────


class TestListJobsSkill:
    """Tests for system.list_jobs skill."""

    def test_empty_store_returns_empty_list(self):
        """No pending/running jobs → empty list."""
        from skills.system.list_jobs import ListJobsSkill

        skill = ListJobsSkill(task_store=_make_store([]))
        result = skill.execute({}, _make_timeline())

        assert isinstance(result, SkillResult)
        assert result.outputs["jobs"] == []

    def test_returns_pending_jobs(self):
        """Pending jobs appear in output."""
        from skills.system.list_jobs import ListJobsSkill

        tasks = [
            _make_task("a", "J-1", "remind water"),
            _make_task("b", "J-2", "play music"),
        ]
        skill = ListJobsSkill(task_store=_make_store(tasks))
        result = skill.execute({}, _make_timeline())

        jobs = result.outputs["jobs"]
        assert len(jobs) == 2
        assert jobs[0]["short_id"] == "J-1"
        assert jobs[1]["short_id"] == "J-2"

    def test_returns_running_jobs(self):
        """Running jobs also appear."""
        from skills.system.list_jobs import ListJobsSkill

        tasks = [
            _make_task("a", "J-1", "test", status=TaskStatus.RUNNING),
        ]
        skill = ListJobsSkill(task_store=_make_store(tasks))
        result = skill.execute({}, _make_timeline())

        jobs = result.outputs["jobs"]
        assert len(jobs) == 1
        assert jobs[0]["status"] == "running"

    def test_excludes_completed_cancelled(self):
        """Completed and cancelled jobs do not appear."""
        from skills.system.list_jobs import ListJobsSkill

        tasks = [
            _make_task("a", "J-1", "done", status=TaskStatus.COMPLETED),
            _make_task("b", "J-2", "nope", status=TaskStatus.CANCELLED),
            _make_task("c", "J-3", "active", status=TaskStatus.PENDING),
        ]
        skill = ListJobsSkill(task_store=_make_store(tasks))
        result = skill.execute({}, _make_timeline())

        jobs = result.outputs["jobs"]
        assert len(jobs) == 1
        assert jobs[0]["short_id"] == "J-3"

    def test_output_contains_all_required_fields(self):
        """Each job dict has: short_id, query, status, type, created_at, next_run, scheduled_at."""
        from skills.system.list_jobs import ListJobsSkill

        now = time.time()
        tasks = [_make_task("a", "J-1", "test query", next_run=now + 60, created_at=now)]
        skill = ListJobsSkill(task_store=_make_store(tasks))
        result = skill.execute({}, _make_timeline())

        job = result.outputs["jobs"][0]
        required_keys = {"short_id", "query", "status", "type", "created_at", "next_run", "scheduled_at"}
        assert required_keys.issubset(set(job.keys())), (
            f"Missing keys: {required_keys - set(job.keys())}"
        )
        assert job["short_id"] == "J-1"
        assert job["query"] == "test query"
        assert job["status"] == "pending"
        assert job["type"] == "delayed"
        assert isinstance(job["created_at"], float)
        assert isinstance(job["next_run"], float)

    def test_contract_matches_skill(self):
        """Skill contract outputs match what execute() returns."""
        from skills.system.list_jobs import ListJobsSkill

        skill = ListJobsSkill(task_store=_make_store([]))
        result = skill.execute({}, _make_timeline())

        # Contract declares {"jobs": "job_list"}
        assert "jobs" in result.outputs
        assert set(result.outputs.keys()) <= set(skill.contract.outputs.keys())


# ──────────────────────────────────────────────────────────────
# B. CancelJobSkill
# ──────────────────────────────────────────────────────────────


class TestCancelJobSkill:
    """Tests for system.cancel_job skill."""

    def _make_skill_and_store(self, tasks=None):
        from skills.system.cancel_job import CancelJobSkill
        store = _make_store(tasks or [])
        skill = CancelJobSkill(task_store=store)
        return skill, store

    def test_cancel_by_short_id(self):
        """Standard J-3 format."""
        tasks = [_make_task("a", "J-3", "remind water")]
        skill, store = self._make_skill_and_store(tasks)

        result = skill.execute({"job_id": "J-3"}, _make_timeline())

        store.cancel.assert_called_once_with("a")
        assert "Cancelled" in result.outputs["cancelled"]
        assert "J-3" in result.outputs["cancelled"]

    def test_cancel_by_lowercase(self):
        """Lowercase j-3."""
        tasks = [_make_task("a", "J-3", "remind water")]
        skill, store = self._make_skill_and_store(tasks)

        result = skill.execute({"job_id": "j-3"}, _make_timeline())

        store.cancel.assert_called_once_with("a")

    def test_cancel_by_bare_number(self):
        """Just '3'."""
        tasks = [_make_task("a", "J-3", "remind water")]
        skill, store = self._make_skill_and_store(tasks)

        result = skill.execute({"job_id": "3"}, _make_timeline())

        store.cancel.assert_called_once_with("a")

    def test_cancel_by_job_number_format(self):
        """'job 3' format that coordinator may emit."""
        tasks = [_make_task("a", "J-3", "remind water")]
        skill, store = self._make_skill_and_store(tasks)

        result = skill.execute({"job_id": "job 3"}, _make_timeline())

        store.cancel.assert_called_once_with("a")

    def test_cancel_by_hash_format(self):
        """'#3' format."""
        tasks = [_make_task("a", "J-3", "remind water")]
        skill, store = self._make_skill_and_store(tasks)

        result = skill.execute({"job_id": "#3"}, _make_timeline())

        store.cancel.assert_called_once_with("a")

    def test_cancel_not_found(self):
        """Non-existent job ID."""
        skill, store = self._make_skill_and_store([])

        result = skill.execute({"job_id": "J-999"}, _make_timeline())

        store.cancel.assert_not_called()
        assert "No job found" in result.outputs["cancelled"]

    def test_cancel_already_completed(self):
        """Completed jobs cannot be cancelled."""
        tasks = [_make_task("a", "J-1", "done", status=TaskStatus.COMPLETED)]
        skill, store = self._make_skill_and_store(tasks)

        result = skill.execute({"job_id": "J-1"}, _make_timeline())

        store.cancel.assert_not_called()
        assert "cannot be cancelled" in result.outputs["cancelled"]

    def test_cancel_emits_event(self):
        """Successful cancel emits job_cancelled to timeline."""
        tasks = [_make_task("a", "J-2", "remind water")]
        skill, store = self._make_skill_and_store(tasks)
        timeline = _make_timeline()

        skill.execute({"job_id": "J-2"}, timeline)

        timeline.emit.assert_called_once()
        call_args = timeline.emit.call_args
        assert call_args.kwargs["event_type"] == "job_cancelled"
        assert call_args.kwargs["source"] == "system.cancel_job"

    def test_contract_matches_skill(self):
        """Skill contract outputs match what execute() returns."""
        from skills.system.cancel_job import CancelJobSkill
        skill = CancelJobSkill(task_store=_make_store([]))

        result = skill.execute({"job_id": "J-999"}, _make_timeline())

        assert "cancelled" in result.outputs
        assert set(result.outputs.keys()) <= set(skill.contract.outputs.keys())


# ──────────────────────────────────────────────────────────────
# C. _build_job_summary
# ──────────────────────────────────────────────────────────────


class TestBuildJobSummary:
    """Tests for RuntimeEventLoop._build_job_summary."""

    @staticmethod
    def _call(query, results_dict):
        from runtime.event_loop import RuntimeEventLoop
        mock_result = MagicMock()
        mock_result.results = results_dict
        return RuntimeEventLoop._build_job_summary(query, mock_result)

    def test_prefers_text_output(self):
        """If a node has 'text' output, use that."""
        summary = self._call("remind water", {
            "n1": {"text": "Time to drink some water!"},
        })
        assert summary == "Time to drink some water!"

    def test_ignores_short_text(self):
        """Text shorter than 6 chars is not considered meaningful."""
        summary = self._call("test query", {
            "n1": {"text": "OK"},
        })
        # Should NOT use "OK" — too short
        # Falls through to long-string check or query fallback
        assert "test query" in summary or len(summary) > 5

    def test_falls_back_to_long_string(self):
        """If no 'text' key, use any string > 20 chars."""
        summary = self._call("test query", {
            "n1": {"result": "This is a sufficiently long output string."},
        })
        assert summary == "This is a sufficiently long output string."

    def test_falls_back_to_query(self):
        """If no meaningful string outputs, use the deferred query."""
        summary = self._call("remind to stretch", {
            "n1": {"count": 42, "flag": True},
        })
        assert "remind to stretch" in summary

    def test_handles_no_results_attribute(self):
        """If exec_result has no .results, use query."""
        from runtime.event_loop import RuntimeEventLoop
        mock_result = MagicMock(spec=[])  # no attributes at all
        summary = RuntimeEventLoop._build_job_summary(
            "test query", mock_result,
        )
        assert "test query" in summary


# ──────────────────────────────────────────────────────────────
# D. _drain_completions routing
# ──────────────────────────────────────────────────────────────


class TestDrainCompletions:
    """Tests for RuntimeEventLoop._drain_completions AttentionManager routing."""

    def _make_event_loop(self, attention_decision):
        """Build a minimal RuntimeEventLoop with mocked deps."""
        from runtime.event_loop import RuntimeEventLoop
        from runtime.completion_event import CompletionEvent

        timeline = MagicMock()
        timeline.all_events.return_value = []

        loop = RuntimeEventLoop.__new__(RuntimeEventLoop)
        loop.timeline = timeline
        loop._scheduler = None
        loop._job_executor = None
        loop._polling_interval = 5.0
        loop._running = False
        loop._thread = None
        loop._event_sources = []
        loop.output_channel = MagicMock()

        # Mock completion queue
        event = CompletionEvent(
            task_id="t1",
            short_id="J-1",
            query="remind water",
            status="completed",
            error=None,
            output="Time to drink water!",
            completed_at=time.time(),
        )
        loop._completion_queue = MagicMock()
        loop._completion_queue.drain.return_value = [event]

        # Mock attention manager
        from runtime.attention import AttentionDecision
        loop.attention_manager = MagicMock()
        loop.attention_manager.decide.return_value = attention_decision

        return loop, event

    def test_interrupt_delivers_immediately(self):
        """INTERRUPT → attention_manager.deliver() called."""
        from runtime.attention import AttentionDecision
        loop, event = self._make_event_loop(AttentionDecision.INTERRUPT)

        loop._drain_completions()

        loop.attention_manager.deliver.assert_called_once()
        delivered_text = loop.attention_manager.deliver.call_args[0][0]
        assert "water" in delivered_text

    def test_queue_enqueues(self):
        """QUEUE → attention_manager.enqueue() called."""
        from runtime.attention import AttentionDecision
        loop, event = self._make_event_loop(AttentionDecision.QUEUE)

        loop._drain_completions()

        loop.attention_manager.enqueue.assert_called_once()
        loop.attention_manager.deliver.assert_not_called()

    def test_suppress_skips(self):
        """SUPPRESS → neither deliver nor enqueue."""
        from runtime.attention import AttentionDecision
        loop, event = self._make_event_loop(AttentionDecision.SUPPRESS)

        loop._drain_completions()

        loop.attention_manager.deliver.assert_not_called()
        loop.attention_manager.enqueue.assert_not_called()

    def test_failed_event_uses_error_text(self):
        """Failed events generate error text, not output text."""
        from runtime.event_loop import RuntimeEventLoop
        from runtime.completion_event import CompletionEvent
        from runtime.attention import AttentionDecision

        loop, _ = self._make_event_loop(AttentionDecision.INTERRUPT)

        # Replace with a failed event
        failed_event = CompletionEvent(
            task_id="t1",
            short_id="J-1",
            query="bad job",
            status="failed",
            error="Timeout after 30s",
            output=None,
            completed_at=time.time(),
        )
        loop._completion_queue.drain.return_value = [failed_event]

        loop._drain_completions()

        delivered_text = loop.attention_manager.deliver.call_args[0][0]
        assert "failed" in delivered_text.lower()
        assert "bad job" in delivered_text

    def test_no_attention_manager_falls_back_to_output_channel(self):
        """Without AttentionManager, drain sends to output_channel directly."""
        from runtime.attention import AttentionDecision
        loop, event = self._make_event_loop(AttentionDecision.INTERRUPT)
        loop.attention_manager = None

        loop._drain_completions()

        loop.output_channel.send.assert_called_once()


# ──────────────────────────────────────────────────────────────
# E. ConsoleOutputChannel thread safety
# ──────────────────────────────────────────────────────────────


class TestConsoleOutputChannelThreadSafety:
    """Verify ConsoleOutputChannel.send() doesn't crash under concurrency."""

    def test_concurrent_sends_no_crash(self, capsys):
        """50 concurrent sends must all complete without exception."""
        from reporting.output import ConsoleOutputChannel

        channel = ConsoleOutputChannel(prefix="TEST")
        errors = []

        def send_msg(i):
            try:
                channel.send(f"Message {i}")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=send_msg, args=(i,))
            for i in range(50)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        captured = capsys.readouterr()
        # All 50 messages should appear (no lost output)
        assert captured.out.count("[TEST]") == 50
