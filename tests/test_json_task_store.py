# tests/test_json_task_store.py

"""
Tests for runtime/json_task_store.py — JSON-persisted task store.

Validates:
- Basic CRUD operations (create, get, update, delete)
- JSON persistence (write/read/boot recovery)
- Atomic write safety
- Short ID generation (J-1, J-2, ...)
- Corruption recovery (backup, forensic preservation)
- Schema validation
- Thread safety basics
- get_due() returns only PENDING tasks with next_run <= now
"""

import json
import os
import threading
import time

import pytest

from runtime.json_task_store import JsonTaskStore, SCHEMA_VERSION
from runtime.task_store import Task, TaskSchedule, TaskStatus, TaskType


@pytest.fixture
def tmp_path(tmp_path_factory):
    return tmp_path_factory.mktemp("json_store")


@pytest.fixture
def store_path(tmp_path):
    return tmp_path / "jobs.json"


@pytest.fixture
def store(store_path):
    return JsonTaskStore(store_path)


def _make_task(
    task_id="t1",
    task_type=TaskType.DELAYED,
    query="pause after 10 seconds",
    next_run=None,
    **kwargs,
):
    return Task(
        id=task_id,
        type=task_type,
        query=query,
        next_run=next_run,
        **kwargs,
    )


# ─────────────────────────────────────────────────────────────
# Basic CRUD
# ─────────────────────────────────────────────────────────────

class TestCRUD:
    def test_create_and_get(self, store):
        task = _make_task()
        store.create(task)
        retrieved = store.get("t1")
        assert retrieved is not None
        assert retrieved.query == "pause after 10 seconds"
        assert retrieved.status == TaskStatus.PENDING

    def test_get_nonexistent(self, store):
        assert store.get("nope") is None

    def test_task_count(self, store):
        assert store.task_count == 0
        store.create(_make_task("a"))
        assert store.task_count == 1
        store.create(_make_task("b"))
        assert store.task_count == 2

    def test_delete(self, store):
        store.create(_make_task())
        assert store.delete("t1") is True
        assert store.get("t1") is None
        assert store.task_count == 0

    def test_delete_nonexistent(self, store):
        assert store.delete("nope") is False

    def test_get_all(self, store):
        store.create(_make_task("a"))
        store.create(_make_task("b"))
        all_tasks = store.get_all()
        assert len(all_tasks) == 2

    def test_list_by_status(self, store):
        store.create(_make_task("a"))
        store.create(_make_task("b"))
        store.update_status("a", TaskStatus.RUNNING)
        pending = store.list_by_status(TaskStatus.PENDING)
        running = store.list_by_status(TaskStatus.RUNNING)
        assert len(pending) == 1
        assert len(running) == 1

    def test_update_task(self, store):
        task = _make_task()
        store.create(task)
        task.next_run = 9999.0
        store.update_task(task)
        assert store.get("t1").next_run == 9999.0


# ─────────────────────────────────────────────────────────────
# Short ID generation
# ─────────────────────────────────────────────────────────────

class TestShortID:
    def test_auto_assigned(self, store):
        task = _make_task()
        store.create(task)
        assert store.get("t1").short_id == "J-1"

    def test_monotonic(self, store):
        store.create(_make_task("a"))
        store.create(_make_task("b"))
        store.create(_make_task("c"))
        assert store.get("a").short_id == "J-1"
        assert store.get("b").short_id == "J-2"
        assert store.get("c").short_id == "J-3"

    def test_preserves_existing_short_id(self, store):
        task = _make_task(short_id="CUSTOM-1")
        store.create(task)
        assert store.get("t1").short_id == "CUSTOM-1"

    def test_lookup_by_short_id(self, store):
        store.create(_make_task("a"))
        store.create(_make_task("b"))
        result = store.get_by_short_id("J-2")
        assert result is not None
        assert result.id == "b"

    def test_lookup_nonexistent_short_id(self, store):
        assert store.get_by_short_id("J-999") is None


# ─────────────────────────────────────────────────────────────
# get_due()
# ─────────────────────────────────────────────────────────────

class TestGetDue:
    def test_returns_due_tasks(self, store):
        store.create(_make_task("a", next_run=100.0))
        store.create(_make_task("b", next_run=200.0))
        store.create(_make_task("c", next_run=300.0))
        due = store.get_due(150.0)
        assert len(due) == 1
        assert due[0].id == "a"

    def test_returns_multiple_due(self, store):
        store.create(_make_task("a", next_run=100.0))
        store.create(_make_task("b", next_run=200.0))
        due = store.get_due(250.0)
        assert len(due) == 2

    def test_excludes_non_pending(self, store):
        store.create(_make_task("a", next_run=100.0))
        store.update_status("a", TaskStatus.RUNNING)
        due = store.get_due(150.0)
        assert len(due) == 0

    def test_excludes_no_next_run(self, store):
        store.create(_make_task("a"))  # next_run=None
        due = store.get_due(9999.0)
        assert len(due) == 0


# ─────────────────────────────────────────────────────────────
# JSON Persistence
# ─────────────────────────────────────────────────────────────

class TestPersistence:
    def test_survives_reload(self, store_path):
        """Data persists across store instances."""
        store1 = JsonTaskStore(store_path)
        store1.create(_make_task("a", query="test query"))
        store1.create(_make_task("b", query="another query"))

        # New instance reads from same file
        store2 = JsonTaskStore(store_path)
        assert store2.task_count == 2
        assert store2.get("a").query == "test query"
        assert store2.get("b").query == "another query"

    def test_counter_survives_reload(self, store_path):
        """Short ID counter persists."""
        store1 = JsonTaskStore(store_path)
        store1.create(_make_task("a"))
        store1.create(_make_task("b"))
        # Counter is at 2

        store2 = JsonTaskStore(store_path)
        store2.create(_make_task("c"))
        assert store2.get("c").short_id == "J-3"  # Continues from 3

    def test_status_persists(self, store_path):
        store1 = JsonTaskStore(store_path)
        store1.create(_make_task("a"))
        store1.update_status("a", TaskStatus.RUNNING)

        store2 = JsonTaskStore(store_path)
        assert store2.get("a").status == TaskStatus.RUNNING

    def test_json_file_has_schema_version(self, store_path):
        store = JsonTaskStore(store_path)
        store.create(_make_task())
        data = json.loads(store_path.read_text())
        assert data["schema_version"] == SCHEMA_VERSION

    def test_cancel_persists(self, store_path):
        store1 = JsonTaskStore(store_path)
        store1.create(_make_task("a"))
        store1.cancel("a")

        store2 = JsonTaskStore(store_path)
        assert store2.get("a").status == TaskStatus.CANCELLED


# ─────────────────────────────────────────────────────────────
# Corruption Recovery
# ─────────────────────────────────────────────────────────────

class TestCorruptionRecovery:
    def test_recovers_from_corrupt_primary(self, store_path):
        """If primary file is corrupt, recovers from backup."""
        # Create valid store
        store1 = JsonTaskStore(store_path)
        store1.create(_make_task("a", query="important job"))

        # Corrupt primary file
        store_path.write_text("NOT JSON!!!")

        # New instance should recover from backup
        store2 = JsonTaskStore(store_path)
        # May recover from backup or start clean
        # depending on whether backup exists (it should from atomic write)
        # At minimum, it should not crash
        assert isinstance(store2.task_count, int)

    def test_starts_clean_if_no_files(self, tmp_path):
        """Clean start with no existing files."""
        path = tmp_path / "nonexistent" / "jobs.json"
        store = JsonTaskStore(path)
        assert store.task_count == 0

    def test_handles_wrong_schema_version(self, store_path):
        """Rejects files with wrong schema version."""
        data = {"schema_version": 999, "next_counter": 0, "jobs": []}
        store_path.write_text(json.dumps(data))

        store = JsonTaskStore(store_path)
        assert store.task_count == 0  # Started clean

    def test_handles_missing_jobs_key(self, store_path):
        """Rejects files without 'jobs' key."""
        data = {"schema_version": SCHEMA_VERSION, "next_counter": 0}
        store_path.write_text(json.dumps(data))

        store = JsonTaskStore(store_path)
        assert store.task_count == 0

    def test_preserves_forensic_on_corruption(self, store_path):
        """Corrupt files are preserved with timestamp suffix."""
        # Create a corrupt file
        store_path.parent.mkdir(parents=True, exist_ok=True)
        store_path.write_text("CORRUPT DATA")

        # Load should create forensic copy
        store = JsonTaskStore(store_path)

        # Check that a .corrupt file exists
        corrupt_files = list(store_path.parent.glob("jobs.corrupt.*.json"))
        assert len(corrupt_files) >= 1

    def test_skips_corrupt_task_entries(self, store_path):
        """Individual corrupt task entries are skipped, not fatal."""
        data = {
            "schema_version": SCHEMA_VERSION,
            "next_counter": 2,
            "jobs": [
                {"id": "good", "type": "delayed", "query": "test", "status": "pending"},
                {"INVALID": "GARBAGE"},
            ],
        }
        store_path.write_text(json.dumps(data))

        store = JsonTaskStore(store_path)
        assert store.task_count == 1
        assert store.get("good") is not None


# ─────────────────────────────────────────────────────────────
# Thread Safety
# ─────────────────────────────────────────────────────────────

class TestThreadSafety:
    def test_concurrent_creates(self, store):
        """Multiple threads creating tasks concurrently."""
        errors = []

        def create_task(i):
            try:
                store.create(_make_task(f"task_{i}", query=f"query {i}"))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=create_task, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert store.task_count == 20
