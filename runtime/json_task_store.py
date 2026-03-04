# runtime/json_task_store.py

"""
JsonTaskStore — Persistent JSON-backed task store.

Write-aside persistence: in-memory dict for speed,
periodic flush to state/jobs/jobs.json on every mutation.

This is INFRASTRUCTURE, not cognition.
No LLM. No cortex. No IR imports.

Safety:
    - Atomic write (tmp → fsync → rename)
    - Corruption detection + backup recovery
    - Forensic preservation of corrupt files
    - Schema versioning for future migration
    - Thread-safe (RLock)
    - Monotonic short_id counter (J-1, J-2, ...)
"""

import json
import logging
import os
import shutil
import threading
import time as _time
from pathlib import Path
from typing import Dict, List, Optional

from runtime.task_store import (
    Task,
    TaskSchedule,
    TaskStatus,
    TaskStore,
    TaskType,
)

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1


class JsonTaskStore(TaskStore):
    """JSON-persisted task store with in-memory fast access.

    Architecture:
        - All reads: from in-memory dict (fast, O(1) by ID)
        - All writes: update dict, then flush to JSON (write-aside)
        - Boot: load from JSON file into memory
        - Thread-safe: RLock protects all operations

    File format:
        {
            "schema_version": 1,
            "next_counter": 5,
            "jobs": [
                { ... Task dict ... },
                ...
            ]
        }
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._lock = threading.RLock()
        self._tasks: Dict[str, Task] = {}
        self._counter: int = 0  # Monotonic counter for short_id

        # Ensure directory exists
        self._path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing state
        self._load()

    # ─────────────────────────────────────────────────────────
    # TaskStore interface implementation
    # ─────────────────────────────────────────────────────────

    def create(self, task: Task) -> str:
        with self._lock:
            # Assign short_id if not set
            if not task.short_id:
                self._counter += 1
                task.short_id = f"J-{self._counter}"
            self._tasks[task.id] = task
            self._flush()
            return task.id

    def get(self, task_id: str) -> Optional[Task]:
        with self._lock:
            return self._tasks.get(task_id)

    def update_status(
        self,
        task_id: str,
        status: TaskStatus,
        error: Optional[str] = None,
    ) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return
            task.status = status
            if error is not None:
                task.error = error
            if status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                task.completed_at = _time.time()
            self._flush()

    def list_by_status(self, status: TaskStatus) -> List[Task]:
        with self._lock:
            return [t for t in self._tasks.values() if t.status == status]

    def cancel(self, task_id: str) -> bool:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task.status != TaskStatus.PENDING:
                return False
            task.status = TaskStatus.CANCELLED
            self._flush()
            return True

    def get_due(self, now: float) -> List[Task]:
        with self._lock:
            return [
                t for t in self._tasks.values()
                if t.status == TaskStatus.PENDING
                and t.next_run is not None
                and t.next_run <= now
            ]

    def get_all(self) -> List[Task]:
        with self._lock:
            return list(self._tasks.values())

    def delete(self, task_id: str) -> bool:
        with self._lock:
            if task_id in self._tasks:
                del self._tasks[task_id]
                self._flush()
                return True
            return False

    def update_task(self, task: Task) -> None:
        with self._lock:
            if task.id in self._tasks:
                self._tasks[task.id] = task
                self._flush()

    @property
    def task_count(self) -> int:
        with self._lock:
            return len(self._tasks)

    # ─────────────────────────────────────────────────────────
    # Short ID lookup
    # ─────────────────────────────────────────────────────────

    def get_by_short_id(self, short_id: str) -> Optional[Task]:
        """Find a task by its human-friendly short ID (e.g., 'J-3')."""
        with self._lock:
            for task in self._tasks.values():
                if task.short_id == short_id:
                    return task
            return None

    # ─────────────────────────────────────────────────────────
    # Persistence — Atomic write + corruption recovery
    # ─────────────────────────────────────────────────────────

    def _flush(self) -> None:
        """Atomic write: tmp file → fsync → rename over original.

        If this fails, in-memory state is still correct.
        The file will be retried on next mutation.
        """
        data = {
            "schema_version": SCHEMA_VERSION,
            "next_counter": self._counter,
            "jobs": [
                task.model_dump(mode="json")
                for task in self._tasks.values()
            ],
        }

        tmp_path = self._path.with_suffix(".tmp")
        try:
            raw = json.dumps(data, indent=2, default=str)
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(raw)
                f.flush()
                os.fsync(f.fileno())

            # Atomic rename (on Windows, may need to delete target first)
            if os.name == "nt" and self._path.exists():
                # Keep a backup before overwriting
                backup = self._path.with_suffix(".bak")
                shutil.copy2(str(self._path), str(backup))
                self._path.unlink()

            tmp_path.rename(self._path)

        except Exception as e:
            logger.error(
                "[JSON_STORE] Flush failed: %s. In-memory state preserved.", e,
            )
            # Clean up tmp if it exists
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

    def _load(self) -> None:
        """Load tasks from JSON file.

        Recovery order:
            1. Try primary file
            2. Try backup (.bak)
            3. If both corrupt → preserve forensic data, start clean

        Never silently resets without preserving corrupt files.
        """
        data = self._try_load_file(self._path)

        if data is None:
            # Try backup
            backup = self._path.with_suffix(".bak")
            data = self._try_load_file(backup)
            if data is not None:
                logger.warning(
                    "[JSON_STORE] Primary file corrupt/missing. "
                    "Recovered from backup: %s", backup,
                )

        if data is None:
            # Both files corrupt or missing — start clean
            if self._path.exists():
                # Preserve forensic data
                corrupt_name = (
                    f"jobs.corrupt.{int(_time.time())}.json"
                )
                corrupt_path = self._path.parent / corrupt_name
                try:
                    shutil.copy2(str(self._path), str(corrupt_path))
                    logger.critical(
                        "[JSON_STORE] Primary and backup corrupted. "
                        "Preserved forensic data at %s. Starting clean.",
                        corrupt_path,
                    )
                except OSError:
                    logger.critical(
                        "[JSON_STORE] All files corrupt. Starting clean.",
                    )
            self._tasks = {}
            self._counter = 0
            return

        # Parse tasks from validated data
        self._counter = data.get("next_counter", 0)
        self._tasks = {}

        for job_dict in data.get("jobs", []):
            try:
                task = Task.model_validate(job_dict)
                self._tasks[task.id] = task
            except Exception as e:
                logger.warning(
                    "[JSON_STORE] Skipping corrupt task entry: %s", e,
                )

        logger.info(
            "[JSON_STORE] Loaded %d tasks (counter=%d) from %s",
            len(self._tasks), self._counter, self._path,
        )

    @staticmethod
    def _try_load_file(path: Path) -> Optional[dict]:
        """Try to load and validate a JSON file.

        Returns parsed dict or None if corrupt/missing.
        """
        if not path.exists():
            return None

        try:
            raw = path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(
                "[JSON_STORE] Cannot parse %s: %s", path, e,
            )
            return None

        # Basic schema validation
        if not isinstance(data, dict):
            return None
        if "schema_version" not in data:
            return None
        if data["schema_version"] != SCHEMA_VERSION:
            logger.warning(
                "[JSON_STORE] Schema version mismatch: expected %d, got %s",
                SCHEMA_VERSION, data.get("schema_version"),
            )
            return None
        if not isinstance(data.get("jobs"), list):
            return None

        return data
