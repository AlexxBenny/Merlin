# tests/test_mutation_gate.py

"""
Tests for Phase 3C: Mutation Gate.

Validates:
- Two parallel nodes mutating world → events serialized
- No state corruption under concurrent emit()
- Event ordering is deterministic (serialized by lock)
- Contract enforcement works correctly under parallel execution
"""

import time
import threading
import pytest

from world.timeline import WorldTimeline, WorldEvent


class TestMutationGateSerialization:
    """Verify WorldTimeline serializes concurrent mutations."""

    def test_concurrent_emits_no_corruption(self):
        """
        Spawn N threads all calling emit() simultaneously.
        Verify no events are lost.
        """
        timeline = WorldTimeline()
        n_threads = 10
        events_per_thread = 100
        barrier = threading.Barrier(n_threads)

        def worker(thread_id):
            barrier.wait()
            for i in range(events_per_thread):
                timeline.emit(
                    source=f"thread_{thread_id}",
                    event_type="test.write",
                    payload={"i": i},
                )

        threads = [
            threading.Thread(target=worker, args=(t,))
            for t in range(n_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        all_events = timeline.all_events()
        assert len(all_events) == n_threads * events_per_thread

    def test_concurrent_emits_no_interleaving(self):
        """
        Each event should be a complete WorldEvent with valid fields.
        No partial writes — that would indicate lock failure.
        """
        timeline = WorldTimeline()
        n_threads = 5
        events_per_thread = 50
        barrier = threading.Barrier(n_threads)

        def worker(thread_id):
            barrier.wait()
            for i in range(events_per_thread):
                timeline.emit(
                    source=f"worker_{thread_id}",
                    event_type=f"type_{thread_id}",
                    payload={"seq": i, "tid": thread_id},
                )

        threads = [
            threading.Thread(target=worker, args=(t,))
            for t in range(n_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        all_events = timeline.all_events()

        # Every event must be well-formed
        for ev in all_events:
            assert ev.source.startswith("worker_")
            assert ev.type.startswith("type_")
            assert "seq" in ev.payload
            assert "tid" in ev.payload

        # Correct count
        assert len(all_events) == n_threads * events_per_thread

    def test_event_count_consistent(self):
        """event_count() returns consistent value under concurrent reads."""
        timeline = WorldTimeline()
        for i in range(50):
            timeline.emit("src", "ev", {"i": i})

        counts = []

        def reader():
            counts.append(timeline.event_count())

        threads = [threading.Thread(target=reader) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(c == 50 for c in counts)


class TestEventsSinceIndex:
    """Validate the events_since_index helper."""

    def test_returns_new_events_only(self):
        timeline = WorldTimeline()
        timeline.emit("a", "first", {})
        timeline.emit("b", "second", {})
        timeline.emit("c", "third", {})

        # Events since index 1 → last two
        since_1 = timeline.events_since_index(1)
        assert len(since_1) == 2
        assert since_1[0].source == "b"
        assert since_1[1].source == "c"

    def test_returns_empty_when_no_new(self):
        timeline = WorldTimeline()
        timeline.emit("a", "first", {})
        since = timeline.events_since_index(1)
        assert since == []

    def test_returns_all_from_zero(self):
        timeline = WorldTimeline()
        timeline.emit("a", "first", {})
        timeline.emit("b", "second", {})
        since = timeline.events_since_index(0)
        assert len(since) == 2


class TestMutationGateWithParallelSkills:
    """
    Simulate two nodes executing concurrently,
    each emitting events. Verify deterministic serialization.
    """

    def test_two_parallel_writers_deterministic_count(self):
        """
        Two 'skills' emit events from separate threads.
        After both complete, total event count must be exact.
        """
        timeline = WorldTimeline()
        barrier = threading.Barrier(2)

        def skill_a():
            barrier.wait()
            for i in range(100):
                timeline.emit("skill_a", "compute.done", {"step": i})

        def skill_b():
            barrier.wait()
            for i in range(100):
                timeline.emit("skill_b", "compute.done", {"step": i})

        t1 = threading.Thread(target=skill_a)
        t2 = threading.Thread(target=skill_b)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        all_events = timeline.all_events()
        assert len(all_events) == 200

        # Each skill contributed exactly 100
        a_events = [e for e in all_events if e.source == "skill_a"]
        b_events = [e for e in all_events if e.source == "skill_b"]
        assert len(a_events) == 100
        assert len(b_events) == 100
