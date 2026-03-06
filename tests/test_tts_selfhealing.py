# tests/test_tts_selfhealing.py

"""
Tests for Pyttsx3TTS self-healing, queue cap, and generation tracking.

These tests mock pyttsx3 to avoid requiring audio hardware.
"""

import queue
import threading
import time
from unittest.mock import MagicMock, patch

import pytest


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _make_tts():
    """Create a Pyttsx3TTS with mocked pyttsx3."""
    from reporting.engines.pyttsx3_tts import Pyttsx3TTS
    return Pyttsx3TTS(rate=175, voice_id=None)


# ─────────────────────────────────────────────────────────────
# Tests: Queue cap
# ─────────────────────────────────────────────────────────────

class TestQueueCap:
    """Queue size is bounded to prevent memory growth."""

    def test_queue_has_maxsize(self):
        tts = _make_tts()
        from reporting.engines.pyttsx3_tts import _MAX_QUEUE_SIZE
        assert tts._queue.maxsize == _MAX_QUEUE_SIZE

    def test_speak_drops_oldest_when_full(self):
        """When queue is full, oldest item is dropped to make room."""
        from reporting.engines.pyttsx3_tts import Pyttsx3TTS, _MAX_QUEUE_SIZE

        tts = Pyttsx3TTS(rate=175)
        # Mark as started with a dead worker to trigger self-heal
        # But override _ensure_worker to be a no-op for this test
        tts._started = True
        tts._worker = MagicMock()
        tts._worker.is_alive.return_value = True

        # Fill queue manually
        for i in range(_MAX_QUEUE_SIZE):
            tts._queue.put_nowait(f"item_{i}")

        assert tts._queue.full()

        # speak() should drop oldest and add new
        tts.speak("new_item")

        # Queue should still be full but contain new_item
        assert tts._queue.full()

        # Drain and check last item
        items = []
        while not tts._queue.empty():
            items.append(tts._queue.get_nowait())
        assert items[-1] == "new_item"
        assert items[0] == "item_1"  # item_0 was dropped


# ─────────────────────────────────────────────────────────────
# Tests: Generation tracking
# ─────────────────────────────────────────────────────────────

class TestGenerationTracking:
    """Worker generation increments on each start/restart."""

    def test_initial_generation_is_zero(self):
        tts = _make_tts()
        assert tts._generation == 0

    def test_generation_increments_on_ensure_worker(self):
        """Each _ensure_worker call that starts a thread increments gen."""
        tts = _make_tts()

        # Mock the worker thread to avoid actually starting pyttsx3
        with patch("threading.Thread") as mock_thread_cls:
            mock_thread = MagicMock()
            mock_thread.is_alive.return_value = True
            mock_thread_cls.return_value = mock_thread

            tts._ensure_worker()
            assert tts._generation == 1

            # Simulate worker death
            mock_thread.is_alive.return_value = False

            tts._ensure_worker()
            assert tts._generation == 2


# ─────────────────────────────────────────────────────────────
# Tests: Worker self-healing
# ─────────────────────────────────────────────────────────────

class TestWorkerSelfHealing:
    """Dead worker is detected and restarted on next speak()."""

    def test_dead_worker_triggers_restart(self):
        """When worker dies, next _ensure_worker spawns a new one."""
        tts = _make_tts()

        with patch("threading.Thread") as mock_thread_cls:
            # First worker
            mock_thread_1 = MagicMock()
            mock_thread_1.is_alive.return_value = True
            mock_thread_cls.return_value = mock_thread_1

            tts._ensure_worker()
            assert tts._started is True
            assert tts._generation == 1

            # Worker dies
            mock_thread_1.is_alive.return_value = False

            # Second worker
            mock_thread_2 = MagicMock()
            mock_thread_2.is_alive.return_value = True
            mock_thread_cls.return_value = mock_thread_2

            tts._ensure_worker()
            assert tts._generation == 2
            mock_thread_2.start.assert_called_once()

    def test_dead_worker_drains_stale_queue(self):
        """On restart, stale queue items are drained."""
        tts = _make_tts()

        with patch("threading.Thread") as mock_thread_cls:
            mock_thread = MagicMock()
            mock_thread.is_alive.return_value = True
            mock_thread_cls.return_value = mock_thread

            tts._ensure_worker()

            # Enqueue items
            tts._queue.put_nowait("stale_1")
            tts._queue.put_nowait("stale_2")

            # Worker dies
            mock_thread.is_alive.return_value = False

            # New worker
            mock_thread_2 = MagicMock()
            mock_thread_2.is_alive.return_value = True
            mock_thread_cls.return_value = mock_thread_2

            tts._ensure_worker()

            # Queue should be drained
            assert tts._queue.empty()

    def test_alive_worker_not_restarted(self):
        """Healthy worker is not touched."""
        tts = _make_tts()

        with patch("threading.Thread") as mock_thread_cls:
            mock_thread = MagicMock()
            mock_thread.is_alive.return_value = True
            mock_thread_cls.return_value = mock_thread

            tts._ensure_worker()
            gen1 = tts._generation

            tts._ensure_worker()
            assert tts._generation == gen1  # No restart

            # Thread.start called only once
            mock_thread.start.assert_called_once()
