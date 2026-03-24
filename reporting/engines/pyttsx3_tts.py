# reporting/engines/pyttsx3_tts.py

"""
Pyttsx3TTS — offline TTS engine using pyttsx3 (Windows SAPI5).

Thread-safe via dedicated worker thread + per-call engine re-init.

Two problems, two fixes:

1. COM apartment affinity (threading):
   pyttsx3 uses SAPI5 via COM. COM objects created in one thread's
   apartment cannot be used from another. MERLIN's multi-threaded
   runtime corrupts the COM state if the engine is shared.
   Fix: dedicated daemon worker thread owns all COM interactions.

2. Engine state corruption (pyttsx3 bug):
   After runAndWait() completes, the pyttsx3 internal event loop
   state becomes invalid. Subsequent runAndWait() returns in ~100ms
   without producing audio — a known pyttsx3/SAPI5 bug.
   Fix: re-create the engine for each speak call. Each cycle gets
   a fresh COM event loop state. Slightly slower (~300ms init
   overhead) but guaranteed correct.
"""

import logging
import queue
import threading
from typing import Optional

from reporting.tts_engine import TTSEngine


logger = logging.getLogger(__name__)

# Sentinel value to signal the worker thread to shut down.
_SHUTDOWN = object()


# Max queued speech items — prevents memory growth if worker dies.
_MAX_QUEUE_SIZE = 50


class Pyttsx3TTS(TTSEngine):
    """
    TTSEngine backed by pyttsx3, with dedicated worker thread.

    Engine is re-created per speak call to avoid SAPI5 state corruption.
    speak() is non-blocking — enqueues text, returns immediately.
    Worker thread dequeues and speaks in order.

    Self-healing: if the worker thread dies (COM crash, unhandled
    exception), speak() detects it and restarts the worker.
    A generation counter prevents stale workers from consuming
    items enqueued after their death.
    """

    def __init__(self, rate: int = 175, voice_id: Optional[str] = None):
        self._rate = rate
        self._voice_id = voice_id
        self._queue: queue.Queue = queue.Queue(maxsize=_MAX_QUEUE_SIZE)
        self._worker: Optional[threading.Thread] = None
        self._started = False
        self._ready = threading.Event()  # signaled when engine inits
        self._generation = 0  # incremented on each worker restart
        self._lock = threading.Lock()  # guards worker lifecycle
        self._permanently_unavailable = False  # set if pyttsx3 missing

    def _ensure_worker(self) -> None:
        """Start or restart the TTS worker thread.

        If a previous worker died, drains the stale queue, increments
        the generation counter, and spawns a new worker.
        """
        with self._lock:
            if self._permanently_unavailable:
                return  # pyttsx3 not installed — don't retry
            if self._worker is not None and self._worker.is_alive():
                return  # healthy worker — nothing to do

            if self._started and self._worker is not None:
                # Worker was started but died — self-heal
                logger.warning(
                    "pyttsx3 worker thread died (gen=%d) — restarting",
                    self._generation,
                )
                # Drain stale queue items
                drained = 0
                while not self._queue.empty():
                    try:
                        self._queue.get_nowait()
                        self._queue.task_done()
                        drained += 1
                    except queue.Empty:
                        break
                if drained:
                    logger.info(
                        "pyttsx3: drained %d stale items from queue",
                        drained,
                    )
                self._ready.clear()

            self._generation += 1
            gen = self._generation
            self._started = True
            self._worker = threading.Thread(
                target=self._worker_loop,
                args=(gen,),
                name=f"tts-worker-gen{gen}",
                daemon=True,
            )
            self._worker.start()
            logger.info(
                "pyttsx3 TTS worker thread started: gen=%d rate=%d voice=%s",
                gen, self._rate, self._voice_id or "default",
            )

    def start(self, timeout: float = 5.0) -> bool:
        """
        Eagerly start the worker thread and wait for engine readiness.

        Call this at boot for:
        - Deterministic startup (no lazy latency)
        - Fail-fast (COM/audio errors surface at init, not first speak)
        - World-ready guarantee (MERLIN — Online means speech is live)

        Returns True if engine initialized within timeout, False otherwise.
        """
        self._ensure_worker()
        ready = self._ready.wait(timeout=timeout)
        if not ready:
            logger.error(
                "pyttsx3 engine failed to initialize within %.1fs — "
                "speech may not work",
                timeout,
            )
        return ready

    def _speak_once(self, text: str) -> None:
        """
        Create a fresh pyttsx3 engine, speak, destroy.

        Each call gets a clean COM event loop — avoids the known
        pyttsx3 bug where runAndWait() silently fails after first use.
        """
        import pyttsx3

        logger.debug(
            "pyttsx3: creating engine for %d chars", len(text),
        )
        engine = pyttsx3.init()
        try:
            engine.setProperty('rate', self._rate)
            if self._voice_id:
                engine.setProperty('voice', self._voice_id)
            engine.say(text)
            logger.debug("pyttsx3: calling runAndWait (%d chars)", len(text))
            engine.runAndWait()
            logger.debug("pyttsx3: runAndWait returned (%d chars)", len(text))
        finally:
            try:
                engine.stop()
            except Exception:
                pass
            del engine

    def _worker_loop(self, generation: int) -> None:
        """
        Worker thread main loop.

        All pyttsx3 engine creation and usage happens here, on this
        thread's COM apartment. Never exits until _SHUTDOWN sentinel
        or unrecoverable exception.

        Args:
            generation: Worker generation number for logging.
        """
        logger.info(
            "pyttsx3 worker thread active (gen=%d, tid=%s)",
            generation, threading.current_thread().ident,
        )

        # Validate engine can init — do one throwaway init to surface errors
        try:
            import pyttsx3
            test_engine = pyttsx3.init()
            test_engine.stop()
            del test_engine
            logger.info(
                "pyttsx3 engine validated on worker thread (gen=%d)",
                generation,
            )
            self._ready.set()  # signal: engine is usable
        except ImportError:
            logger.warning(
                "pyttsx3 not installed (gen=%d) — TTS disabled. "
                "Install with: pip install merlin-assistant[voice]",
                generation,
            )
            self._permanently_unavailable = True
            self._ready.set()  # unblock start() even on failure
            return
        except Exception:
            logger.exception(
                "pyttsx3 engine validation failed (gen=%d)", generation,
            )
            self._ready.set()  # unblock start() even on failure
            return

        while True:
            try:
                text = self._queue.get()
            except Exception:
                logger.exception(
                    "pyttsx3 worker: queue.get failed (gen=%d)",
                    generation,
                )
                break

            if text is _SHUTDOWN:
                logger.info(
                    "pyttsx3 worker: shutdown received (gen=%d)",
                    generation,
                )
                self._queue.task_done()
                break

            try:
                logger.info(
                    "pyttsx3: speaking %d chars on worker thread (gen=%d)",
                    len(text), generation,
                )
                self._speak_once(text)
                logger.info(
                    "pyttsx3: speech delivered (%d chars, gen=%d)",
                    len(text), generation,
                )
            except Exception:
                logger.exception(
                    "pyttsx3 worker: speech failed for %d chars (gen=%d)",
                    len(text), generation,
                )
            finally:
                self._queue.task_done()

        logger.warning(
            "pyttsx3 worker loop exiting (gen=%d) — will self-heal on next speak()",
            generation,
        )

    def speak(self, text: str) -> None:
        """
        Enqueue text for speech. Non-blocking.

        Returns immediately. Worker thread speaks in order.
        If the worker thread has died, restarts it automatically.
        If the queue is full, drops the oldest item.
        """
        self._ensure_worker()
        try:
            self._queue.put_nowait(text)
        except queue.Full:
            # Drop oldest to make room — prevent memory growth
            try:
                dropped = self._queue.get_nowait()
                self._queue.task_done()
                logger.warning(
                    "pyttsx3: queue full — dropped oldest item (%d chars)",
                    len(dropped) if isinstance(dropped, str) else 0,
                )
            except queue.Empty:
                pass
            self._queue.put_nowait(text)

    def speak_sync(self, text: str) -> None:
        """
        Enqueue text and block until spoken.

        Useful for critical announcements or shutdown.
        """
        self._ensure_worker()
        self._queue.put(text)
        self._queue.join()

    def shutdown(self) -> None:
        """Signal worker to stop after current speech."""
        if self._worker and self._worker.is_alive():
            self._queue.put(_SHUTDOWN)
            self._worker.join(timeout=5.0)
            if self._worker.is_alive():
                logger.warning("pyttsx3 worker thread did not exit cleanly")

    def is_available(self) -> bool:
        """Check if pyttsx3 is installed."""
        try:
            import pyttsx3  # noqa: F401
            return True
        except ImportError:
            return False
