# perception/perception_orchestrator.py

"""
PerceptionOrchestrator — concurrent multi-modal input.

Manages text + speech channels. First percept wins per cycle.
Explicit cancellation semantics — no zombie threads, no implicit behavior.

Concurrency invariants:
1. CancellationToken is atomic, shared per cycle.
2. Speech listener runs as daemon thread.
3. When one percept resolves → token cancels → other channel aborts.
4. Orchestrator join()s speech thread (no timeout — cancellation is reliable).
5. First-percept-wins: if user types while speaking, text wins,
   speech recording aborted via AudioRecorder.stop(), partial discarded.

Limitation (Phase 1, documented):
    Hybrid mode is text-dominant. input() blocks until Enter is pressed
    and cannot be cancelled programmatically. When speech wins first,
    the main thread remains blocked on input() until the user presses
    Enter. The speech percept is returned, but the stale text input
    from that Enter press is discarded on the next cycle.
    This is an acceptable CLI limitation — resolved in a future GUI/TUI.
"""

import logging
import queue
import threading
from typing import Optional

from brain.core import Percept
from perception.text import TextPerception
from perception.speech import SpeechPerception

try:
    from prompt_toolkit import PromptSession
except ImportError:
    PromptSession = None  # type: ignore[misc,assignment]


logger = logging.getLogger(__name__)


class CancellationToken:
    """Atomic flag shared between channels in a single listen cycle."""

    def __init__(self):
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()


class PerceptionOrchestrator:
    """
    Owns all perception channels. Returns exactly one Percept per cycle.

    Modes:
    - text_only: self._speech is None → standard input() loop.
    - voice_only: self._text is None → speech listen() loop.
    - hybrid: both channels active → first wins.
    """

    def __init__(
        self,
        text: Optional[TextPerception] = None,
        speech: Optional[SpeechPerception] = None,
        prompt_session: Optional["PromptSession"] = None,
    ):
        if text is None and speech is None:
            raise ValueError(
                "PerceptionOrchestrator requires at least one channel."
            )
        self._text = text
        self._speech = speech
        self._session = prompt_session

    def _read_input(self) -> str:
        """Read user input via PromptSession (if available) or raw input()."""
        if self._session is not None:
            return self._session.prompt("You: ").strip()
        return input("You: ").strip()

    def next_percept(self) -> Percept:
        """
        One listen cycle. Exactly one Percept returned.

        Routing:
        - Text-only → blocking input()
        - Voice-only → blocking listen()
        - Hybrid → concurrent, first wins, other cancelled
        """
        # ── Text-only mode ──
        if self._speech is None:
            raw = self._read_input()
            return self._text.perceive(raw)

        # ── Voice-only mode ──
        if self._text is None:
            return self._speech.listen()

        # ── Hybrid mode ──
        return self._hybrid_listen()

    def _hybrid_listen(self) -> Percept:
        """
        Concurrent text + speech.

        Limitation: input() is non-cancellable. See module docstring.
        When speech wins, the main thread remains on input() until
        the user presses Enter. That stale input is discarded.
        """
        token = CancellationToken()
        result_queue: queue.Queue[Percept] = queue.Queue(maxsize=1)

        # Speech: daemon thread, checks token via AudioRecorder._stop_flag
        speech_thread = threading.Thread(
            target=self._listen_speech,
            args=(token, result_queue),
            daemon=True,
        )
        speech_thread.start()

        # Text: main thread (blocking — cannot be cancelled)
        try:
            raw = self._read_input()
            if raw and not token.is_cancelled:
                # Text won — cancel speech
                token.cancel()
                self._speech._recorder.stop()
                speech_thread.join()  # reliable — stop() triggers _stop_flag
                return self._text.perceive(raw)
        except EOFError:
            pass

        # Text didn't win — wait for speech
        try:
            percept = result_queue.get(timeout=60)
        except queue.Empty:
            logger.warning("No percept received within timeout.")
            percept = Percept(
                modality="speech", payload="", confidence=0.0,
            )

        token.cancel()
        self._speech._recorder.stop()
        speech_thread.join()  # cleanup — no zombies
        return percept

    def _listen_speech(
        self,
        token: CancellationToken,
        q: queue.Queue,
    ) -> None:
        """Speech listener thread. Puts percept if it wins the race."""
        try:
            percept = self._speech.listen()
            if not token.is_cancelled and percept.payload:
                try:
                    q.put_nowait(percept)
                    token.cancel()
                except queue.Full:
                    pass  # text already won
        except Exception:
            logger.exception("Speech listen failed in background thread.")
