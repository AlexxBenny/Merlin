# reporting/output.py

"""
OutputChannel — How MERLIN speaks.

This is the LAST MILE of the system.
Everything upstream (brain, cortex, executor, reporter) is pure logic.
This is the only place where the outside world hears anything.

Design rules:
- OutputChannel is dumb — it delivers, it does not decide.
- Silence is explicit (send_silent), not an empty string.
- Multiple channels can compose (text + speech simultaneously).
"""

from abc import ABC, abstractmethod
from typing import List
import logging
import threading


logger = logging.getLogger(__name__)


class OutputChannel(ABC):
    """
    Abstract base for all output delivery mechanisms.

    Every channel must implement:
    - send(): deliver a message to the user
    - send_silent(): explicitly do nothing (for audit/logging)
    """

    @abstractmethod
    def send(self, message: str) -> None:
        """Deliver a message to the user."""

    def send_silent(self) -> None:
        """
        Explicitly record that the system chose silence.
        Override for audit logging; default is no-op.
        """
        logger.debug("OutputChannel: silence chosen")


class ConsoleOutputChannel(OutputChannel):
    """
    Prints to stdout. Primary channel during development.
    Thread-safe: uses a lock to prevent interleaved output.
    """

    def __init__(self, prefix: str = "MERLIN"):
        self.prefix = prefix
        self._lock = threading.Lock()

    def send(self, message: str) -> None:
        with self._lock:
            print(f"[{self.prefix}] {message}")

    def send_silent(self) -> None:
        logger.debug("ConsoleOutputChannel: silence")


class SpeechOutputChannel(OutputChannel):
    """
    TTS delivery channel.

    Depends on TTSEngine interface — never on concrete engines.
    Engine injected by VoiceEngineFactory from config.
    TTS failures are caught and logged — never crash MERLIN.
    """

    def __init__(self, tts_engine: "TTSEngine"):
        from reporting.tts_engine import TTSEngine  # type check only
        self._tts = tts_engine

    def send(self, message: str) -> None:
        try:
            logger.info(
                "SpeechOutputChannel: enqueuing %d chars for speech",
                len(message),
            )
            self._tts.speak(message)
        except Exception:
            logger.exception(
                "TTS engine failure — message not spoken: %s",
                message[:80],
            )

    def send_silent(self) -> None:
        logger.debug("SpeechOutputChannel: silence")


class CompositeOutputChannel(OutputChannel):
    """
    Delivers to multiple channels simultaneously.

    Example: text on screen + voice at the same time.
    If any child channel fails, the others still deliver.
    """

    def __init__(self, channels: List[OutputChannel]):
        if not channels:
            raise ValueError("CompositeOutputChannel requires at least one channel")
        self.channels = channels

    def send(self, message: str) -> None:
        for channel in self.channels:
            try:
                channel.send(message)
            except Exception:
                logger.warning(
                    "OutputChannel %s failed to send, continuing",
                    type(channel).__name__,
                )

    def send_silent(self) -> None:
        for channel in self.channels:
            try:
                channel.send_silent()
            except Exception:
                pass
