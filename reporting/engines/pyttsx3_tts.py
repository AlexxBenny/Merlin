# reporting/engines/pyttsx3_tts.py

"""
Pyttsx3TTS — offline TTS engine using pyttsx3 (Windows SAPI5).

Bootstrap TTS engine for Phase 1. Swappable via config for
CoquiTTS, EdgeTTS, etc. without code changes.
"""

import logging
from typing import Optional

from reporting.tts_engine import TTSEngine


logger = logging.getLogger(__name__)


class Pyttsx3TTS(TTSEngine):
    """
    TTSEngine backed by pyttsx3.

    Uses Windows SAPI5 voices. Configurable rate and voice.
    Lazy engine init — creates pyttsx3 engine on first speak().
    """

    def __init__(self, rate: int = 175, voice_id: Optional[str] = None):
        self._rate = rate
        self._voice_id = voice_id
        self._engine: Optional[object] = None  # lazy

    def _ensure_engine(self) -> None:
        """Initialize pyttsx3 engine on first use."""
        if self._engine is not None:
            return

        import pyttsx3

        self._engine = pyttsx3.init()
        self._engine.setProperty('rate', self._rate)

        if self._voice_id:
            self._engine.setProperty('voice', self._voice_id)

        logger.info(
            "pyttsx3 TTS engine initialized: rate=%d voice=%s",
            self._rate, self._voice_id or "default",
        )

    def speak(self, text: str) -> None:
        """Speak text using pyttsx3. Blocks until playback completes."""
        self._ensure_engine()
        self._engine.say(text)
        self._engine.runAndWait()

    def is_available(self) -> bool:
        """Check if pyttsx3 is installed."""
        try:
            import pyttsx3  # noqa: F401
            return True
        except ImportError:
            return False
