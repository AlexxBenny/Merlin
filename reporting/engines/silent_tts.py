# reporting/engines/silent_tts.py

"""
SilentTTS — no-op TTS engine for tests and headless mode.

Logs the text that would be spoken without producing audio.
"""

import logging

from reporting.tts_engine import TTSEngine


logger = logging.getLogger(__name__)


class SilentTTS(TTSEngine):
    """TTSEngine that logs instead of speaking. For tests and headless runs."""

    def speak(self, text: str) -> None:
        logger.debug("[SilentTTS] Would speak: %s", text[:100])

    def is_available(self) -> bool:
        return True
