# reporting/tts_engine.py

"""
Text-to-Speech engine abstraction.

Concrete engines (Pyttsx3TTS, CoquiTTS, etc.) implement this interface.
SpeechOutputChannel depends ONLY on this ABC — never on concrete engines.
Engines are created by VoiceEngineFactory from config.
"""

from abc import ABC, abstractmethod


class TTSEngine(ABC):
    """
    Abstract base class for Text-to-Speech engines.

    Contract:
    - speak() blocks until audio playback completes.
    - is_available() checks if the engine can run.
    - Callers MUST wrap speak() in try/except — engine failures
      must never crash MERLIN.
    """

    @abstractmethod
    def speak(self, text: str) -> None:
        """
        Speak the given text aloud.

        Blocks until playback completes.
        Raises on hardware or engine failure — caller must catch.
        """

    @abstractmethod
    def is_available(self) -> bool:
        """Check whether this engine is ready to speak."""
