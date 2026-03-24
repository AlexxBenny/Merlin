# perception/stt_engine.py

"""
Speech-to-Text engine abstraction.

Concrete engines (WhisperSTT, VoskSTT, etc.) implement this interface.
SpeechPerception depends ONLY on this ABC — never on concrete engines.
Engines are created by VoiceEngineFactory from config.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True)
class TranscriptionSegment:
    """A single segment of transcribed speech with timestamps."""
    start: float   # seconds
    end: float     # seconds
    text: str


@dataclass(frozen=True)
class TranscriptionResult:
    """Immutable result from a single STT transcription."""
    text: str
    confidence: float
    language: str
    duration: float = 0.0
    segments: Tuple[TranscriptionSegment, ...] = ()


class STTEngine(ABC):
    """
    Abstract base class for Speech-to-Text engines.

    Contract:
    - transcribe() accepts raw audio as numpy array + sample rate.
    - is_available() checks if the engine can run (deps installed, model loaded).
    - Capability methods have safe defaults — override only in capable engines.
    """

    @abstractmethod
    def transcribe(self, audio: np.ndarray, sample_rate: int) -> TranscriptionResult:
        """
        Transcribe audio samples to text.

        Args:
            audio: 1D float32 numpy array of audio samples.
            sample_rate: Sample rate in Hz (typically 16000).

        Returns:
            TranscriptionResult with text, confidence, language.
        """

    @abstractmethod
    def is_available(self) -> bool:
        """Check whether this engine is ready to transcribe."""

    def supports_streaming(self) -> bool:
        """Override in engines that support incremental decoding."""
        return False

    def supports_language(self, language: str) -> bool:
        """Override to declare supported languages explicitly."""
        return True  # optimistic default
