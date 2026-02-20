# perception/engines/mock_stt.py

"""
MockSTT — test-only STTEngine that returns predefined transcripts.

No hardware or dependencies required.
Used for unit testing SpeechPerception and PerceptionOrchestrator
without monkeypatching.
"""

from typing import List

import numpy as np

from perception.stt_engine import STTEngine, TranscriptionResult


class MockSTT(STTEngine):
    """
    Returns transcripts from a predefined list, one per call.

    After the list is exhausted, returns empty strings.
    """

    def __init__(self, responses: List[str], confidence: float = 1.0):
        self._responses = iter(responses)
        self._confidence = confidence

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> TranscriptionResult:
        text = next(self._responses, "")
        return TranscriptionResult(
            text=text,
            confidence=self._confidence,
            language="en",
        )

    def is_available(self) -> bool:
        return True

    def supports_streaming(self) -> bool:
        return False
