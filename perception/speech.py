# perception/speech.py

"""
SpeechPerception — converts microphone audio to Percept objects.

Depends ONLY on STTEngine interface and AudioRecorder.
Never imports concrete engines — those are injected by VoiceEngineFactory.

Follows the same pattern as TextPerception: one method → one Percept.
"""

import logging
import time

from brain.core import Percept
from perception.stt_engine import STTEngine
from perception.audio_recorder import AudioRecorder


logger = logging.getLogger(__name__)


class SpeechPerception:
    """
    Produces Percepts from microphone audio via STT.

    Design:
    - listen() is blocking in Phase 1.
    - Designed for eventual listen_stream() without interface breaks.
    - Caller controls concurrency (PerceptionOrchestrator).
    """

    def __init__(self, stt_engine: STTEngine, recorder: AudioRecorder):
        self._stt = stt_engine
        self._recorder = recorder

    def listen(self) -> Percept:
        """
        Record from mic until silence, transcribe, return Percept.

        Returns a Percept with modality="speech" and the transcribed text
        as payload. Confidence comes from the STT engine.

        Returns empty-payload Percept if recording was cancelled or
        no speech detected.
        """
        audio = self._recorder.record_until_silence()

        if audio.size == 0:
            logger.debug("No audio captured — returning empty Percept.")
            return Percept(
                modality="speech",
                payload="",
                confidence=0.0,
                timestamp=time.time(),
            )

        result = self._stt.transcribe(audio, self._recorder.sample_rate)

        logger.info(
            "Speech transcribed: '%s' (confidence=%.3f, lang=%s)",
            result.text[:80], result.confidence, result.language,
        )

        return Percept(
            modality="speech",
            payload=result.text.strip(),
            confidence=result.confidence,
            timestamp=time.time(),
        )
