# perception/engines/whisper_stt.py

"""
WhisperSTT — faster-whisper implementation of STTEngine.

Lazy-loads the model on first transcribe() call, not at init.
Config-driven: model size, device, compute type set via VoiceEngineFactory.
"""

import logging
from typing import Optional

import numpy as np

from perception.stt_engine import STTEngine, TranscriptionResult


logger = logging.getLogger(__name__)


class WhisperSTT(STTEngine):
    """
    STTEngine backed by faster-whisper.

    Model loading is lazy — the first transcribe() call loads the model.
    Subsequent calls reuse the cached instance.
    """

    def __init__(
        self,
        model_size: str = "small",
        device: str = "cuda",
        compute_type: str = "float16",
        language: str = "en",
    ):
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._language = language
        self._model: Optional[object] = None  # lazy

    def _ensure_model(self) -> None:
        """Load the Whisper model on first use."""
        if self._model is not None:
            return

        from faster_whisper import WhisperModel

        logger.info(
            "Loading Whisper model: size=%s device=%s compute=%s",
            self._model_size, self._device, self._compute_type,
        )
        self._model = WhisperModel(
            self._model_size,
            device=self._device,
            compute_type=self._compute_type,
        )
        logger.info("Whisper model loaded.")

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> TranscriptionResult:
        """
        Transcribe audio using faster-whisper.

        Audio must be float32, mono. Sample rate should be 16000
        (faster-whisper resamples internally if needed).
        """
        self._ensure_model()

        segments, info = self._model.transcribe(
            audio,
            language=self._language,
            beam_size=5,
            vad_filter=True,
        )

        # Collect all segment texts
        text_parts = []
        total_prob = 0.0
        segment_count = 0

        for segment in segments:
            text_parts.append(segment.text)
            total_prob += segment.avg_logprob
            segment_count += 1

        text = " ".join(text_parts).strip()
        # Convert avg log prob to rough confidence (0-1 range)
        avg_prob = total_prob / max(segment_count, 1)
        confidence = min(1.0, max(0.0, 1.0 + avg_prob))  # log prob is negative

        detected_language = info.language if info.language else self._language

        return TranscriptionResult(
            text=text,
            confidence=round(confidence, 3),
            language=detected_language,
        )

    def is_available(self) -> bool:
        """Check if faster-whisper is installed."""
        try:
            import faster_whisper  # noqa: F401
            return True
        except ImportError:
            return False

    def supports_streaming(self) -> bool:
        return False

    def supports_language(self, language: str) -> bool:
        # Whisper supports 99 languages
        return True
