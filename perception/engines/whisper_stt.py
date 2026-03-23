# perception/engines/whisper_stt.py

"""
WhisperSTT — faster-whisper implementation of STTEngine.

Lazy-loads the model on first transcribe() call, not at init.
Config-driven: model size, device, compute type set via VoiceEngineFactory.
"""

import logging
from typing import Optional

import numpy as np

from perception.stt_engine import STTEngine, TranscriptionResult, TranscriptionSegment


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
        model_path: Optional[str] = None,
    ):
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._language = language
        self._model_path = model_path
        self._model: Optional[object] = None  # lazy

    def _ensure_model(self) -> None:
        """Load the Whisper model on first use."""
        if self._model is not None:
            return

        from pathlib import Path
        from faster_whisper import WhisperModel

        # Resolve model download directory
        download_root = None
        if self._model_path:
            root = Path(__file__).resolve().parent.parent.parent
            model_dir = root / self._model_path
            model_dir.mkdir(parents=True, exist_ok=True)
            download_root = str(model_dir)
            logger.info(
                "Loading Whisper model: size=%s device=%s compute=%s path=%s",
                self._model_size, self._device, self._compute_type, download_root,
            )
        else:
            logger.info(
                "Loading Whisper model: size=%s device=%s compute=%s",
                self._model_size, self._device, self._compute_type,
            )

        kwargs = dict(
            device=self._device,
            compute_type=self._compute_type,
        )
        if download_root:
            kwargs["download_root"] = download_root

        self._model = WhisperModel(
            self._model_size,
            **kwargs,
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
        result_segments = []
        total_prob = 0.0
        segment_count = 0

        for segment in segments:
            text_parts.append(segment.text)
            total_prob += segment.avg_logprob
            segment_count += 1
            result_segments.append(TranscriptionSegment(
                start=round(segment.start, 3),
                end=round(segment.end, 3),
                text=segment.text.strip(),
            ))

        text = " ".join(text_parts).strip()
        # Convert avg log prob to rough confidence (0-1 range)
        avg_prob = total_prob / max(segment_count, 1)
        confidence = min(1.0, max(0.0, 1.0 + avg_prob))  # log prob is negative

        detected_language = info.language if info.language else self._language
        duration = getattr(info, 'duration', 0.0) or 0.0

        return TranscriptionResult(
            text=text,
            confidence=round(confidence, 3),
            language=detected_language,
            duration=round(duration, 3),
            segments=tuple(result_segments),
        )

    def transcribe_file(self, audio_file) -> TranscriptionResult:
        """Transcribe audio from a file-like object (BinaryIO).

        Accepts any format faster-whisper/PyAV can decode:
        webm, opus, wav, mp3, ogg, etc.

        Used by the API server for direct uploads — avoids
        manual audio decoding / numpy conversion.
        """
        self._ensure_model()

        segments, info = self._model.transcribe(
            audio_file,
            language=self._language,
            beam_size=5,
            vad_filter=True,
        )

        text_parts = []
        result_segments = []
        total_prob = 0.0
        segment_count = 0

        for segment in segments:
            text_parts.append(segment.text)
            total_prob += segment.avg_logprob
            segment_count += 1
            result_segments.append(TranscriptionSegment(
                start=round(segment.start, 3),
                end=round(segment.end, 3),
                text=segment.text.strip(),
            ))

        text = " ".join(text_parts).strip()
        avg_prob = total_prob / max(segment_count, 1)
        confidence = min(1.0, max(0.0, 1.0 + avg_prob))
        detected_language = info.language if info.language else self._language
        duration = getattr(info, 'duration', 0.0) or 0.0

        return TranscriptionResult(
            text=text,
            confidence=round(confidence, 3),
            language=detected_language,
            duration=round(duration, 3),
            segments=tuple(result_segments),
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
