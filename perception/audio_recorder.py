# perception/audio_recorder.py

"""
AudioRecorder — Microphone capture with WebRTC VAD silence detection.

Uses sounddevice for capture + webrtcvad for robust voice activity detection.
Supports early cancellation via stop() for PerceptionOrchestrator concurrency.

Design contract:
- record_until_silence() blocks until silence detected OR stop() called.
- _stop_flag is checked every frame — cancellation is prompt, not deferred.
- Method signature designed for eventual record_stream() extension.
"""

import logging
import struct
import threading
from typing import Optional

import numpy as np


logger = logging.getLogger(__name__)


class AudioRecorder:
    """
    Captures audio from the default microphone with WebRTC VAD.

    Cancellation: Call stop() from another thread to abort recording.
    The _stop_flag is checked every VAD frame (~30ms), so cancellation
    latency is bounded.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        silence_duration: float = 1.5,
        max_record_seconds: float = 30.0,
        vad_mode: int = 2,
    ):
        self.sample_rate = sample_rate
        self._silence_duration = silence_duration
        self._max_record_seconds = max_record_seconds
        self._vad_mode = vad_mode
        self._stop_flag = threading.Event()

    def record_until_silence(self) -> np.ndarray:
        """
        Record from mic until silence detected, stop() called, or max time.

        Returns:
            1D float32 numpy array of audio samples.
            Empty array if cancelled before any speech detected.
        """
        import sounddevice as sd
        import webrtcvad

        self._stop_flag.clear()

        vad = webrtcvad.Vad(self._vad_mode)

        # WebRTC VAD requires 16-bit PCM in 10/20/30ms frames
        frame_duration_ms = 30
        frame_size = int(self.sample_rate * frame_duration_ms / 1000)
        max_frames = int(
            self._max_record_seconds * 1000 / frame_duration_ms
        )

        frames: list = []
        silence_frames = 0
        silence_threshold = int(
            self._silence_duration * 1000 / frame_duration_ms
        )
        speech_detected = False

        logger.debug(
            "Recording: sr=%d vad_mode=%d silence=%.1fs max=%.0fs",
            self.sample_rate, self._vad_mode,
            self._silence_duration, self._max_record_seconds,
        )

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
                blocksize=frame_size,
            ) as stream:
                for _ in range(max_frames):
                    if self._stop_flag.is_set():
                        logger.debug("Recording cancelled via stop().")
                        break

                    data, overflowed = stream.read(frame_size)
                    if overflowed:
                        logger.debug("Audio buffer overflowed.")

                    # Flatten to 1D int16
                    pcm = data.flatten()
                    raw_bytes = struct.pack(
                        f"{len(pcm)}h", *pcm.tolist()
                    )

                    is_speech = vad.is_speech(
                        raw_bytes, self.sample_rate
                    )

                    if is_speech:
                        speech_detected = True
                        silence_frames = 0
                        frames.append(pcm)
                    elif speech_detected:
                        silence_frames += 1
                        frames.append(pcm)  # keep trailing silence
                        if silence_frames >= silence_threshold:
                            logger.debug(
                                "Silence detected after %.1fs of speech.",
                                len(frames) * frame_duration_ms / 1000,
                            )
                            break

        except Exception:
            logger.exception("Audio recording error")
            return np.array([], dtype=np.float32)

        if not frames:
            return np.array([], dtype=np.float32)

        # Convert int16 → float32 (whisper expects float32 in [-1, 1])
        audio = np.concatenate(frames).astype(np.float32) / 32768.0
        logger.debug("Recorded %.2fs of audio.", len(audio) / self.sample_rate)
        return audio

    def stop(self) -> None:
        """Cancel current recording immediately."""
        self._stop_flag.set()
