# infrastructure/voice_factory.py

"""
VoiceEngineFactory — config-driven creation of voice engines.

Reads execution.yaml voice section, creates STT and TTS engine instances.
Returns None on failure (ImportError, init error, unknown engine) —
callers must handle graceful degradation.

Design:
- Lazy imports — engine modules only imported when that engine is selected.
- ImportError caught — missing dependency disables voice, doesn't crash.
- Unknown engine → logged, returns None.
"""

import logging
from typing import Optional

from perception.stt_engine import STTEngine
from reporting.tts_engine import TTSEngine


logger = logging.getLogger(__name__)


class VoiceEngineFactory:
    """
    Creates STT and TTS engines from config.

    Returns None if engine unavailable — caller falls back to text-only.
    """

    @staticmethod
    def create_stt(voice_config: dict) -> Optional[STTEngine]:
        """
        Create an STT engine from config.

        Returns None if engine cannot be created (missing deps, bad config).
        main.py must check for None and disable voice input gracefully.
        """
        engine_name = voice_config.get("stt", {}).get("engine", "")

        try:
            if engine_name == "faster-whisper":
                from perception.engines.whisper_stt import WhisperSTT
                return WhisperSTT(
                    model_size=voice_config["stt"].get("model", "small"),
                    device=voice_config["stt"].get("device", "cuda"),
                    compute_type=voice_config["stt"].get(
                        "compute_type", "float16"
                    ),
                    language=voice_config["stt"].get("language", "en"),
                )
        except ImportError:
            logger.error(
                "STT engine '%s' unavailable — missing dependency. "
                "Install with: pip install faster-whisper",
                engine_name,
            )
            return None
        except Exception as e:
            logger.error(
                "STT engine '%s' init failed: %s", engine_name, e,
            )
            return None

        logger.error("Unknown STT engine: '%s'", engine_name)
        return None

    @staticmethod
    def create_tts(voice_config: dict) -> Optional[TTSEngine]:
        """
        Create a TTS engine from config.

        Returns None if engine cannot be created.
        main.py must check for None and use console-only output.
        """
        engine_name = voice_config.get("tts", {}).get("engine", "")

        try:
            if engine_name == "pyttsx3":
                from reporting.engines.pyttsx3_tts import Pyttsx3TTS
                return Pyttsx3TTS(
                    rate=voice_config["tts"].get("rate", 175),
                    voice_id=voice_config["tts"].get("voice_id"),
                )
            elif engine_name == "silent":
                from reporting.engines.silent_tts import SilentTTS
                return SilentTTS()
        except ImportError:
            logger.error(
                "TTS engine '%s' unavailable — missing dependency. "
                "Install with: pip install %s",
                engine_name, engine_name,
            )
            return None
        except Exception as e:
            logger.error(
                "TTS engine '%s' init failed: %s", engine_name, e,
            )
            return None

        logger.error("Unknown TTS engine: '%s'", engine_name)
        return None
