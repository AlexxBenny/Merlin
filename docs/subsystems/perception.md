# Perception

**Location**: `perception/`

Layer 1. Converts external signals into `Percept` objects. No reasoning, no state, no routing.

## Components

### PerceptionOrchestrator (`perception/perception_orchestrator.py`)

Multi-modal input orchestration. Routes between text and speech input.

### TextPerception (`perception/text.py`)

Console text input → `Percept` object. Minimal — wraps raw text.

### SpeechPerception (`perception/speech.py`)

Audio → text transcription → `Percept` object.

### AudioRecorder (`perception/audio_recorder.py`)

Microphone input with voice activity detection. Records audio chunks for STT processing.

### InputNormalizer (`perception/normalize.py`)

Text cleanup: whitespace normalization, encoding fixes.

## STT Engines (`perception/engines/`)

| Engine | File | Description |
|--------|------|-----------|
| `WhisperSTT` | `whisper_stt.py` | OpenAI Whisper model for speech-to-text |
| `MockSTT` | `mock_stt.py` | Testing stub that returns predefined text |

Both implement `STTEngine` base class (`perception/stt_engine.py`).
