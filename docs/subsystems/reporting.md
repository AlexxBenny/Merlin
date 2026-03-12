# Reporting System

**Location**: `reporting/`

Transforms raw skill results into natural language responses.

## Components

### ReportBuilder (`reporting/report_builder.py`)

LLM-based narration of execution results. Takes skill outputs and generates user-friendly summaries.

**Features**:
- Skill result aggregation
- Error message humanization
- Partial success reporting
- Context-aware narration (uses conversation history)

### NarrationPolicy (`reporting/narration.py`)

Controls narration verbosity and style.

| Style | Behavior |
|-------|---------|
| `terse` | Minimal confirmation ("Done.") |
| `normal` | Natural summary |
| `verbose` | Detailed step-by-step |

Skills declare their preferred `output_style` in their contract.

### NotificationPolicy (`reporting/notification_policy.py`)

Decides what to speak aloud vs. what to show in console only.

| Decision | Criteria |
|----------|---------|
| Speak | Short responses, confirmations, errors |
| Show only | Long outputs, file contents, lists |
| Both | Important notifications |

### OutputChannel (`reporting/output.py`)

Console + TTS output multiplexing. Sends responses to:
- Console (always)
- TTS engine (based on notification policy)

### TTSEngine (`reporting/tts_engine.py`)

Abstract TTS interface.

## TTS Engines (`reporting/engines/`)

| Engine | File | Description |
|--------|------|-----------|
| `Pyttsx3TTS` | `pyttsx3_tts.py` | Offline TTS using pyttsx3 library. Worker thread for non-blocking speech |
| `SilentTTS` | `silent_tts.py` | Testing stub — no audio output |
