# Environment Configuration

MERLIN uses a `.env` file for secrets and environment-specific settings.

## Required Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `OPENROUTER_API_KEY` | OpenRouter API access | `sk-or-v1-...` |
| `GEMINI_API_KEY` | Google Gemini API access | `AIza...` |

## Optional Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `OLLAMA_HOST` | Ollama server URL | `http://localhost:11434` |
| `HUGGINGFACE_API_KEY` | HuggingFace Inference API | — |
| `BROWSER_HEADLESS` | Run browser headless | `false` |
| `BROWSER_DISABLE_SECURITY` | Disable browser security | `false` |
| `LOG_LEVEL` | Logging verbosity | `INFO` |
| `TTS_ENGINE` | TTS engine selection | `pyttsx3` |
| `STT_ENGINE` | STT engine selection | `whisper` |
| `MAX_WORKERS` | Parallel execution threads | `4` |
| `NODE_TIMEOUT` | Per-node execution timeout (seconds) | `60` |

## Setup

1. Copy `.env.example` to `.env`
2. Fill in API keys
3. Adjust optional settings as needed

```bash
cp .env.example .env
# Edit .env with your API keys
```

## API Key Requirements

- At least **one** LLM provider API key is required
- `OPENROUTER_API_KEY` is recommended (access to multiple models)
- `KeyPool` supports multiple keys per provider for rate limit management
