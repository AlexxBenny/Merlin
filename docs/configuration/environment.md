# Environment Configuration

MERLIN uses a `.env` file for secrets and environment-specific settings.

## Setup

**For pip-installed users:**

`merlin init` creates the `.env` file automatically during setup.

Location:
- **Windows:** `%APPDATA%\Local\merlin\.env`
- **Linux/macOS:** `~/.config/merlin/.env`

**For developers (cloned repo):**

```bash
cp .env.example .env
# Edit .env with your API keys
```

The `.env` at the repository root is used in development mode.

## Required Variables

At least **one** LLM provider API key is required. `OPENROUTER_API_KEY` is recommended (access to multiple models).

| Variable | Purpose | Example |
|----------|---------|---------|
| `OPENROUTER_API_KEY` | OpenRouter API access | `sk-or-v1-...` |
| `GEMINI_API_KEY` | Google Gemini API access | `AIza...` |

## Optional Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `TELEGRAM_BOT_TOKEN` | Telegram bot adapter | — |
| `EMAIL_ADDRESS` | Email sender address | — |
| `EMAIL_APP_PASSWORD` | Email SMTP app password | — |
| `OLLAMA_HOST` | Ollama server URL | `http://localhost:11434` |
| `HUGGINGFACE_API_KEY` | HuggingFace Inference API | — |
| `BROWSER_HEADLESS` | Run browser headless | `false` |
| `BROWSER_DISABLE_SECURITY` | Disable browser security | `false` |
| `LOG_LEVEL` | Logging verbosity | `INFO` |
| `TTS_ENGINE` | TTS engine selection | `pyttsx3` |
| `STT_ENGINE` | STT engine selection | `whisper` |
| `MAX_WORKERS` | Parallel execution threads | `4` |
| `NODE_TIMEOUT` | Per-node execution timeout (seconds) | `60` |

## API Key Management

- `KeyPool` supports multiple keys per provider for rate limit management
