# Model Layer

**Location**: `models/`

LLM adapters. MERLIN supports multiple LLM providers with role-based routing.

## Components

### LLMClient (`models/base.py`)

Abstract base class for all LLM clients:

```python
class LLMClient:
    def generate(prompt: str, max_tokens: int, temperature: float) -> str
```

### ModelRouter (`models/router.py`)

Role-based model selection. Maps cognitive roles to specific models via `config/models.yaml`.

**Roles**:
| Role | Purpose | Typical Model |
|------|---------|---------------|
| `cognitive_coordinator` | Reasoning, direct answers | gemini-2.5-flash |
| `mission_compiler` | Plan compilation | gemini-2.5-flash |
| `report_builder` | Result narration | gemini-2.5-flash |
| `recovery_reasoner` | Tier 2 failure diagnosis (DecisionEngine) | gemini-2.5-flash |
| `fallback` | Graceful degradation | gemini-2.5-flash |

### LLM Clients

| Client | File | Provider |
|--------|------|----------|
| `OpenRouterClient` | `openrouter_client.py` | OpenRouter API (access to GPT, Claude, Gemini, etc.) |
| `GeminiClient` | `gemini_client.py` | Google Gemini API direct |
| `OllamaClient` | `ollama_client.py` | Local Ollama models |
| `HuggingFaceClient` | `huggingface_client.py` | Hugging Face Inference API |

### KeyPool (`models/key_pool.py`)

API key rotation and rate limit management. Cycles through multiple API keys to avoid rate limits.

## Dynamic Token Management

`max_tokens` is calculated dynamically based on prompt size and model context limit to prevent 402 errors:

```
available = model_context_limit - prompt_tokens
max_tokens = min(requested_max, available - safety_margin)
```
