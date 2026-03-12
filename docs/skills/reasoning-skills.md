# Reasoning Skills

**Location**: `skills/reasoning/`

## `reasoning.generate_text` (`generate_text.py`)

Generate text using an LLM.

| Input | Type | Description |
|-------|------|-----------|
| `prompt` | `text_content` | Generation prompt |
| `max_tokens` | `integer` | Maximum output tokens (optional) |

| Output | Type | Description |
|--------|------|-----------|
| `text` | `text_content` | Generated text |

**Use case**: When the user asks MERLIN to write, summarize, or generate content.

**Note**: This is the only skill that directly invokes an LLM. All other LLM usage is in the cortex layer (compilation, coordination, reporting).
