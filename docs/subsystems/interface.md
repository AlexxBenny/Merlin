# Interface Layer (`interface/`)

> The API boundary between MERLIN core and external frontends.

---

## Overview

The interface layer provides a **decoupled communication bridge** between the MERLIN runtime process and external consumers (dashboard, widget, Telegram bot, third-party tools). It uses **filesystem-based IPC** (via a shared `ipc.py` module) for state export and a **command queue** for exactly-once command execution.

Key design constraint: the API server runs as a **separate process** — it never imports or touches MERLIN core internals directly.

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│              MERLIN Process                      │
│                                                  │
│  LogBufferHandler → RingBuffer (deque, 500)     │
│                         │                        │
│  MerlinBridge (daemon thread)                   │
│    ├── Export loop (1s): state → JSON files      │
│    └── Command loop (0.3s): queue → execute      │
│                         │                        │
│              state/api/*.json                    │
└─────────────────┬───────────────────────────────┘
                  │  Filesystem IPC
                  ▼
┌─────────────────────────────────────────────────┐
│         API Server (separate process)            │
│         python -m interface.api_server           │
│         FastAPI on port 8420                     │
│                                                  │
│  Reads:  state/api/*.json                       │
│  Writes: state/api/command_queue/cmd_*.json     │
│  Reads:  state/api/responses/cmd_*.json         │
└─────────────────────────────────────────────────┘
                  │  Same IPC
                  ▼
┌─────────────────────────────────────────────────┐
│       Telegram Bot (separate process)            │
│       python -m interface.telegram_bot           │
│       Whitelist-secured, serialized              │
│                                                  │
│  Writes: state/api/command_queue/cmd_*.json     │
│  Reads:  state/api/responses/cmd_*.json         │
└─────────────────────────────────────────────────┘
```

---

## Components

### `log_buffer.py` — Log Capture

Runs **inside** the MERLIN process. Attaches to the Python root logger.

| Class | Purpose |
|-------|---------|
| `RingBuffer` | Thread-safe fixed-size buffer (`collections.deque(maxlen=500)`) |
| `LogBufferHandler` | `logging.Handler` subclass — serializes records into the ring buffer |
| `install_log_buffer()` | Module-level convenience: installs handler on root logger, returns buffer |

Each log entry: `{timestamp, level, module, logger, message, lineno}`.

### `bridge.py` — IPC Bridge

Runs as a **daemon thread** inside the MERLIN process. Holds a reference to the live `Merlin` conductor instance.

**Export loop** (every 1s):
| File | Source |
|------|--------|
| `state/api/system.json` | `psutil` metrics + mission state |
| `state/api/jobs.json` | `TickSchedulerManager._store.get_all()` |
| `state/api/memory.json` | `UserKnowledgeStore` (5 domains) |
| `state/api/world.json` | `WorldState.from_events()` |
| `state/api/missions.json` | Last 20 `MissionOutcome` records |
| `state/api/logs.json` | `RingBuffer.get_all()` |

All writes use **atomic tmp→rename** for crash safety.

**Command loop** (every 0.3s):
- Polls `state/api/command_queue/cmd_*.json`
- Executes command → writes response to `state/api/responses/{cmd_id}.json`
- Deletes command file (exactly-once: consume → delete)

Supported commands: `chat`, `cancel_job`, `pause_job`, `resume_job`, `update_config`.

### `config_schema.py` — Config Validation

Pydantic models for validated configuration editing.

| Model | Fields |
|-------|--------|
| `EditableExecutorConfig` | `max_workers`, `node_timeout_seconds` |
| `EditableSchedulerConfig` | `enabled`, `max_retry_attempts`, `max_concurrent_jobs` |
| `EditableAttentionConfig` | `cooldown_seconds`, `max_queue_size`, `merge_duplicates` |
| `EditableNarrationConfig` | `enabled`, `single_node_silent`, `compression_threshold` |
| `EditableVoiceConfig` | `enabled`, `tts_enabled`, `tts_rate` |
| `EditableBrowserConfig` | `headless` |

`apply_config_update()` uses `ruamel.yaml` for comment-preserving round-trip YAML editing. Falls back to standard `yaml` if `ruamel` is unavailable.

### `ipc.py` — Shared IPC Module

Used by **both** the API server and the Telegram adapter. Zero MERLIN core imports.

| Function | Purpose |
|--------|---------|
| `read_json(path)` | Safe JSON read (returns None on failure) |
| `write_json(path, data)` | Atomic tmp→rename JSON write |
| `submit_command(type, payload, dir)` | Write command to queue, returns cmd_id |
| `wait_for_response(cmd_id, dir, timeout)` | Async poll for response |
| `queue_depth(dir)` | Count pending commands |
| `is_bridge_alive(state_dir)` | Check system.json freshness |

**Protocol version**: 1. Command format: `{id, type, payload, created_at, protocol_version}`.

### `telegram_bot.py` — Telegram Adapter

Runs as a **separate subprocess** (`python -m interface.telegram_bot`). Whitelist-secured.

| Feature | Implementation |
|---------|---------------|
| Security | `allowed_user_ids` whitelist from `config/telegram.yaml` |
| Serialization | `asyncio.Lock` — one message at a time |
| Queue guard | Rejects if `queue_depth > max_queue_depth` |
| Liveness | Checks `system.json` freshness before submitting |
| Truncation | Response capped at 4000 chars (Telegram limit) |
| Logging | Structured `[TELEGRAM]` logs with user_id, cmd_id, latency |

### `api_server.py` — FastAPI Server

Runs as a **separate process** (`python -m interface.api_server`).

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/v1/chat` | Send message, get response |
| POST | `/api/v1/chat/stream` | SSE streaming response |
| GET | `/api/v1/chat/history` | Session chat history |
| POST | `/api/v1/chat/new_session` | Archive + reset session |
| GET | `/api/v1/system` | System metrics |
| GET | `/api/v1/jobs` | List scheduled jobs |
| DELETE | `/api/v1/jobs/{id}` | Cancel a job |
| PATCH | `/api/v1/jobs/{id}` | Pause/resume a job |
| GET | `/api/v1/memory` | User knowledge store |
| GET | `/api/v1/missions` | Recent mission history |
| GET | `/api/v1/missions/{id}` | Single mission detail |
| GET | `/api/v1/world` | World state snapshot |
| GET | `/api/v1/config` | Config values (secrets masked) |
| PATCH | `/api/v1/config` | Update config with validation |
| GET | `/api/v1/logs` | Recent log entries |
| GET | `/api/v1/health` | Health check (widget heartbeat) |
| WS | `/ws/logs` | Real-time log stream |
| WS | `/ws/events` | System/jobs/missions updates |

All endpoints are **versioned** under `/api/v1/`.

---

## Command Queue Protocol

Commands follow an exactly-once execution pattern:

1. **API server** writes `state/api/command_queue/cmd_{timestamp}_{uuid}.json`
2. **Bridge** reads + executes the command
3. **Bridge** writes response to `state/api/responses/{cmd_id}.json`
4. **Bridge** deletes the command file
5. **API server** polls for response, returns to client

On crash between steps 2–4, the command file is still consumed (step 4) with an error response. No duplicate execution.

---

## Chat Sessions

Chat history is **session-scoped** and stored at `state/api/chat_sessions/current_session.json`. Sessions are capped at 200 messages. The `/chat/new_session` endpoint archives the current session and starts fresh.

This is separate from the long-term `UserKnowledgeStore` — chat history is ephemeral per session.

---

## Security

- API keys in `.env` are **never** exposed via the config endpoint
- Config values containing `KEY`, `SECRET`, `TOKEN`, or `PASSWORD` are automatically masked (`****` + last 4 chars)
- `.env` is **not editable** via the API
