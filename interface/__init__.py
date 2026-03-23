# interface/ — MERLIN Frontend API boundary layer.
#
# This package contains:
# - bridge.py: IPC bridge (runs inside MERLIN process)
# - api_server.py: FastAPI server (runs as separate process)
# - telegram_bot.py: Telegram bot adapter (runs as separate process)
# - ipc.py: Shared file-based IPC helpers (used by api_server + telegram_bot)
# - log_buffer.py: Ring buffer log handler
# - config_schema.py: Pydantic config validation schemas
