# interface/ — MERLIN Frontend API boundary layer.
#
# This package contains:
# - bridge.py: IPC bridge (runs inside MERLIN process)
# - api_server.py: FastAPI server (runs as separate process)
# - log_buffer.py: Ring buffer log handler
# - config_schema.py: Pydantic config validation schemas
