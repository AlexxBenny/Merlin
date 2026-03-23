# interface/ipc.py

"""
Shared IPC helpers for MERLIN's file-based command/response protocol.

Used by:
    - api_server.py (HTTP → command queue)
    - telegram_bot.py (Telegram → command queue)

Protocol version: 1
    Command: {"id": str, "type": str, "payload": dict, "created_at": float, "protocol_version": 1}
    Response: {"id": str, "status": str, "response": str, "completed_at": float}

This module has ZERO MERLIN core imports. Safe to use in any process.
"""

import asyncio
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

PROTOCOL_VERSION = 1

# ─────────────────────────────────────────────────────────────
# File I/O helpers
# ─────────────────────────────────────────────────────────────

def read_json(path: Path) -> Any:
    """Read a JSON file safely. Returns None on failure."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def write_json(path: Path, data: Any) -> None:
    """Write JSON atomically via tmp → rename."""
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    if os.path.exists(str(path)):
        os.replace(tmp, str(path))
    else:
        os.rename(tmp, str(path))


# ─────────────────────────────────────────────────────────────
# Command queue interface
# ─────────────────────────────────────────────────────────────

def submit_command(
    cmd_type: str,
    payload: Dict[str, Any],
    command_dir: Path,
) -> str:
    """Submit a command to the queue. Returns command ID."""
    cmd_id = f"cmd_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    cmd = {
        "id": cmd_id,
        "type": cmd_type,
        "payload": payload,
        "created_at": time.time(),
        "protocol_version": PROTOCOL_VERSION,
    }
    write_json(command_dir / f"{cmd_id}.json", cmd)
    return cmd_id


async def wait_for_response(
    cmd_id: str,
    response_dir: Path,
    timeout: float = 60.0,
    poll_interval: float = 0.2,
) -> Dict[str, Any]:
    """Poll for a command response asynchronously."""
    response_path = response_dir / f"{cmd_id}.json"
    start = time.time()

    while time.time() - start < timeout:
        data = read_json(response_path)
        if data is not None:
            # Clean up response file
            try:
                response_path.unlink(missing_ok=True)
            except OSError:
                pass
            return data
        await asyncio.sleep(poll_interval)

    return {
        "id": cmd_id,
        "status": "timeout",
        "response": "Command timed out.",
        "completed_at": time.time(),
    }


def queue_depth(command_dir: Path) -> int:
    """Count pending commands in the queue."""
    try:
        return len(list(command_dir.glob("cmd_*.json")))
    except OSError:
        return 0


def is_bridge_alive(
    state_dir: Path,
    max_stale_seconds: float = 10.0,
) -> bool:
    """Check if the bridge is alive by verifying system.json freshness.

    The bridge writes system.json every ~1 second. If the timestamp
    is stale or the file is missing, the bridge is not running.
    """
    system_data = read_json(state_dir / "system.json")
    if system_data is None:
        return False

    ts = system_data.get("timestamp")
    if ts is None:
        return False

    return (time.time() - ts) < max_stale_seconds
