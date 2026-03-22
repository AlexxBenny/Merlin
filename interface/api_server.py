# interface/api_server.py

"""
MERLIN API Server — Separate-process FastAPI server.

Runs independently from MERLIN core:
    python -m interface.api_server

Reads state from state/api/*.json (written by bridge.py).
Writes commands to state/api/command_queue/ (read by bridge.py).
Reads responses from state/api/responses/ (written by bridge.py).

No MERLIN core imports. No direct access to MERLIN internals.
"""

import asyncio
import json
import logging
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Paths (relative to MERLIN root)
# ─────────────────────────────────────────────────────────────

_BASE_PATH = Path(__file__).resolve().parent.parent
_STATE_DIR = _BASE_PATH / "state" / "api"
_COMMAND_DIR = _STATE_DIR / "command_queue"
_RESPONSE_DIR = _STATE_DIR / "responses"
_CHAT_DIR = _STATE_DIR / "chat_sessions"
_CONFIG_DIR = _BASE_PATH / "config"

# Ensure directories exist
for _d in [_STATE_DIR, _COMMAND_DIR, _RESPONSE_DIR, _CHAT_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# File I/O helpers
# ─────────────────────────────────────────────────────────────

def _read_json(path: Path) -> Any:
    """Read a JSON file safely. Returns None on failure."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _write_json(path: Path, data: Any) -> None:
    """Write JSON atomically."""
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    if os.path.exists(str(path)):
        os.replace(tmp, str(path))
    else:
        os.rename(tmp, str(path))


def _mask_secrets(data: Dict[str, Any]) -> Dict[str, Any]:
    """Mask API keys and secrets in config data."""
    masked = {}
    for key, value in data.items():
        if isinstance(value, dict):
            masked[key] = _mask_secrets(value)
        elif isinstance(value, str) and any(
            s in key.upper() for s in ["KEY", "SECRET", "TOKEN", "PASSWORD"]
        ):
            if len(value) > 4:
                masked[key] = "****" + value[-4:]
            else:
                masked[key] = "****"
        else:
            masked[key] = value
    return masked


# ─────────────────────────────────────────────────────────────
# Command queue interface
# ─────────────────────────────────────────────────────────────

def _submit_command(
    cmd_type: str,
    payload: Dict[str, Any],
) -> str:
    """Submit a command to the queue. Returns command ID."""
    cmd_id = f"cmd_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    cmd = {
        "id": cmd_id,
        "type": cmd_type,
        "payload": payload,
        "created_at": time.time(),
    }
    _write_json(_COMMAND_DIR / f"{cmd_id}.json", cmd)
    return cmd_id


async def _wait_for_response(
    cmd_id: str,
    timeout: float = 60.0,
    poll_interval: float = 0.2,
) -> Dict[str, Any]:
    """Poll for a command response."""
    response_path = _RESPONSE_DIR / f"{cmd_id}.json"
    start = time.time()

    while time.time() - start < timeout:
        data = _read_json(response_path)
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


# ─────────────────────────────────────────────────────────────
# FastAPI Application
# ─────────────────────────────────────────────────────────────

try:
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel
except ImportError:
    raise ImportError(
        "FastAPI is required for the API server. "
        "Install with: pip install fastapi uvicorn"
    )


app = FastAPI(
    title="MERLIN API",
    version="1.0.0",
    description="MERLIN AI Assistant — Frontend API",
)

# CORS for dashboard (Vite dev server on :5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────
# Request/Response Models
# ─────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    id: str
    status: str
    response: str


class JobActionRequest(BaseModel):
    action: str  # "pause" | "resume"


class ConfigUpdatePayload(BaseModel):
    updates: Dict[str, Any]


# ─────────────────────────────────────────────────────────────
# Chat Endpoints
# ─────────────────────────────────────────────────────────────

@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a message to MERLIN and get a response."""
    cmd_id = _submit_command("chat", {"message": request.message})
    result = await _wait_for_response(cmd_id, timeout=120.0)

    # Store in chat session
    _store_chat_message(request.message, result.get("response", ""))

    return ChatResponse(
        id=result.get("id", cmd_id),
        status=result.get("status", "unknown"),
        response=result.get("response", "No response."),
    )


@app.post("/api/v1/chat/stream")
async def chat_stream(request: ChatRequest):
    """Send a message and stream the response via SSE."""
    cmd_id = _submit_command("chat", {"message": request.message})

    async def event_generator():
        yield f"data: {json.dumps({'type': 'start', 'id': cmd_id})}\n\n"

        result = await _wait_for_response(cmd_id, timeout=120.0)
        response_text = result.get("response", "No response.")

        # Store in chat session
        _store_chat_message(request.message, response_text)

        # Stream response progressively (simulate typing)
        words = response_text.split()
        chunk = []
        for i, word in enumerate(words):
            chunk.append(word)
            if len(chunk) >= 3 or i == len(words) - 1:
                partial = " ".join(chunk)
                yield f"data: {json.dumps({'type': 'chunk', 'text': partial})}\n\n"
                chunk = []
                await asyncio.sleep(0.05)

        yield f"data: {json.dumps({'type': 'done', 'full_response': response_text})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


def _store_chat_message(user_msg: str, assistant_msg: str) -> None:
    """Store chat message in the current session."""
    session_file = _CHAT_DIR / "current_session.json"
    session = _read_json(session_file) or {"messages": []}
    session["messages"].append({
        "role": "user",
        "content": user_msg,
        "timestamp": time.time(),
    })
    session["messages"].append({
        "role": "assistant",
        "content": assistant_msg,
        "timestamp": time.time(),
    })
    # Cap at 200 messages per session
    if len(session["messages"]) > 200:
        session["messages"] = session["messages"][-200:]
    _write_json(session_file, session)


@app.get("/api/v1/chat/history")
async def chat_history():
    """Get current chat session history."""
    session = _read_json(_CHAT_DIR / "current_session.json")
    return session or {"messages": []}


@app.post("/api/v1/chat/new_session")
async def new_chat_session():
    """Start a new chat session, archiving the current one."""
    current = _read_json(_CHAT_DIR / "current_session.json")
    if current and current.get("messages"):
        archive_name = f"session_{int(time.time())}.json"
        _write_json(_CHAT_DIR / archive_name, current)
    _write_json(_CHAT_DIR / "current_session.json", {"messages": []})
    return {"status": "ok", "message": "New session started."}


# ─────────────────────────────────────────────────────────────
# System Endpoint
# ─────────────────────────────────────────────────────────────

@app.get("/api/v1/system")
async def get_system():
    """Get system metrics and MERLIN runtime state."""
    data = _read_json(_STATE_DIR / "system.json")
    if data is None:
        raise HTTPException(503, "System state not available. Is MERLIN running?")
    return data


# ─────────────────────────────────────────────────────────────
# Jobs Endpoints
# ─────────────────────────────────────────────────────────────

@app.get("/api/v1/jobs")
async def get_jobs():
    """List all scheduled jobs."""
    data = _read_json(_STATE_DIR / "jobs.json")
    return data or []


@app.delete("/api/v1/jobs/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a pending job."""
    cmd_id = _submit_command("cancel_job", {"job_id": job_id})
    result = await _wait_for_response(cmd_id, timeout=10.0)
    return {
        "status": result.get("status", "unknown"),
        "message": result.get("response", ""),
    }


@app.patch("/api/v1/jobs/{job_id}")
async def update_job(job_id: str, request: JobActionRequest):
    """Pause or resume a job."""
    if request.action == "pause":
        cmd_type = "pause_job"
    elif request.action == "resume":
        cmd_type = "resume_job"
    else:
        raise HTTPException(400, f"Unknown action: {request.action}")

    cmd_id = _submit_command(cmd_type, {"job_id": job_id})
    result = await _wait_for_response(cmd_id, timeout=10.0)
    return {
        "status": result.get("status", "unknown"),
        "message": result.get("response", ""),
    }


# ─────────────────────────────────────────────────────────────
# Memory Endpoint
# ─────────────────────────────────────────────────────────────

@app.get("/api/v1/memory")
async def get_memory():
    """Get user knowledge store."""
    data = _read_json(_STATE_DIR / "memory.json")
    return data or {
        "preferences": {}, "facts": {}, "traits": {},
        "policies": {}, "relationships": {},
    }


# ─────────────────────────────────────────────────────────────
# Missions Endpoints
# ─────────────────────────────────────────────────────────────

@app.get("/api/v1/missions")
async def get_missions():
    """Get recent mission history."""
    data = _read_json(_STATE_DIR / "missions.json")
    return data or []


@app.get("/api/v1/missions/{mission_id}")
async def get_mission(mission_id: str):
    """Get a specific mission by ID."""
    data = _read_json(_STATE_DIR / "missions.json")
    if data is None:
        raise HTTPException(404, "No missions data available.")

    for mission in data:
        if mission.get("mission_id") == mission_id:
            return mission

    raise HTTPException(404, f"Mission {mission_id} not found.")


# ─────────────────────────────────────────────────────────────
# World State Endpoint
# ─────────────────────────────────────────────────────────────

@app.get("/api/v1/world")
async def get_world():
    """Get current world state."""
    data = _read_json(_STATE_DIR / "world.json")
    return data or {}


# ─────────────────────────────────────────────────────────────
# Config Endpoints
# ─────────────────────────────────────────────────────────────

@app.get("/api/v1/config")
async def get_config():
    """Get all config values (secrets masked)."""
    config = {}

    # Load execution.yaml
    exec_path = _CONFIG_DIR / "execution.yaml"
    if exec_path.exists():
        with open(exec_path, "r", encoding="utf-8") as f:
            exec_config = yaml.safe_load(f) or {}
        config["execution"] = _mask_secrets(exec_config)

    # Load models.yaml if exists
    models_path = _CONFIG_DIR / "models.yaml"
    if models_path.exists():
        with open(models_path, "r", encoding="utf-8") as f:
            models_config = yaml.safe_load(f) or {}
        config["models"] = _mask_secrets(models_config)

    # Load email.yaml if exists
    email_path = _CONFIG_DIR / "email.yaml"
    if email_path.exists():
        with open(email_path, "r", encoding="utf-8") as f:
            email_config = yaml.safe_load(f) or {}
        config["email"] = _mask_secrets(email_config.get("email", email_config))

    # Load whatsapp.yaml if exists
    wa_path = _CONFIG_DIR / "whatsapp.yaml"
    if wa_path.exists():
        with open(wa_path, "r", encoding="utf-8") as f:
            wa_config = yaml.safe_load(f) or {}
        config["whatsapp"] = _mask_secrets(wa_config.get("whatsapp", wa_config))

    # Load field metadata for dashboard display
    from interface.config_schema import CONFIG_FIELD_METADATA
    config["_field_metadata"] = CONFIG_FIELD_METADATA

    return config


@app.patch("/api/v1/config")
async def update_config(payload: ConfigUpdatePayload):
    """Update config values with validation."""
    cmd_id = _submit_command("update_config", payload.updates)
    result = await _wait_for_response(cmd_id, timeout=10.0)
    return {
        "status": result.get("status", "unknown"),
        "message": result.get("response", ""),
    }


# ─────────────────────────────────────────────────────────────
# Logs Endpoints
# ─────────────────────────────────────────────────────────────

@app.get("/api/v1/logs")
async def get_logs(
    n: int = 100,
    level: Optional[str] = None,
):
    """Get recent log entries."""
    data = _read_json(_STATE_DIR / "logs.json")
    if data is None:
        return []

    # Filter by level if specified
    if level:
        level_upper = level.upper()
        data = [e for e in data if e.get("level") == level_upper]

    # Return last N
    return data[-n:]


# ─────────────────────────────────────────────────────────────
# WebSocket: Live log stream
# ─────────────────────────────────────────────────────────────

@app.websocket("/ws/logs")
async def ws_logs(websocket: WebSocket):
    """Stream log entries in real-time."""
    await websocket.accept()

    last_count = 0
    try:
        while True:
            data = _read_json(_STATE_DIR / "logs.json")
            if data and len(data) > last_count:
                # Send only new entries
                new_entries = data[last_count:]
                for entry in new_entries:
                    await websocket.send_json(entry)
                last_count = len(data)
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────
# WebSocket: Live events (system/jobs/missions)
# ─────────────────────────────────────────────────────────────

@app.websocket("/ws/events")
async def ws_events(websocket: WebSocket):
    """Stream system/jobs/missions updates in real-time."""
    await websocket.accept()

    last_hashes: Dict[str, str] = {}

    try:
        while True:
            for name in ["system", "jobs", "missions"]:
                path = _STATE_DIR / f"{name}.json"
                data = _read_json(path)
                if data is not None:
                    # Simple change detection via JSON hash
                    data_str = json.dumps(data, sort_keys=True, default=str)
                    data_hash = str(hash(data_str))
                    if last_hashes.get(name) != data_hash:
                        last_hashes[name] = data_hash
                        await websocket.send_json({
                            "type": name,
                            "data": data,
                            "timestamp": time.time(),
                        })
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────────────────────

@app.get("/api/v1/health")
async def health():
    """Health check endpoint for widget heartbeat."""
    system = _read_json(_STATE_DIR / "system.json")
    merlin_running = system is not None
    return {
        "status": "ok" if merlin_running else "degraded",
        "merlin_connected": merlin_running,
        "timestamp": time.time(),
    }


# ─────────────────────────────────────────────────────────────
# Draft management endpoints (email integration)
# ─────────────────────────────────────────────────────────────

_DRAFTS_SUMMARY = _STATE_DIR / "drafts_summary.json"
_DRAFTS_STATE_DIR = _BASE_PATH / "state" / "email" / "drafts"


class DraftUpdate(BaseModel):
    """Pydantic model for draft PATCH requests."""
    recipient: Optional[str] = None
    cc: Optional[str] = None
    bcc: Optional[str] = None
    subject: Optional[str] = None
    body: Optional[str] = None
    status: Optional[str] = None


@app.get("/api/v1/drafts")
async def get_drafts():
    """List all drafts (read from bridge-exported summary)."""
    data = _read_json(_DRAFTS_SUMMARY)
    if data is None:
        # Fallback: try reading individual draft files directly
        _DRAFTS_STATE_DIR.mkdir(parents=True, exist_ok=True)
        drafts = []
        for f in sorted(_DRAFTS_STATE_DIR.glob("d-*.json"), reverse=True):
            d = _read_json(f)
            if d:
                drafts.append(d)
        return drafts
    return data


@app.get("/api/v1/drafts/{draft_id}")
async def get_draft(draft_id: str):
    """Get a specific draft."""
    # Try exported summary first
    drafts = _read_json(_DRAFTS_SUMMARY)
    if drafts:
        for d in drafts:
            if d.get("id") == draft_id:
                return d

    # Fallback to direct file
    path = _DRAFTS_STATE_DIR / f"{draft_id}.json"
    data = _read_json(path)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Draft {draft_id} not found")
    return data


@app.patch("/api/v1/drafts/{draft_id}")
async def update_draft(draft_id: str, update: DraftUpdate):
    """Update draft fields via bridge command."""
    updates = {k: v for k, v in update.model_dump().items() if v is not None}
    if not updates:
        raise HTTPException(status_code=400, detail="No updates provided")

    # Validate status transitions
    if "status" in updates:
        valid_statuses = {"pending_review", "approved", "discarded"}
        if updates["status"] not in valid_statuses:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status. Must be one of: {valid_statuses}",
            )

    cmd_id = _submit_command("update_draft", {
        "draft_id": draft_id,
        **updates,
    })
    response = await _wait_for_response(cmd_id, timeout=10.0)
    return response


@app.delete("/api/v1/drafts/{draft_id}")
async def delete_draft(draft_id: str):
    """Discard a draft via bridge command."""
    cmd_id = _submit_command("discard_draft", {"draft_id": draft_id})
    response = await _wait_for_response(cmd_id, timeout=10.0)
    return response


@app.post("/api/v1/drafts/{draft_id}/send")
async def send_draft(draft_id: str):
    """Send an approved draft via bridge command."""
    cmd_id = _submit_command("send_draft", {"draft_id": draft_id})
    response = await _wait_for_response(cmd_id, timeout=30.0)
    return response


# ─────────────────────────────────────────────────────────────
# WhatsApp endpoints
# ─────────────────────────────────────────────────────────────

_WA_STATUS_FILE = _STATE_DIR / "whatsapp_status.json"
_WA_MESSAGES_FILE = _STATE_DIR / "whatsapp_messages.json"


class WhatsAppSendRequest(BaseModel):
    """Pydantic model for WhatsApp send requests."""
    contact: str
    text: str


@app.get("/api/v1/whatsapp/status")
async def whatsapp_status():
    """Get WhatsApp connection status."""
    data = _read_json(_WA_STATUS_FILE)
    if data is None:
        return {
            "connected": False,
            "messages_sent_today": 0,
            "total_messages": 0,
            "rate_limit_remaining": 0,
        }
    return data


@app.get("/api/v1/whatsapp/messages")
async def whatsapp_messages():
    """Get WhatsApp message history."""
    data = _read_json(_WA_MESSAGES_FILE)
    if data is None:
        return []
    return data


@app.post("/api/v1/whatsapp/send")
async def whatsapp_send(request: WhatsAppSendRequest):
    """Send a WhatsApp message via bridge command."""
    cmd_id = _submit_command("wa_send", {
        "contact": request.contact,
        "text": request.text,
    })
    response = await _wait_for_response(cmd_id, timeout=30.0)
    return response


# ─────────────────────────────────────────────────────────────
# Module entrypoint
# ─────────────────────────────────────────────────────────────

def main():
    """Run the API server."""
    try:
        import uvicorn
    except ImportError:
        raise ImportError(
            "uvicorn is required for the API server. "
            "Install with: pip install uvicorn"
        )

    print("=" * 50)
    print("  MERLIN API Server")
    print(f"  http://localhost:8420/api/v1/")
    print(f"  Docs: http://localhost:8420/docs")
    print("=" * 50)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8420,
        log_level="info",
    )


if __name__ == "__main__":
    main()
