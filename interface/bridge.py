# interface/bridge.py

"""
MerlinBridge — IPC bridge between MERLIN core and the API server.

Runs as a daemon thread INSIDE the MERLIN process.
Holds a reference to the live Merlin instance.

Two responsibilities:
1. EXPORT: Periodically serialize system state → state/api/*.json
2. IMPORT: Poll command queue → execute commands → write responses

File I/O uses atomic writes (tmp → rename) to prevent partial reads.
The API server (separate process) reads these files independently.
"""

import json
import logging
import os
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    pass  # No MERLIN imports at module level — all via __init__ args

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

_EXPORT_INTERVAL = 1.0       # seconds between state exports
_COMMAND_POLL_INTERVAL = 0.3 # seconds between command queue polls
_STATE_DIR = "state/api"
_COMMAND_DIR = "state/api/command_queue"
_RESPONSE_DIR = "state/api/responses"
_CHAT_DIR = "state/api/chat_sessions"


# ─────────────────────────────────────────────────────────────
# Atomic file write utility
# ─────────────────────────────────────────────────────────────

def _atomic_write_json(path: str, data: Any) -> None:
    """Write JSON atomically via tmp → rename.

    Same pattern used by JsonTaskStore for crash safety.
    """
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        # Atomic rename (on Windows, need to remove target first)
        if os.path.exists(path):
            os.replace(tmp_path, path)
        else:
            os.rename(tmp_path, path)
    except Exception as e:
        logger.debug("Atomic write failed for %s: %s", path, e)
        # Clean up tmp file
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _safe_read_json(path: str) -> Any:
    """Safely read a JSON file. Returns None on failure."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


# ─────────────────────────────────────────────────────────────
# MerlinBridge
# ─────────────────────────────────────────────────────────────

class MerlinBridge:
    """IPC bridge — runs inside MERLIN process as a daemon thread.

    Args:
        merlin: The live Merlin conductor instance.
        base_path: Root path of the MERLIN project (for resolving state/).
        log_buffer: The RingBuffer from log_buffer.py (for log export).
    """

    def __init__(
        self,
        merlin: Any,
        base_path: str,
        log_buffer: Any = None,
    ) -> None:
        self._merlin = merlin
        self._base_path = Path(base_path)
        self._log_buffer = log_buffer

        # Resolve all paths
        self._state_dir = self._base_path / _STATE_DIR
        self._command_dir = self._base_path / _COMMAND_DIR
        self._response_dir = self._base_path / _RESPONSE_DIR
        self._chat_dir = self._base_path / _CHAT_DIR

        # Create directories
        for d in [self._state_dir, self._command_dir,
                  self._response_dir, self._chat_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Threading
        self._running = False
        self._export_thread: Optional[threading.Thread] = None
        self._command_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    # ─────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the bridge threads."""
        if self._running:
            return

        self._running = True
        self._stop_event.clear()

        self._export_thread = threading.Thread(
            target=self._export_loop,
            name="merlin-bridge-export",
            daemon=True,
        )
        self._command_thread = threading.Thread(
            target=self._command_loop,
            name="merlin-bridge-commands",
            daemon=True,
        )

        self._export_thread.start()
        self._command_thread.start()
        logger.info("[BRIDGE] Started (state_dir=%s)", self._state_dir)

    def stop(self) -> None:
        """Stop the bridge threads."""
        self._running = False
        self._stop_event.set()

        if self._export_thread and self._export_thread.is_alive():
            self._export_thread.join(timeout=3.0)
        if self._command_thread and self._command_thread.is_alive():
            self._command_thread.join(timeout=3.0)

        logger.info("[BRIDGE] Stopped")

    # ─────────────────────────────────────────────────────────
    # State export loop
    # ─────────────────────────────────────────────────────────

    def _export_loop(self) -> None:
        """Periodically export MERLIN state to JSON files."""
        while not self._stop_event.is_set():
            try:
                self._export_system()
                self._export_jobs()
                self._export_memory()
                self._export_world()
                self._export_missions()
                self._export_logs()
                self._export_drafts()
                self._export_whatsapp()
            except Exception as e:
                logger.debug("[BRIDGE] Export error: %s", e)

            self._stop_event.wait(timeout=_EXPORT_INTERVAL)

    def _export_system(self) -> None:
        """Export system metrics + MERLIN runtime state."""
        import psutil

        data = {
            "cpu_percent": psutil.cpu_percent(interval=None),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage("/").percent if os.name != "nt"
                           else psutil.disk_usage("C:\\").percent,
            "uptime_seconds": time.time() - self._merlin._start_time
                             if hasattr(self._merlin, "_start_time") else 0,
            "mission_state": self._get_mission_state(),
            "timestamp": time.time(),
        }

        # Battery (may not exist on desktop)
        battery = psutil.sensors_battery()
        if battery:
            data["battery_percent"] = battery.percent
            data["battery_charging"] = battery.power_plugged

        _atomic_write_json(
            str(self._state_dir / "system.json"), data,
        )

    def _export_jobs(self) -> None:
        """Export scheduler jobs."""
        scheduler = getattr(self._merlin, "scheduler", None)
        if scheduler is None:
            _atomic_write_json(
                str(self._state_dir / "jobs.json"), [],
            )
            return

        store = getattr(scheduler, "_store", None)
        if store is None:
            _atomic_write_json(
                str(self._state_dir / "jobs.json"), [],
            )
            return

        tasks = store.get_all()
        jobs_data = []
        for task in tasks:
            try:
                jobs_data.append(task.model_dump(mode="json"))
            except Exception:
                jobs_data.append({
                    "id": task.id,
                    "query": task.query,
                    "status": task.status.value if hasattr(task.status, "value") else str(task.status),
                    "type": task.type.value if hasattr(task.type, "value") else str(task.type),
                })

        _atomic_write_json(
            str(self._state_dir / "jobs.json"), jobs_data,
        )

    def _export_memory(self) -> None:
        """Export user knowledge store."""
        knowledge = getattr(self._merlin, "_user_knowledge", None)
        if knowledge is None:
            _atomic_write_json(
                str(self._state_dir / "memory.json"),
                {"preferences": {}, "facts": {}, "traits": {},
                 "policies": {}, "relationships": {}},
            )
            return

        try:
            # Access internal dicts directly — UserKnowledgeStore has no get_domain()
            # Domains: _preferences, _facts, _traits, _relationships are Dict[str, KnowledgeEntry]
            # _policies is List[Policy]
            prefs = {k: v.to_dict() for k, v in knowledge._preferences.items()}
            facts = {k: v.to_dict() for k, v in knowledge._facts.items()}
            traits = {k: v.to_dict() for k, v in knowledge._traits.items()}
            relationships = {k: v.to_dict() for k, v in knowledge._relationships.items()}
            policies = [p.to_dict() for p in knowledge._policies]

            data = {
                "preferences": prefs,
                "facts": facts,
                "traits": traits,
                "policies": policies,
                "relationships": relationships,
            }
        except Exception:
            data = {"preferences": {}, "facts": {}, "traits": {},
                    "policies": {}, "relationships": {}}

        _atomic_write_json(
            str(self._state_dir / "memory.json"), data,
        )

    def _export_world(self) -> None:
        """Export current world state."""
        try:
            timeline = getattr(self._merlin, "timeline", None)
            if timeline is None:
                timeline = getattr(
                    getattr(self._merlin, "orchestrator", None),
                    "timeline", None,
                )

            if timeline is None:
                _atomic_write_json(
                    str(self._state_dir / "world.json"), {},
                )
                return

            # Import WorldState here to avoid module-level core imports
            from world.state import WorldState

            events = timeline.all_events()
            state = WorldState.from_events(events)
            _atomic_write_json(
                str(self._state_dir / "world.json"),
                state.model_dump(mode="json"),
            )
        except Exception as e:
            logger.debug("[BRIDGE] World export error: %s", e)

    def _export_missions(self) -> None:
        """Export recent mission history with DAG data."""
        try:
            conversation = getattr(self._merlin, "conversation", None)
            if conversation is None:
                _atomic_write_json(
                    str(self._state_dir / "missions.json"), [],
                )
                return

            outcomes = getattr(conversation, "outcomes", [])
            missions_data = []

            for outcome in outcomes[-20:]:  # Last 20 missions
                try:
                    mission_entry = {
                        "mission_id": outcome.mission_id,
                        "timestamp": outcome.timestamp,
                        "nodes_executed": outcome.nodes_executed,
                        "nodes_skipped": outcome.nodes_skipped,
                        "nodes_failed": outcome.nodes_failed,
                        "nodes_timed_out": getattr(outcome, "nodes_timed_out", []),
                        "active_entity": outcome.active_entity,
                        "active_domain": outcome.active_domain,
                        "recovery_attempted": getattr(outcome, "recovery_attempted", False),
                    }
                    missions_data.append(mission_entry)
                except Exception:
                    pass

            _atomic_write_json(
                str(self._state_dir / "missions.json"), missions_data,
            )
        except Exception as e:
            logger.debug("[BRIDGE] Missions export error: %s", e)

    def _export_logs(self) -> None:
        """Export log buffer to JSON."""
        if self._log_buffer is None:
            return

        logs = self._log_buffer.get_all()
        _atomic_write_json(
            str(self._state_dir / "logs.json"), logs,
        )

    # ─────────────────────────────────────────────────────────
    # Command queue processing
    # ─────────────────────────────────────────────────────────

    def _command_loop(self) -> None:
        """Poll command queue and execute commands."""
        while not self._stop_event.is_set():
            try:
                self._process_commands()
            except Exception as e:
                logger.debug("[BRIDGE] Command loop error: %s", e)

            self._stop_event.wait(timeout=_COMMAND_POLL_INTERVAL)

    def _process_commands(self) -> None:
        """Process all pending commands in the queue."""
        if not self._command_dir.exists():
            return

        # List command files, sorted by name (timestamp-based ordering)
        cmd_files = sorted(self._command_dir.glob("cmd_*.json"))

        for cmd_file in cmd_files:
            try:
                cmd = _safe_read_json(str(cmd_file))
                if cmd is None:
                    # Invalid file — remove it
                    cmd_file.unlink(missing_ok=True)
                    continue

                cmd_id = cmd.get("id", "unknown")
                cmd_type = cmd.get("type", "unknown")

                # Execute command
                response = self._execute_command(cmd_type, cmd)

                # Write response
                response_data = {
                    "id": cmd_id,
                    "status": "completed",
                    "response": response,
                    "completed_at": time.time(),
                }
                _atomic_write_json(
                    str(self._response_dir / f"{cmd_id}.json"),
                    response_data,
                )

                # Delete command file (exactly-once: consume → delete)
                cmd_file.unlink(missing_ok=True)

                logger.info(
                    "[BRIDGE] Processed command %s (type=%s)",
                    cmd_id, cmd_type,
                )

            except Exception as e:
                logger.warning(
                    "[BRIDGE] Failed to process command %s: %s",
                    cmd_file.name, e,
                )
                # Write error response
                cmd = _safe_read_json(str(cmd_file)) or {}
                cmd_id = cmd.get("id", cmd_file.stem)
                _atomic_write_json(
                    str(self._response_dir / f"{cmd_id}.json"),
                    {
                        "id": cmd_id,
                        "status": "error",
                        "response": f"Command failed: {e}",
                        "completed_at": time.time(),
                    },
                )
                cmd_file.unlink(missing_ok=True)

    def _execute_command(self, cmd_type: str, cmd: Dict[str, Any]) -> str:
        """Execute a command and return the response string."""
        if cmd_type == "chat":
            return self._handle_chat(cmd)
        elif cmd_type == "cancel_job":
            return self._handle_cancel_job(cmd)
        elif cmd_type == "pause_job":
            return self._handle_pause_job(cmd)
        elif cmd_type == "resume_job":
            return self._handle_resume_job(cmd)
        elif cmd_type == "update_config":
            return self._handle_update_config(cmd)
        elif cmd_type == "update_draft":
            return self._handle_update_draft(cmd)
        elif cmd_type == "discard_draft":
            return self._handle_discard_draft(cmd)
        elif cmd_type == "send_draft":
            return self._handle_send_draft(cmd)
        elif cmd_type == "wa_send":
            return self._handle_wa_send(cmd)
        else:
            return f"Unknown command type: {cmd_type}"

    # ─────────────────────────────────────────────────────────
    # Command handlers
    # ─────────────────────────────────────────────────────────

    def _handle_chat(self, cmd: Dict[str, Any]) -> str:
        """Handle a chat command by routing through MERLIN's percept handler."""
        message = cmd.get("payload", {}).get("message", "")
        if not message:
            return "Empty message."

        try:
            # Create a Percept and route through MERLIN
            from brain.core import Percept

            percept = Percept(
                modality="text",
                payload=message,
                confidence=1.0,
                timestamp=time.time(),
            )

            # Capture the response via output channel interception
            response_parts: list[str] = []
            original_send = self._merlin.output_channel.send

            def _capture_send(text: str) -> None:
                response_parts.append(text)
                original_send(text)  # Preserve TTS + terminal output

            self._merlin.output_channel.send = _capture_send
            try:
                result = self._merlin.handle_percept(percept)
            finally:
                self._merlin.output_channel.send = original_send

            # Use captured response, fall back to result
            if response_parts:
                return " ".join(response_parts)
            return result or "No response."

        except Exception as e:
            logger.error("[BRIDGE] Chat command failed: %s", e, exc_info=True)
            return f"Error: {e}"

    def _handle_cancel_job(self, cmd: Dict[str, Any]) -> str:
        """Cancel a scheduled job."""
        job_id = cmd.get("payload", {}).get("job_id", "")
        scheduler = getattr(self._merlin, "scheduler", None)
        if scheduler is None:
            return "Scheduler not available."
        if scheduler.cancel(job_id):
            return f"Job {job_id} cancelled."
        return f"Could not cancel job {job_id}."

    def _handle_pause_job(self, cmd: Dict[str, Any]) -> str:
        """Pause a pending job."""
        job_id = cmd.get("payload", {}).get("job_id", "")
        scheduler = getattr(self._merlin, "scheduler", None)
        if scheduler is None:
            return "Scheduler not available."
        if scheduler.pause(job_id):
            return f"Job {job_id} paused."
        return f"Could not pause job {job_id}."

    def _handle_resume_job(self, cmd: Dict[str, Any]) -> str:
        """Resume a paused job."""
        job_id = cmd.get("payload", {}).get("job_id", "")
        scheduler = getattr(self._merlin, "scheduler", None)
        if scheduler is None:
            return "Scheduler not available."
        if scheduler.resume(job_id):
            return f"Job {job_id} resumed."
        return f"Could not resume job {job_id}."

    def _handle_update_config(self, cmd: Dict[str, Any]) -> str:
        """Update config via validated schema."""
        from interface.config_schema import ConfigUpdateRequest, apply_config_update

        payload = cmd.get("payload", {})
        try:
            update = ConfigUpdateRequest(**payload)

            all_changes = {}

            # Apply non-email changes to execution.yaml
            config_path = str(self._base_path / "config" / "execution.yaml")
            changes = apply_config_update(config_path, update)
            all_changes.update(changes)

            # Apply email changes to email.yaml
            email_changes = self._apply_email_config(payload)
            all_changes.update(email_changes)

            if all_changes:
                return f"Updated: {', '.join(f'{k}={v}' for k, v in all_changes.items())}"
            return "No changes applied."
        except Exception as e:
            return f"Config update failed: {e}"

    def _apply_email_config(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Apply email-specific config updates to email.yaml."""
        # Map flat payload keys to email.yaml nested structure
        mapping = {
            "email": None,           # top-level email fields
            "email_smtp": "smtp",    # email → smtp
            "email_imap": "imap",    # email → imap
            "email_defaults": "defaults",
        }

        updates_for_email = {}
        for payload_key, yaml_key in mapping.items():
            section_data = payload.get(payload_key)
            if not section_data or not isinstance(section_data, dict):
                continue
            for k, v in section_data.items():
                if v is not None:
                    if yaml_key:
                        updates_for_email.setdefault(yaml_key, {})[k] = v
                    else:
                        updates_for_email[k] = v

        if not updates_for_email:
            return {}

        email_path = self._base_path / "config" / "email.yaml"
        try:
            import yaml
            with open(email_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            email_section = data.get("email", {})

            changes = {}
            for key, value in updates_for_email.items():
                if isinstance(value, dict):
                    # Nested section (smtp, imap, defaults)
                    if key not in email_section:
                        email_section[key] = {}
                    for sub_k, sub_v in value.items():
                        email_section[key][sub_k] = sub_v
                        changes[f"email.{key}.{sub_k}"] = sub_v
                else:
                    email_section[key] = value
                    changes[f"email.{key}"] = value

            data["email"] = email_section
            with open(email_path, "w", encoding="utf-8") as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)

            return changes
        except Exception as e:
            logger.warning("[BRIDGE] Email config update failed: %s", e)
            return {}

    # ─────────────────────────────────────────────────────────
    # Draft command handlers (email integration)
    # ─────────────────────────────────────────────────────────

    def _export_drafts(self) -> None:
        """Export drafts from MERLIN-owned store to API-readable summary."""
        drafts_dir = self._base_path / "state" / "email" / "drafts"
        if not drafts_dir.exists():
            return

        drafts = []
        for f in sorted(drafts_dir.glob("d-*.json"), reverse=True):
            data = _safe_read_json(str(f))
            if data:
                drafts.append(data)

        summary_path = self._state_dir / "drafts_summary.json"
        _atomic_write_json(str(summary_path), drafts)

    def _handle_update_draft(self, cmd: Dict[str, Any]) -> str:
        """Update draft fields in MERLIN-owned store."""
        payload = cmd.get("payload", {})
        draft_id = payload.pop("draft_id", None)
        if not draft_id:
            return "Missing draft_id"

        draft_path = self._base_path / "state" / "email" / "drafts" / f"{draft_id}.json"
        draft = _safe_read_json(str(draft_path))
        if draft is None:
            return f"Draft {draft_id} not found"

        # Apply allowed updates
        allowed = {"recipient", "cc", "bcc", "subject", "body", "status"}
        for key, value in payload.items():
            if key in allowed:
                draft[key] = value

        import time as _time
        draft["updated_at"] = _time.time()
        _atomic_write_json(str(draft_path), draft)
        return f"Draft {draft_id} updated"

    def _handle_discard_draft(self, cmd: Dict[str, Any]) -> str:
        """Mark a draft as discarded."""
        draft_id = cmd.get("payload", {}).get("draft_id")
        if not draft_id:
            return "Missing draft_id"

        draft_path = self._base_path / "state" / "email" / "drafts" / f"{draft_id}.json"
        draft = _safe_read_json(str(draft_path))
        if draft is None:
            return f"Draft {draft_id} not found"

        import time as _time
        draft["status"] = "discarded"
        draft["updated_at"] = _time.time()
        _atomic_write_json(str(draft_path), draft)
        return f"Draft {draft_id} discarded"

    def _handle_send_draft(self, cmd: Dict[str, Any]) -> str:
        """Send an approved draft via EmailClient."""
        draft_id = cmd.get("payload", {}).get("draft_id")
        if not draft_id:
            return "Missing draft_id"

        draft_path = self._base_path / "state" / "email" / "drafts" / f"{draft_id}.json"
        draft = _safe_read_json(str(draft_path))
        if draft is None:
            return f"Draft {draft_id} not found"

        if draft.get("status") != "approved":
            return (
                f"Draft {draft_id} is '{draft.get('status')}', not 'approved'. "
                f"Approve the draft before sending."
            )

        # Try to get email_client from MERLIN
        # email_client is in skill_deps and accessible via the executor's registry
        try:
            from providers.email.client import EmailClient as _EmailClient
            # Search for a skill that has an email_client attribute
            email_client = None
            if hasattr(self._merlin, "executor") and hasattr(self._merlin.executor, "registry"):
                for skill in self._merlin.executor.registry._skills.values():
                    if hasattr(skill, "_email_client"):
                        email_client = skill._email_client
                        break

            if email_client is None:
                return "Email client not available — email is disabled in config"

            result = email_client.send_draft(draft_id)
            return f"Email sent to {draft.get('recipient', 'unknown')}. Message ID: {result.get('message_id', 'N/A')}"
        except Exception as e:
            return f"Send failed: {e}"

    # ─────────────────────────────────────────────────────────
    # WhatsApp command handlers
    # ─────────────────────────────────────────────────────────

    def _export_whatsapp(self) -> None:
        """Export WhatsApp status + message history to API-readable JSON."""
        messages_dir = self._base_path / "state" / "whatsapp" / "messages"

        # Export message history summary
        messages = []
        if messages_dir.exists():
            for f in sorted(messages_dir.glob("wa-*.json"), reverse=True):
                data = _safe_read_json(str(f))
                if data:
                    messages.append(data)
        # Cap at 200 most recent
        messages = messages[:200]
        _atomic_write_json(
            str(self._state_dir / "whatsapp_messages.json"), messages,
        )

        # Export connection status
        wa_client = self._get_whatsapp_client()
        if wa_client:
            status = wa_client.get_status()
        else:
            status = {
                "connected": False,
                "messages_sent_today": 0,
                "total_messages": len(messages),
                "rate_limit_remaining": 0,
            }
        _atomic_write_json(
            str(self._state_dir / "whatsapp_status.json"), status,
        )

    def _handle_wa_send(self, cmd: Dict[str, Any]) -> str:
        """Send a WhatsApp message from the dashboard.

        Includes dedup check via state/whatsapp/sent_commands/{cmd_id}.json.
        """
        import time as _time

        cmd_id = cmd.get("id", "")
        payload = cmd.get("payload", {})
        contact = payload.get("contact")
        text = payload.get("text", "")

        if not contact:
            return "Missing contact"
        if not text:
            return "Missing message text"

        # Dedup check
        dedup_dir = self._base_path / "state" / "whatsapp" / "sent_commands"
        dedup_dir.mkdir(parents=True, exist_ok=True)
        dedup_path = dedup_dir / f"{cmd_id}.json"
        if dedup_path.exists():
            return f"Duplicate command {cmd_id} — already processed"

        # Get WhatsApp client
        wa_client = self._get_whatsapp_client()
        if wa_client is None:
            return "WhatsApp client not available — WhatsApp is disabled in config"

        try:
            msg = wa_client.send_text(contact, text)

            # Record successful processing for dedup
            _atomic_write_json(str(dedup_path), {
                "cmd_id": cmd_id,
                "processed_at": _time.time(),
                "status": msg.status,
                "message_id": msg.id,
            })

            if msg.status == "sent":
                return f"WhatsApp message sent to {msg.contact_name}"
            else:
                return f"WhatsApp send failed: {msg.error}"
        except Exception as e:
            return f"WhatsApp send failed: {e}"

    def _get_whatsapp_client(self):
        """Get WhatsAppClient from skill registry (same pattern as email)."""
        try:
            if hasattr(self._merlin, "executor") and hasattr(
                self._merlin.executor, "registry"
            ):
                for skill in self._merlin.executor.registry._skills.values():
                    if hasattr(skill, "_client") and hasattr(
                        skill._client, "is_connected"
                    ):
                        # Found a WhatsApp skill with a client
                        return skill._client
        except Exception:
            pass
        return None

    # ─────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────

    def _get_mission_state(self) -> str:
        """Get current mission state from AttentionManager."""
        am = getattr(self._merlin, "attention_manager", None)
        if am is None:
            return "unknown"
        return am.mission_state.value
