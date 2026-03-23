# skills/whatsapp/send_file.py

"""
WhatsAppSendFileSkill — Send a file via WhatsApp.

Uses the same ask-back mechanism as send_message for ambiguous contacts.
Uses the same file resolution pattern as DraftMessageSkill for attachments.

risk_level="destructive" → REQUIRES_CONFIRMATION supervisor guard.
"""

import logging
import mimetypes
from pathlib import Path
from typing import Any, Dict, Optional

from skills.skill_result import SkillResult
from skills.base import Skill
from skills.contract import SkillContract, FailurePolicy
from ir.mission import ExecutionMode
from world.timeline import WorldTimeline
from world.snapshot import WorldSnapshot
from providers.whatsapp.client import WhatsAppClient
from providers.whatsapp.contact_resolver import (
    ContactNotFoundError,
    ContactAmbiguousError,
)

logger = logging.getLogger(__name__)


class WhatsAppSendFileSkill(Skill):
    """Send a file via WhatsApp.

    Inputs:
        contact:   Contact name or phone number
        file_path: Absolute or relative path to the file

    Optional:
        caption: Caption text to include with the file

    Outputs:
        send_status: Human-readable result message
    """

    contract = SkillContract(
        name="whatsapp.send_file",
        action="send_file",
        target_type="whatsapp",
        description="Send a file via WhatsApp",
        narration_template="sending file to {contact} via WhatsApp",
        risk_level="destructive",
        intent_verbs=[
            "send", "share", "forward", "attach",
        ],
        intent_keywords=[
            "whatsapp", "whats app", "wa", "file",
            "document", "photo", "image", "video",
            "attachment", "pdf",
        ],
        verb_specificity="generic",
        domain="whatsapp",
        requires_focus=False,
        resource_cost="medium",
        inputs={
            "contact": "whatsapp_contact",
            "file_path": "file_path_input",
        },
        optional_inputs={
            "caption": "text_prompt",
        },
        outputs={
            "send_status": "info_string",
        },
        allowed_modes={ExecutionMode.foreground},
        failure_policy={
            ExecutionMode.foreground: FailurePolicy.FAIL,
        },
        emits_events=[],
        mutates_world=True,
        idempotent=False,
        data_freshness="snapshot",
        output_style="terse",
        requires=[],
        produces=["whatsapp_file_sent"],
        effect_type="create",
    )

    def __init__(self, whatsapp_client: WhatsAppClient,
                 location_config=None, file_index=None):
        self._client = whatsapp_client
        self._location_config = location_config
        self._file_index = file_index

    def execute(
        self,
        inputs: Dict[str, Any],
        world: WorldTimeline,
        snapshot: Optional[WorldSnapshot] = None,
    ) -> SkillResult:
        contact = inputs["contact"]
        file_path_input = inputs["file_path"]
        caption = inputs.get("caption", "")

        logger.info(
            "[TRACE] WhatsAppSendFileSkill.execute: "
            "contact=%s, file=%s",
            contact, file_path_input,
        )

        # Resolve file path
        file_path = self._resolve_file_path(file_path_input)
        if file_path is None:
            return SkillResult(
                outputs={
                    "send_status": f"File not found: {file_path_input}",
                },
                metadata={"domain": "whatsapp"},
            )

        # Read file data
        try:
            file_data = file_path.read_bytes()
        except (OSError, PermissionError) as e:
            return SkillResult(
                outputs={
                    "send_status": f"Cannot read file: {e}",
                },
                metadata={"domain": "whatsapp"},
            )

        # Detect MIME type
        mime_type = (
            mimetypes.guess_type(str(file_path))[0]
            or "application/octet-stream"
        )

        # Send via client (handles contact resolution)
        try:
            msg = self._client.send_file(
                contact, file_data, file_path.name,
                mime_type, caption or None,
            )
        except ContactAmbiguousError as e:
            return SkillResult(
                outputs={},
                status="no_op",
                metadata={
                    "domain": "whatsapp",
                    "reason": "ambiguous_input",
                    "message": e.user_message,
                    "matches": e.candidates,
                    "options": [
                        f"  {i+1}. {c['name']} ({c['phone']})"
                        for i, c in enumerate(e.candidates)
                    ],
                },
            )
        except ContactNotFoundError as e:
            return SkillResult(
                outputs={},
                status="no_op",
                metadata={
                    "domain": "whatsapp",
                    "reason": "ambiguous_input",
                    "message": e.user_message,
                    "options": [],
                },
            )

        if msg.status == "failed":
            return SkillResult(
                outputs={
                    "send_status": f"Failed to send file: {msg.error}",
                },
                metadata={
                    "domain": "whatsapp",
                    "message_id": msg.id,
                },
            )

        return SkillResult(
            outputs={
                "send_status": (
                    f"File '{file_path.name}' sent to "
                    f"{msg.contact_name} via WhatsApp"
                ),
            },
            metadata={
                "domain": "whatsapp",
                "message_id": msg.id,
            },
        )

    def _resolve_file_path(self, file_input: Any) -> Optional[Path]:
        """Resolve a file input to an absolute Path.

        Handles:
        - Absolute paths
        - Relative paths via location_config
        - FileRef dicts with anchor/relative_path
        - String file names via FileIndex (same as DraftMessageSkill)
        """
        if isinstance(file_input, dict):
            anchor = file_input.get("anchor", "WORKSPACE")
            rel_path = file_input.get(
                "relative_path", file_input.get("name", ""),
            )
            if self._location_config:
                p = self._location_config.resolve(anchor) / rel_path
                if p.exists():
                    return p

        if isinstance(file_input, str):
            # Try as absolute path
            p = Path(file_input)
            if p.is_absolute() and p.exists():
                return p

            # Try via FileIndex (same as DraftMessageSkill)
            if self._file_index and self._location_config:
                matches = self._file_index.search(
                    file_input.strip(),
                    location_config=self._location_config,
                )
                if matches:
                    best = matches[0]
                    resolved = (
                        self._location_config.resolve(best.anchor)
                        / best.relative_path
                    )
                    if resolved.exists():
                        return resolved

            # Try as relative path from cwd
            p = Path(file_input)
            if p.exists():
                return p.resolve()

        return None
