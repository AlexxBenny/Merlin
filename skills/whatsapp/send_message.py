# skills/whatsapp/send_message.py

"""
WhatsAppSendMessageSkill — Send a text message via WhatsApp.

Uses the existing ask-back mechanism for ambiguous contacts:
- ContactAmbiguousError → SkillResult(status="no_op", reason="ambiguous_input")
- ContactNotFoundError  → SkillResult(status="no_op", reason="ambiguous_input")

Both trigger MERLIN's PendingMission → _resume_from_clarification flow
with ZERO changes to merlin.py, executor, or orchestrator.

risk_level="destructive" → REQUIRES_CONFIRMATION supervisor guard.
"""

import logging
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


class WhatsAppSendMessageSkill(Skill):
    """Send a text message via WhatsApp.

    Inputs:
        contact: Contact name ("Mom"), alias ("amma"), or phone number
        message_text: Text content to send

    Outputs:
        send_status: Human-readable result message
    """

    contract = SkillContract(
        name="whatsapp.send_message",
        action="send_message",
        target_type="whatsapp",
        description="Send a WhatsApp text message",
        narration_template="sending WhatsApp message to {contact}",
        risk_level="destructive",
        intent_verbs=[
            "send", "message", "text", "whatsapp", "tell",
            "say", "ping", "msg",
        ],
        intent_keywords=[
            "whatsapp", "whats app", "wa", "message",
            "text", "msg", "chat",
        ],
        verb_specificity="generic",
        domain="whatsapp",
        requires_focus=False,
        resource_cost="low",
        inputs={
            "contact": "whatsapp_contact",
            "message_text": "text_prompt",
        },
        optional_inputs={},
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
        produces=["whatsapp_message_sent"],
        effect_type="create",
    )

    def __init__(self, whatsapp_client: WhatsAppClient):
        self._client = whatsapp_client

    def execute(
        self,
        inputs: Dict[str, Any],
        world: WorldTimeline,
        snapshot: Optional[WorldSnapshot] = None,
    ) -> SkillResult:
        contact = inputs["contact"]
        message_text = inputs["message_text"]

        logger.info(
            "[TRACE] WhatsAppSendMessageSkill.execute: "
            "contact=%s, text=%r",
            contact, message_text[:80],
        )

        try:
            msg = self._client.send_text(contact, message_text)
        except ContactAmbiguousError as e:
            # Multiple contacts match → trigger ask-back
            logger.info(
                "[WHATSAPP] Ambiguous contact '%s': %s",
                contact, e.user_message,
            )
            return SkillResult(
                outputs={},
                status="no_op",
                metadata={
                    "domain": "whatsapp",
                    "reason": "ambiguous_input",
                    "message": e.user_message,
                    "options": [
                        f"  {i+1}. {c['name']} ({c['phone']})"
                        for i, c in enumerate(e.candidates)
                    ],
                },
            )
        except ContactNotFoundError as e:
            # No contact found → ask user for number
            logger.info(
                "[WHATSAPP] Contact not found '%s': %s",
                contact, e.user_message,
            )
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

        # Check if send failed
        if msg.status == "failed":
            return SkillResult(
                outputs={
                    "send_status": f"Failed to send: {msg.error}",
                },
                metadata={
                    "domain": "whatsapp",
                    "message_id": msg.id,
                },
            )

        return SkillResult(
            outputs={
                "send_status": (
                    f"WhatsApp message sent to {msg.contact_name}"
                ),
            },
            metadata={
                "domain": "whatsapp",
                "message_id": msg.id,
            },
        )
