# skills/email/send_message.py

"""
SendMessageSkill — Send an approved email draft.

Two-stage safety:
1. Validates draft status == "approved" (dashboard review)
2. risk_level="destructive" → REQUIRES_CONFIRMATION supervisor guard

A malicious prompt cannot bypass both stages.
"""

import logging
from typing import Any, Dict, Optional

from skills.skill_result import SkillResult
from skills.base import Skill
from skills.contract import SkillContract, FailurePolicy
from ir.mission import ExecutionMode
from world.timeline import WorldTimeline
from world.snapshot import WorldSnapshot
from providers.email.client import EmailClient

logger = logging.getLogger(__name__)


class SendMessageSkill(Skill):
    """Send an approved email draft.

    Inputs:
        draft_id: ULID of the draft to send

    Outputs:
        send_status: Human-readable result message
    """

    contract = SkillContract(
        name="email.send_message",
        action="send_message",
        target_type="email",
        description="Send an approved email",
        narration_template="sending email draft {draft_id}",
        risk_level="destructive",
        intent_verbs=["send", "deliver", "dispatch"],
        intent_keywords=["email", "emails", "mail", "mails", "draft",
                         "drafts", "message", "messages"],
        verb_specificity="generic",
        domain="email",
        requires_focus=False,
        resource_cost="low",
        inputs={
            "draft_id": "draft_identifier",
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
    )

    def __init__(self, email_client: EmailClient):
        self._email_client = email_client

    def execute(
        self,
        inputs: Dict[str, Any],
        world: WorldTimeline,
        snapshot: Optional[WorldSnapshot] = None,
    ) -> SkillResult:
        draft_id = inputs["draft_id"]

        logger.info(
            "[TRACE] SendMessageSkill.execute: draft_id=%s", draft_id,
        )

        # send_draft validates approved status internally
        result = self._email_client.send_draft(draft_id)

        return SkillResult(
            outputs={
                "send_status": (
                    f"Email sent successfully. Message ID: "
                    f"{result.get('message_id', 'unknown')}"
                ),
            },
            metadata={"domain": "email"},
        )
