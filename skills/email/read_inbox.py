# skills/email/read_inbox.py

"""
ReadInboxSkill — Fetch recent email headers.

Returns RAW email data only — no summarization.
Summarization is handled by reasoning.generate_text via mission DAG
composition (keeps data retrieval separate from reasoning).

Header-only fetch for scaling — full bodies loaded on demand.
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


class ReadInboxSkill(Skill):
    """Fetch recent email headers from inbox.

    Inputs: (none required)

    Optional:
        limit: Max emails to fetch (default 10)

    Outputs:
        emails: List of email header dicts
    """

    contract = SkillContract(
        name="email.read_inbox",
        action="read_inbox",
        target_type="email",
        description="Read recent emails from inbox",
        narration_template="checking inbox",
        intent_verbs=["read", "check", "show", "get", "list", "fetch"],
        intent_keywords=["email", "emails", "inbox", "mail", "mails",
                          "messages", "unread", "mailbox", "latest"],
        verb_specificity="generic",
        domain="email",
        requires_focus=False,
        resource_cost="medium",
        inputs={},
        optional_inputs={
            "limit": "step_limit",
        },
        outputs={
            "emails": "email_list",
        },
        allowed_modes={ExecutionMode.foreground},
        failure_policy={
            ExecutionMode.foreground: FailurePolicy.FAIL,
        },
        emits_events=[],
        mutates_world=False,
        idempotent=True,
        data_freshness="live",
        output_style="rich",
        requires=[],
        produces=["email_list"],
        effect_type="reveal",
    )

    def __init__(self, email_client: EmailClient):
        self._email_client = email_client

    def execute(
        self,
        inputs: Dict[str, Any],
        world: WorldTimeline,
        snapshot: Optional[WorldSnapshot] = None,
    ) -> SkillResult:
        limit = inputs.get("limit", 10)
        if isinstance(limit, str):
            try:
                limit = int(limit)
            except ValueError:
                limit = 10

        logger.info(
            "[TRACE] ReadInboxSkill.execute: limit=%d", limit,
        )

        emails = self._email_client.fetch_inbox(limit=limit)

        logger.info(
            "[TRACE] ReadInboxSkill: fetched %d emails", len(emails),
        )

        return SkillResult(
            outputs={"emails": emails},
            metadata={
                "domain": "email",
                "count": len(emails),
            },
        )
