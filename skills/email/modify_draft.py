# skills/email/modify_draft.py

"""
ModifyDraftSkill — Conversational email draft editing.

Users say things like "make it more formal", "change the second paragraph",
"remove the last sentence". The LLM rewrites the draft body with guardrails
that prevent unintended changes to recipient, subject, or factual details.
"""

import logging
from typing import Any, Dict, Optional

from skills.skill_result import SkillResult
from skills.base import Skill
from skills.contract import SkillContract, FailurePolicy
from ir.mission import ExecutionMode
from world.timeline import WorldTimeline
from world.snapshot import WorldSnapshot
from models.base import LLMClient
from providers.email.client import EmailClient

logger = logging.getLogger(__name__)


class ModifyDraftSkill(Skill):
    """Edit an existing email draft via LLM instruction.

    Inputs:
        draft_id:    ULID of the draft to modify
        instruction: What to change (e.g., "make it more formal")

    Outputs:
        draft_id:      ULID of the updated draft
        draft_preview: First 200 chars of the new body
    """

    contract = SkillContract(
        name="email.modify_draft",
        action="modify_draft",
        target_type="email",
        description="Edit an email draft",
        narration_template="editing draft {draft_id}",
        intent_verbs=["change", "edit", "modify", "rewrite", "update",
                       "fix", "revise", "rephrase", "improve"],
        intent_keywords=["email", "draft", "message", "mail", "tone",
                          "formal", "casual", "paragraph"],
        verb_specificity="generic",
        domain="email",
        requires_focus=False,
        resource_cost="medium",
        inputs={
            "draft_id": "draft_identifier",
            "instruction": "text_prompt",
        },
        optional_inputs={},
        outputs={
            "draft_id": "draft_identifier",
            "draft_preview": "generated_text",
        },
        allowed_modes={ExecutionMode.foreground},
        failure_policy={
            ExecutionMode.foreground: FailurePolicy.FAIL,
        },
        emits_events=[],
        mutates_world=True,
        idempotent=False,
        data_freshness="snapshot",
        output_style="rich",
    )

    def __init__(self, content_llm: LLMClient, email_client: EmailClient):
        self._llm = content_llm
        self._email_client = email_client

    def execute(
        self,
        inputs: Dict[str, Any],
        world: WorldTimeline,
        snapshot: Optional[WorldSnapshot] = None,
    ) -> SkillResult:
        draft_id = inputs["draft_id"]
        instruction = inputs["instruction"]

        logger.info(
            "[TRACE] ModifyDraftSkill.execute: draft_id=%s, instruction=%r",
            draft_id, instruction[:80],
        )

        # Load existing draft
        draft = self._email_client.get_draft(draft_id)
        if draft is None:
            raise RuntimeError(f"Draft {draft_id} not found")

        if draft["status"] not in ("pending_review", "approved"):
            raise RuntimeError(
                f"Cannot modify draft {draft_id}: status is '{draft['status']}'"
            )

        # Build guardrailed prompt — includes full context
        prompt = (
            "You are editing an email draft. Apply the user's instruction.\n\n"
            f"Recipient: {draft['recipient']}\n"
            f"Subject: {draft['subject']}\n"
            f"Current body:\n{draft['body']}\n\n"
            f"User instruction: \"{instruction}\"\n\n"
            "Rules:\n"
            "- Rewrite the email body applying the instruction.\n"
            "- Do NOT change the recipient or factual details unless "
            "explicitly requested.\n"
            "- Preserve the overall intent and structure unless "
            "instructed otherwise.\n"
            "- Return ONLY the new email body. No commentary, "
            "no 'Subject:' prefix, no explanations."
        )

        new_body = self._llm.complete(prompt)

        if not new_body or not new_body.strip():
            raise RuntimeError(
                f"Draft modification returned empty response for: "
                f"{instruction[:100]}"
            )

        new_body = new_body.strip()

        # Update draft — reset to pending_review if was approved
        self._email_client.update_draft(draft_id, {
            "body": new_body,
            "status": "pending_review",
        })

        logger.info(
            "[TRACE] ModifyDraftSkill: draft %s updated (%d chars)",
            draft_id, len(new_body),
        )

        return SkillResult(
            outputs={
                "draft_id": draft_id,
                "draft_preview": new_body[:200],
            },
            metadata={"domain": "email"},
        )
