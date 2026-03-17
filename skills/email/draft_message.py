# skills/email/draft_message.py

"""
DraftMessageSkill — Generate an email draft via LLM.

Uses content_llm (same DI pattern as GenerateTextSkill) to generate
subject + body, then persists the draft via EmailClient.

The draft appears in the dashboard Mail page for user review.
MERLIN does NOT send emails autonomously.
"""

import logging
import time
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


class DraftMessageSkill(Skill):
    """Generate an email draft using an LLM.

    Inputs:
        prompt:    What the email should say / accomplish
        recipient: Email address of the recipient

    Optional:
        subject: Override subject (LLM generates if omitted)
        style:   Tone modifier (e.g., "formal", "casual")

    Outputs:
        draft_id:      ULID of the created draft
        draft_preview: First 200 chars of generated body
    """

    contract = SkillContract(
        name="email.draft_message",
        action="draft_message",
        target_type="email",
        description="Draft an email message",
        narration_template="drafting email to {recipient}",
        intent_verbs=["email", "write", "draft", "compose", "send", "mail"],
        intent_keywords=["email", "mail", "message", "inbox", "outlook",
                         "gmail", "recipient", "subject"],
        verb_specificity="generic",
        domain="email",
        requires_focus=False,
        resource_cost="medium",
        inputs={
            "prompt": "text_prompt",
            "recipient": "email_address",
        },
        optional_inputs={
            "subject": "email_subject",
            "style": "text_prompt",
        },
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

    def __init__(self, content_llm: LLMClient, email_client: EmailClient,
                 user_knowledge=None):
        self._llm = content_llm
        self._email_client = email_client
        self._user_knowledge = user_knowledge

    def execute(
        self,
        inputs: Dict[str, Any],
        world: WorldTimeline,
        snapshot: Optional[WorldSnapshot] = None,
    ) -> SkillResult:
        prompt = inputs["prompt"]
        recipient = inputs["recipient"]
        subject = inputs.get("subject", "")
        style = inputs.get("style", "")

        logger.info(
            "[TRACE] DraftMessageSkill.execute: recipient=%s, prompt=%r",
            recipient, prompt[:80],
        )

        # Build user context from structured memory (allow-listed + sanitized)
        user_context = ""
        if self._user_knowledge:
            user_context = self._user_knowledge.format_profile_for_prompt()

        # Build LLM prompt for email generation
        system = (
            "You are MERLIN, an intelligent desktop automation assistant. "
            "Generate a professional email based on the user's request.\n"
        )
        if user_context:
            system += (
                f"\nUser identity:\n{user_context}\n"
                "Use this information to personalize the email "
                "(e.g., sign with the user's name, use correct dates).\n"
            )
        system += (
            "\nReturn the email in this exact format:\n"
            "SUBJECT: <subject line>\n"
            "BODY:\n<email body>\n\n"
            "Do not add meta-commentary. Just produce the email."
        )
        if style:
            system += f" Write in a {style} style."
        if subject:
            system += f" Use this subject: {subject}"

        full_prompt = (
            f"{system}\n\n"
            f"Recipient: {recipient}\n"
            f"User request: {prompt}"
        )

        text = self._llm.complete(full_prompt)

        if not text or not text.strip():
            raise RuntimeError(
                f"Email generation returned empty response for: {prompt[:100]}"
            )

        # Parse subject and body from LLM output
        generated_subject, generated_body = self._parse_email(text.strip())

        # Use provided subject if given, otherwise use LLM-generated
        final_subject = subject if subject else generated_subject

        # Create draft via EmailClient
        draft = self._email_client.create_draft(
            recipient=recipient,
            subject=final_subject,
            body=generated_body,
            source_query=prompt,
            intent_source="email.draft_message",
        )

        logger.info(
            "[TRACE] DraftMessageSkill: draft %s created (%d chars)",
            draft["id"], len(generated_body),
        )

        return SkillResult(
            outputs={
                "draft_id": draft["id"],
                "draft_preview": generated_body[:200],
            },
            metadata={"domain": "email"},
        )

    @staticmethod
    def _parse_email(text: str) -> tuple:
        """Parse LLM output into (subject, body).

        Expected format:
            SUBJECT: ...
            BODY:
            ...

        Falls back to empty subject + full text as body.
        """
        subject = ""
        body = text

        lines = text.split("\n")
        body_start = 0

        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.upper().startswith("SUBJECT:"):
                subject = stripped[len("SUBJECT:"):].strip()
            elif stripped.upper() == "BODY:" or stripped.upper().startswith("BODY:"):
                body_content = stripped[len("BODY:"):].strip()
                remaining = "\n".join(lines[i + 1:]).strip()
                body = (body_content + "\n" + remaining).strip() if body_content else remaining
                break

        if not subject:
            subject = "No Subject"

        return subject, body
