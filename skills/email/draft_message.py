# skills/email/draft_message.py

"""
DraftMessageSkill — Generate an email draft via LLM.

Uses content_llm (same DI pattern as GenerateTextSkill) to generate
subject + body, then persists the draft via EmailClient.

The draft appears in the dashboard Mail page for user review.
MERLIN does NOT send emails autonomously.
"""

import logging
import mimetypes
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

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
        intent_keywords=["email", "emails", "mail", "mails", "message",
                         "messages", "inbox", "outlook", "gmail",
                         "recipient", "subject"],
        verb_specificity="generic",
        domain="email",
        requires_focus=False,
        resource_cost="medium",
        inputs={
            "recipient": "email_address",
        },
        optional_inputs={
            "prompt": "text_prompt",
            "subject": "email_subject",
            "style": "text_prompt",
            "attachments": "file_ref_list",
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
        requires=[],
        produces=["email_draft"],
        effect_type="create",
    )

    def __init__(self, content_llm: LLMClient, email_client: EmailClient,
                 user_knowledge=None, location_config=None, file_index=None):
        self._llm = content_llm
        self._email_client = email_client
        self._user_knowledge = user_knowledge
        self._location_config = location_config
        self._file_index = file_index

    def execute(
        self,
        inputs: Dict[str, Any],
        world: WorldTimeline,
        snapshot: Optional[WorldSnapshot] = None,
    ) -> SkillResult:
        prompt = inputs.get("prompt", "")
        recipient = inputs["recipient"]
        subject = inputs.get("subject", "")
        style = inputs.get("style", "")
        raw_attachments = inputs.get("attachments", [])

        # Auto-generate forwarding prompt when only attachments are given
        if not prompt and raw_attachments:
            file_names = []
            for a in raw_attachments:
                if isinstance(a, dict):
                    file_names.append(a.get("name", str(a)))
                else:
                    file_names.append(str(a))
            prompt = f"Forward the attached file(s): {', '.join(file_names)}"
        elif not prompt:
            prompt = "Write a brief professional email"

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

        # Validate and resolve attachments (FileRef → absolute path)
        validated_attachments = None
        if raw_attachments:
            # Resolve string file names to FileRef dicts at execution time
            resolved = self._resolve_string_attachments(raw_attachments)
            try:
                validated_attachments = self._validate_attachments(resolved)
            except ValueError as ve:
                if "Duplicate attachment name" in str(ve):
                    # Multiple files with same name → ambiguity.
                    # Signal via no_op so recovery loop asks the user.
                    from pathlib import Path as _P
                    dup_name = str(ve).split(": ", 1)[-1] if ": " in str(ve) else "?"
                    dupes = [
                        r for r in resolved
                        if _P(r.get("relative_path", r.get("name", ""))).name == dup_name
                    ]
                    options = [
                        f"  {i+1}. {d.get('relative_path', d.get('name', '?'))}"
                        for i, d in enumerate(dupes[:5])
                    ]
                    question = (
                        f"Multiple files named '{dup_name}' — "
                        f"which one should I attach?\n"
                        + "\n".join(options)
                    )
                    return SkillResult(
                        outputs={},
                        status="no_op",
                        metadata={
                            "domain": "email",
                            "reason": "ambiguous_input",
                            "message": question,
                            "options": options,
                        },
                    )
                raise  # Re-raise non-duplicate ValueErrors
            logger.info(
                "[TRACE] DraftMessageSkill: %d attachment(s) validated",
                len(validated_attachments),
            )

        # Create draft via EmailClient
        draft = self._email_client.create_draft(
            recipient=recipient,
            subject=final_subject,
            body=generated_body,
            source_query=prompt,
            intent_source="email.draft_message",
            attachments=validated_attachments,
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

    # ── Attachment validation ────────────────────────────────

    MAX_ATTACHMENT_SIZE = 25 * 1024 * 1024   # 25 MB per file
    MAX_TOTAL_SIZE = 25 * 1024 * 1024        # 25 MB total

    def _resolve_string_attachments(
        self, raw_attachments: List[Any],
    ) -> List[Dict[str, Any]]:
        """Resolve string file names to FileRef dicts at execution time.

        The LLM may pass attachment names as bare strings (e.g., ["resume.pdf"])
        instead of structured FileRef dicts. This method resolves them
        dynamically using FileIndex — same index used by fs.search_file.

        Architecture note: the DECOMPOSER should NOT plan fs.search_file
        for attachments. File resolution happens HERE, at execution time.
        This is the same principle as Phase 9E entity resolution but for
        file_ref_list inputs instead of file_path_input.
        """
        resolved: List[Dict[str, Any]] = []
        for item in raw_attachments:
            if isinstance(item, dict):
                # Already a FileRef dict — pass through
                resolved.append(item)
            elif isinstance(item, str) and item.strip():
                # Bare file name — resolve via FileIndex
                if self._file_index and self._location_config:
                    matches = self._file_index.search(
                        item.strip(),
                        location_config=self._location_config,
                    )
                    if matches:
                        best = matches[0]
                        resolved.append(best.to_output_dict())
                        logger.info(
                            "[TRACE] Resolved attachment '%s' → %s/%s (confidence=%.2f)",
                            item, best.anchor, best.relative_path, best.confidence,
                        )
                    else:
                        # Not found in index — pass as minimal dict,
                        # _validate_attachments will raise FileNotFoundError
                        resolved.append({"name": item.strip()})
                        logger.warning(
                            "[TRACE] Attachment '%s' not found in FileIndex",
                            item,
                        )
                else:
                    # No FileIndex available — pass as minimal dict
                    resolved.append({"name": item.strip()})
            else:
                logger.warning(
                    "[TRACE] Skipping invalid attachment entry: %r", item,
                )
        return resolved

    def _validate_attachments(
        self, file_refs: List[Dict[str, Any]],
    ) -> List[Dict[str, str]]:
        """Validate and resolve FileRef dicts to attachment metadata.

        Resolves via location_config.resolve(anchor) / relative_path
        (same pattern as read_file, write_file, create_folder).

        Checks: existence, permissions, size, MIME type, duplicates.
        """
        validated: List[Dict[str, str]] = []
        total_size = 0
        seen_names: set = set()

        for ref in file_refs:
            # Resolve FileRef to absolute path
            anchor = ref.get("anchor", "WORKSPACE")
            rel_path = ref.get("relative_path", ref.get("name", ""))

            if self._location_config:
                p = self._location_config.resolve(anchor) / rel_path
            else:
                p = Path(rel_path)

            # 1. Existence
            if not p.exists():
                raise FileNotFoundError(
                    f"Attachment not found: {p}"
                )
            # 2. Permissions
            if not os.access(p, os.R_OK):
                raise PermissionError(
                    f"Cannot read attachment: {p}"
                )
            # 3. Size
            size = p.stat().st_size
            if size > self.MAX_ATTACHMENT_SIZE:
                raise ValueError(
                    f"Attachment too large "
                    f"({size / (1024 * 1024):.1f}MB): {p.name}"
                )
            total_size += size
            if total_size > self.MAX_TOTAL_SIZE:
                raise ValueError(
                    f"Total attachments exceed "
                    f"{self.MAX_TOTAL_SIZE / (1024 * 1024):.0f}MB limit"
                )
            # 4. MIME type
            mime_type = (
                mimetypes.guess_type(str(p))[0]
                or "application/octet-stream"
            )
            # 5. Duplicate name
            if p.name in seen_names:
                raise ValueError(
                    f"Duplicate attachment name: {p.name}"
                )
            seen_names.add(p.name)

            validated.append({
                "path": str(p),
                "name": p.name,
                "mime_type": mime_type,
                "size": str(size),
            })

        return validated
