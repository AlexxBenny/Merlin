# skills/email/search_email.py

"""
SearchEmailSkill — Search emails by natural language query.

Pipeline: LLM → IMAP query builder → validate_imap_query → provider.search

The LLM converts natural language to IMAP criteria syntax. The
validate_imap_query function catches malformed output before it
reaches the IMAP server.
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
from providers.email.client import EmailClient, validate_imap_query

logger = logging.getLogger(__name__)


# Prompt template for LLM → IMAP conversion
_IMAP_BUILDER_PROMPT = """\
Convert this email search request to an IMAP search criteria string.

User query: "{query}"

Rules:
- Use standard IMAP search keys: FROM, TO, SUBJECT, SINCE, BEFORE, \
ON, BODY, TEXT, UNSEEN, SEEN, ANSWERED, FLAGGED
- Date format: DD-Mon-YYYY (e.g., 01-Jan-2026)
- Wrap string values in double quotes
- Combine multiple criteria with spaces (implicit AND)
- For OR, use: OR <criteria1> <criteria2>

Examples:
- "emails from Alex" → FROM "Alex"
- "unread emails about MERLIN" → UNSEEN SUBJECT "MERLIN"
- "emails from HR since last week" → FROM "HR" SINCE 09-Mar-2026
- "messages about report before January" → SUBJECT "report" BEFORE 01-Jan-2026

Return ONLY the IMAP criteria string, nothing else.
"""


class SearchEmailSkill(Skill):
    """Search emails using natural language queries.

    Inputs:
        query: Natural language search description

    Outputs:
        emails: List of matching email header dicts
    """

    contract = SkillContract(
        name="email.search_email",
        action="search_email",
        target_type="email",
        description="Search emails by criteria",
        narration_template="searching emails: {query}",
        intent_verbs=["search", "find", "look", "locate"],
        intent_keywords=["email", "mail", "from", "about", "subject",
                          "message", "sent"],
        verb_specificity="generic",
        domain="email",
        requires_focus=False,
        resource_cost="medium",
        inputs={
            "query": "email_search_query",
        },
        optional_inputs={},
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
        query = inputs["query"]

        logger.info(
            "[TRACE] SearchEmailSkill.execute: query=%r", query[:80],
        )

        # Stage 1: LLM converts natural language → IMAP criteria
        llm_prompt = _IMAP_BUILDER_PROMPT.format(query=query)
        imap_criteria = self._llm.complete(llm_prompt)

        if not imap_criteria or not imap_criteria.strip():
            raise RuntimeError(
                f"IMAP query builder returned empty result for: {query[:100]}"
            )

        imap_criteria = imap_criteria.strip()
        logger.info(
            "[TRACE] SearchEmailSkill: LLM → IMAP: %r", imap_criteria,
        )

        # Stage 2: Validate IMAP criteria (catches LLM hallucinations)
        try:
            validated = validate_imap_query(imap_criteria)
        except ValueError as e:
            logger.warning(
                "[EMAIL] IMAP query validation failed: %s. "
                "Falling back to SUBJECT search.", e,
            )
            # Fallback: use the original query as a SUBJECT search
            validated = f'SUBJECT "{query}"'

        # Stage 3: Execute search via provider
        emails = self._email_client.search(validated)

        logger.info(
            "[TRACE] SearchEmailSkill: found %d results", len(emails),
        )

        return SkillResult(
            outputs={"emails": emails},
            metadata={
                "domain": "email",
                "count": len(emails),
                "imap_criteria": validated,
            },
        )
