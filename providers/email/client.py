# providers/email/client.py

"""
EmailClient — Facade over EmailProvider + draft persistence.

Skills depend on this, never on providers directly.
Owns draft state at state/email/drafts/ (MERLIN-owned store).
API server accesses drafts through bridge, never directly.

Draft IDs use ULID format for:
- Collision-free concurrent creation
- Lexicographic time ordering
- Cross-session uniqueness
"""

import json
import logging
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from providers.email.base import EmailProvider

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# ULID-style ID generation (no external dependency)
# ─────────────────────────────────────────────────────────────
# Format: d-{timestamp_ms_base32}-{random_base32}
# Example: d-01HZXKJ1Y5-Z7Z8K8J1

_CROCKFORD_BASE32 = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def _generate_draft_id() -> str:
    """Generate a ULID-style draft ID.

    Time-ordered, collision-resistant, no external deps.
    Format: d-{10-char timestamp}{16-char random}
    """
    ts_ms = int(time.time() * 1000)

    # Encode timestamp (48 bits → 10 base32 chars)
    ts_chars = []
    for _ in range(10):
        ts_chars.append(_CROCKFORD_BASE32[ts_ms & 0x1F])
        ts_ms >>= 5
    ts_chars.reverse()

    # 80 bits of randomness → 16 base32 chars
    rand_bytes = os.urandom(10)
    rand_val = int.from_bytes(rand_bytes, "big")
    rand_chars = []
    for _ in range(16):
        rand_chars.append(_CROCKFORD_BASE32[rand_val & 0x1F])
        rand_val >>= 5
    rand_chars.reverse()

    return f"d-{''.join(ts_chars)}{''.join(rand_chars)}"


# ─────────────────────────────────────────────────────────────
# IMAP query validation
# ─────────────────────────────────────────────────────────────

# Valid IMAP search keys (RFC 3501 §6.4.4)
_VALID_IMAP_KEYS = {
    "ALL", "ANSWERED", "BCC", "BEFORE", "BODY", "CC", "DELETED",
    "DRAFT", "FLAGGED", "FROM", "HEADER", "KEYWORD", "LARGER",
    "NEW", "NOT", "OLD", "ON", "OR", "RECENT", "SEEN", "SENTBEFORE",
    "SENTON", "SENTSINCE", "SINCE", "SMALLER", "SUBJECT", "TEXT",
    "TO", "UID", "UNANSWERED", "UNDELETED", "UNDRAFT", "UNFLAGGED",
    "UNKEYWORD", "UNSEEN",
}

# IMAP date format: DD-Mon-YYYY
_IMAP_DATE_PATTERN = re.compile(
    r"\d{1,2}-(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-\d{4}",
    re.IGNORECASE,
)


def validate_imap_query(criteria: str) -> str:
    """Validate and sanitize an LLM-generated IMAP search query.

    Checks for:
    - Invalid/dangerous tokens
    - Malformed date formats
    - Injection attempts

    Returns:
        Sanitized criteria string.

    Raises:
        ValueError: If criteria contains invalid tokens.
    """
    if not criteria or not criteria.strip():
        raise ValueError("Empty IMAP search criteria")

    # Strip outer parens if present
    clean = criteria.strip()
    if clean.startswith("(") and clean.endswith(")"):
        clean = clean[1:-1].strip()

    # Tokenize — split on spaces, respecting quoted strings
    tokens = []
    current = ""
    in_quotes = False
    for char in clean:
        if char == '"':
            in_quotes = not in_quotes
            current += char
        elif char == " " and not in_quotes:
            if current:
                tokens.append(current)
                current = ""
        else:
            current += char
    if current:
        tokens.append(current)

    # Validate each non-quoted token is a valid IMAP key, date, or number
    for token in tokens:
        if token.startswith('"') and token.endswith('"'):
            continue  # Quoted string — allowed
        upper = token.upper()
        if upper in _VALID_IMAP_KEYS:
            continue
        if _IMAP_DATE_PATTERN.match(token):
            continue
        if token.isdigit():
            continue
        # Check for partial date without validation
        if re.match(r"\d{1,2}-\w{3}-\d{4}", token):
            raise ValueError(
                f"Invalid IMAP date format '{token}'. "
                f"Expected DD-Mon-YYYY (e.g., 01-Jan-2026)"
            )
        raise ValueError(
            f"Invalid IMAP search token: '{token}'. "
            f"Valid keys: {', '.join(sorted(_VALID_IMAP_KEYS))}"
        )

    return clean


# ─────────────────────────────────────────────────────────────
# Atomic file I/O with locking
# ─────────────────────────────────────────────────────────────

def _atomic_write_json(path: str, data: Any) -> None:
    """Write JSON atomically via tmp → rename."""
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        if os.path.exists(path):
            os.replace(tmp_path, path)
        else:
            os.rename(tmp_path, path)
    except Exception as e:
        logger.debug("Atomic write failed for %s: %s", path, e)
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
# EmailClient
# ─────────────────────────────────────────────────────────────

class EmailClient:
    """Facade over EmailProvider + draft persistence.

    Skills call this. Never the provider directly.
    Drafts stored at {drafts_dir}/*.json (MERLIN-owned).

    Args:
        provider: The concrete EmailProvider (SMTPProvider, etc.)
        drafts_dir: Path to draft storage directory.
        from_address: Default sender address.
    """

    def __init__(
        self,
        provider: EmailProvider,
        drafts_dir: str,
        from_address: str = "",
        signature: str = "",
    ):
        self._provider = provider
        self._drafts_dir = Path(drafts_dir)
        self._drafts_dir.mkdir(parents=True, exist_ok=True)
        self._from_address = from_address or os.environ.get(
            "EMAIL_FROM_ADDRESS", ""
        )
        self._signature = signature

    @property
    def is_configured(self) -> bool:
        """Whether the underlying provider is configured."""
        return self._provider.is_configured()

    # ─────────────────────────────────────────────────────────
    # Draft management
    # ─────────────────────────────────────────────────────────

    def create_draft(
        self,
        recipient: str,
        subject: str,
        body: str,
        source_query: str = "",
        intent_source: str = "email.draft_message",
        cc: str = "",
        bcc: str = "",
        attachments: Optional[List[Dict[str, str]]] = None,
        reply_to_message_id: str = "",
        thread_id: str = "",
    ) -> Dict[str, Any]:
        """Create a new draft and persist it.

        Returns the full draft dict.
        """
        draft_id = _generate_draft_id()
        now = time.time()

        draft = {
            "id": draft_id,
            "recipient": recipient,
            "cc": cc,
            "bcc": bcc,
            "subject": subject,
            "body": body,
            "status": "pending_review",
            "attachments": attachments or [],
            "source_query": source_query,
            "intent_source": intent_source,
            "reply_to_message_id": reply_to_message_id,
            "thread_id": thread_id,
            "created_at": now,
            "updated_at": now,
        }

        path = str(self._drafts_dir / f"{draft_id}.json")
        _atomic_write_json(path, draft)
        logger.info("[EMAIL] Draft created: %s → %s", draft_id, recipient)
        return draft

    def get_draft(self, draft_id: str) -> Optional[Dict[str, Any]]:
        """Load a single draft by ID."""
        path = str(self._drafts_dir / f"{draft_id}.json")
        return _safe_read_json(path)

    def update_draft(
        self, draft_id: str, updates: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Update draft fields. Returns updated draft or None."""
        draft = self.get_draft(draft_id)
        if draft is None:
            return None

        # Only allow updating safe fields
        allowed_fields = {
            "recipient", "cc", "bcc", "subject", "body",
            "status", "attachments",
        }
        for key, value in updates.items():
            if key in allowed_fields:
                draft[key] = value

        draft["updated_at"] = time.time()

        path = str(self._drafts_dir / f"{draft_id}.json")
        _atomic_write_json(path, draft)
        logger.info("[EMAIL] Draft updated: %s", draft_id)
        return draft

    def list_drafts(self) -> List[Dict[str, Any]]:
        """List all drafts, sorted newest first."""
        drafts = []
        for f in self._drafts_dir.glob("d-*.json"):
            data = _safe_read_json(str(f))
            if data:
                drafts.append(data)

        # Sort by created_at descending (ULID IDs are time-ordered,
        # but explicit sort is safer)
        drafts.sort(key=lambda d: d.get("created_at", 0), reverse=True)
        return drafts

    def discard_draft(self, draft_id: str) -> bool:
        """Mark a draft as discarded."""
        draft = self.get_draft(draft_id)
        if draft is None:
            return False
        draft["status"] = "discarded"
        draft["updated_at"] = time.time()
        path = str(self._drafts_dir / f"{draft_id}.json")
        _atomic_write_json(path, draft)
        logger.info("[EMAIL] Draft discarded: %s", draft_id)
        return True

    # ─────────────────────────────────────────────────────────
    # Provider delegation
    # ─────────────────────────────────────────────────────────

    def send(
        self,
        to: str,
        subject: str,
        body: str,
        cc: Optional[str] = None,
        bcc: Optional[str] = None,
        attachments: Optional[List[Dict[str, str]]] = None,
        reply_to_message_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send email via provider."""
        return self._provider.send(
            to=to,
            subject=subject,
            body=body,
            from_address=self._from_address,
            cc=cc,
            bcc=bcc,
            attachments=attachments,
            reply_to_message_id=reply_to_message_id,
        )

    def send_draft(self, draft_id: str) -> Dict[str, Any]:
        """Send an approved draft.

        Validates status == "approved" before sending.
        Updates draft status to "sent" after successful send.

        Raises:
            ValueError: If draft not found or not approved.
        """
        draft = self.get_draft(draft_id)
        if draft is None:
            raise ValueError(f"Draft {draft_id} not found")
        if draft["status"] != "approved":
            raise ValueError(
                f"Draft {draft_id} is '{draft['status']}', not 'approved'. "
                f"Approve the draft before sending."
            )

        # Append signature if configured
        body = draft["body"]
        if self._signature:
            body = f"{body}\n\n{self._signature}"

        result = self.send(
            to=draft["recipient"],
            subject=draft["subject"],
            body=body,
            cc=draft.get("cc"),
            bcc=draft.get("bcc"),
            attachments=draft.get("attachments"),
            reply_to_message_id=draft.get("reply_to_message_id"),
        )

        # Update status to sent
        self.update_draft(draft_id, {"status": "sent"})
        logger.info("[EMAIL] Draft sent: %s", draft_id)
        return result

    def fetch_inbox(
        self, limit: int = 10, folder: str = "INBOX",
    ) -> List[Dict[str, Any]]:
        """Fetch inbox headers via provider."""
        return self._provider.fetch_inbox(limit=limit, folder=folder)

    def search(
        self, imap_criteria: str, folder: str = "INBOX", limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search emails via provider with validated criteria."""
        validated = validate_imap_query(imap_criteria)
        return self._provider.search(
            imap_criteria=validated, folder=folder, limit=limit,
        )
