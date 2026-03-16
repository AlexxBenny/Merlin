# providers/email/base.py

"""
EmailProvider — Abstract interface for pluggable email backends.

Skills and EmailClient depend on this interface, never on concrete
providers. Adding a new provider (Gmail API, Microsoft Graph) requires
only implementing this ABC — zero skill rewrites.

v1: SMTPProvider (smtplib + imaplib)
v2: GmailAPIProvider, MicrosoftGraphProvider (OAuth2)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class EmailProvider(ABC):
    """Abstract email backend.

    Implementations handle transport-level concerns:
    - Connection management (SMTP, IMAP, REST API)
    - Authentication (app passwords, OAuth tokens)
    - Protocol-specific formatting

    Skills never see these details — they interact via EmailClient.
    """

    @abstractmethod
    def send(
        self,
        to: str,
        subject: str,
        body: str,
        from_address: str,
        cc: Optional[str] = None,
        bcc: Optional[str] = None,
        attachments: Optional[List[Dict[str, str]]] = None,
        reply_to_message_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send an email.

        Args:
            to: Recipient email address.
            subject: Email subject line.
            body: Email body (plain text).
            from_address: Sender email address.
            cc: CC recipients (comma-separated).
            bcc: BCC recipients (comma-separated).
            attachments: List of {path, filename, mime_type} dicts.
            reply_to_message_id: Message-ID header for threading replies.

        Returns:
            {"message_id": str, "status": "sent"}

        Raises:
            ConnectionError: SMTP/API connection failed.
            RuntimeError: Send failed after connection.
        """
        raise NotImplementedError

    @abstractmethod
    def fetch_inbox(
        self,
        limit: int = 10,
        folder: str = "INBOX",
    ) -> List[Dict[str, Any]]:
        """Fetch recent email headers (not full bodies).

        Returns list of:
            {
                "uid": str,
                "message_id": str,
                "thread_id": str,       # References/In-Reply-To derived
                "from": str,
                "to": str,
                "subject": str,
                "date": str,            # ISO format
                "snippet": str,         # first ~200 chars of body
                "has_attachments": bool,
            }

        Header-only fetch for scaling — full body loaded on demand.
        """
        raise NotImplementedError

    @abstractmethod
    def fetch_message(self, uid: str, folder: str = "INBOX") -> Dict[str, Any]:
        """Fetch full message body by UID.

        Returns:
            {
                "uid": str,
                "message_id": str,
                "thread_id": str,
                "from": str,
                "to": str,
                "subject": str,
                "date": str,
                "body": str,            # full plain text body
                "html_body": str,       # HTML body if available
                "attachments": list,
                "has_attachments": bool,
            }
        """
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        imap_criteria: str,
        folder: str = "INBOX",
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search emails using pre-built IMAP criteria.

        Args:
            imap_criteria: IMAP search string (e.g., 'FROM "alex" SINCE 01-Jan-2026').
                           Built by the skill's LLM→IMAP converter, NOT raw user text.
            folder: IMAP folder to search.
            limit: Max results to return.

        Returns:
            Same format as fetch_inbox (header-only).
        """
        raise NotImplementedError

    @abstractmethod
    def is_configured(self) -> bool:
        """Whether this provider has valid credentials/config."""
        raise NotImplementedError
