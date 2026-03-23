# providers/communication/base.py

"""
CommunicationChannel — ABC for real-time messaging platforms.

Parallel to EmailProvider (which is SMTP/IMAP-specific).
This abstraction covers conversational, identity-bound channels:
WhatsApp, Telegram, Slack, Discord, etc.

Design rules:
- send_text / send_file always return ChannelMessage (never raise)
- Failures are captured in ChannelMessage.status + .error
- Provider owns rate limiting internally
- No persistence — client layer handles message logging
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class ChannelMessage:
    """Unified message record for any communication channel.

    Immutable after creation (frozen dataclass).
    Used for both outbound sends and future inbound receives.
    """
    id: str                           # ULID-style message ID
    channel: str                      # "whatsapp", "telegram", "slack"
    recipient_id: str                 # Channel-native ID (JID, user_id, etc.)
    contact_name: str                 # Resolved human name (e.g., "Mom")
    direction: str                    # "outbound" | "inbound"
    content_type: str                 # "text" | "file" | "image" | "audio" | "video"
    content: str                      # Text content or file path
    status: str                       # "sent" | "failed" | "pending"
    timestamp: float                  # Unix timestamp
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None       # Error message if status == "failed"


class CommunicationChannel(ABC):
    """ABC for real-time messaging channels.

    NOT EmailProvider. Email is document-like (subject, CC, BCC, IMAP).
    This is for conversational, identity-bound channels.

    Implementations must:
    - Never raise on send failures (return ChannelMessage with status="failed")
    - Handle rate limiting internally
    - Be thread-safe
    """

    @abstractmethod
    def send_text(self, recipient_id: str, text: str) -> ChannelMessage:
        """Send a text message. Never raises — returns failed ChannelMessage on error."""
        ...

    @abstractmethod
    def send_file(
        self,
        recipient_id: str,
        file_data: bytes,
        filename: str,
        mime_type: str,
        caption: Optional[str] = None,
    ) -> ChannelMessage:
        """Send a file attachment. Never raises — returns failed ChannelMessage on error."""
        ...

    @abstractmethod
    def is_connected(self) -> bool:
        """Whether the channel has an active connection."""
        ...

    @property
    @abstractmethod
    def channel_name(self) -> str:
        """Channel identifier (e.g., 'whatsapp', 'telegram')."""
        ...
