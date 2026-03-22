# providers/whatsapp/client.py

"""
WhatsAppClient — Facade over NeonizeProvider + ContactResolver + persistence.

Skills depend on this, never on providers directly.
Same pattern as EmailClient (facade over EmailProvider + draft persistence).

Responsibilities:
- Contact resolution (name → JID via ContactResolver)
- Message delegation to NeonizeProvider
- Message history persistence at state/whatsapp/messages/
- Connection status delegation to ConnectionManager
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from providers.communication.base import ChannelMessage
from providers.whatsapp.contact_resolver import (
    ContactResolver,
    ContactNotFoundError,
    ContactAmbiguousError,
)
from providers.whatsapp.neonize_provider import NeonizeProvider

logger = logging.getLogger(__name__)


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


class WhatsAppClient:
    """Facade over NeonizeProvider + ContactResolver + message persistence.

    Skills call this. Never the provider directly.
    Messages stored at {messages_dir}/*.json (MERLIN-owned).

    Args:
        provider: The NeonizeProvider instance.
        contact_resolver: Resolves names to phone numbers.
        messages_dir: Path to message history directory.
    """

    def __init__(
        self,
        provider: NeonizeProvider,
        contact_resolver: ContactResolver,
        messages_dir: str = "state/whatsapp/messages",
    ):
        self._provider = provider
        self._resolver = contact_resolver
        self._messages_dir = Path(messages_dir)
        self._messages_dir.mkdir(parents=True, exist_ok=True)
        # Also ensure sent_commands dir for dedup
        self._dedup_dir = Path(messages_dir).parent / "sent_commands"
        self._dedup_dir.mkdir(parents=True, exist_ok=True)

    @property
    def is_connected(self) -> bool:
        """Whether WhatsApp is currently connected."""
        return self._provider.is_connected()

    def send_text(
        self,
        contact: str,
        text: str,
    ) -> ChannelMessage:
        """Resolve contact and send a text message.

        Args:
            contact: Human name ("Mom"), alias ("amma"), or phone number.
            text: Message text to send.

        Returns:
            ChannelMessage with the send result.

        Raises:
            ContactNotFoundError: No number found (skill converts to no_op).
            ContactAmbiguousError: Multiple matches (skill converts to no_op).
        """
        # Resolve contact to phone number
        phone, display_name = self._resolver.resolve(contact)

        # Send via provider
        msg = self._provider.send_text(phone, text)

        # Enrich with resolved contact name
        msg = ChannelMessage(
            id=msg.id,
            channel=msg.channel,
            recipient_id=msg.recipient_id,
            contact_name=display_name,
            direction=msg.direction,
            content_type=msg.content_type,
            content=msg.content,
            status=msg.status,
            timestamp=msg.timestamp,
            metadata=msg.metadata,
            error=msg.error,
        )

        # Persist message record
        self._persist_message(msg)

        # Stage 4: Learning — cache successful resolution
        if msg.status == "sent":
            self._resolver.learn_contact(contact, phone, source="send")

        return msg

    def send_file(
        self,
        contact: str,
        file_data: bytes,
        filename: str,
        mime_type: str,
        caption: Optional[str] = None,
    ) -> ChannelMessage:
        """Resolve contact and send a file.

        Args:
            contact: Human name or phone number.
            file_data: Raw file bytes.
            filename: Display filename.
            mime_type: MIME type string.
            caption: Optional caption text.

        Returns:
            ChannelMessage with the send result.

        Raises:
            ContactNotFoundError: No number found.
            ContactAmbiguousError: Multiple matches.
        """
        phone, display_name = self._resolver.resolve(contact)
        msg = self._provider.send_file(
            phone, file_data, filename, mime_type, caption,
        )

        # Enrich with resolved contact name
        msg = ChannelMessage(
            id=msg.id,
            channel=msg.channel,
            recipient_id=msg.recipient_id,
            contact_name=display_name,
            direction=msg.direction,
            content_type=msg.content_type,
            content=msg.content,
            status=msg.status,
            timestamp=msg.timestamp,
            metadata=msg.metadata,
            error=msg.error,
        )

        self._persist_message(msg)

        # Stage 4: Learning — cache successful resolution
        if msg.status == "sent":
            self._resolver.learn_contact(contact, phone, source="send")

        return msg

    def _persist_message(self, msg: ChannelMessage) -> None:
        """Persist a message record to state/whatsapp/messages/."""
        record = {
            "id": msg.id,
            "channel": msg.channel,
            "recipient_id": msg.recipient_id,
            "contact_name": msg.contact_name,
            "direction": msg.direction,
            "content_type": msg.content_type,
            "content": msg.content,
            "status": msg.status,
            "timestamp": msg.timestamp,
            "metadata": msg.metadata,
            "error": msg.error,
        }

        path = str(self._messages_dir / f"{msg.id}.json")
        _atomic_write_json(path, record)
        logger.info(
            "[WHATSAPP] Message persisted: %s → %s (%s)",
            msg.id, msg.contact_name, msg.status,
        )

    def get_messages(self, limit: int = 200) -> List[Dict[str, Any]]:
        """List recent messages, sorted newest first."""
        messages = []
        for f in self._messages_dir.glob("wa-*.json"):
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                    messages.append(data)
            except (json.JSONDecodeError, OSError):
                continue

        messages.sort(
            key=lambda m: m.get("timestamp", 0), reverse=True,
        )
        return messages[:limit]

    def get_status(self) -> Dict[str, Any]:
        """Get WhatsApp connection status summary."""
        messages = self.get_messages(limit=1000)
        today_start = time.time() - (time.time() % 86400)
        sent_today = sum(
            1 for m in messages
            if m.get("timestamp", 0) >= today_start
            and m.get("status") == "sent"
        )

        return {
            "connected": self.is_connected,
            "messages_sent_today": sent_today,
            "total_messages": len(messages),
            "rate_limit_remaining": self._provider._rate_limiter.remaining,
        }
