# providers/whatsapp/neonize_provider.py

"""
NeonizeProvider — Concrete CommunicationChannel using neonize (Whatsmeow).

Wraps the neonize client for WhatsApp message sending.
NEVER raises on send failures — always returns ChannelMessage with status.

Design rules:
- send_text / send_file return ChannelMessage (never raise)
- Rate limiting is enforced before every send
- Connection manager provides the underlying client
- Thread-safe (neonize send_message is documented as thread-safe)
"""

import logging
import os
import time
from typing import Optional

from providers.communication.base import CommunicationChannel, ChannelMessage
from providers.whatsapp.connection_manager import WhatsAppConnectionManager
from providers.whatsapp.rate_limiter import (
    WhatsAppRateLimiter,
    RateLimitExceeded,
)

logger = logging.getLogger(__name__)

# Crockford Base32 for message ID generation (same as EmailClient)
_CROCKFORD_BASE32 = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def _generate_message_id() -> str:
    """Generate a ULID-style message ID. Format: wa-{timestamp}{random}."""
    ts_ms = int(time.time() * 1000)
    ts_chars = []
    for _ in range(10):
        ts_chars.append(_CROCKFORD_BASE32[ts_ms & 0x1F])
        ts_ms >>= 5
    ts_chars.reverse()

    rand_bytes = os.urandom(10)
    rand_val = int.from_bytes(rand_bytes, "big")
    rand_chars = []
    for _ in range(16):
        rand_chars.append(_CROCKFORD_BASE32[rand_val & 0x1F])
        rand_val >>= 5
    rand_chars.reverse()

    return f"wa-{''.join(ts_chars)}{''.join(rand_chars)}"


class NeonizeProvider(CommunicationChannel):
    """WhatsApp provider using neonize (Whatsmeow wrapper).

    Args:
        connection_manager: Manages the neonize client lifecycle.
        rate_limiter: Token-bucket rate limiter.
    """

    def __init__(
        self,
        connection_manager: WhatsAppConnectionManager,
        rate_limiter: WhatsAppRateLimiter,
    ):
        self._conn_manager = connection_manager
        self._rate_limiter = rate_limiter

    @property
    def channel_name(self) -> str:
        return "whatsapp"

    def is_connected(self) -> bool:
        return self._conn_manager.is_connected

    def send_text(self, recipient_id: str, text: str) -> ChannelMessage:
        """Send a text message to a WhatsApp JID.

        Never raises. Returns ChannelMessage with status="failed" on error.
        """
        msg_id = _generate_message_id()
        now = time.time()

        try:
            # Rate limit check
            self._rate_limiter.acquire_or_raise()

            # Get connected client
            client = self._conn_manager.get_client()

            # Build JID and send
            from neonize.utils import build_jid
            jid = build_jid(recipient_id)
            result = client.send_message(jid, text)

            logger.info(
                "[WHATSAPP] Text sent to %s (msg_id=%s)",
                recipient_id, msg_id,
            )

            return ChannelMessage(
                id=msg_id,
                channel="whatsapp",
                recipient_id=recipient_id,
                contact_name="",  # Filled by client layer
                direction="outbound",
                content_type="text",
                content=text,
                status="sent",
                timestamp=now,
                metadata={
                    "neonize_response": str(result)[:200],
                    "rate_limit_remaining": self._rate_limiter.remaining,
                },
            )

        except RateLimitExceeded as e:
            logger.warning("[WHATSAPP] Rate limit exceeded: %s", e)
            return ChannelMessage(
                id=msg_id,
                channel="whatsapp",
                recipient_id=recipient_id,
                contact_name="",
                direction="outbound",
                content_type="text",
                content=text,
                status="failed",
                timestamp=now,
                error=str(e),
            )

        except RuntimeError as e:
            # Connection not available
            logger.error("[WHATSAPP] Not connected: %s", e)
            return ChannelMessage(
                id=msg_id,
                channel="whatsapp",
                recipient_id=recipient_id,
                contact_name="",
                direction="outbound",
                content_type="text",
                content=text,
                status="failed",
                timestamp=now,
                error=str(e),
            )

        except Exception as e:
            logger.error(
                "[WHATSAPP] Send failed to %s: %s",
                recipient_id, e, exc_info=True,
            )
            return ChannelMessage(
                id=msg_id,
                channel="whatsapp",
                recipient_id=recipient_id,
                contact_name="",
                direction="outbound",
                content_type="text",
                content=text,
                status="failed",
                timestamp=now,
                error=f"Send failed: {e}",
            )

    def send_file(
        self,
        recipient_id: str,
        file_data: bytes,
        filename: str,
        mime_type: str,
        caption: Optional[str] = None,
    ) -> ChannelMessage:
        """Send a file to a WhatsApp JID.

        Automatically selects the correct neonize builder based on MIME type:
        - image/* → build_image_message
        - video/* → build_video_message
        - audio/* → build_audio_message
        - * → build_document_message

        Never raises. Returns ChannelMessage with status="failed" on error.
        """
        msg_id = _generate_message_id()
        now = time.time()

        try:
            self._rate_limiter.acquire_or_raise()
            client = self._conn_manager.get_client()

            from neonize.utils import build_jid
            jid = build_jid(recipient_id)

            # Select the correct message builder based on MIME type
            if mime_type.startswith("image/"):
                msg = client.build_image_message(
                    file_data, caption=caption or "", mime=mime_type,
                )
            elif mime_type.startswith("video/"):
                msg = client.build_video_message(
                    file_data, caption=caption or "", mime=mime_type,
                )
            elif mime_type.startswith("audio/"):
                msg = client.build_audio_message(
                    file_data, mime=mime_type,
                )
            else:
                msg = client.build_document_message(
                    file_data, filename=filename,
                    caption=caption or "", mime=mime_type,
                )

            result = client.send_message(jid, msg)

            logger.info(
                "[WHATSAPP] File '%s' sent to %s (msg_id=%s)",
                filename, recipient_id, msg_id,
            )

            return ChannelMessage(
                id=msg_id,
                channel="whatsapp",
                recipient_id=recipient_id,
                contact_name="",
                direction="outbound",
                content_type="file",
                content=filename,
                status="sent",
                timestamp=now,
                metadata={
                    "filename": filename,
                    "mime_type": mime_type,
                    "size": len(file_data),
                    "caption": caption or "",
                    "neonize_response": str(result)[:200],
                    "rate_limit_remaining": self._rate_limiter.remaining,
                },
            )

        except RateLimitExceeded as e:
            return ChannelMessage(
                id=msg_id,
                channel="whatsapp",
                recipient_id=recipient_id,
                contact_name="",
                direction="outbound",
                content_type="file",
                content=filename,
                status="failed",
                timestamp=now,
                error=str(e),
            )

        except RuntimeError as e:
            return ChannelMessage(
                id=msg_id,
                channel="whatsapp",
                recipient_id=recipient_id,
                contact_name="",
                direction="outbound",
                content_type="file",
                content=filename,
                status="failed",
                timestamp=now,
                error=str(e),
            )

        except Exception as e:
            logger.error(
                "[WHATSAPP] File send failed to %s: %s",
                recipient_id, e, exc_info=True,
            )
            return ChannelMessage(
                id=msg_id,
                channel="whatsapp",
                recipient_id=recipient_id,
                contact_name="",
                direction="outbound",
                content_type="file",
                content=filename,
                status="failed",
                timestamp=now,
                error=f"File send failed: {e}",
            )
