# providers/email/smtp_provider.py

"""
SMTPProvider — v1 email backend using smtplib + imaplib.

Uses Python stdlib only (no external dependencies).
Credentials from environment variables / config.
Connection-per-operation (no persistent connections).

Threading safety: each call creates its own connection.
"""

import email as email_lib
import email.header
import email.utils
import imaplib
import logging
import os
import re
import smtplib
import ssl
from datetime import datetime
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional

from providers.email.base import EmailProvider

logger = logging.getLogger(__name__)


class SMTPProvider(EmailProvider):
    """Email provider using SMTP (send) + IMAP (read/search).

    Configuration:
        smtp.host, smtp.port, smtp.use_tls
        imap.host, imap.port, imap.use_ssl
        Credentials from EMAIL_USERNAME / EMAIL_PASSWORD env vars.
    """

    def __init__(self, config: dict):
        self._smtp_config = config.get("smtp", {})
        self._imap_config = config.get("imap", {})
        self._defaults = config.get("defaults", {})

        # Credentials from env (never stored in config YAML)
        self._username = os.environ.get("EMAIL_USERNAME", "")
        self._password = os.environ.get("EMAIL_PASSWORD", "")

    def is_configured(self) -> bool:
        """Check if minimum SMTP config exists."""
        return bool(
            self._smtp_config.get("host")
            and self._username
            and self._password
        )

    # ─────────────────────────────────────────────────────────
    # Send
    # ─────────────────────────────────────────────────────────

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
        """Send email via SMTP."""
        host = self._smtp_config.get("host", "")
        port = self._smtp_config.get("port", 587)
        use_tls = self._smtp_config.get("use_tls", True)

        if not host:
            raise ConnectionError("SMTP host not configured")

        # Build message
        if attachments:
            msg = MIMEMultipart()
            msg.attach(MIMEText(body, "plain", "utf-8"))
            for att in attachments:
                path = att.get("path", "")
                filename = att.get("filename", os.path.basename(path))
                mime_type = att.get("mime_type", "application/octet-stream")
                try:
                    with open(path, "rb") as f:
                        part = MIMEApplication(f.read(), Name=filename)
                    part["Content-Disposition"] = f'attachment; filename="{filename}"'
                    msg.attach(part)
                except FileNotFoundError:
                    logger.warning("Attachment not found: %s", path)
        else:
            msg = MIMEText(body, "plain", "utf-8")

        msg["Subject"] = subject
        msg["From"] = from_address
        msg["To"] = to
        if cc:
            msg["Cc"] = cc
        if reply_to_message_id:
            msg["In-Reply-To"] = reply_to_message_id
            msg["References"] = reply_to_message_id

        # Add Message-ID
        domain = from_address.split("@")[-1] if "@" in from_address else "merlin.local"
        msg["Message-ID"] = email.utils.make_msgid(domain=domain)

        # Collect all recipients
        all_recipients = [to]
        if cc:
            all_recipients.extend(r.strip() for r in cc.split(","))
        if bcc:
            all_recipients.extend(r.strip() for r in bcc.split(","))

        # Send
        try:
            if use_tls:
                context = ssl.create_default_context()
                with smtplib.SMTP(host, port, timeout=30) as server:
                    server.ehlo()
                    server.starttls(context=context)
                    server.ehlo()
                    server.login(self._username, self._password)
                    server.sendmail(from_address, all_recipients, msg.as_string())
            else:
                with smtplib.SMTP(host, port, timeout=30) as server:
                    server.login(self._username, self._password)
                    server.sendmail(from_address, all_recipients, msg.as_string())

            logger.info("[EMAIL] Sent to %s: %s", to, subject[:50])
            return {
                "message_id": msg["Message-ID"],
                "status": "sent",
            }

        except smtplib.SMTPAuthenticationError as e:
            raise ConnectionError(f"SMTP authentication failed: {e}") from e
        except smtplib.SMTPException as e:
            raise RuntimeError(f"SMTP send failed: {e}") from e

    # ─────────────────────────────────────────────────────────
    # Fetch inbox (header-only for scaling)
    # ─────────────────────────────────────────────────────────

    def fetch_inbox(
        self,
        limit: int = 10,
        folder: str = "INBOX",
    ) -> List[Dict[str, Any]]:
        """Fetch recent email headers via IMAP."""
        conn = self._imap_connect()
        try:
            conn.select(folder, readonly=True)
            _, data = conn.search(None, "ALL")
            uids = data[0].split()

            # Take last N (most recent)
            uids = uids[-limit:] if len(uids) > limit else uids
            uids.reverse()  # newest first

            results = []
            for uid in uids:
                headers = self._fetch_headers(conn, uid)
                if headers:
                    results.append(headers)

            return results
        finally:
            try:
                conn.close()
                conn.logout()
            except Exception:
                pass

    def fetch_message(self, uid: str, folder: str = "INBOX") -> Dict[str, Any]:
        """Fetch full message body by UID."""
        conn = self._imap_connect()
        try:
            conn.select(folder, readonly=True)
            _, data = conn.fetch(uid.encode(), "(RFC822)")
            if not data or data[0] is None:
                raise ValueError(f"Message {uid} not found")

            raw = data[0][1]
            msg = email_lib.message_from_bytes(raw)
            return self._parse_full_message(msg, uid)
        finally:
            try:
                conn.close()
                conn.logout()
            except Exception:
                pass

    # ─────────────────────────────────────────────────────────
    # Search
    # ─────────────────────────────────────────────────────────

    def search(
        self,
        imap_criteria: str,
        folder: str = "INBOX",
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search emails using pre-built IMAP criteria."""
        conn = self._imap_connect()
        try:
            conn.select(folder, readonly=True)
            _, data = conn.search(None, imap_criteria)
            uids = data[0].split()

            uids = uids[-limit:] if len(uids) > limit else uids
            uids.reverse()

            results = []
            for uid in uids:
                headers = self._fetch_headers(conn, uid)
                if headers:
                    results.append(headers)

            return results
        finally:
            try:
                conn.close()
                conn.logout()
            except Exception:
                pass

    # ─────────────────────────────────────────────────────────
    # IMAP helpers
    # ─────────────────────────────────────────────────────────

    def _imap_connect(self) -> imaplib.IMAP4_SSL:
        """Create an authenticated IMAP connection."""
        host = self._imap_config.get("host", "")
        port = self._imap_config.get("port", 993)

        if not host:
            raise ConnectionError("IMAP host not configured")

        try:
            conn = imaplib.IMAP4_SSL(host, port, timeout=30)
            conn.login(self._username, self._password)
            return conn
        except imaplib.IMAP4.error as e:
            raise ConnectionError(f"IMAP connection failed: {e}") from e

    def _fetch_headers(
        self, conn: imaplib.IMAP4_SSL, uid: bytes,
    ) -> Optional[Dict[str, Any]]:
        """Fetch headers + snippet for a single message UID."""
        try:
            # Fetch headers + first part for snippet
            _, data = conn.fetch(uid, "(BODY.PEEK[HEADER] BODY.PEEK[TEXT]<0.400>)")
            if not data or data[0] is None:
                return None

            # Parse the response — IMAP fetch returns tuples
            header_data = b""
            snippet_data = b""
            for part in data:
                if isinstance(part, tuple):
                    descriptor = part[0].decode("utf-8", errors="replace").upper()
                    if "HEADER" in descriptor:
                        header_data = part[1]
                    elif "TEXT" in descriptor:
                        snippet_data = part[1]

            if not header_data:
                return None

            msg = email_lib.message_from_bytes(header_data)

            # Decode subject
            subject = self._decode_header(msg.get("Subject", ""))
            from_addr = self._decode_header(msg.get("From", ""))
            to_addr = self._decode_header(msg.get("To", ""))
            date_str = msg.get("Date", "")
            message_id = msg.get("Message-ID", "")
            references = msg.get("References", "")
            in_reply_to = msg.get("In-Reply-To", "")

            # Derive thread_id from References or Message-ID
            thread_id = ""
            if references:
                # First reference is the thread root
                refs = references.strip().split()
                thread_id = refs[0] if refs else message_id
            elif in_reply_to:
                thread_id = in_reply_to
            else:
                thread_id = message_id

            # Parse date
            iso_date = ""
            if date_str:
                try:
                    parsed = email.utils.parsedate_to_datetime(date_str)
                    iso_date = parsed.isoformat()
                except Exception:
                    iso_date = date_str

            # Snippet from body
            snippet = ""
            if snippet_data:
                try:
                    snippet = snippet_data.decode("utf-8", errors="replace")
                    snippet = re.sub(r"\s+", " ", snippet).strip()[:200]
                except Exception:
                    snippet = ""

            return {
                "uid": uid.decode() if isinstance(uid, bytes) else str(uid),
                "message_id": message_id,
                "thread_id": thread_id,
                "from": from_addr,
                "to": to_addr,
                "subject": subject,
                "date": iso_date,
                "snippet": snippet,
                "has_attachments": False,  # Header-only — no attachment detection
            }
        except Exception as e:
            logger.debug("[EMAIL] Failed to fetch headers for %s: %s", uid, e)
            return None

    def _parse_full_message(
        self, msg: email_lib.message.Message, uid: str,
    ) -> Dict[str, Any]:
        """Parse a full RFC822 message into a structured dict."""
        subject = self._decode_header(msg.get("Subject", ""))
        from_addr = self._decode_header(msg.get("From", ""))
        to_addr = self._decode_header(msg.get("To", ""))
        date_str = msg.get("Date", "")
        message_id = msg.get("Message-ID", "")
        references = msg.get("References", "")
        in_reply_to = msg.get("In-Reply-To", "")

        thread_id = ""
        if references:
            refs = references.strip().split()
            thread_id = refs[0] if refs else message_id
        elif in_reply_to:
            thread_id = in_reply_to
        else:
            thread_id = message_id

        iso_date = ""
        if date_str:
            try:
                parsed = email.utils.parsedate_to_datetime(date_str)
                iso_date = parsed.isoformat()
            except Exception:
                iso_date = date_str

        # Extract body
        body = ""
        html_body = ""
        attachments = []

        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                disposition = str(part.get("Content-Disposition", ""))

                if "attachment" in disposition:
                    attachments.append({
                        "filename": part.get_filename() or "unknown",
                        "mime_type": content_type,
                    })
                elif content_type == "text/plain" and not body:
                    payload = part.get_payload(decode=True)
                    if payload:
                        body = payload.decode("utf-8", errors="replace")
                elif content_type == "text/html" and not html_body:
                    payload = part.get_payload(decode=True)
                    if payload:
                        html_body = payload.decode("utf-8", errors="replace")
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                body = payload.decode("utf-8", errors="replace")

        return {
            "uid": uid,
            "message_id": message_id,
            "thread_id": thread_id,
            "from": from_addr,
            "to": to_addr,
            "subject": subject,
            "date": iso_date,
            "body": body,
            "html_body": html_body,
            "attachments": attachments,
            "has_attachments": len(attachments) > 0,
        }

    @staticmethod
    def _decode_header(value: str) -> str:
        """Decode RFC2047 encoded header value."""
        if not value:
            return ""
        try:
            decoded_parts = email.header.decode_header(value)
            parts = []
            for part, charset in decoded_parts:
                if isinstance(part, bytes):
                    parts.append(part.decode(charset or "utf-8", errors="replace"))
                else:
                    parts.append(part)
            return " ".join(parts)
        except Exception:
            return value
