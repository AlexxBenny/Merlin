# interface/telegram_bot.py

"""
TelegramAdapter — Telegram input adapter for MERLIN.

Runs as a SEPARATE PROCESS (no MERLIN core imports).
Communicates via the same file-based IPC as api_server.py:
    - Writes command JSON to state/api/command_queue/
    - Polls response JSON from state/api/responses/

Telegram messages go through the identical pipeline as UI messages:
    Telegram → command_queue → Bridge → handle_percept → response

Safeguards:
    - Whitelist-only access (allowed_user_ids)
    - asyncio.Lock for message serialization
    - Queue pressure guard (max_queue_depth)
    - Bridge liveness check (system.json freshness)
    - Response truncation (Telegram 4096 char limit)
    - Structured [TELEGRAM] logging
"""

import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Set

logger = logging.getLogger(__name__)

# Telegram message length limit
_TELEGRAM_MAX_LEN = 4000  # leave margin under 4096 for safety

# ─────────────────────────────────────────────────────────────
# Paths (relative to MERLIN root)
# ─────────────────────────────────────────────────────────────

_BASE_PATH = Path(__file__).resolve().parent.parent
_STATE_DIR = _BASE_PATH / "state" / "api"
_COMMAND_DIR = _STATE_DIR / "command_queue"
_RESPONSE_DIR = _STATE_DIR / "responses"

# Ensure directories exist
for _d in [_STATE_DIR, _COMMAND_DIR, _RESPONSE_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# TelegramAdapter
# ─────────────────────────────────────────────────────────────

class TelegramAdapter:
    """Telegram input adapter for MERLIN.

    Uses python-telegram-bot v21.x (fully async).
    Reuses file-based IPC via interface.ipc module.
    """

    def __init__(
        self,
        token: str,
        allowed_user_ids: Set[int],
        max_queue_depth: int = 5,
        response_timeout: float = 120.0,
    ) -> None:
        self._token = token
        self._allowed_users = allowed_user_ids
        self._max_queue = max_queue_depth
        self._timeout = response_timeout
        self._lock = asyncio.Lock()

    async def _handle_start(self, update, context) -> None:
        """Handle /start command."""
        from telegram import Update

        user = update.effective_user
        if not self._is_allowed(user.id):
            await update.message.reply_text("⛔ Unauthorized.")
            logger.warning(
                "[TELEGRAM] Rejected /start from user=%d (%s)",
                user.id, user.username or "?",
            )
            return

        await update.message.reply_text(
            "🔵 MERLIN connected.\n\n"
            "Send any message to interact with MERLIN.\n"
            "Your messages are routed through the same pipeline as the UI."
        )
        logger.info(
            "[TELEGRAM] /start from user=%d (%s)",
            user.id, user.username or "?",
        )

    async def _handle_message(self, update, context) -> None:
        """Handle incoming text messages."""
        from interface.ipc import (
            submit_command, wait_for_response,
            queue_depth, is_bridge_alive,
        )

        user = update.effective_user
        text = update.message.text

        if not text:
            return

        # ── Security: whitelist check ──
        if not self._is_allowed(user.id):
            await update.message.reply_text("⛔ Unauthorized.")
            logger.warning(
                "[TELEGRAM] Rejected message from user=%d (%s): \"%s\"",
                user.id, user.username or "?", text[:50],
            )
            return

        # ── Serialize: one message at a time ──
        async with self._lock:
            start_time = time.time()

            # ── Bridge liveness check ──
            if not is_bridge_alive(_STATE_DIR):
                await update.message.reply_text(
                    "⚠️ MERLIN is not running. Start with:\n"
                    "`python main.py --telegram`",
                    parse_mode="Markdown",
                )
                logger.warning(
                    "[TELEGRAM] Bridge not alive, rejecting user=%d cmd=\"%s\"",
                    user.id, text[:50],
                )
                return

            # ── Queue pressure check ──
            depth = queue_depth(_COMMAND_DIR)
            if depth >= self._max_queue:
                await update.message.reply_text(
                    "⏳ System busy. Try again in a moment."
                )
                logger.warning(
                    "[TELEGRAM] Queue full (%d/%d), rejecting user=%d cmd=\"%s\"",
                    depth, self._max_queue, user.id, text[:50],
                )
                return

            # ── Submit command (enriched payload) ──
            payload = {
                "message": text,
                "source": "telegram",
                "user_id": user.id,
                "username": user.username or "",
            }
            cmd_id = submit_command("chat", payload, _COMMAND_DIR)

            logger.info(
                "[TELEGRAM] user=%d cmd_id=%s \"%s\"",
                user.id, cmd_id, text[:80],
            )

            # ── Wait for response ──
            result = await wait_for_response(
                cmd_id, _RESPONSE_DIR,
                timeout=self._timeout,
            )

            latency = time.time() - start_time
            status = result.get("status", "unknown")
            response = result.get("response", "No response.")

            logger.info(
                "[TELEGRAM] cmd_id=%s status=%s latency=%.1fs",
                cmd_id, status, latency,
            )

            # ── Truncate for Telegram ──
            if len(response) > _TELEGRAM_MAX_LEN:
                response = response[:_TELEGRAM_MAX_LEN] + "\n\n[truncated]"

            # ── Send response ──
            await update.message.reply_text(response)

    def _is_allowed(self, user_id: int) -> bool:
        """Check if a Telegram user ID is in the whitelist."""
        if not self._allowed_users:
            return False  # empty whitelist = reject everyone
        return user_id in self._allowed_users

    def run(self) -> None:
        """Start the Telegram bot polling loop."""
        try:
            from telegram.ext import (
                ApplicationBuilder, CommandHandler, MessageHandler, filters,
            )
        except ImportError:
            raise ImportError(
                "python-telegram-bot is required for the Telegram adapter. "
                "Install with: pip install python-telegram-bot"
            )

        app = ApplicationBuilder().token(self._token).build()
        app.add_handler(CommandHandler("start", self._handle_start))
        app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message)
        )

        logger.info("[TELEGRAM] Bot starting (allowed_users=%s)", self._allowed_users)
        print("=" * 50)
        print("  MERLIN Telegram Bot")
        print(f"  Allowed users: {self._allowed_users}")
        print("=" * 50)

        app.run_polling(drop_pending_updates=True)


# ─────────────────────────────────────────────────────────────
# Module entrypoint
# ─────────────────────────────────────────────────────────────

def main() -> None:
    """Run the Telegram adapter as a standalone process."""
    import yaml
    from dotenv import load_dotenv

    # Load environment
    env_path = _BASE_PATH / ".env"
    if env_path.exists():
        load_dotenv(str(env_path))

    # Load config
    config_path = _BASE_PATH / "config" / "telegram.yaml"
    if not config_path.exists():
        print(f"ERROR: {config_path} not found.")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    telegram_config = config.get("telegram", {})

    if not telegram_config.get("enabled", False):
        print("ERROR: Telegram is disabled in config/telegram.yaml")
        sys.exit(1)

    # Read token from env
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    if not token:
        print("ERROR: TELEGRAM_BOT_TOKEN not set in .env")
        sys.exit(1)

    # Read whitelist
    allowed_ids = telegram_config.get("allowed_user_ids", [])
    if not allowed_ids:
        print(
            "ERROR: allowed_user_ids is empty in config/telegram.yaml. "
            "MERLIN has system-level execution — open access is a vulnerability. "
            "Add your Telegram user ID (message @userinfobot on Telegram to find it)."
        )
        sys.exit(1)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )

    # Suppress noisy third-party loggers (httpx polls every ~10s,
    # telegram.ext logs application lifecycle). Only [TELEGRAM] logs show.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("telegram").setLevel(logging.WARNING)
    logging.getLogger("telegram.ext").setLevel(logging.WARNING)

    adapter = TelegramAdapter(
        token=token,
        allowed_user_ids=set(int(uid) for uid in allowed_ids),
        max_queue_depth=telegram_config.get("max_queue_depth", 5),
        response_timeout=telegram_config.get("response_timeout", 120),
    )
    adapter.run()


if __name__ == "__main__":
    main()
