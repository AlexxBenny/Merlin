# providers/whatsapp/rate_limiter.py

"""
WhatsAppRateLimiter — Token-bucket rate limiter for WhatsApp sends.

WhatsApp actively detects automated clients. This prevents account bans
by enforcing a maximum number of messages per time window.

Default: 10 messages per 60 seconds. Configurable via config/whatsapp.yaml.
Thread-safe (uses threading.Lock — same pattern as SessionManager).
"""

import threading
import time
from typing import Optional


class RateLimitExceeded(Exception):
    """Raised when the rate limit is exceeded."""

    def __init__(self, retry_after: float):
        self.retry_after = retry_after
        super().__init__(
            f"WhatsApp rate limit exceeded. "
            f"Try again in {retry_after:.1f} seconds."
        )


class WhatsAppRateLimiter:
    """Token-bucket rate limiter.

    Tracks message timestamps in a sliding window. On each acquire(),
    prunes expired timestamps and checks the count against the limit.

    Args:
        max_messages: Maximum messages allowed in the window.
        window_seconds: Size of the sliding window in seconds.
    """

    def __init__(
        self,
        max_messages: int = 10,
        window_seconds: int = 60,
    ):
        self._max_messages = max_messages
        self._window_seconds = window_seconds
        self._timestamps: list[float] = []
        self._lock = threading.Lock()

    def acquire(self) -> bool:
        """Try to acquire a send token.

        Returns True if allowed, False if rate exceeded.
        Does NOT block.
        """
        with self._lock:
            now = time.monotonic()
            self._prune(now)
            if len(self._timestamps) >= self._max_messages:
                return False
            self._timestamps.append(now)
            return True

    def acquire_or_raise(self) -> None:
        """Acquire a token or raise RateLimitExceeded."""
        if not self.acquire():
            retry_after = self.seconds_until_available()
            raise RateLimitExceeded(retry_after)

    def seconds_until_available(self) -> float:
        """Seconds until the next token becomes available."""
        with self._lock:
            now = time.monotonic()
            self._prune(now)
            if len(self._timestamps) < self._max_messages:
                return 0.0
            # Oldest timestamp in the window — when it expires, a slot opens
            oldest = self._timestamps[0]
            return max(0.0, (oldest + self._window_seconds) - now)

    @property
    def remaining(self) -> int:
        """Number of remaining tokens in the current window."""
        with self._lock:
            self._prune(time.monotonic())
            return max(0, self._max_messages - len(self._timestamps))

    def _prune(self, now: float) -> None:
        """Remove expired timestamps from the window."""
        cutoff = now - self._window_seconds
        self._timestamps = [
            ts for ts in self._timestamps if ts > cutoff
        ]
