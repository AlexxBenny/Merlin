# interface/log_buffer.py

"""
LogBufferHandler — Ring-buffer log capture for MERLIN frontend.

Runs INSIDE the MERLIN process. Attaches to the root Python logger.
Captures log records into a thread-safe ring buffer (collections.deque).
The bridge export loop periodically writes the buffer to state/api/logs.json.

Architecture (3 decoupled layers):
    Python root logger → LogBufferHandler.emit() → RingBuffer (deque)
                                                        │
                                              bridge.py reads buffer
                                                        │
                                              writes state/api/logs.json

No async code. No event loop. No WebSocket.
The API server reads logs.json independently and streams to clients.
"""

import logging
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────
# Ring Buffer (thread-safe via deque atomic operations)
# ─────────────────────────────────────────────────────────────

class RingBuffer:
    """Fixed-size ring buffer for log entries.

    Uses collections.deque(maxlen=N) which provides thread-safe
    atomic append. No explicit locking needed for append/read.

    Position tracking via _write_count allows the bridge to
    detect new entries without re-reading the entire buffer.
    """

    def __init__(self, maxlen: int = 500) -> None:
        self._buffer: deque = deque(maxlen=maxlen)
        self._write_count: int = 0
        self._lock = threading.Lock()

    def append(self, entry: Dict[str, Any]) -> None:
        """Append a log entry. Thread-safe."""
        with self._lock:
            self._buffer.append(entry)
            self._write_count += 1

    def get_recent(
        self,
        n: int = 100,
        level_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return the N most recent entries, optionally filtered by level.

        Args:
            n: Maximum number of entries to return.
            level_filter: If provided, only return entries with this level
                         (case-insensitive: "DEBUG", "INFO", "WARNING", "ERROR").
        """
        with self._lock:
            entries = list(self._buffer)

        if level_filter:
            level_upper = level_filter.upper()
            entries = [e for e in entries if e.get("level") == level_upper]

        return entries[-n:]

    def get_all(self) -> List[Dict[str, Any]]:
        """Return all entries in the buffer."""
        with self._lock:
            return list(self._buffer)

    @property
    def write_count(self) -> int:
        """Total number of entries ever written (monotonic)."""
        return self._write_count

    @property
    def size(self) -> int:
        """Current number of entries in the buffer."""
        return len(self._buffer)


# ─────────────────────────────────────────────────────────────
# Log Handler (attaches to Python logging)
# ─────────────────────────────────────────────────────────────

class LogBufferHandler(logging.Handler):
    """Logging handler that captures records into a RingBuffer.

    Usage:
        buffer = RingBuffer(maxlen=500)
        handler = LogBufferHandler(buffer)
        logging.getLogger().addHandler(handler)

    Each log record is serialized to a dict with:
        timestamp, level, module, message, logger_name
    """

    def __init__(self, buffer: RingBuffer, level: int = logging.DEBUG) -> None:
        super().__init__(level)
        self._buffer = buffer

    def emit(self, record: logging.LogRecord) -> None:
        """Serialize and buffer a log record. Never raises."""
        try:
            entry = {
                "timestamp": record.created,
                "level": record.levelname,
                "module": record.module,
                "logger": record.name,
                "message": self.format(record) if self.formatter else record.getMessage(),
                "lineno": record.lineno,
            }
            self._buffer.append(entry)
        except Exception:
            # Never let logging handler crash the application
            pass

    @property
    def buffer(self) -> RingBuffer:
        """Access the underlying ring buffer."""
        return self._buffer


# ─────────────────────────────────────────────────────────────
# Module-level convenience
# ─────────────────────────────────────────────────────────────

# Singleton buffer for the MERLIN process
_global_buffer: Optional[RingBuffer] = None


def install_log_buffer(maxlen: int = 500) -> RingBuffer:
    """Install the log buffer handler on the root logger.

    Returns the RingBuffer instance for the bridge to read.
    Idempotent — safe to call multiple times.
    """
    global _global_buffer
    if _global_buffer is not None:
        return _global_buffer

    _global_buffer = RingBuffer(maxlen=maxlen)
    handler = LogBufferHandler(_global_buffer)
    handler.setLevel(logging.DEBUG)

    # Format: same as MERLIN's existing log format
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)

    logging.getLogger().addHandler(handler)
    return _global_buffer


def get_global_buffer() -> Optional[RingBuffer]:
    """Get the global log buffer (None if not installed)."""
    return _global_buffer
