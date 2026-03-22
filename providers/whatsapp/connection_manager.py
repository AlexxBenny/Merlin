# providers/whatsapp/connection_manager.py

"""
WhatsAppConnectionManager — Lifecycle management for neonize client.

NOT SessionManager (which tracks UI environment handles like app windows).
This manages a persistent WebSocket connection to WhatsApp servers.

Responsibilities:
- start(): Creates neonize client, runs connect() in daemon thread
- shutdown(): Graceful disconnect + thread join
- get_client(): Returns connected client (raises if not connected)
- Reconnect: On disconnect event, exponential backoff retry (max 5 attempts)
- QR code: On first run, logs QR code for user to scan

Thread-safe state via threading.Lock (same pattern as SessionManager).
"""

import logging
import threading
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Neonize imports are deferred to start() to avoid import errors
# when neonize is not installed (whatsapp.enabled: false).


class WhatsAppConnectionManager:
    """Manages neonize client lifecycle in a background thread.

    Args:
        session_name: Name for the WhatsApp session (used for SQLite DB).
        database_path: Path to the neonize SQLite database.
    """

    def __init__(
        self,
        session_name: str = "merlin_whatsapp",
        database_path: str = "state/whatsapp/neonize.db",
    ):
        self._session_name = session_name
        self._database_path = database_path
        self._client = None           # neonize.NewClient instance
        self._thread: Optional[threading.Thread] = None
        self._connected = False
        self._lock = threading.Lock()
        self._reconnect_count = 0
        self._max_reconnects = 5
        self._shutdown_event = threading.Event()

    @property
    def is_connected(self) -> bool:
        """Thread-safe connection status."""
        with self._lock:
            return self._connected

    def get_client(self):
        """Get the connected neonize client.

        Raises:
            RuntimeError: If not connected.
        """
        with self._lock:
            if not self._connected or self._client is None:
                raise RuntimeError(
                    "WhatsApp is not connected. "
                    "Check the connection status in the dashboard."
                )
            return self._client

    def start(self) -> None:
        """Start the neonize client in a daemon thread.

        Creates the database directory, initializes the client,
        registers event handlers, and starts the connection thread.
        """
        try:
            from neonize.client import NewClient
            from neonize.events import (
                ConnectedEv,
                DisconnectedEv,
                QREv,
            )
        except ImportError:
            logger.error(
                "[WHATSAPP] neonize not installed. "
                "Run: pip install neonize"
            )
            raise

        # Ensure database directory exists
        db_path = Path(self._database_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # Create client — name param IS the SQLite file path
        self._client = NewClient(str(db_path))

        # Register event handlers
        @self._client.event(ConnectedEv)
        def on_connected(_client, _event):
            with self._lock:
                self._connected = True
                self._reconnect_count = 0
            logger.info(
                "[WHATSAPP] Connected successfully (session=%s)",
                self._session_name,
            )

        @self._client.event(DisconnectedEv)
        def on_disconnected(_client, _event):
            with self._lock:
                self._connected = False
            logger.warning("[WHATSAPP] Disconnected")
            if not self._shutdown_event.is_set():
                self._attempt_reconnect()

        @self._client.event(QREv)
        def on_qr(_client, event):
            # Log QR code data for terminal scanning.
            # In future, this can be displayed in the dashboard.
            logger.info(
                "[WHATSAPP] QR Code received. Scan this with your "
                "WhatsApp mobile app to connect."
            )
            try:
                # neonize provides QR as bytes or string
                qr_data = getattr(event, "QR", None) or str(event)
                # Try to generate terminal-friendly QR
                try:
                    import segno
                    qr = segno.make(qr_data)
                    qr.terminal(compact=True)
                except ImportError:
                    logger.info(
                        "[WHATSAPP] QR data: %s", str(qr_data)[:200],
                    )
            except Exception as e:
                logger.warning(
                    "[WHATSAPP] Could not display QR: %s", e,
                )

        # Start connection in daemon thread
        self._thread = threading.Thread(
            target=self._run_client,
            name="whatsapp-neonize",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "[WHATSAPP] Connection thread started (session=%s, db=%s)",
            self._session_name, self._database_path,
        )

    def _run_client(self) -> None:
        """Run the neonize client's blocking connect loop.

        This runs in a daemon thread and blocks until disconnect.
        """
        try:
            self._client.connect()
        except Exception as e:
            logger.error(
                "[WHATSAPP] Client connection failed: %s", e,
                exc_info=True,
            )
            with self._lock:
                self._connected = False

    def _attempt_reconnect(self) -> None:
        """Attempt reconnection with exponential backoff."""
        with self._lock:
            if self._reconnect_count >= self._max_reconnects:
                logger.error(
                    "[WHATSAPP] Max reconnect attempts (%d) exceeded. "
                    "Manual restart required.",
                    self._max_reconnects,
                )
                return
            self._reconnect_count += 1
            attempt = self._reconnect_count

        backoff = min(2 ** attempt, 60)  # Cap at 60 seconds
        logger.info(
            "[WHATSAPP] Reconnect attempt %d/%d in %ds...",
            attempt, self._max_reconnects, backoff,
        )

        if self._shutdown_event.wait(timeout=backoff):
            return  # Shutdown requested during wait

        try:
            self._thread = threading.Thread(
                target=self._run_client,
                name=f"whatsapp-neonize-reconnect-{attempt}",
                daemon=True,
            )
            self._thread.start()
        except Exception as e:
            logger.error(
                "[WHATSAPP] Reconnect attempt %d failed: %s",
                attempt, e,
            )

    def shutdown(self) -> None:
        """Gracefully disconnect and stop the background thread."""
        logger.info("[WHATSAPP] Shutting down...")
        self._shutdown_event.set()

        with self._lock:
            self._connected = False

        if self._client:
            try:
                self._client.disconnect()
            except Exception as e:
                logger.warning(
                    "[WHATSAPP] Error during disconnect: %s", e,
                )

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

        logger.info("[WHATSAPP] Shutdown complete")
