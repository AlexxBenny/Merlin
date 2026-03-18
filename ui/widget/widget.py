# ui/widget/widget.py

"""
MERLIN Desktop Widget — Floating assistant orb with chat panel.

PySide6-based frameless, always-on-top circular orb that:
- Sits in the top-right corner of the primary screen
- Idle: dark translucent circle with MERLIN icon text
- Processing: cyan glow pulse animation
- Disconnected: grey appearance
- Click: expands to ~300×400 chat panel with SSE streaming
- Escape/click outside: collapses back to orb

Communicates exclusively through the API server (http://localhost:8420).
No MERLIN core imports.
"""

import json
import logging
import sys
import time
import threading
from typing import Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Check for PySide6
# ─────────────────────────────────────────────────────────────

try:
    from PySide6.QtWidgets import (
        QApplication, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QLineEdit, QPushButton, QScrollArea,
        QGraphicsDropShadowEffect, QSizePolicy,
    )
    from PySide6.QtCore import (
        Qt, QTimer, QPropertyAnimation, QEasingCurve,
        QPoint, QSize, Signal, QThread, QUrl, QEvent,
    )
    from PySide6.QtGui import (
        QColor, QPainter, QBrush, QPen, QFont, QFontMetrics,
        QLinearGradient, QRadialGradient, QIcon, QCursor,
    )
    from PySide6.QtNetwork import QNetworkAccessManager, QNetworkRequest, QNetworkReply
    HAS_PYSIDE6 = True
except ImportError:
    HAS_PYSIDE6 = False
    print("PySide6 not installed. Widget unavailable.")
    print("Install with: pip install PySide6")
    sys.exit(1)

try:
    import requests as _requests_mod
    HAS_REQUESTS = True
except ImportError:
    _requests_mod = None  # type: ignore[assignment]
    HAS_REQUESTS = False


# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

API_BASE = "http://localhost:8420"
HEALTH_INTERVAL_MS = 5000
ORB_SIZE = 60
PANEL_WIDTH = 320
PANEL_HEIGHT = 440

# Colors
COLOR_BG_DARK = QColor(18, 18, 24)
COLOR_BG_PANEL = QColor(24, 24, 32, 240)
COLOR_ACCENT = QColor(0, 200, 255)       # Cyan
COLOR_ACCENT_DIM = QColor(0, 120, 180)
COLOR_GREY = QColor(80, 80, 90)
COLOR_TEXT = QColor(220, 220, 230)
COLOR_TEXT_DIM = QColor(140, 140, 155)
COLOR_USER_BUBBLE = QColor(45, 45, 60)
COLOR_BOT_BUBBLE = QColor(30, 80, 100)
COLOR_INPUT_BG = QColor(35, 35, 48)
COLOR_HOVER = QColor(0, 200, 255, 30)


# ─────────────────────────────────────────────────────────────
# Chat Worker (background thread for API calls)
# ─────────────────────────────────────────────────────────────

class ChatWorker(QThread):
    """Background thread for chat API calls."""
    response_ready = Signal(str)
    error_occurred = Signal(str)

    def __init__(self, message: str, parent=None):
        super().__init__(parent)
        self.message = message

    def run(self):
        try:
            if _requests_mod is None:
                self.error_occurred.emit("requests library not installed")
                return

            resp = _requests_mod.post(
                f"{API_BASE}/api/v1/chat",
                json={"message": self.message},
                timeout=120,
            )
            if resp.status_code == 200:
                data = resp.json()
                self.response_ready.emit(data.get("response", "No response."))
            else:
                self.error_occurred.emit(f"API error: {resp.status_code}")
        except Exception as e:
            if 'ConnectionError' in type(e).__name__:
                self.error_occurred.emit("Cannot connect to MERLIN API.")
            else:
                self.error_occurred.emit(str(e))


# ─────────────────────────────────────────────────────────────
# Chat Bubble Widget
# ─────────────────────────────────────────────────────────────

class ChatBubble(QWidget):
    """Individual chat message bubble."""

    def __init__(self, text: str, is_user: bool, parent=None):
        super().__init__(parent)
        self.text = text
        self.is_user = is_user
        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)

        label = QLabel(self.text)
        label.setWordWrap(True)
        label.setFont(QFont("Segoe UI", 9))
        label.setStyleSheet(f"""
            QLabel {{
                background-color: {COLOR_USER_BUBBLE.name() if self.is_user else COLOR_BOT_BUBBLE.name()};
                color: {COLOR_TEXT.name()};
                border-radius: 10px;
                padding: 8px 12px;
            }}
        """)
        label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        label.setMaximumWidth(PANEL_WIDTH - 60)

        if self.is_user:
            layout.addStretch()
            layout.addWidget(label)
        else:
            layout.addWidget(label)
            layout.addStretch()


# ─────────────────────────────────────────────────────────────
# MERLIN Orb Widget
# ─────────────────────────────────────────────────────────────

class MerlinOrb(QWidget):
    """Floating MERLIN assistant orb with expandable chat panel."""

    def __init__(self):
        super().__init__()
        self._expanded = False
        self._connected = False
        self._processing = False
        self._glow_opacity = 0.0
        self._chat_worker: Optional[ChatWorker] = None

        self._setup_window()
        self._setup_orb_ui()
        self._setup_panel_ui()
        self._setup_health_timer()
        self._position_window()

    # ─── Window setup ───────────────────────────────────────

    def _setup_window(self):
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool  # Don't show in taskbar
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(ORB_SIZE, ORB_SIZE)

    def _position_window(self):
        """Position in top-right corner of primary screen."""
        screen = QApplication.primaryScreen()
        if screen:
            geo = screen.availableGeometry()
            x = geo.right() - ORB_SIZE - 20
            y = geo.top() + 40
            self.move(x, y)

    # ─── Orb UI ─────────────────────────────────────────────

    def _setup_orb_ui(self):
        """Set up the circular orb."""
        # Glow animation
        self._glow_timer = QTimer(self)
        self._glow_timer.timeout.connect(self._animate_glow)
        self._glow_phase = 0.0

    def paintEvent(self, event):
        """Custom paint for the orb."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if self._expanded:
            # Draw panel background
            painter.setBrush(QBrush(COLOR_BG_PANEL))
            painter.setPen(QPen(COLOR_ACCENT_DIM if self._connected else COLOR_GREY, 1))
            painter.drawRoundedRect(0, 0, self.width(), self.height(), 16, 16)
            return

        # Draw orb
        center_x = self.width() // 2
        center_y = self.height() // 2
        radius = ORB_SIZE // 2 - 2

        # Glow effect (when processing)
        if self._processing and self._connected:
            glow_color = QColor(COLOR_ACCENT)
            glow_color.setAlphaF(self._glow_opacity * 0.3)
            glow_radius = radius + 8
            gradient = QRadialGradient(center_x, center_y, glow_radius)
            gradient.setColorAt(0, glow_color)
            gradient.setColorAt(1, QColor(0, 0, 0, 0))
            painter.setBrush(QBrush(gradient))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(
                center_x - glow_radius, center_y - glow_radius,
                glow_radius * 2, glow_radius * 2,
            )

        # Main circle
        if self._connected:
            bg = QColor(COLOR_BG_DARK)
            border = COLOR_ACCENT if self._processing else COLOR_ACCENT_DIM
        else:
            bg = QColor(50, 50, 55)
            border = COLOR_GREY

        painter.setBrush(QBrush(bg))
        painter.setPen(QPen(border, 2))
        painter.drawEllipse(center_x - radius, center_y - radius, radius * 2, radius * 2)

        # Text
        painter.setPen(QPen(COLOR_ACCENT if self._connected else COLOR_GREY))
        font = QFont("Segoe UI", 10, QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "M")

    def _animate_glow(self):
        """Pulse glow animation."""
        import math
        self._glow_phase += 0.1
        self._glow_opacity = (math.sin(self._glow_phase) + 1) / 2
        self.update()

    # ─── Panel UI ───────────────────────────────────────────

    def _setup_panel_ui(self):
        """Set up the expandable chat panel (hidden initially)."""
        self._panel_widget = QWidget(self)
        self._panel_widget.hide()

        layout = QVBoxLayout(self._panel_widget)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # Header
        header = QHBoxLayout()
        title = QLabel("MERLIN")
        title.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {COLOR_ACCENT.name()};")
        header.addWidget(title)

        self._status_label = QLabel("●")
        self._status_label.setFont(QFont("Segoe UI", 8))
        self._status_label.setStyleSheet(f"color: {COLOR_GREY.name()};")
        header.addStretch()
        header.addWidget(self._status_label)

        close_btn = QPushButton("✕")
        close_btn.setFixedSize(24, 24)
        close_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                color: {COLOR_TEXT_DIM.name()};
                border: none;
                font-size: 14px;
            }}
            QPushButton:hover {{
                color: {COLOR_TEXT.name()};
            }}
        """)
        close_btn.clicked.connect(self._collapse)
        header.addWidget(close_btn)

        layout.addLayout(header)

        # Chat area (scrollable)
        self._scroll_area = QScrollArea()
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setStyleSheet("""
            QScrollArea {
                background: transparent;
                border: none;
            }
            QScrollBar:vertical {
                width: 6px;
                background: transparent;
            }
            QScrollBar::handle:vertical {
                background: rgba(100, 100, 120, 80);
                border-radius: 3px;
            }
        """)

        self._chat_container = QWidget()
        self._chat_layout = QVBoxLayout(self._chat_container)
        self._chat_layout.setContentsMargins(0, 0, 0, 0)
        self._chat_layout.setSpacing(4)
        self._chat_layout.addStretch()

        self._scroll_area.setWidget(self._chat_container)
        layout.addWidget(self._scroll_area)

        # Input area
        input_layout = QHBoxLayout()
        input_layout.setSpacing(6)

        self._input_field = QLineEdit()
        self._input_field.setPlaceholderText("Ask MERLIN...")
        self._input_field.setFont(QFont("Segoe UI", 10))
        self._input_field.setStyleSheet(f"""
            QLineEdit {{
                background-color: {COLOR_INPUT_BG.name()};
                color: {COLOR_TEXT.name()};
                border: 1px solid {COLOR_ACCENT_DIM.name()};
                border-radius: 8px;
                padding: 8px 12px;
            }}
            QLineEdit:focus {{
                border-color: {COLOR_ACCENT.name()};
            }}
        """)
        self._input_field.returnPressed.connect(self._send_message)
        input_layout.addWidget(self._input_field)

        send_btn = QPushButton("→")
        send_btn.setFixedSize(36, 36)
        send_btn.setFont(QFont("Segoe UI", 14))
        send_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLOR_ACCENT_DIM.name()};
                color: {COLOR_BG_DARK.name()};
                border: none;
                border-radius: 8px;
            }}
            QPushButton:hover {{
                background-color: {COLOR_ACCENT.name()};
            }}
        """)
        send_btn.clicked.connect(self._send_message)
        input_layout.addWidget(send_btn)

        layout.addLayout(input_layout)

    # ─── Expand / Collapse ──────────────────────────────────

    def _expand(self):
        """Expand from orb to chat panel."""
        if self._expanded:
            return

        self._expanded = True
        screen = QApplication.primaryScreen()
        if screen:
            geo = screen.availableGeometry()
            x = geo.right() - PANEL_WIDTH - 20
            y = geo.top() + 40
            self.move(x, y)

        self.setFixedSize(PANEL_WIDTH, PANEL_HEIGHT)
        self._panel_widget.setGeometry(0, 0, PANEL_WIDTH, PANEL_HEIGHT)
        self._panel_widget.show()
        self._input_field.setFocus()
        # Install global event filter to detect clicks outside
        QApplication.instance().installEventFilter(self)
        self.update()

    def _collapse(self):
        """Collapse from chat panel to orb."""
        if not self._expanded:
            return

        self._expanded = False
        self._panel_widget.hide()
        # Remove global event filter
        app = QApplication.instance()
        if app:
            app.removeEventFilter(self)
        self.setFixedSize(ORB_SIZE, ORB_SIZE)
        self._position_window()
        self.update()

    # ─── Mouse events (drag-to-move + click-to-expand) ─────

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start_pos = event.globalPosition().toPoint()
            self._drag_origin = self.pos()
            self._dragging = False
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if hasattr(self, '_drag_start_pos') and self._drag_start_pos is not None:
            delta = event.globalPosition().toPoint() - self._drag_start_pos
            # Start dragging only after moving > 5px (avoids accidental drags)
            if not self._dragging and (abs(delta.x()) > 5 or abs(delta.y()) > 5):
                self._dragging = True
            if self._dragging:
                self.move(self._drag_origin + delta)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if not self._dragging and not self._expanded:
                self._expand()
            self._drag_start_pos = None
            self._dragging = False
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self._collapse()
        super().keyPressEvent(event)

    def eventFilter(self, obj, event):
        """Catch global mouse clicks — collapse if click is outside this widget."""
        if (
            self._expanded
            and event.type() == QEvent.Type.MouseButtonPress
            and not self.geometry().contains(QCursor.pos())
        ):
            self._collapse()
        return super().eventFilter(obj, event)

    # ─── Chat ───────────────────────────────────────────────

    def _send_message(self):
        """Send a chat message to MERLIN."""
        text = self._input_field.text().strip()
        if not text:
            return

        self._input_field.clear()
        self._add_bubble(text, is_user=True)

        # Start processing
        self._processing = True
        self._glow_timer.start(50)
        self.update()

        # Send via background thread
        self._chat_worker = ChatWorker(text)
        self._chat_worker.response_ready.connect(self._on_response)
        self._chat_worker.error_occurred.connect(self._on_error)
        self._chat_worker.start()

    def _on_response(self, response: str):
        """Handle chat response."""
        self._processing = False
        self._glow_timer.stop()
        self._glow_opacity = 0.0
        self.update()
        self._add_bubble(response, is_user=False)

    def _on_error(self, error: str):
        """Handle chat error."""
        self._processing = False
        self._glow_timer.stop()
        self._glow_opacity = 0.0
        self.update()
        self._add_bubble(f"⚠ {error}", is_user=False)

    def _add_bubble(self, text: str, is_user: bool):
        """Add a chat bubble to the panel."""
        bubble = ChatBubble(text, is_user)
        # Insert before the stretch at the end
        self._chat_layout.insertWidget(
            self._chat_layout.count() - 1, bubble,
        )
        # Auto-scroll to bottom
        QTimer.singleShot(50, lambda: self._scroll_area.verticalScrollBar().setValue(
            self._scroll_area.verticalScrollBar().maximum()
        ))

    # ─── Health heartbeat ───────────────────────────────────

    def _setup_health_timer(self):
        """Poll API health every 5 seconds."""
        self._health_timer = QTimer(self)
        self._health_timer.timeout.connect(self._check_health)
        self._health_timer.start(HEALTH_INTERVAL_MS)
        # Initial check
        QTimer.singleShot(500, self._check_health)

    def _check_health(self):
        """Check if MERLIN API is reachable."""
        def _do_check():
            try:
                if HAS_REQUESTS and _requests_mod is not None:
                    resp = _requests_mod.get(
                        f"{API_BASE}/api/v1/health",
                        timeout=3,
                    )
                    self._connected = resp.status_code == 200
                else:
                    self._connected = False
            except Exception:
                self._connected = False

            # Update status indicator (must be on main thread)
            if hasattr(self, '_status_label'):
                color = COLOR_ACCENT.name() if self._connected else COLOR_GREY.name()
                self._status_label.setStyleSheet(f"color: {color};")
            self.update()

        # Run in background to avoid blocking UI
        thread = threading.Thread(target=_do_check, daemon=True)
        thread.start()


# ─────────────────────────────────────────────────────────────
# Module entrypoint
# ─────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("MERLIN Widget")

    orb = MerlinOrb()
    orb.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
