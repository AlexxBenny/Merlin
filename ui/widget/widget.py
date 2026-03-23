"""
MERLIN Desktop Widget — Floating assistant orb with chat panel.

PySide6-based frameless, always-on-top circular orb that:
- Sits in the top-right corner of the primary screen
- Idle:       spinning conic-gradient ring around dark core with "M" letter
- Processing: particles orbit + fast ring spin + cyan inner pulse
- Disconnected: grey muted appearance
- Click: expands to ~300×420 chat panel with typing indicator & message bubbles
- Escape/click outside: collapses back to orb

Communicates exclusively through the API server (http://localhost:8420).
No MERLIN core imports.
"""

import json
import logging
import math
import sys
import time
import threading
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from PySide6.QtWidgets import (
        QApplication, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QLineEdit, QPushButton, QScrollArea,
        QSizePolicy, QGraphicsDropShadowEffect,
    )
    from PySide6.QtCore import (
        Qt, QTimer, QPoint, QPointF, QRectF, QSizeF,
        Signal, QThread, QEvent, QPropertyAnimation,
        QEasingCurve,
    )
    from PySide6.QtGui import (
        QColor, QPainter, QBrush, QPen, QFont, QFontMetrics,
        QLinearGradient, QRadialGradient, QConicalGradient,
        QIcon, QCursor, QPainterPath, QPixmap,
    )
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

API_BASE          = "http://localhost:8420"
HEALTH_INTERVAL   = 5000   # ms
ORB_SIZE          = 80     # slightly bigger for the ring
PANEL_WIDTH       = 320
PANEL_HEIGHT      = 460
ANIM_INTERVAL     = 20     # ms  → ~50 fps

# ── Palette ────────────────────────────────────────────────
C_BG_CORE      = QColor(13,  13,  20)
C_BG_PANEL     = QColor(16,  16,  26,  245)
C_ACCENT       = QColor(0,   200, 255)
C_ACCENT_DIM   = QColor(0,   110, 190)
C_ACCENT_GLOW  = QColor(0,   180, 255, 60)
C_GREY         = QColor(70,  70,  82)
C_RING_1       = QColor(0,   200, 255)
C_RING_2       = QColor(0,    80, 180)
C_TEXT         = QColor(215, 228, 245)
C_TEXT_DIM     = QColor(130, 150, 175)
C_USER_BUBBLE  = QColor(0,   90,  160, 60)
C_BOT_BUBBLE   = QColor(255, 255, 255, 10)
C_INPUT_BG     = QColor(28,  28,  42)
C_BORDER_SOFT  = QColor(0,   180, 255, 40)
C_SEPARATOR    = QColor(255, 255, 255, 13)


# ─────────────────────────────────────────────────────────────
# Chat Worker
# ─────────────────────────────────────────────────────────────

class ChatWorker(QThread):
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
                self.error_occurred.emit(f"API error {resp.status_code}")
        except Exception as e:
            if "ConnectionError" in type(e).__name__:
                self.error_occurred.emit("Cannot connect to MERLIN API.")
            else:
                self.error_occurred.emit(str(e))


class MicWorker(QThread):
    """Records mic audio with VAD, uploads to STT endpoint.

    States: recording → uploading → done/error
    Stops on: manual stop(), 1.2s silence, or 10s max duration.
    """

    transcription_ready = Signal(str)   # emitted with transcribed text
    error_occurred = Signal(str)        # emitted with error message
    status_changed = Signal(str)        # "listening" | "processing"

    # Recording params
    SAMPLE_RATE = 16000
    CHANNELS = 1
    BLOCK_SIZE = 480          # 30ms frames for VAD
    SILENCE_LIMIT = 1.2       # seconds of silence before auto-stop
    MAX_DURATION = 10.0       # seconds max recording

    def __init__(self, parent=None):
        super().__init__(parent)
        self._stop_flag = False

    def stop_recording(self):
        """Signal the worker to stop recording (button re-click)."""
        self._stop_flag = True

    def run(self):
        self.status_changed.emit("listening")
        audio_data = self._record()
        if audio_data is None:
            return  # error already emitted

        if len(audio_data) < self.SAMPLE_RATE * 0.3:  # < 300ms
            self.error_occurred.emit("Recording too short")
            return

        self.status_changed.emit("processing")
        self._upload_and_transcribe(audio_data)

    def _record(self):
        """Record mic audio with silence detection."""
        try:
            import sounddevice as sd
        except ImportError:
            self.error_occurred.emit("sounddevice not installed")
            return None

        frames = []
        silence_frames = 0
        silence_threshold = int(self.SILENCE_LIMIT * self.SAMPLE_RATE / self.BLOCK_SIZE)
        max_frames = int(self.MAX_DURATION * self.SAMPLE_RATE / self.BLOCK_SIZE)
        frame_count = 0

        # Energy-based silence detection (simpler, no webrtcvad dep)
        ENERGY_THRESHOLD = 0.01

        try:
            with sd.InputStream(
                samplerate=self.SAMPLE_RATE,
                channels=self.CHANNELS,
                dtype='float32',
                blocksize=self.BLOCK_SIZE,
            ) as stream:
                while not self._stop_flag and frame_count < max_frames:
                    data, overflowed = stream.read(self.BLOCK_SIZE)
                    frames.append(data.copy())
                    frame_count += 1

                    # Energy check
                    import numpy as np
                    energy = np.sqrt(np.mean(data ** 2))
                    if energy < ENERGY_THRESHOLD:
                        silence_frames += 1
                    else:
                        silence_frames = 0

                    # Auto-stop after sustained silence
                    if silence_frames >= silence_threshold and frame_count > 10:
                        logger.debug("MicWorker: auto-stop after %.1fs silence",
                                     self.SILENCE_LIMIT)
                        break

        except sd.PortAudioError as e:
            err = str(e)
            if "permission" in err.lower() or "denied" in err.lower():
                self.error_occurred.emit("Microphone permission denied")
            elif "not found" in err.lower() or "no device" in err.lower():
                self.error_occurred.emit("No microphone device found")
            else:
                self.error_occurred.emit(f"Mic error: {err}")
            return None
        except Exception as e:
            self.error_occurred.emit(f"Recording error: {e}")
            return None

        if not frames:
            self.error_occurred.emit("No audio captured")
            return None

        import numpy as np
        audio = np.concatenate(frames, axis=0)
        # sounddevice returns (N, channels) — flatten to 1D for mono
        if audio.ndim > 1:
            audio = audio.flatten()
        return audio

    def _upload_and_transcribe(self, audio_data):
        """Upload audio to STT endpoint and emit result."""
        import io
        import struct

        if _requests_mod is None:
            self.error_occurred.emit("requests library not installed")
            return

        # Convert float32 numpy to WAV bytes in memory
        try:
            wav_buffer = io.BytesIO()
            import numpy as np
            # Convert float32 [-1,1] to int16
            pcm = (audio_data * 32767).astype(np.int16)
            # Write WAV header manually (avoids soundfile dep)
            num_samples = len(pcm)
            data_size = num_samples * 2  # 16-bit = 2 bytes per sample
            wav_buffer.write(b'RIFF')
            wav_buffer.write(struct.pack('<I', 36 + data_size))
            wav_buffer.write(b'WAVE')
            wav_buffer.write(b'fmt ')
            wav_buffer.write(struct.pack('<I', 16))       # chunk size
            wav_buffer.write(struct.pack('<H', 1))        # PCM
            wav_buffer.write(struct.pack('<H', 1))        # mono
            wav_buffer.write(struct.pack('<I', self.SAMPLE_RATE))
            wav_buffer.write(struct.pack('<I', self.SAMPLE_RATE * 2))
            wav_buffer.write(struct.pack('<H', 2))        # block align
            wav_buffer.write(struct.pack('<H', 16))       # bits per sample
            wav_buffer.write(b'data')
            wav_buffer.write(struct.pack('<I', data_size))
            wav_buffer.write(pcm.tobytes())
            wav_buffer.seek(0)
        except Exception as e:
            self.error_occurred.emit(f"Audio encoding error: {e}")
            return

        try:
            resp = _requests_mod.post(
                f"{API_BASE}/api/v1/stt/transcribe",
                files={"audio": ("recording.wav", wav_buffer, "audio/wav")},
                timeout=30,
            )

            if resp.status_code == 200:
                data = resp.json()
                text = data.get("text", "").strip()
                if text:
                    logger.info(
                        "MicWorker: transcribed %d chars (confidence=%.3f, "
                        "duration=%.1fs, language=%s)",
                        len(text), data.get("confidence", 0),
                        data.get("duration", 0), data.get("language", "?"),
                    )
                    self.transcription_ready.emit(text)
                else:
                    self.error_occurred.emit("No speech detected")
            elif resp.status_code == 503:
                self.error_occurred.emit("STT engine unavailable on server")
            else:
                self.error_occurred.emit(f"STT error: HTTP {resp.status_code}")

        except Exception as e:
            if "ConnectionError" in type(e).__name__:
                self.error_occurred.emit("Cannot connect to MERLIN API")
            elif "Timeout" in type(e).__name__:
                self.error_occurred.emit("STT request timed out")
            else:
                self.error_occurred.emit(f"STT error: {e}")


# ─────────────────────────────────────────────────────────────
# Chat Bubble
# ─────────────────────────────────────────────────────────────

class ChatBubble(QWidget):
    """Rounded message bubble — user right / bot left / error left-red."""

    # kind: "user" | "bot" | "error"
    def __init__(self, text: str, kind: str = "bot", parent=None):
        super().__init__(parent)
        self.text = text
        self.kind = kind
        self._setup()

    def _setup(self):
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 3, 10, 3)

        label = QLabel()
        # ── CRITICAL: always plain text — never let Qt parse HTML/rich ──
        label.setTextFormat(Qt.TextFormat.PlainText)
        label.setText(self.text)
        label.setWordWrap(True)
        label.setFont(QFont("Segoe UI", 10))
        label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        # max bubble width = 78% of panel, leaving margin on opposite side
        label.setMaximumWidth(int(PANEL_WIDTH * 0.78))

        if self.kind == "user":
            label.setStyleSheet("""
                QLabel {
                    background: qlineargradient(x1:0,y1:0,x2:1,y2:1,
                        stop:0 rgba(0,130,230,90), stop:1 rgba(0,85,170,70));
                    color: rgba(220,238,255,235);
                    border-radius: 16px;
                    border-bottom-right-radius: 4px;
                    border: 1px solid rgba(0,190,255,65);
                    padding: 9px 13px;
                }
            """)
            lay.addStretch()
            lay.addWidget(label)

        elif self.kind == "error":
            # Distinct red-tinted style for API / connection errors
            label.setStyleSheet("""
                QLabel {
                    background: rgba(160,30,30,55);
                    color: rgba(255,190,185,230);
                    border-radius: 16px;
                    border-bottom-left-radius: 4px;
                    border: 1px solid rgba(220,80,70,60);
                    padding: 9px 13px;
                }
            """)
            lay.addWidget(label)
            lay.addStretch()

        else:  # bot
            label.setStyleSheet("""
                QLabel {
                    background: rgba(255,255,255,8);
                    color: rgba(205,222,242,220);
                    border-radius: 16px;
                    border-bottom-left-radius: 4px;
                    border: 1px solid rgba(255,255,255,16);
                    padding: 9px 13px;
                }
            """)
            lay.addWidget(label)
            lay.addStretch()


class TypingBubble(QWidget):
    """Animated three-dot typing indicator — left-aligned inside a bot-style pill."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._phase = 0.0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(35)
        self.setFixedHeight(40)

    def _tick(self):
        self._phase = (self._phase + 0.09) % (2 * math.pi)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        pill_w, pill_h = 62, 32
        pill_x = 10
        pill_y = (self.height() - pill_h) // 2
        path = QPainterPath()
        path.addRoundedRect(QRectF(pill_x, pill_y, pill_w, pill_h), 16, 16)
        p.fillPath(path, QBrush(QColor(255, 255, 255, 10)))
        p.setPen(QPen(QColor(255, 255, 255, 18), 1))
        p.drawPath(path)
        dot_cx = pill_x + 14
        cy = self.height() / 2.0
        for i in range(3):
            t      = self._phase + i * 1.05
            offset = -3.5 * math.sin(t)
            alpha  = int(140 + 100 * (0.5 + 0.5 * math.sin(t)))
            c = QColor(0, 200, 255, max(40, min(255, alpha)))
            p.setBrush(QBrush(c))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(QPointF(dot_cx + i * 16, cy + offset), 3.5, 3.5)

    def stop(self):
        self._timer.stop()


# ─────────────────────────────────────────────────────────────
# MERLIN Orb — main widget
# ─────────────────────────────────────────────────────────────

class MerlinOrb(QWidget):

    def __init__(self):
        super().__init__()
        self._expanded    = False
        self._connected   = False
        self._processing  = False

        # Orb animation state
        self._ring_angle   = 0.0      # outer ring rotation (deg)
        self._inner_angle  = 0.0      # inner highlight rotation
        self._glow_phase   = 0.0      # pulsing glow
        self._morph_phase  = 0.0      # blob morphing
        self._particles: list[dict] = self._init_particles()

        self._typing_widget: Optional[TypingBubble] = None
        self._chat_worker:  Optional[ChatWorker]    = None
        self._mic_worker:   Optional[MicWorker]     = None
        self._mic_recording = False

        self._setup_window()
        self._setup_panel()
        self._setup_anim_timer()
        self._setup_health_timer()
        self._position_window()

    # ── window ───────────────────────────────────────────────

    def _setup_window(self):
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(ORB_SIZE, ORB_SIZE)

    def _position_window(self):
        screen = QApplication.primaryScreen()
        if screen:
            geo = screen.availableGeometry()
            self.move(geo.right() - ORB_SIZE - 20, geo.top() + 40)

    # ── particles ────────────────────────────────────────────

    def _init_particles(self) -> list[dict]:
        particles = []
        for i in range(5):
            particles.append({
                "angle":  (360 / 5) * i,
                "speed":  0.6 + i * 0.15,
                "radius": 36 + (i % 3) * 4,
                "size":   2.0 + (i % 2) * 1.0,
                "life":   (i / 5.0),
            })
        return particles

    # ── animation timer ──────────────────────────────────────

    def _setup_anim_timer(self):
        self._anim_timer = QTimer(self)
        self._anim_timer.timeout.connect(self._tick_anim)
        self._anim_timer.start(ANIM_INTERVAL)

    def _tick_anim(self):
        speed = 2.5 if self._processing else 1.0

        self._ring_angle   = (self._ring_angle  + 0.9  * speed) % 360
        self._inner_angle  = (self._inner_angle - 1.3  * speed) % 360
        self._glow_phase  += 0.04 * speed
        self._morph_phase += 0.015

        for pt in self._particles:
            pt["angle"] = (pt["angle"] + pt["speed"] * speed) % 360
            pt["life"]  = (pt["life"]  + 0.003 * speed) % 1.0

        if not self._expanded:
            self.update()

    # ── paint ────────────────────────────────────────────────

    def paintEvent(self, event):
        if self._expanded:
            self._paint_panel_bg()
            return
        self._paint_orb()

    def _paint_panel_bg(self):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        path = QPainterPath()
        path.addRoundedRect(QRectF(0, 0, self.width(), self.height()), 18, 18)
        p.fillPath(path, QBrush(C_BG_PANEL))
        # top accent line
        grad = QLinearGradient(0, 0, self.width(), 0)
        grad.setColorAt(0.0, QColor(0, 0, 0, 0))
        grad.setColorAt(0.5, QColor(0, 200, 255, 90))
        grad.setColorAt(1.0, QColor(0, 0, 0, 0))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(QBrush(grad), 1))
        p.drawLine(18, 0, self.width() - 18, 0)
        # border
        border_col = C_ACCENT if self._connected else C_GREY
        border_col.setAlpha(50)
        p.setPen(QPen(border_col, 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(path)

    def _paint_orb(self):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        cx = self.width()  / 2.0
        cy = self.height() / 2.0
        r  = ORB_SIZE / 2.0 - 4

        # ── ambient glow ──────────────────────────────────
        glow_a = int(30 + 20 * math.sin(self._glow_phase)) if self._connected else 10
        glow_r = r + 14
        glow_g = QRadialGradient(cx, cy, glow_r)
        glow_g.setColorAt(0.0, QColor(0, 180, 255, glow_a))
        glow_g.setColorAt(1.0, QColor(0, 0, 0, 0))
        p.setBrush(QBrush(glow_g))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(QPointF(cx, cy), glow_r, glow_r)

        # ── outer spinning ring (conic simulation) ────────
        if self._connected:
            ring_pen_w = 2.5
            steps = 60
            for i in range(steps):
                angle_deg = self._ring_angle + (360.0 / steps) * i
                angle_rad = math.radians(angle_deg)
                t = (i / steps)
                # colour cycles blue → cyan → blue
                alpha = int(60 + 195 * (0.5 + 0.5 * math.sin(t * 2 * math.pi)))
                hue   = int(195 + 25 * math.sin(t * math.pi))
                seg_c = QColor.fromHsv(hue, 220, 255, alpha)
                p.setPen(QPen(seg_c, ring_pen_w, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                a1 = -self._ring_angle - (360.0 / steps) * i
                span = 360.0 / steps
                p.drawArc(
                    QRectF(cx - r, cy - r, r * 2, r * 2),
                    int(a1 * 16), int(span * 16),
                )
        else:
            p.setPen(QPen(C_GREY, 1.5))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(QPointF(cx, cy), r, r)

        # ── inner highlight arc ───────────────────────────
        if self._connected:
            hi_g = QConicalGradient(cx, cy, self._inner_angle)
            hi_g.setColorAt(0.0, QColor(0, 220, 255, 90))
            hi_g.setColorAt(0.25, QColor(0, 80,  200, 30))
            hi_g.setColorAt(0.5, QColor(0, 220, 255, 10))
            hi_g.setColorAt(1.0, QColor(0, 220, 255, 90))
            p.setPen(QPen(QBrush(hi_g), 1.0))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(QPointF(cx, cy), r - 3, r - 3)

        # ── core circle ───────────────────────────────────
        core_g = QRadialGradient(cx - r * 0.2, cy - r * 0.25, r * 1.2)
        core_g.setColorAt(0.0, QColor(22, 35, 52))
        core_g.setColorAt(0.6, QColor(13, 13, 20))
        core_g.setColorAt(1.0, QColor(8,  8,  14))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(core_g))
        p.drawEllipse(QPointF(cx, cy), r - 2, r - 2)

        # ── inner glow pulse ──────────────────────────────
        if self._connected:
            pulse_a = int(18 + 14 * math.sin(self._glow_phase * 1.3))
            inner_g = QRadialGradient(cx, cy, r * 0.75)
            inner_g.setColorAt(0.0, QColor(0, 200, 255, pulse_a))
            inner_g.setColorAt(1.0, QColor(0,   0,   0, 0))
            p.setBrush(QBrush(inner_g))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(QPointF(cx, cy), r - 2, r - 2)

        # ── "M" letter ────────────────────────────────────
        letter_color = C_ACCENT if self._connected else C_GREY
        p.setPen(QPen(letter_color))
        font = QFont("Segoe UI", 16, QFont.Weight.Bold)
        p.setFont(font)
        p.drawText(QRectF(0, 0, self.width(), self.height()),
                   Qt.AlignmentFlag.AlignCenter, "M")

        # ── orbiting particles ────────────────────────────
        if self._connected:
            for pt in self._particles:
                life    = pt["life"]
                alpha   = int(200 * math.sin(life * math.pi))
                if alpha < 20:
                    continue
                ang_rad = math.radians(pt["angle"])
                px  = cx + pt["radius"] * math.cos(ang_rad)
                py  = cy + pt["radius"] * math.sin(ang_rad)
                sz  = pt["size"] * (0.5 + 0.5 * math.sin(life * math.pi))
                pc  = QColor(0, 200, 255, alpha)
                p.setBrush(QBrush(pc))
                p.setPen(Qt.PenStyle.NoPen)
                p.drawEllipse(QPointF(px, py), sz, sz)

    # ── panel UI ─────────────────────────────────────────────

    def _setup_panel(self):
        self._panel = QWidget(self)
        self._panel.hide()

        root = QVBoxLayout(self._panel)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── header ──
        hdr_widget = QWidget()
        hdr_widget.setFixedHeight(54)
        hdr_widget.setStyleSheet("background: transparent;")
        hdr = QHBoxLayout(hdr_widget)
        hdr.setContentsMargins(14, 0, 14, 0)
        hdr.setSpacing(10)

        # mini orb in panel header
        mini_orb = _MiniOrb(self)
        hdr.addWidget(mini_orb)

        title = QLabel("MERLIN")
        title.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {C_ACCENT.name()}; letter-spacing: 2px;")
        hdr.addWidget(title)
        hdr.addStretch()

        self._status_dot = QLabel("●")
        self._status_dot.setFont(QFont("Segoe UI", 8))
        self._status_dot.setStyleSheet(f"color: {C_GREY.name()};")
        hdr.addWidget(self._status_dot)

        close_btn = QPushButton("✕")
        close_btn.setFixedSize(26, 26)
        close_btn.setStyleSheet(f"""
            QPushButton {{
                background: rgba(255,255,255,8);
                color: {C_TEXT_DIM.name()};
                border: 1px solid rgba(255,255,255,15);
                border-radius: 13px;
                font-size: 11px;
            }}
            QPushButton:hover {{
                background: rgba(255,255,255,18);
                color: {C_TEXT.name()};
            }}
        """)
        close_btn.clicked.connect(self._collapse)
        hdr.addWidget(close_btn)
        root.addWidget(hdr_widget)

        # separator
        sep = QWidget()
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background: rgba(255,255,255,13);")
        root.addWidget(sep)

        # ── scroll area ──
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setStyleSheet("""
            QScrollArea { background: transparent; border: none; }
            QScrollBar:vertical { width: 5px; background: transparent; margin: 4px 0; }
            QScrollBar::handle:vertical {
                background: rgba(0,180,255,60);
                border-radius: 2px;
                min-height: 20px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
        """)

        self._chat_container = QWidget()
        self._chat_container.setStyleSheet("background: transparent;")
        self._chat_layout = QVBoxLayout(self._chat_container)
        self._chat_layout.setContentsMargins(0, 10, 0, 10)
        self._chat_layout.setSpacing(4)
        self._chat_layout.addStretch()

        self._scroll.setWidget(self._chat_container)
        root.addWidget(self._scroll)

        # separator
        sep2 = QWidget()
        sep2.setFixedHeight(1)
        sep2.setStyleSheet("background: rgba(255,255,255,13);")
        root.addWidget(sep2)

        # ── input bar ──
        bar = QWidget()
        bar.setFixedHeight(60)
        bar.setStyleSheet("background: transparent;")
        bar_lay = QHBoxLayout(bar)
        bar_lay.setContentsMargins(12, 10, 12, 10)
        bar_lay.setSpacing(8)

        self._input = QLineEdit()
        self._input.setPlaceholderText("Ask MERLIN…")
        self._input.setFont(QFont("Segoe UI", 10))
        self._input.setStyleSheet(f"""
            QLineEdit {{
                background: {C_INPUT_BG.name()};
                color: rgba(215,230,248,225);
                border: 1px solid rgba(0,170,255,40);
                border-radius: 10px;
                padding: 8px 12px;
                selection-background-color: rgba(0,150,220,80);
            }}
            QLineEdit:focus {{
                border-color: rgba(0,200,255,100);
                background: rgba(30,32,48,255);
            }}
            QLineEdit:disabled {{
                background: rgba(20,20,32,180);
                color: rgba(255,255,255,50);
                border-color: rgba(255,255,255,12);
            }}
        """)
        self._input.returnPressed.connect(self._send_message)
        bar_lay.addWidget(self._input)

        # ── mic button ──
        self._mic_btn = QPushButton("\U0001F3A4")  # 🎤
        self._mic_btn.setFixedSize(38, 38)
        self._mic_btn.setFont(QFont("Segoe UI", 13))
        self._mic_btn.setStyleSheet("""
            QPushButton {
                background: rgba(0,160,220,45);
                color: rgba(0,200,255,220);
                border: 1px solid rgba(0,185,255,55);
                border-radius: 10px;
            }
            QPushButton:hover {
                background: rgba(0,180,240,75);
                border-color: rgba(0,210,255,100);
            }
            QPushButton:pressed { padding-top: 2px; }
            QPushButton:disabled {
                background: rgba(40,40,60,120);
                color: rgba(255,255,255,50);
                border-color: rgba(255,255,255,15);
            }
        """)
        self._mic_btn.setToolTip("Voice input")
        self._mic_btn.clicked.connect(self._toggle_mic)
        bar_lay.addWidget(self._mic_btn)

        self._send_btn = QPushButton("→")
        self._send_btn.setFixedSize(38, 38)
        self._send_btn.setFont(QFont("Segoe UI", 15))
        self._send_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:1,
                    stop:0 rgba(0,110,200,210), stop:1 rgba(0,75,160,210));
                color: rgba(200,235,255,235);
                border: 1px solid rgba(0,185,255,65);
                border-radius: 10px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:1,
                    stop:0 rgba(0,150,230,230), stop:1 rgba(0,110,200,230));
                border-color: rgba(0,210,255,110);
            }
            QPushButton:pressed { padding-top: 2px; }
            QPushButton:disabled {
                background: rgba(40,40,60,120);
                color: rgba(255,255,255,50);
                border-color: rgba(255,255,255,15);
            }
        """)
        self._send_btn.clicked.connect(self._send_message)
        bar_lay.addWidget(self._send_btn)

        root.addWidget(bar)

    # ── expand / collapse ─────────────────────────────────────

    def _expand(self):
        if self._expanded:
            return
        self._expanded = True
        screen = QApplication.primaryScreen()
        if screen:
            geo = screen.availableGeometry()
            self.move(geo.right() - PANEL_WIDTH - 20, geo.top() + 40)

        self.setFixedSize(PANEL_WIDTH, PANEL_HEIGHT)
        self._panel.setGeometry(0, 0, PANEL_WIDTH, PANEL_HEIGHT)
        self._panel.show()
        self._input.setFocus()
        QApplication.instance().installEventFilter(self)
        self.update()

    def _collapse(self):
        if not self._expanded:
            return
        self._expanded = False
        self._panel.hide()
        app = QApplication.instance()
        if app:
            app.removeEventFilter(self)
        self.setFixedSize(ORB_SIZE, ORB_SIZE)
        self._position_window()
        self.update()

    # ── drag ─────────────────────────────────────────────────

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start  = event.globalPosition().toPoint()
            self._drag_origin = self.pos()
            self._dragging    = False
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if hasattr(self, "_drag_start") and self._drag_start:
            delta = event.globalPosition().toPoint() - self._drag_start
            if not self._dragging and (abs(delta.x()) > 5 or abs(delta.y()) > 5):
                self._dragging = True
            if self._dragging:
                self.move(self._drag_origin + delta)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if not getattr(self, "_dragging", False) and not self._expanded:
                self._expand()
            self._drag_start = None
            self._dragging   = False
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self._collapse()
        super().keyPressEvent(event)

    def eventFilter(self, obj, event):
        if (
            self._expanded
            and event.type() == QEvent.Type.MouseButtonPress
            and not self.geometry().contains(QCursor.pos())
        ):
            self._collapse()
        return super().eventFilter(obj, event)

    # ── chat ─────────────────────────────────────────────────

    def _set_input_enabled(self, enabled: bool):
        self._input.setEnabled(enabled)
        self._send_btn.setEnabled(enabled)
        if enabled:
            self._input.setPlaceholderText("Ask MERLIN…")
        else:
            self._input.setPlaceholderText("Waiting for response…")

    def _send_message(self):
        text = self._input.text().strip()
        if not text:
            return
        self._input.clear()
        self._add_bubble(text, kind="user")
        self._processing = True
        self._set_input_enabled(False)

        # typing indicator
        self._typing_widget = TypingBubble()
        self._chat_layout.insertWidget(self._chat_layout.count() - 1, self._typing_widget)
        self._scroll_to_bottom()

        self._chat_worker = ChatWorker(text)
        self._chat_worker.response_ready.connect(self._on_response)
        self._chat_worker.error_occurred.connect(self._on_error)
        self._chat_worker.start()

    def _remove_typing(self):
        if self._typing_widget:
            self._typing_widget.stop()
            self._typing_widget.setParent(None)
            self._typing_widget.deleteLater()
            self._typing_widget = None

    def _on_response(self, response: str):
        self._processing = False
        self._remove_typing()
        self._add_bubble(response, kind="bot")
        self._set_input_enabled(True)
        self._input.setFocus()

    def _on_error(self, error: str):
        self._processing = False
        self._remove_typing()
        self._add_bubble(error, kind="error")
        self._set_input_enabled(True)
        self._input.setFocus()

    # ── mic ───────────────────────────────────────────────

    def _toggle_mic(self):
        """Toggle mic recording on/off."""
        if self._mic_recording:
            # Stop recording
            if self._mic_worker:
                self._mic_worker.stop_recording()
            return

        if self._processing:
            return

        # Start recording
        self._mic_worker = MicWorker()
        self._mic_worker.transcription_ready.connect(self._on_mic_result)
        self._mic_worker.error_occurred.connect(self._on_mic_error)
        self._mic_worker.status_changed.connect(self._on_mic_status)
        self._mic_worker.finished.connect(self._on_mic_thread_finished)
        self._mic_worker.start()
        self._mic_recording = True

        # Visual: red recording state
        self._mic_btn.setText("\u25FC")  # ◼ stop icon
        self._mic_btn.setStyleSheet("""
            QPushButton {
                background: rgba(220,50,50,65);
                color: rgba(255,130,130,230);
                border: 1px solid rgba(220,70,70,80);
                border-radius: 10px;
            }
            QPushButton:hover {
                background: rgba(220,50,50,100);
            }
        """)
        self._input.setPlaceholderText("Listening\u2026")

    def _on_mic_status(self, status: str):
        if status == "listening":
            self._input.setPlaceholderText("Listening\u2026")
        elif status == "processing":
            self._input.setPlaceholderText("Transcribing\u2026")
            self._mic_btn.setText("\u27F3")  # ⟳

    def _on_mic_result(self, text: str):
        """Handle successful transcription: populate + auto-send."""
        self._reset_mic_ui()
        self._input.setText(text)
        self._send_message()

    def _on_mic_error(self, error: str):
        """Handle mic/STT error: show as error bubble."""
        self._reset_mic_ui()
        self._add_bubble(f"\U0001F3A4 {error}", kind="error")

    def _reset_mic_ui(self):
        """Reset mic button to idle state.

        NOTE: does NOT set _mic_worker = None here.
        The QThread may still be finishing its run() method.
        Cleanup happens in _on_mic_thread_finished via the
        finished signal, which fires after run() returns.
        """
        self._mic_recording = False
        self._mic_btn.setText("\U0001F3A4")  # 🎤
        self._mic_btn.setStyleSheet("""
            QPushButton {
                background: rgba(0,160,220,45);
                color: rgba(0,200,255,220);
                border: 1px solid rgba(0,185,255,55);
                border-radius: 10px;
            }
            QPushButton:hover {
                background: rgba(0,180,240,75);
                border-color: rgba(0,210,255,100);
            }
            QPushButton:pressed { padding-top: 2px; }
            QPushButton:disabled {
                background: rgba(40,40,60,120);
                color: rgba(255,255,255,50);
                border-color: rgba(255,255,255,15);
            }
        """)
        self._input.setPlaceholderText("Ask MERLIN\u2026")

    def _on_mic_thread_finished(self):
        """Safely clean up the MicWorker after its thread exits."""
        if self._mic_worker is not None:
            self._mic_worker.deleteLater()
            self._mic_worker = None

    def _add_bubble(self, text: str, kind: str = "bot"):
        bubble = ChatBubble(text, kind)
        self._chat_layout.insertWidget(self._chat_layout.count() - 1, bubble)
        self._scroll_to_bottom()

    def _scroll_to_bottom(self):
        QTimer.singleShot(60, lambda: self._scroll.verticalScrollBar().setValue(
            self._scroll.verticalScrollBar().maximum()
        ))

    # ── health ───────────────────────────────────────────────

    def _setup_health_timer(self):
        self._health_timer = QTimer(self)
        self._health_timer.timeout.connect(self._check_health)
        self._health_timer.start(HEALTH_INTERVAL)
        QTimer.singleShot(600, self._check_health)

    def _check_health(self):
        def _run():
            try:
                if HAS_REQUESTS and _requests_mod:
                    resp = _requests_mod.get(f"{API_BASE}/api/v1/health", timeout=3)
                    self._connected = resp.status_code == 200
                else:
                    self._connected = False
            except Exception:
                self._connected = False

            color = C_ACCENT.name() if self._connected else C_GREY.name()
            if hasattr(self, "_status_dot"):
                self._status_dot.setStyleSheet(f"color: {color};")

        threading.Thread(target=_run, daemon=True).start()


# ─────────────────────────────────────────────────────────────
# Mini animated orb for the panel header
# ─────────────────────────────────────────────────────────────

class _MiniOrb(QWidget):
    def __init__(self, parent_orb: MerlinOrb):
        super().__init__()
        self._orb   = parent_orb
        self._phase = 0.0
        self.setFixedSize(28, 28)
        t = QTimer(self)
        t.timeout.connect(self._tick)
        t.start(40)

    def _tick(self):
        self._phase = (self._phase + 0.08) % (2 * math.pi)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        cx, cy, r = 14.0, 14.0, 11.0

        # glow
        glow_a = int(20 + 15 * math.sin(self._phase))
        if self._orb._connected:
            gg = QRadialGradient(cx, cy, r + 5)
            gg.setColorAt(0, QColor(0, 200, 255, glow_a))
            gg.setColorAt(1, QColor(0, 0, 0, 0))
            p.setBrush(QBrush(gg))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(QPointF(cx, cy), r + 5, r + 5)

        # core
        core_g = QRadialGradient(cx - 3, cy - 3, r * 1.2)
        core_g.setColorAt(0, QColor(22, 35, 52))
        core_g.setColorAt(1, QColor(10, 10, 18))
        p.setBrush(QBrush(core_g))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(QPointF(cx, cy), r, r)

        # ring
        ring_c = QColor(0, 200, 255, 160) if self._orb._connected else QColor(70, 70, 82, 120)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(ring_c, 1.2))
        p.drawEllipse(QPointF(cx, cy), r, r)

        # letter
        p.setPen(QPen(C_ACCENT if self._orb._connected else C_GREY))
        p.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        p.drawText(QRectF(0, 0, 28, 28), Qt.AlignmentFlag.AlignCenter, "M")


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("MERLIN Widget")
    # dark palette baseline
    orb = MerlinOrb()
    orb.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
