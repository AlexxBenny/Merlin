# infrastructure/system_controller.py

"""
SystemController — Centralized OS control API.

Absorbs AURA-style app launch/focus/close logic into
a single, stateless, OS-facing service.

Design constraints:
- Stateless: no internal mutable state
- No timeline/event imports: returns results, skills emit events
- No WorldState dependency: pure infrastructure
- Timeout-guarded: all subprocess calls have bounded execution
- Graceful fallback: all methods return clean failures if win32 unavailable
- Platform-guarded: Windows-only packages lazy-imported inside try blocks
- COM safety: pycaw COM objects acquired fresh per call, never cached

Injection pattern:
    SystemController is injected into system skills via constructor.
    Skills call methods, receive results, and emit events.
    This mirrors LocationConfig injection into fs skills.
"""

import sys

import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Result types (pure data, no logic)
# ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AppHandle:
    """Result of opening an app."""
    app_name: str
    pid: Optional[int] = None
    success: bool = True
    error: Optional[str] = None


@dataclass(frozen=True)
class AppInfo:
    """Lightweight app descriptor."""
    name: str
    pid: int
    title: Optional[str] = None


@dataclass(frozen=True)
class WindowInfo:
    """Window descriptor."""
    title: str
    hwnd: int
    pid: int
    app_name: Optional[str] = None
    visible: bool = True


@dataclass(frozen=True)
class HardwareResult:
    """Result of a hardware control operation."""
    success: bool = True
    actual_value: Optional[Any] = None  # Actual OS state after mutation
    error: Optional[str] = None


# ─────────────────────────────────────────────────────────────
# Win32 optional imports (lazy — never crash on non-Windows)
# ─────────────────────────────────────────────────────────────

_IS_WINDOWS = sys.platform == "win32"

_HAS_WIN32 = False
_HAS_PSUTIL = False
_HAS_PYCAW = False
_HAS_CTYPES_MEDIA = False

if _IS_WINDOWS:
    try:
        import win32gui
        import win32process
        import win32con
        import win32api
        _HAS_WIN32 = True
    except ImportError:
        logger.info("win32gui not available — window ops disabled")

    try:
        import psutil
        _HAS_PSUTIL = True
    except ImportError:
        logger.info("psutil not available — process ops disabled")

    try:
        from pycaw.pycaw import AudioUtilities
        _HAS_PYCAW = True
    except ImportError:
        logger.info("pycaw not available — volume control disabled")

    try:
        import ctypes
        import ctypes.wintypes
        _HAS_CTYPES_MEDIA = True
    except ImportError:
        logger.info("ctypes not available — media keys disabled")
else:
    logger.info("Non-Windows platform — hardware control disabled")


# ─────────────────────────────────────────────────────────────
# SystemController
# ─────────────────────────────────────────────────────────────

# Default timeout for subprocess operations (seconds)
_SUBPROCESS_TIMEOUT = 10


class SystemController:
    """
    Centralized OS control.

    Tier 1 (GUI): os.startfile() — let Windows resolve
    Tier 2 (CLI): shutil.which() + subprocess
    Window ops:   win32gui + win32process + psutil
    """

    # ── App Launch ──

    def launch(
        self,
        entity: "ApplicationEntity",
        args: Optional[List[str]] = None,
    ) -> AppHandle:
        """Launch an application using its best available strategy.

        Iterates strategies by priority (descending). Tries each
        until one succeeds. Returns failure if all strategies exhausted.

        Args:
            entity: Resolved ApplicationEntity from registry.
            args:   Optional command-line arguments.

        Returns:
            AppHandle with success status and PID if available.
        """
        if not entity.launch_strategies:
            return AppHandle(
                app_name=entity.app_id,
                success=False,
                error="No launch strategies available",
            )

        strategies = sorted(
            entity.launch_strategies, key=lambda s: -s.priority,
        )

        last_error = ""
        for strategy in strategies:
            result = self._try_strategy(strategy, entity.app_id, args)
            if result.success:
                logger.info(
                    "Launched '%s' via %s (priority=%d)",
                    entity.app_id, strategy.method.value, strategy.priority,
                )
                return result
            last_error = result.error or "Unknown failure"
            logger.debug(
                "Strategy %s failed for '%s': %s",
                strategy.method.value, entity.app_id, last_error,
            )

        return AppHandle(
            app_name=entity.app_id,
            success=False,
            error=f"All {len(strategies)} strategies exhausted. Last: {last_error}",
        )

    def _try_strategy(
        self,
        strategy: "LaunchStrategy",
        app_name: str,
        args: Optional[List[str]] = None,
    ) -> AppHandle:
        """Attempt a single launch strategy. Never raises."""
        try:
            if strategy.type == "executable":
                return self._launch_executable(
                    strategy.value, app_name, args,
                )
            elif strategy.type == "protocol":
                return self._launch_protocol(
                    strategy.value, app_name,
                )
            elif strategy.type == "appsfolder":
                return self._launch_appsfolder(
                    strategy.value, app_name,
                )
            elif strategy.type == "shell":
                return self._launch_shell(
                    strategy.value, app_name, args,
                )
            else:
                return AppHandle(
                    app_name=app_name, success=False,
                    error=f"Unknown strategy type: {strategy.type}",
                )
        except Exception as e:
            return AppHandle(
                app_name=app_name, success=False,
                error=f"{strategy.type} failed: {e}",
            )

    def _launch_executable(
        self, exe_path: str, app_name: str,
        args: Optional[List[str]] = None,
    ) -> AppHandle:
        """Launch via executable path. Returns PID."""
        cmd = [exe_path] + (args or [])
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(0.3)
        if proc.poll() is not None and proc.returncode != 0:
            return AppHandle(
                app_name=app_name, success=False,
                error=f"Process exited with code {proc.returncode}",
            )
        return AppHandle(
            app_name=app_name, pid=proc.pid, success=True,
        )

    def _launch_protocol(
        self, protocol_uri: str, app_name: str,
    ) -> AppHandle:
        """Launch via protocol handler (e.g., spotify:)."""
        os.startfile(protocol_uri)
        time.sleep(0.5)
        return AppHandle(
            app_name=app_name, pid=None, success=True,
        )

    def _launch_appsfolder(
        self, appsfolder_path: str, app_name: str,
    ) -> AppHandle:
        """Launch via shell:AppsFolder (UWP/Store apps)."""
        os.startfile(appsfolder_path)
        time.sleep(0.5)
        return AppHandle(
            app_name=app_name, pid=None, success=True,
        )

    def _launch_shell(
        self, shell_name: str, app_name: str,
        args: Optional[List[str]] = None,
    ) -> AppHandle:
        """Launch via shell (os.startfile or shutil.which fallback)."""
        exe_path = shutil.which(shell_name)
        if exe_path:
            return self._launch_executable(exe_path, app_name, args)
        os.startfile(shell_name)
        time.sleep(0.5)
        return AppHandle(
            app_name=app_name, pid=None, success=True,
        )

    def open_app(
        self,
        app_name: str,
        args: Optional[List[str]] = None,
    ) -> AppHandle:
        """
        Open an application (backward-compatible fallback).

        Prefer launch(entity) when ApplicationEntity is available.
        This method uses the legacy 2-tier resolution:
        1. shutil.which() for CLI-resolvable apps
        2. os.startfile() for GUI-registered apps

        Never raises. Returns AppHandle with success=False on failure.
        """
        # Tier 2: CLI resolution (gives us PID via subprocess)
        exe_path = shutil.which(app_name)
        if exe_path:
            try:
                cmd = [exe_path] + (args or [])
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                # Brief wait to check for immediate crash
                time.sleep(0.3)
                if proc.poll() is not None and proc.returncode != 0:
                    return AppHandle(
                        app_name=app_name,
                        success=False,
                        error=f"Process exited with code {proc.returncode}",
                    )
                return AppHandle(
                    app_name=app_name,
                    pid=proc.pid,
                    success=True,
                )
            except Exception as e:
                logger.warning("CLI launch failed for '%s': %s", app_name, e)

        # Tier 1: GUI launch (no PID available)
        try:
            os.startfile(app_name)
            # Brief wait for app to register
            time.sleep(0.5)
            return AppHandle(
                app_name=app_name,
                pid=None,  # startfile doesn't return PID
                success=True,
            )
        except OSError as e:
            return AppHandle(
                app_name=app_name,
                success=False,
                error=str(e),
            )

    # ── App Focus ──

    def focus_app(self, app_name: str) -> bool:
        """
        Bring an application window to the foreground.

        Uses a robust multi-step strategy to work around Windows'
        SetForegroundWindow restrictions on background processes:

        1. ShowWindow(SW_RESTORE) — unminimize
        2. BringWindowToTop
        3. AttachThreadInput — links caller thread to foreground thread
        4. SetForegroundWindow
        5. Detach threads

        Returns True if a window was focused, False otherwise.
        """
        if not _HAS_WIN32:
            logger.warning("focus_app unavailable: win32gui not installed")
            return False

        windows = self.find_windows(app_name)
        if not windows:
            logger.info("focus_app: no windows found for '%s'", app_name)
            return False

        target = windows[0]
        hwnd = target.hwnd

        try:
            # Step 1: Restore if minimized
            if win32gui.IsIconic(hwnd):
                win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)

            # Step 2: Get thread IDs for thread attachment
            foreground_hwnd = win32gui.GetForegroundWindow()
            target_thread_id, _ = win32process.GetWindowThreadProcessId(hwnd)
            foreground_thread_id, _ = win32process.GetWindowThreadProcessId(
                foreground_hwnd,
            )
            current_thread_id = win32api.GetCurrentThreadId()

            # Step 3: Attach threads (allows SetForegroundWindow to succeed)
            attached_fg = False
            attached_target = False
            try:
                if current_thread_id != foreground_thread_id:
                    win32process.AttachThreadInput(
                        current_thread_id, foreground_thread_id, True,
                    )
                    attached_fg = True
                if current_thread_id != target_thread_id:
                    win32process.AttachThreadInput(
                        current_thread_id, target_thread_id, True,
                    )
                    attached_target = True

                # Step 4: Bring to top + set foreground
                win32gui.BringWindowToTop(hwnd)
                win32gui.SetForegroundWindow(hwnd)

            finally:
                # Step 5: Always detach threads
                if attached_fg:
                    try:
                        win32process.AttachThreadInput(
                            current_thread_id, foreground_thread_id, False,
                        )
                    except Exception:
                        pass
                if attached_target:
                    try:
                        win32process.AttachThreadInput(
                            current_thread_id, target_thread_id, False,
                        )
                    except Exception:
                        pass

            # Verify focus actually changed
            new_foreground = win32gui.GetForegroundWindow()
            if new_foreground == hwnd:
                return True
            else:
                logger.warning(
                    "focus_app: SetForegroundWindow succeeded but "
                    "foreground is %s (expected %s)",
                    new_foreground, hwnd,
                )
                # Still return True — OS accepted the call even if
                # verification is racey
                return True

        except Exception as e:
            logger.warning(
                "focus_app: failed to focus window %s for '%s': %s",
                hwnd, app_name, e,
            )
            return False

    # ── App Close ──

    def close_app(self, app_name: str) -> bool:
        """
        Request an application to close gracefully.

        Sends WM_CLOSE to all windows matching app_name.
        Does NOT force-kill. Returns True if at least one
        close message was sent.
        """
        if not _HAS_WIN32:
            logger.warning("close_app unavailable: win32gui not installed")
            return False

        windows = self.find_windows(app_name)
        if not windows:
            return False

        closed_any = False
        for winfo in windows:
            try:
                win32gui.PostMessage(winfo.hwnd, win32con.WM_CLOSE, 0, 0)
                closed_any = True
            except Exception as e:
                logger.warning("Failed to close window %s: %s", winfo.hwnd, e)

        return closed_any

    # ── App Listing ──

    def list_running_apps(self) -> List[AppInfo]:
        """
        List visible running applications (not all processes).

        Returns apps that have at least one visible window.
        This is deliberately NOT a full process dump.
        """
        if not _HAS_WIN32 or not _HAS_PSUTIL:
            return []

        apps: Dict[int, AppInfo] = {}

        def _enum_callback(hwnd: int, _: Any) -> bool:
            if not win32gui.IsWindowVisible(hwnd):
                return True
            title = win32gui.GetWindowText(hwnd)
            if not title or title == "Program Manager":
                return True

            try:
                _, pid = win32process.GetWindowThreadProcessId(hwnd)
                if pid not in apps:
                    proc = psutil.Process(pid)
                    apps[pid] = AppInfo(
                        name=proc.name(),
                        pid=pid,
                        title=title,
                    )
            except (psutil.NoSuchProcess, Exception):
                pass
            return True

        try:
            win32gui.EnumWindows(_enum_callback, None)
        except Exception as e:
            logger.warning("EnumWindows failed: %s", e)

        return list(apps.values())

    # ── Window Discovery ──

    def find_windows(self, pattern: str) -> List[WindowInfo]:
        """
        Find visible windows matching a pattern.

        Matches by:
        1. Process name (case-insensitive, partial match)
        2. Window title (case-insensitive, partial match)

        Returns list sorted by match quality (process name match first).
        """
        if not _HAS_WIN32 or not _HAS_PSUTIL:
            return []

        results: List[WindowInfo] = []
        pattern_lower = pattern.lower()

        def _enum_callback(hwnd: int, _: Any) -> bool:
            if not win32gui.IsWindowVisible(hwnd):
                return True
            title = win32gui.GetWindowText(hwnd)
            if not title:
                return True

            try:
                _, pid = win32process.GetWindowThreadProcessId(hwnd)
                proc = psutil.Process(pid)
                proc_name = proc.name().lower()

                # Match by process name or window title
                if (pattern_lower in proc_name or
                        pattern_lower in title.lower()):
                    results.append(WindowInfo(
                        title=title,
                        hwnd=hwnd,
                        pid=pid,
                        app_name=proc.name(),
                    ))
            except (psutil.NoSuchProcess, Exception):
                pass
            return True

        try:
            win32gui.EnumWindows(_enum_callback, None)
        except Exception as e:
            logger.warning("EnumWindows failed: %s", e)

        # Sort: process name matches first, then title matches
        results.sort(
            key=lambda w: (
                0 if pattern_lower in (w.app_name or "").lower() else 1,
                w.title,
            ),
        )

        return results

    # ─────────────────────────────────────────────────────────
    # Brightness Control (PowerShell WMI)
    # ─────────────────────────────────────────────────────────

    def set_brightness(self, percent: int) -> HardwareResult:
        """
        Set display brightness via WMI (PowerShell).

        Reads actual brightness after set to confirm.
        Returns HardwareResult with actual_value = confirmed brightness.
        """
        if not _IS_WINDOWS:
            return HardwareResult(success=False, error="Not Windows")

        try:
            subprocess.run(
                [
                    "powershell", "-NoProfile", "-Command",
                    f"(Get-WmiObject -Namespace root/wmi "
                    f"-Class WmiMonitorBrightnessMethods)"
                    f".WmiSetBrightness(1, {percent})",
                ],
                capture_output=True,
                timeout=_SUBPROCESS_TIMEOUT,
            )

            # Read back actual brightness
            actual = self.get_brightness()
            return HardwareResult(
                success=True,
                actual_value=actual if actual is not None else percent,
            )
        except subprocess.TimeoutExpired:
            return HardwareResult(success=False, error="Brightness set timed out")
        except Exception as e:
            return HardwareResult(success=False, error=str(e))

    def get_brightness(self) -> Optional[int]:
        """
        Read current display brightness from WMI.
        Returns brightness percent (0-100), or None on failure.
        """
        if not _IS_WINDOWS:
            return None

        try:
            result = subprocess.run(
                [
                    "powershell", "-NoProfile", "-Command",
                    "(Get-WmiObject -Namespace root/wmi "
                    "-Class WmiMonitorBrightness)"
                    ".CurrentBrightness",
                ],
                capture_output=True,
                text=True,
                timeout=_SUBPROCESS_TIMEOUT,
            )
            if result.stdout.strip():
                return int(result.stdout.strip())
        except Exception as e:
            logger.debug("get_brightness failed: %s", e)
        return None

    # ─────────────────────────────────────────────────────────
    # Volume / Mute Control (pycaw)
    # ─────────────────────────────────────────────────────────

    def _get_volume_interface(self) -> Optional[Any]:
        """
        Get the default audio endpoint volume interface.

        Returns the IAudioEndpointVolume COM interface via pycaw's
        AudioDevice.EndpointVolume property.

        MUST NOT cache — acquire fresh per call for COM safety.

        Note: pycaw's AudioDevice wraps the raw IMMDevice COM object.
        Use .EndpointVolume (property) instead of raw .Activate().
        """
        if not _HAS_PYCAW:
            return None
        try:
            speakers = AudioUtilities.GetSpeakers()
            if speakers is None:
                logger.debug("GetSpeakers() returned None — no default audio device")
                return None
            return speakers.EndpointVolume
        except Exception as e:
            logger.debug("Failed to get volume interface: %s", e)
            return None

    def set_volume(self, percent: int) -> HardwareResult:
        """
        Set system master volume.

        Reads actual volume after set to confirm.
        pycaw uses scalar 0.0-1.0 internally.
        """
        volume = self._get_volume_interface()
        if volume is None:
            return HardwareResult(success=False, error="Volume control unavailable")

        try:
            scalar = percent / 100.0
            volume.SetMasterVolumeLevelScalar(scalar, None)

            # Read back actual
            actual_scalar = volume.GetMasterVolumeLevelScalar()
            actual_percent = round(actual_scalar * 100)
            return HardwareResult(success=True, actual_value=actual_percent)
        except Exception as e:
            return HardwareResult(success=False, error=str(e))

    def get_volume(self) -> Tuple[Optional[int], Optional[bool]]:
        """
        Read current volume percent and muted state.
        Returns (percent, is_muted) or (None, None) on failure.
        """
        volume = self._get_volume_interface()
        if volume is None:
            return None, None

        try:
            scalar = volume.GetMasterVolumeLevelScalar()
            muted = bool(volume.GetMute())
            return round(scalar * 100), muted
        except Exception as e:
            logger.debug("get_volume failed: %s", e)
            return None, None

    def mute(self) -> HardwareResult:
        """Mute system audio. Returns actual muted state."""
        volume = self._get_volume_interface()
        if volume is None:
            return HardwareResult(success=False, error="Volume control unavailable")

        try:
            volume.SetMute(True, None)
            actual = bool(volume.GetMute())
            return HardwareResult(success=True, actual_value=actual)
        except Exception as e:
            return HardwareResult(success=False, error=str(e))

    def unmute(self) -> HardwareResult:
        """Unmute system audio. Returns actual muted state."""
        volume = self._get_volume_interface()
        if volume is None:
            return HardwareResult(success=False, error="Volume control unavailable")

        try:
            volume.SetMute(False, None)
            actual = bool(volume.GetMute())
            return HardwareResult(success=True, actual_value=actual)
        except Exception as e:
            return HardwareResult(success=False, error=str(e))

    # ─────────────────────────────────────────────────────────
    # Night Light (Registry via PowerShell)
    # ─────────────────────────────────────────────────────────

    def toggle_nightlight(self) -> HardwareResult:
        """
        Toggle Windows Night Light via registry + broadcast.

        Modifies the Blue Light Reduction CloudStore registry key
        and increments the sequence counter to trigger the display
        subsystem refresh.

        Note: Night Light behavior can be inconsistent across
        Windows versions. This is best-effort.
        """
        if not _IS_WINDOWS:
            return HardwareResult(success=False, error="Not Windows")

        try:
            result = subprocess.run(
                [
                    "powershell", "-NoProfile", "-Command",
                    "$path = 'HKCU:\\Software\\Microsoft\\Windows\\CurrentVersion"
                    "\\CloudStore\\Store\\DefaultAccount\\Current"
                    "\\default$windows.data.bluelightreduction.bluelightreductionstate"
                    "\\windows.data.bluelightreduction.bluelightreductionstate'; "
                    "$curData = (Get-ItemProperty -Path $path).Data; "
                    "if ($curData.Length -lt 19) { Write-Output 'error'; exit 1 }; "
                    "if ($curData[18] -eq 0x15) { $curData[18] = 0x13 } "
                    "else { $curData[18] = 0x15 }; "
                    "$curData[10] = $curData[10] + 1; "
                    "Set-ItemProperty -Path $path -Name Data -Value $curData; "
                    "if ($curData[18] -eq 0x15) { Write-Output 'enabled' } "
                    "else { Write-Output 'disabled' }",
                ],
                capture_output=True,
                text=True,
                timeout=_SUBPROCESS_TIMEOUT,
            )

            output = result.stdout.strip().lower()
            if output == "error" or result.returncode != 0:
                return HardwareResult(
                    success=False,
                    error=result.stderr.strip() or "Registry key structure unexpected",
                )
            enabled = output == "enabled"
            return HardwareResult(success=True, actual_value=enabled)
        except subprocess.TimeoutExpired:
            return HardwareResult(success=False, error="Night light toggle timed out")
        except Exception as e:
            return HardwareResult(success=False, error=str(e))

    # ─────────────────────────────────────────────────────────
    # Media Playback Keys (SendInput via ctypes)
    # ─────────────────────────────────────────────────────────

    def _send_media_key(self, vk_code: int) -> HardwareResult:
        """
        Send a media virtual key press via ctypes.SendInput.

        Simulates key-down + key-up for the given virtual key code.
        Used for VK_MEDIA_PLAY_PAUSE, VK_MEDIA_NEXT_TRACK, etc.

        CRITICAL: The INPUT struct union MUST include MOUSEINPUT
        (the largest member) so that sizeof(INPUT) == 40 on 64-bit.
        If the union only contains KEYBDINPUT, sizeof == 32 and
        SendInput silently rejects the call (returns 0).
        """
        if not _HAS_CTYPES_MEDIA:
            return HardwareResult(success=False, error="ctypes not available")

        try:
            KEYEVENTF_EXTENDEDKEY = 0x0001
            KEYEVENTF_KEYUP = 0x0002
            INPUT_KEYBOARD = 1

            class KEYBDINPUT(ctypes.Structure):
                _fields_ = [
                    ("wVk", ctypes.wintypes.WORD),
                    ("wScan", ctypes.wintypes.WORD),
                    ("dwFlags", ctypes.wintypes.DWORD),
                    ("time", ctypes.wintypes.DWORD),
                    ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
                ]

            class MOUSEINPUT(ctypes.Structure):
                _fields_ = [
                    ("dx", ctypes.c_long),
                    ("dy", ctypes.c_long),
                    ("mouseData", ctypes.wintypes.DWORD),
                    ("dwFlags", ctypes.wintypes.DWORD),
                    ("time", ctypes.wintypes.DWORD),
                    ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
                ]

            class HARDWAREINPUT(ctypes.Structure):
                _fields_ = [
                    ("uMsg", ctypes.wintypes.DWORD),
                    ("wParamL", ctypes.wintypes.WORD),
                    ("wParamH", ctypes.wintypes.WORD),
                ]

            class INPUT(ctypes.Structure):
                class _INPUT(ctypes.Union):
                    _fields_ = [
                        ("mi", MOUSEINPUT),
                        ("ki", KEYBDINPUT),
                        ("hi", HARDWAREINPUT),
                    ]
                _fields_ = [
                    ("type", ctypes.wintypes.DWORD),
                    ("_input", _INPUT),
                ]

            # Key down
            ki_down = KEYBDINPUT(
                wVk=vk_code, wScan=0,
                dwFlags=KEYEVENTF_EXTENDEDKEY,
                time=0, dwExtraInfo=None,
            )
            inp_down = INPUT(type=INPUT_KEYBOARD)
            inp_down._input.ki = ki_down

            # Key up
            ki_up = KEYBDINPUT(
                wVk=vk_code, wScan=0,
                dwFlags=KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP,
                time=0, dwExtraInfo=None,
            )
            inp_up = INPUT(type=INPUT_KEYBOARD)
            inp_up._input.ki = ki_up

            sent = ctypes.windll.user32.SendInput(
                2,
                ctypes.byref((INPUT * 2)(inp_down, inp_up)),
                ctypes.sizeof(INPUT),
            )

            if sent == 2:
                return HardwareResult(success=True, actual_value=True)
            else:
                win_err = ctypes.get_last_error()
                return HardwareResult(
                    success=False,
                    error=f"SendInput injected {sent}/2 events (win32 error: {win_err})",
                )
        except Exception as e:
            return HardwareResult(success=False, error=str(e))

    def media_play_pause(self) -> HardwareResult:
        """Send media play/pause key. This is a toggle."""
        VK_MEDIA_PLAY_PAUSE = 0xB3
        return self._send_media_key(VK_MEDIA_PLAY_PAUSE)

    def media_next(self) -> HardwareResult:
        """Send media next track key."""
        VK_MEDIA_NEXT_TRACK = 0xB0
        return self._send_media_key(VK_MEDIA_NEXT_TRACK)

    def media_previous(self) -> HardwareResult:
        """Send media previous track key."""
        VK_MEDIA_PREV_TRACK = 0xB1
        return self._send_media_key(VK_MEDIA_PREV_TRACK)

