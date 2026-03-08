# infrastructure/app_discovery.py

"""
ApplicationDiscoveryService — Boot-time OS application scanner.

Scans the Windows OS environment to discover installed applications
and constructs ApplicationEntity objects with multiple launch strategies.

Design rules:
- Runs at boot, NOT at launch time
- Produces ApplicationEntity objects and hands them to ApplicationRegistry
- Merges results: one entity can have strategies from multiple sources
- Binds AppCapabilities during entity construction
- Never called during mission execution
- All OS access wrapped in try/except — failures never crash boot

Resolution strategies (adapted from app_resolver.py reference):
1. Protocol handlers (HKCR registry)
2. App Paths registry (HKLM/HKCU)
3. Start Menu shortcuts (.lnk files)
4. AppsFolder enumeration (UWP/Store apps)
5. Known install locations
6. CLI PATH resolution (shutil.which)
"""

import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from infrastructure.application_registry import (
    ApplicationEntity,
    LaunchStrategy,
    ResolutionMethod,
    STRATEGY_DEFAULTS,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Protocol aliases for non-obvious app→protocol mappings
# ─────────────────────────────────────────────────────────────

KNOWN_PROTOCOL_ALIASES: Dict[str, str] = {
    "settings": "ms-settings",
    "store": "ms-windows-store",
    "mail": "mailto",
    "calculator": "calculator",
    "camera": "microsoft.windows.camera",
}


# ─────────────────────────────────────────────────────────────
# Builder: accumulates partial entity data across strategies
# ─────────────────────────────────────────────────────────────

class _EntityBuilder:
    """Accumulates discovery results for a single app_id."""

    def __init__(self, app_id: str):
        self.app_id = app_id
        self.display_names: Set[str] = set()
        self.strategies: List[LaunchStrategy] = []
        self.executables: Set[str] = set()
        self.protocols: Set[str] = set()
        self.process_names: Set[str] = set()
        self.install_locations: Set[str] = set()
        self.is_uwp: bool = False

    def add_strategy(self, strategy: LaunchStrategy) -> None:
        # Avoid duplicate strategies (same type + normalized value)
        for existing in self.strategies:
            if existing.type == strategy.type:
                # For executables, normalize path case (Windows is case-insensitive)
                if existing.type == "executable":
                    if existing.value.lower() == strategy.value.lower():
                        return
                elif existing.value == strategy.value:
                    return
        self.strategies.append(strategy)

    def add_executable(self, exe_path: str) -> None:
        self.executables.add(exe_path)
        # Derive canonical process name from executable
        exe_name = os.path.basename(exe_path)
        if exe_name:
            self.process_names.add(exe_name.lower())

    def add_display_name(self, name: str) -> None:
        self.display_names.add(name)

    def build(self, capabilities: Any = None) -> ApplicationEntity:
        """Construct the final ApplicationEntity."""
        # Sort strategies by priority descending
        sorted_strategies = sorted(
            self.strategies, key=lambda s: -s.priority,
        )

        # If no process names were derived from executables,
        # infer from app_id
        if not self.process_names:
            self.process_names.add(f"{self.app_id}.exe")

        # Ensure app_id is in display_names
        self.display_names.add(self.app_id)

        return ApplicationEntity(
            app_id=self.app_id,
            display_names=sorted(self.display_names),
            launch_strategies=sorted_strategies,
            executables=sorted(self.executables),
            protocols=sorted(self.protocols),
            canonical_process_names=sorted(self.process_names),
            install_locations=sorted(self.install_locations),
            capabilities=capabilities,
            is_uwp=self.is_uwp,
        )


# ─────────────────────────────────────────────────────────────
# ApplicationDiscoveryService
# ─────────────────────────────────────────────────────────────

class ApplicationDiscoveryService:
    """Boot-time OS scanner. Discovers installed applications.

    Usage:
        discovery = ApplicationDiscoveryService(capability_registry)
        entities = discovery.discover_all()
        for entity in entities:
            registry.register(entity)
    """

    def __init__(self, capability_registry=None):
        """
        Args:
            capability_registry: AppCapabilityRegistry for binding
                                 capabilities to discovered entities.
        """
        self._cap_registry = capability_registry

        # Start Menu paths (user + system)
        self._start_menu_paths = [
            Path(os.environ.get("APPDATA", ""))
            / "Microsoft" / "Windows" / "Start Menu" / "Programs",
            Path(os.environ.get("PROGRAMDATA", ""))
            / "Microsoft" / "Windows" / "Start Menu" / "Programs",
        ]

        # Common install locations
        self._install_paths = [
            Path(os.environ.get("LOCALAPPDATA", "")),
            Path(os.environ.get("APPDATA", "")),
            Path(os.environ.get("PROGRAMFILES", "")),
            Path(os.environ.get("PROGRAMFILES(X86)", "")),
        ]

    def discover_all(self) -> List[ApplicationEntity]:
        """Run all discovery strategies and produce merged entities.

        Each strategy adds to a shared builder map (app_id → builder).
        After all strategies run, builders produce final entities
        with capabilities bound.
        """
        builders: Dict[str, _EntityBuilder] = {}

        logger.info("ApplicationDiscovery: starting boot-time scan...")

        # Strategy 1: Protocol handlers
        count = self._scan_protocols(builders)
        logger.info("  Protocols: %d entries", count)

        # Strategy 2: App Paths registry
        count = self._scan_app_paths(builders)
        logger.info("  App Paths: %d entries", count)

        # Strategy 3: Start Menu shortcuts
        count = self._scan_start_menu(builders)
        logger.info("  Start Menu: %d entries", count)

        # Strategy 3.5: AppsFolder (UWP/Store apps)
        count = self._scan_appsfolder(builders)
        logger.info("  AppsFolder: %d entries", count)

        # Strategy 4: Known install locations
        count = self._scan_install_locations(builders)
        logger.info("  Install locations: %d entries", count)

        # Strategy 5: CLI PATH resolution
        count = self._scan_cli_path(builders)
        logger.info("  CLI PATH: %d entries", count)

        # Build final entities with capabilities
        entities = []
        for app_id, builder in builders.items():
            caps = None
            if self._cap_registry:
                caps = self._cap_registry.get(app_id)
            entity = builder.build(capabilities=caps)
            entities.append(entity)

        logger.info(
            "ApplicationDiscovery complete: %d entities from %d strategies total",
            len(entities),
            sum(len(b.strategies) for b in builders.values()),
        )

        return entities

    # ── Helper: get or create builder ─────────────────────────

    @staticmethod
    def _get_builder(
        builders: Dict[str, _EntityBuilder], app_id: str,
    ) -> _EntityBuilder:
        """Get existing builder or create a new one."""
        if app_id not in builders:
            builders[app_id] = _EntityBuilder(app_id)
        return builders[app_id]

    # Known multi-word name → canonical app_id mappings.
    # These handle common cases where display name ≠ canonical ID.
    _CANONICAL_NAMES: Dict[str, str] = {
        "google chrome": "chrome",
        "mozilla firefox": "firefox",
        "visual studio code": "vscode",
        "visual studio": "visual_studio",
        "microsoft edge": "msedge",
        "microsoft teams": "teams",
        "microsoft outlook": "outlook",
        "microsoft word": "word",
        "microsoft excel": "excel",
        "microsoft powerpoint": "powerpoint",
        "microsoft onenote": "onenote",
        "windows terminal": "windowsterminal",
        "task manager": "taskmgr",
        "control panel": "control",
        "file explorer": "explorer",
        "command prompt": "cmd",
        "windows powershell": "powershell",
    }

    @staticmethod
    def _derive_app_id(name: str) -> str:
        """Derive canonical app_id from a display name or filename.

        Rules:
        1. Lowercase + strip whitespace
        2. Strip .exe, .lnk suffixes
        3. Check canonical name table for known multi-word → ID mappings
        4. Return normalized name

        Examples:
            'Chrome.exe'         → 'chrome'
            'Google Chrome'      → 'chrome'
            'Notepad.lnk'        → 'notepad'
            'Visual Studio Code' → 'vscode'
            'Spotify'            → 'spotify'
        """
        app_id = name.lower().strip()
        for suffix in (".exe", ".lnk"):
            if app_id.endswith(suffix):
                app_id = app_id[: -len(suffix)]
        app_id = app_id.strip()

        # Check canonical name table
        if app_id in ApplicationDiscoveryService._CANONICAL_NAMES:
            return ApplicationDiscoveryService._CANONICAL_NAMES[app_id]

        return app_id

    # ── Strategy 1: Protocol handlers ─────────────────────────

    def _scan_protocols(self, builders: Dict[str, _EntityBuilder]) -> int:
        """Scan HKCR for protocol handlers."""
        if sys.platform != "win32":
            return 0

        count = 0
        try:
            import winreg

            # Check known protocol aliases first
            for app_name, protocol_name in KNOWN_PROTOCOL_ALIASES.items():
                try:
                    key_path = f"{protocol_name}\\shell\\open\\command"
                    key = winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, key_path)
                    winreg.CloseKey(key)

                    app_id = self._derive_app_id(app_name)
                    builder = self._get_builder(builders, app_id)
                    builder.protocols.add(f"{protocol_name}:")
                    builder.add_display_name(app_name)
                    defaults = STRATEGY_DEFAULTS.get("protocol", {})
                    builder.add_strategy(LaunchStrategy(
                        type="protocol",
                        value=f"{protocol_name}:",
                        method=ResolutionMethod.PROTOCOL,
                        priority=defaults.get("priority", 80),
                        reliability_score=defaults.get("reliability", 70),
                        details=f"HKCR\\{key_path}",
                    ))
                    count += 1
                except (FileNotFoundError, OSError):
                    pass

            # Also check common app names as protocol names
            common_protocol_apps = [
                "spotify", "discord", "slack", "steam", "telegram",
                "zoom", "teams", "skype", "whatsapp",
            ]
            for app_name in common_protocol_apps:
                if app_name in KNOWN_PROTOCOL_ALIASES:
                    continue
                try:
                    key_path = f"{app_name}\\shell\\open\\command"
                    key = winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, key_path)
                    winreg.CloseKey(key)

                    builder = self._get_builder(builders, app_name)
                    builder.protocols.add(f"{app_name}:")
                    builder.add_display_name(app_name)
                    defaults = STRATEGY_DEFAULTS.get("protocol", {})
                    builder.add_strategy(LaunchStrategy(
                        type="protocol",
                        value=f"{app_name}:",
                        method=ResolutionMethod.PROTOCOL,
                        priority=defaults.get("priority", 80),
                        reliability_score=defaults.get("reliability", 70),
                        details=f"HKCR\\{key_path}",
                    ))
                    count += 1
                except (FileNotFoundError, OSError):
                    pass

        except ImportError:
            logger.debug("winreg not available — skipping protocol scan")
        except Exception as e:
            logger.warning("Protocol scan failed: %s", e)

        return count

    # ── Strategy 2: App Paths registry ────────────────────────

    def _scan_app_paths(self, builders: Dict[str, _EntityBuilder]) -> int:
        """Scan HKLM/HKCU App Paths for registered executables."""
        if sys.platform != "win32":
            return 0

        count = 0
        try:
            import winreg

            base_key = "SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\App Paths"

            for hkey, hkey_name in [
                (winreg.HKEY_LOCAL_MACHINE, "HKLM"),
                (winreg.HKEY_CURRENT_USER, "HKCU"),
            ]:
                try:
                    key = winreg.OpenKey(hkey, base_key)
                except FileNotFoundError:
                    continue

                try:
                    i = 0
                    while True:
                        try:
                            subkey_name = winreg.EnumKey(key, i)
                            i += 1
                        except OSError:
                            break

                        try:
                            subkey = winreg.OpenKey(key, subkey_name)
                            value, _ = winreg.QueryValueEx(subkey, "")
                            winreg.CloseKey(subkey)

                            if value:
                                exe_path = value.strip('"')
                                if os.path.exists(exe_path):
                                    app_id = self._derive_app_id(subkey_name)
                                    builder = self._get_builder(builders, app_id)
                                    builder.add_executable(exe_path)
                                    builder.add_display_name(
                                        subkey_name.replace(".exe", "")
                                    )
                                    builder.install_locations.add(
                                        os.path.dirname(exe_path)
                                    )
                                    defaults = STRATEGY_DEFAULTS.get("executable", {})
                                    builder.add_strategy(LaunchStrategy(
                                        type="executable",
                                        value=exe_path,
                                        method=ResolutionMethod.APP_PATHS,
                                        priority=defaults.get("priority", 100),
                                        reliability_score=defaults.get("reliability", 100),
                                        details=f"{hkey_name}\\{base_key}\\{subkey_name}",
                                    ))
                                    count += 1
                        except (FileNotFoundError, OSError):
                            pass
                finally:
                    winreg.CloseKey(key)

        except ImportError:
            logger.debug("winreg not available — skipping App Paths scan")
        except Exception as e:
            logger.warning("App Paths scan failed: %s", e)

        return count

    # ── Strategy 3: Start Menu shortcuts ──────────────────────

    def _scan_start_menu(self, builders: Dict[str, _EntityBuilder]) -> int:
        """Scan Start Menu for .lnk shortcuts and resolve targets."""
        count = 0

        for start_menu_path in self._start_menu_paths:
            if not start_menu_path.exists():
                continue

            try:
                for lnk_file in start_menu_path.rglob("*.lnk"):
                    target_exe = self._parse_shortcut(lnk_file)
                    if target_exe and os.path.exists(target_exe):
                        app_id = self._derive_app_id(lnk_file.stem)
                        builder = self._get_builder(builders, app_id)
                        builder.add_executable(target_exe)
                        builder.add_display_name(lnk_file.stem)
                        builder.install_locations.add(
                            os.path.dirname(target_exe)
                        )
                        defaults = STRATEGY_DEFAULTS.get("executable", {})
                        builder.add_strategy(LaunchStrategy(
                            type="executable",
                            value=target_exe,
                            method=ResolutionMethod.START_MENU,
                            priority=defaults.get("priority", 100),
                            reliability_score=defaults.get("reliability", 100),
                            details=str(lnk_file),
                        ))
                        count += 1
            except PermissionError:
                logger.debug("Permission denied scanning %s", start_menu_path)
            except Exception as e:
                logger.debug("Start Menu scan error in %s: %s", start_menu_path, e)

        return count

    @staticmethod
    def _parse_shortcut(lnk_path: Path) -> Optional[str]:
        """Parse .lnk shortcut to get target executable path.

        Uses comtypes if available, wrapped in try/except.
        COM issues never crash discovery.
        """
        try:
            from comtypes.client import CreateObject
            from comtypes import CoInitialize, CoUninitialize

            CoInitialize()
            try:
                shell = CreateObject("WScript.Shell")
                shortcut = shell.CreateShortcut(str(lnk_path))
                target_path = shortcut.TargetPath
                if target_path:
                    return target_path
            finally:
                CoUninitialize()

        except ImportError:
            pass
        except Exception as e:
            logger.debug("Shortcut parse failed for %s: %s", lnk_path, e)

        return None

    # ── Strategy 3.5: AppsFolder (UWP/Store apps) ─────────────

    def _scan_appsfolder(self, builders: Dict[str, _EntityBuilder]) -> int:
        """Enumerate shell:AppsFolder for UWP/Store apps."""
        count = 0

        try:
            from win32com.client import Dispatch

            shell = Dispatch("Shell.Application")
            folder = shell.NameSpace("shell:AppsFolder")

            if folder is None:
                logger.debug("Could not access shell:AppsFolder")
                return 0

            items = folder.Items()
            for item in items:
                try:
                    display_name = item.Name
                    app_user_model_id = item.Path

                    if not display_name or not app_user_model_id:
                        continue

                    app_id = self._derive_app_id(display_name)
                    builder = self._get_builder(builders, app_id)
                    builder.add_display_name(display_name)
                    builder.is_uwp = True
                    defaults = STRATEGY_DEFAULTS.get("appsfolder", {})
                    builder.add_strategy(LaunchStrategy(
                        type="appsfolder",
                        value=f"shell:AppsFolder\\{app_user_model_id}",
                        method=ResolutionMethod.APPSFOLDER,
                        priority=defaults.get("priority", 90),
                        reliability_score=defaults.get("reliability", 90),
                        details=f"UWP: {display_name}",
                    ))
                    count += 1
                except Exception:
                    continue

        except ImportError:
            logger.debug("win32com not available — skipping AppsFolder scan")
        except Exception as e:
            logger.debug("AppsFolder scan failed: %s", e)

        return count

    # ── Strategy 4: Known install locations ────────────────────

    def _scan_install_locations(
        self, builders: Dict[str, _EntityBuilder],
    ) -> int:
        """Shallow search of common install directories."""
        count = 0

        for install_root in self._install_paths:
            if not install_root.exists():
                continue

            try:
                for folder in install_root.iterdir():
                    if not folder.is_dir():
                        continue

                    folder_lower = folder.name.lower()

                    # Look for {folder_name}/{folder_name}.exe
                    exe_path = folder / f"{folder_lower}.exe"
                    if exe_path.exists():
                        app_id = self._derive_app_id(folder.name)
                        builder = self._get_builder(builders, app_id)
                        builder.add_executable(str(exe_path))
                        builder.add_display_name(folder.name)
                        builder.install_locations.add(str(folder))
                        defaults = STRATEGY_DEFAULTS.get("executable", {})
                        builder.add_strategy(LaunchStrategy(
                            type="executable",
                            value=str(exe_path),
                            method=ResolutionMethod.INSTALL_SEARCH,
                            priority=defaults.get("priority", 100),
                            # Lower reliability — install location match
                            # is less precise than registry
                            reliability_score=defaults.get("reliability", 100) - 20,
                            details=f"Install dir: {folder}",
                        ))
                        count += 1
                        continue

                    # Depth 2: check one level deeper
                    try:
                        for subdir in folder.iterdir():
                            if not subdir.is_dir():
                                continue
                            exe_path = subdir / f"{folder_lower}.exe"
                            if exe_path.exists():
                                app_id = self._derive_app_id(folder.name)
                                builder = self._get_builder(builders, app_id)
                                builder.add_executable(str(exe_path))
                                builder.add_display_name(folder.name)
                                builder.install_locations.add(str(subdir))
                                defaults = STRATEGY_DEFAULTS.get("executable", {})
                                builder.add_strategy(LaunchStrategy(
                                    type="executable",
                                    value=str(exe_path),
                                    method=ResolutionMethod.INSTALL_SEARCH,
                                    priority=defaults.get("priority", 100),
                                    reliability_score=defaults.get("reliability", 100) - 30,
                                    details=f"Install subdir: {subdir}",
                                ))
                                count += 1
                                break
                    except PermissionError:
                        pass

            except PermissionError:
                logger.debug("Permission denied scanning %s", install_root)
            except Exception as e:
                logger.debug("Install scan error in %s: %s", install_root, e)

        return count

    # ── Strategy 5: CLI PATH resolution ───────────────────────

    def _scan_cli_path(self, builders: Dict[str, _EntityBuilder]) -> int:
        """Check PATH for well-known CLI tools."""
        count = 0

        # Common CLI-launchable apps
        cli_apps = [
            "notepad", "calc", "mspaint", "explorer", "cmd",
            "powershell", "code", "git", "python", "node", "npm",
        ]

        for app_name in cli_apps:
            exe_path = shutil.which(app_name)
            if exe_path:
                app_id = self._derive_app_id(app_name)
                builder = self._get_builder(builders, app_id)
                builder.add_executable(exe_path)
                builder.add_display_name(app_name)
                defaults = STRATEGY_DEFAULTS.get("executable", {})
                builder.add_strategy(LaunchStrategy(
                    type="executable",
                    value=exe_path,
                    method=ResolutionMethod.CLI_PATH,
                    priority=defaults.get("priority", 100),
                    reliability_score=defaults.get("reliability", 100),
                    details=f"shutil.which('{app_name}')",
                ))
                count += 1

        return count
