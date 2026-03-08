"""AppResolver - Multi-Strategy Windows Application Resolution

JARVIS Architecture Principle: Deterministic, OS-aware app launching.
NO LLM guessing. This module resolves app names to launchable targets.

Resolution Order (non-negotiable):
1. Known protocols (spotify:, ms-settings:)
2. App Paths registry
3. Start Menu shortcuts (.lnk)
3.5. AppsFolder enumeration (UWP/Store apps)
4. Known install locations
5. Shell fallback (os.startfile)

Each strategy runs in order. First match wins.
"""

import os
import sys
import logging
import winreg
from pathlib import Path
from typing import Optional, Dict, Literal
from dataclasses import dataclass
from enum import Enum


class ResolutionMethod(Enum):
    """How the app was resolved - for diagnostics and logging"""
    PROTOCOL = "protocol"
    APP_PATHS = "app_paths"
    START_MENU = "start_menu"
    APPSFOLDER = "appsfolder"  # UWP/Store apps via shell:AppsFolder
    INSTALL_SEARCH = "install_search"
    SHELL_FALLBACK = "shell_fallback"
    FAILED = "failed"


@dataclass
class LaunchTarget:
    """Result of app resolution"""
    target_type: Literal["protocol", "executable", "shell"]
    value: str  # URI, path, or app_name
    resolution_method: ResolutionMethod
    details: Optional[str] = None  # Additional info (e.g., registry key, shortcut path)
    args: Optional[list] = None  # Optional command-line arguments (populated by launcher, not resolver)
    
    def __repr__(self) -> str:
        args_str = f" args={self.args}" if self.args else ""
        return f"LaunchTarget({self.target_type}:{self.value} via {self.resolution_method.value}{args_str})"


# Protocol aliases for non-obvious mappings
KNOWN_PROTOCOL_ALIASES = {
    "settings": "ms-settings",
    "store": "ms-windows-store",
    "mail": "mailto",
    "calculator": "calculator",
    "camera": "microsoft.windows.camera",
}


class AppResolver:
    """Multi-strategy Windows application resolver.
    
    Resolves app names to launchable targets using a deterministic,
    ordered resolution pipeline. No LLM involvement.
    """
    
    def __init__(self):
        # Resolution cache: app_name -> LaunchTarget
        # Apps don't move often, so this massively improves UX
        self._cache: Dict[str, LaunchTarget] = {}
        
        # Start Menu paths (user + system)
        self._start_menu_paths = [
            Path(os.environ.get("APPDATA", "")) / "Microsoft" / "Windows" / "Start Menu" / "Programs",
            Path(os.environ.get("PROGRAMDATA", "")) / "Microsoft" / "Windows" / "Start Menu" / "Programs",
        ]
        
        # Common install locations for Strategy 4
        self._install_paths = [
            Path(os.environ.get("LOCALAPPDATA", "")),
            Path(os.environ.get("APPDATA", "")),
            Path(os.environ.get("PROGRAMFILES", "")),
            Path(os.environ.get("PROGRAMFILES(X86)", "")),
        ]
        
        # AppsFolder cache: display_name_lower -> (AppUserModelID, display_name)
        # Populated lazily on first use, then cached
        self._appsfolder_cache: Optional[Dict[str, tuple]] = None
        
        logging.info("AppResolver initialized with 6-stage resolution pipeline")
    
    def resolve(self, app_name: str) -> LaunchTarget:
        """Resolve app name to a launchable target.
        
        Resolution order (first match wins):
        1. Protocol detection
        2. App Paths registry
        3. Start Menu shortcuts
        3.5. AppsFolder enumeration (UWP/Store apps)
        4. Known install locations
        5. Shell fallback
        
        Args:
            app_name: Application name (e.g., "spotify", "chrome")
            
        Returns:
            LaunchTarget with target_type, value, and resolution_method
        """
        app_name_lower = app_name.lower().strip()
        
        # Check cache first
        if app_name_lower in self._cache:
            cached = self._cache[app_name_lower]
            logging.debug(f"AppResolver cache hit: {app_name} -> {cached}")
            return cached
        
        # Strategy 1: Protocol detection
        target = self._try_protocol(app_name_lower)
        if target:
            self._cache[app_name_lower] = target
            logging.info(f"Resolved '{app_name}' via {target.resolution_method.value} -> {target.value}")
            return target
        
        # Strategy 2: App Paths registry
        target = self._try_app_paths(app_name_lower)
        if target:
            self._cache[app_name_lower] = target
            logging.info(f"Resolved '{app_name}' via {target.resolution_method.value} -> {target.value}")
            return target
        
        # Strategy 3: Start Menu shortcuts
        target = self._try_start_menu(app_name_lower)
        if target:
            self._cache[app_name_lower] = target
            logging.info(f"Resolved '{app_name}' via {target.resolution_method.value} -> {target.value}")
            return target
        
        # Strategy 3.5: AppsFolder enumeration (UWP/Store apps)
        target = self._try_appsfolder(app_name_lower)
        if target:
            self._cache[app_name_lower] = target
            logging.info(f"Resolved '{app_name}' via {target.resolution_method.value} -> {target.value}")
            return target
        
        # Strategy 4: Known install locations
        target = self._try_install_locations(app_name_lower)
        if target:
            self._cache[app_name_lower] = target
            logging.info(f"Resolved '{app_name}' via {target.resolution_method.value} -> {target.value}")
            return target
        
        # Strategy 5: Shell fallback (let Windows try)
        target = LaunchTarget(
            target_type="shell",
            value=app_name,
            resolution_method=ResolutionMethod.SHELL_FALLBACK,
            details="No specific resolution found, falling back to OS shell"
        )
        self._cache[app_name_lower] = target
        logging.info(f"Resolved '{app_name}' via {target.resolution_method.value} (fallback)")
        return target
    
    def _try_protocol(self, app_name: str) -> Optional[LaunchTarget]:
        """Strategy 1: Check if a protocol handler exists.
        
        Checks HKCR\{protocol} for shell\open\command
        """
        # Check alias first
        protocol_name = KNOWN_PROTOCOL_ALIASES.get(app_name, app_name)
        
        try:
            # Check if protocol exists in HKCR
            key_path = f"{protocol_name}\\shell\\open\\command"
            key = winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, key_path)
            winreg.CloseKey(key)
            
            # Protocol exists
            return LaunchTarget(
                target_type="protocol",
                value=f"{protocol_name}:",
                resolution_method=ResolutionMethod.PROTOCOL,
                details=f"HKCR\\{key_path}"
            )
        except FileNotFoundError:
            pass
        except OSError as e:
            logging.debug(f"Protocol check failed for {protocol_name}: {e}")
        
        return None
    
    def _try_app_paths(self, app_name: str) -> Optional[LaunchTarget]:
        """Strategy 2: Check App Paths registry.
        
        Checks both HKLM and HKCU for App Paths entries.
        """
        exe_name = f"{app_name}.exe" if not app_name.endswith('.exe') else app_name
        
        # Registry locations to check
        reg_locations = [
            (winreg.HKEY_LOCAL_MACHINE, f"SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\App Paths\\{exe_name}"),
            (winreg.HKEY_CURRENT_USER, f"SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\App Paths\\{exe_name}"),
        ]
        
        for hkey, key_path in reg_locations:
            try:
                key = winreg.OpenKey(hkey, key_path)
                value, _ = winreg.QueryValueEx(key, "")  # Default value is the path
                winreg.CloseKey(key)
                
                if value and os.path.exists(value.strip('"')):
                    exe_path = value.strip('"')
                    return LaunchTarget(
                        target_type="executable",
                        value=exe_path,
                        resolution_method=ResolutionMethod.APP_PATHS,
                        details=f"{'HKLM' if hkey == winreg.HKEY_LOCAL_MACHINE else 'HKCU'}\\{key_path}"
                    )
            except FileNotFoundError:
                pass
            except OSError as e:
                logging.debug(f"App Paths check failed at {key_path}: {e}")
        
        return None
    
    def _try_start_menu(self, app_name: str) -> Optional[LaunchTarget]:
        """Strategy 3: Search Start Menu shortcuts.
        
        OPTIMIZED: Only matches filenames containing app_name (no eager parsing).
        Then parses matched .lnk files with comtypes.
        """
        # Search patterns - match filenames
        search_patterns = [
            f"{app_name}.lnk",
            f"{app_name}*.lnk",
        ]
        
        matched_shortcuts = []
        
        for start_menu_path in self._start_menu_paths:
            if not start_menu_path.exists():
                continue
            
            try:
                # Find matching .lnk files (shallow and nested)
                for lnk_file in start_menu_path.rglob("*.lnk"):
                    lnk_name_lower = lnk_file.stem.lower()
                    
                    # Match: filename contains app_name or exact match
                    if app_name in lnk_name_lower or lnk_name_lower == app_name:
                        matched_shortcuts.append(lnk_file)
                        
                        # Limit matches to prevent performance issues
                        if len(matched_shortcuts) >= 5:
                            break
                
            except PermissionError:
                logging.debug(f"Permission denied scanning {start_menu_path}")
            except Exception as e:
                logging.debug(f"Start Menu scan error in {start_menu_path}: {e}")
        
        # Parse matched shortcuts to find target executable
        for shortcut_path in matched_shortcuts:
            target_exe = self._parse_shortcut(shortcut_path)
            if target_exe and os.path.exists(target_exe):
                return LaunchTarget(
                    target_type="executable",
                    value=target_exe,
                    resolution_method=ResolutionMethod.START_MENU,
                    details=str(shortcut_path)
                )
        
        return None
    
    def _parse_shortcut(self, lnk_path: Path) -> Optional[str]:
        """Parse Windows shortcut (.lnk) to get target executable.
        
        Uses comtypes.shelllink if available, falls back to shell approach.
        Wrapped in try/except - COM issues should never break launching.
        """
        try:
            # Try comtypes (already in project dependencies)
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
            logging.debug("comtypes not available for shortcut parsing")
        except Exception as e:
            logging.debug(f"Shortcut parsing failed for {lnk_path}: {e}")
        
        return None
    
    def _try_appsfolder(self, app_name: str) -> Optional[LaunchTarget]:
        """Strategy 3.5: Search AppsFolder for UWP/Store apps.
        
        Enumerates installed apps via shell:AppsFolder and matches by display name.
        This is how Windows launches Store apps like Telegram, WhatsApp, Photos.
        
        Cache is built lazily on first call, then reused.
        """
        try:
            # Build cache on first use
            if self._appsfolder_cache is None:
                self._appsfolder_cache = self._build_appsfolder_cache()
            
            # Normalize search terms
            search_terms = [app_name]
            
            # Add common variations
            if " " not in app_name:
                # "telegram" -> also try "telegram desktop", "telegram messenger"
                search_terms.extend([
                    f"{app_name} desktop",
                    f"{app_name} messenger", 
                    f"{app_name} app",
                ])
            
            # Search cache
            for term in search_terms:
                if term in self._appsfolder_cache:
                    app_id, display_name = self._appsfolder_cache[term]
                    return LaunchTarget(
                        target_type="shell",
                        value=f"shell:AppsFolder\\{app_id}",
                        resolution_method=ResolutionMethod.APPSFOLDER,
                        details=f"UWP App: {display_name}"
                    )
            
            # Try partial match (app_name appears in display name)
            for cached_name, (app_id, display_name) in self._appsfolder_cache.items():
                if app_name in cached_name:
                    return LaunchTarget(
                        target_type="shell",
                        value=f"shell:AppsFolder\\{app_id}",
                        resolution_method=ResolutionMethod.APPSFOLDER,
                        details=f"UWP App: {display_name} (partial match)"
                    )
                    
        except Exception as e:
            # COM issues should never break app launching
            logging.debug(f"AppsFolder enumeration failed: {e}")
        
        return None
    
    def _build_appsfolder_cache(self) -> Dict[str, tuple]:
        """Build cache of installed apps from shell:AppsFolder.
        
        Uses COM Shell.Application to enumerate AppsFolder.
        Returns dict: lowercase_display_name -> (AppUserModelID, original_display_name)
        """
        cache = {}
        
        try:
            from win32com.client import Dispatch
            
            shell = Dispatch("Shell.Application")
            folder = shell.NameSpace("shell:AppsFolder")
            
            if folder is None:
                logging.debug("Could not access shell:AppsFolder")
                return cache
            
            items = folder.Items()
            
            for item in items:
                try:
                    display_name = item.Name
                    app_id = item.Path  # This is the AppUserModelID
                    
                    if display_name and app_id:
                        # Normalize: lowercase, strip whitespace
                        key = display_name.lower().strip()
                        cache[key] = (app_id, display_name)
                        
                except Exception as e:
                    # Skip problematic items
                    continue
            
            logging.info(f"AppsFolder cache built: {len(cache)} apps indexed")
            
        except ImportError:
            logging.debug("win32com not available for AppsFolder enumeration")
        except Exception as e:
            logging.debug(f"Failed to build AppsFolder cache: {e}")
        
        return cache
    
    def _try_install_locations(self, app_name: str) -> Optional[LaunchTarget]:
        """Strategy 4: Search known install locations.
        
        SHALLOW SEARCH: Max depth 2, matches folder/exe name only.
        This is a fallback, not a crawler.
        """
        exe_name = f"{app_name}.exe"
        
        for install_root in self._install_paths:
            if not install_root.exists():
                continue
            
            try:
                # Depth 1: Check for {app_name}/{app_name}.exe or {app_name}/*.exe
                app_folder = install_root / app_name
                if app_folder.is_dir():
                    # Check for exact exe match
                    exe_path = app_folder / exe_name
                    if exe_path.exists():
                        return LaunchTarget(
                            target_type="executable",
                            value=str(exe_path),
                            resolution_method=ResolutionMethod.INSTALL_SEARCH,
                            details=f"Found at {app_folder}"
                        )
                    
                    # Depth 2: Check one level deeper for common patterns
                    for subdir in app_folder.iterdir():
                        if subdir.is_dir():
                            exe_path = subdir / exe_name
                            if exe_path.exists():
                                return LaunchTarget(
                                    target_type="executable",
                                    value=str(exe_path),
                                    resolution_method=ResolutionMethod.INSTALL_SEARCH,
                                    details=f"Found at {subdir}"
                                )
                
                # Also check for folders that START with app_name (e.g., "Spotify" folder)
                for folder in install_root.iterdir():
                    if folder.is_dir() and folder.name.lower().startswith(app_name):
                        exe_path = folder / exe_name
                        if exe_path.exists():
                            return LaunchTarget(
                                target_type="executable",
                                value=str(exe_path),
                                resolution_method=ResolutionMethod.INSTALL_SEARCH,
                                details=f"Found at {folder}"
                            )
                        # Also check for any .exe matching app name in folder root
                        for f in folder.iterdir():
                            if f.is_file() and f.suffix.lower() == '.exe' and app_name in f.stem.lower():
                                return LaunchTarget(
                                    target_type="executable",
                                    value=str(f),
                                    resolution_method=ResolutionMethod.INSTALL_SEARCH,
                                    details=f"Found at {folder}"
                                )
                                
            except PermissionError:
                logging.debug(f"Permission denied scanning {install_root}")
            except Exception as e:
                logging.debug(f"Install location scan error in {install_root}: {e}")
        
        return None
    
    def clear_cache(self) -> None:
        """Clear the resolution cache (useful after app install/uninstall)."""
        self._cache.clear()
        logging.info("AppResolver cache cleared")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics for diagnostics."""
        by_method = {}
        for target in self._cache.values():
            method = target.resolution_method.value
            by_method[method] = by_method.get(method, 0) + 1
        return {
            "total_cached": len(self._cache),
            "by_method": by_method
        }


# Module-level singleton for consistent caching
_resolver: Optional[AppResolver] = None


def get_app_resolver() -> AppResolver:
    """Get the singleton AppResolver instance."""
    global _resolver
    if _resolver is None:
        _resolver = AppResolver()
    return _resolver
