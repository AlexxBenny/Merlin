"""Tool: system.apps.launch.shell

OS-native shell launch for GUI applications.
Uses multi-strategy resolution via AppResolver, then launches appropriately.

Category: system
Risk Level: medium
Side Effects: changes_focus, changes_ui_state, launches_process

WHEN TO USE:
- Launching GUI applications by name (chrome, notepad, spotify)
- When no command-line arguments are needed
- Default choice for user-facing application launches

WHEN NOT TO USE:
- CLI tools that need arguments (use system.apps.launch.path instead)
- When explicit executable path is provided
"""

import os
import time
import subprocess
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from urllib.parse import quote as url_encode
from tools.base import Tool
from tools.system.apps.utils import find_windows
from tools.system.apps.app_handle import AppHandle, HandleRegistry
from tools.system.apps.app_resolver import get_app_resolver, LaunchTarget, ResolutionMethod


# ===== App-specific launch configuration =====
# Loaded from config/apps.yaml - determines default args for browsers, etc.
# This logic lives HERE (not in app_resolver) per separation of concerns:
#   - AppResolver: finds WHAT to launch
#   - LaunchShell: decides HOW to launch it (args, flags, profiles)

def _load_app_config() -> Dict[str, Any]:
    """Load app-specific configuration from config/apps.yaml"""
    config_path = Path(__file__).parent.parent.parent.parent / "config" / "apps.yaml"
    if config_path.exists():
        try:
            import yaml
            with open(config_path, encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
                logging.debug(f"Loaded app config from {config_path}")
                return config
        except Exception as e:
            logging.warning(f"Failed to load app config: {e}")
    return {}


# Load config at module level (cached)
_APP_CONFIG: Optional[Dict[str, Any]] = None

def _get_app_config() -> Dict[str, Any]:
    """Get cached app config (loads lazily)"""
    global _APP_CONFIG
    if _APP_CONFIG is None:
        _APP_CONFIG = _load_app_config()
    return _APP_CONFIG


class LaunchAppShell(Tool):
    """Launch GUI application via OS shell (os.startfile)"""
    
    @property
    def name(self) -> str:
        return "system.apps.launch.shell"
    
    @property
    def description(self) -> str:
        return (
            "Launches a GUI application using OS-native shell resolution. "
            "Works with apps registered in Windows (Chrome, Spotify, etc.). "
            "Do NOT use for CLI tools with arguments."
        )
    
    @property
    def risk_level(self) -> str:
        return "medium"
        
    @property
    def side_effects(self) -> list[str]:
        return ["changes_focus", "changes_ui_state", "resource_usage"]
        
    @property
    def stabilization_time_ms(self) -> int:
        return 2000
        
    @property
    def reversible(self) -> bool:
        return True

    @property
    def requires_visual_confirmation(self) -> bool:
        return True

    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "app_name": {
                    "type": "string",
                    "description": "Name of application to launch (e.g., 'chrome', 'notepad', 'spotify'). For search, can be a search engine name like 'youtube', 'google'."
                },
                "url": {
                    "type": "string",
                    "description": "Optional URL to open in browser (e.g., 'https://google.com', 'example.com'). Ignored if search_query is provided."
                },
                "search_query": {
                    "type": "string",
                    "description": "Optional search query to perform (e.g., 'nvidia drivers', 'weather'). Takes precedence over url."
                },
                "wait_for_window": {
                    "type": "boolean",
                    "default": True,
                    "description": "Wait for a visible window to appear?"
                },
                "timeout_ms": {
                    "type": "integer",
                    "default": 10000,
                    "description": "Timeout for wait_for_window"
                }
            },
            "required": ["app_name"]
        }
    
    def execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute app launch using multi-strategy resolution."""
        if not self.validate_args(args):
            return {
                "status": "error",
                "error": "Invalid arguments",
                "failure_class": "logical"  # Invalid input
            }
        
        app_name = args["app_name"]
        url = args.get("url")
        search_query = args.get("search_query")
        wait = args.get("wait_for_window", True)
        timeout_sec = args.get("timeout_ms", 10000) / 1000.0
        
        config = _get_app_config()
        search_config = config.get("search", {})
        search_engines = search_config.get("engines", {})
        browsers = config.get("browsers", {})
        
        # Determine effective app name and search engine
        effective_app_name = app_name.lower()
        search_engine = None
        
        # RULE: Global search is the default
        # Only use site-specific search if app_name is a KNOWN search engine
        # This prevents "search viswajyothi college" from becoming viswajyothi.ac.in/search?q=...
        
        if effective_app_name in search_engines:
            # app_name is a known search engine (youtube, google, duckduckgo, etc.)
            search_engine = effective_app_name
            effective_app_name = search_config.get("default_browser", "chrome")
            logging.info(f"'{app_name}' is a known search engine, using browser '{effective_app_name}'")
        
        elif search_query and effective_app_name not in browsers:
            # search_query is provided but app_name is NOT a browser or known search engine
            # → Default to Google search on default browser
            # This is the fix for "search viswajyothi college" which should NOT
            # try to open viswajyothi.ac.in/search?q=...
            search_engine = "google"  # Default to Google
            effective_app_name = search_config.get("default_browser", "chrome")
            logging.info(f"Search query provided with unknown app '{app_name}', defaulting to Google search")
        
        # Resolve via multi-strategy pipeline
        resolver = get_app_resolver()
        target = resolver.resolve(effective_app_name)
        
        logging.info(f"Resolved '{effective_app_name}' via {target.resolution_method.value} -> {target.value}")
        
        # Apply app-specific args (browser profile, URL, search)
        # NOTE: Args are only applied to executable targets, not protocols
        target = self._apply_browser_args(
            effective_app_name, 
            target, 
            url=url, 
            search_query=search_query,
            search_engine=search_engine
        )
        
        # Execute based on target type
        success, error = self._execute_target(target)
        if not success:
            # Determine failure class from error
            # FileNotFoundError → logical (app not found)
            # OSError with "Access is denied" → permission
            # Other OSError → environmental (transient OS state)
            if "Access is denied" in error or "Permission" in error:
                failure_class = "permission"
            elif "No valid launch method" in error or "not found" in error.lower():
                failure_class = "logical"  # App doesn't exist
            else:
                failure_class = "environmental"  # Transient OS state
            return {
                "status": "error",
                "error": error,
                "error_type": "environment",
                "failure_class": failure_class,
                "launch_method": target.resolution_method.value,
                "resolution_details": target.details
            }
        
        # Wait for window if requested
        if wait:
            result = self._wait_for_window(effective_app_name, timeout_sec)
            result["resolution_method"] = target.resolution_method.value
            if target.args:
                result["browser_args"] = target.args
            return result
        else:
            handle = AppHandle.create(effective_app_name, f"{target.resolution_method.value}:{target.value}")
            HandleRegistry.register(handle)
            return {
                "status": "success",
                "launch_method": target.resolution_method.value,
                "resolution_target": target.value,
                "app_handle": handle.to_dict(),
                "browser_args": target.args,
                "note": f"Launched via {target.resolution_method.value}, did not wait for window"
            }
    
    def _apply_browser_args(
        self, 
        app_name: str, 
        target: LaunchTarget,
        url: Optional[str] = None,
        search_query: Optional[str] = None,
        search_engine: Optional[str] = None
    ) -> LaunchTarget:
        """Apply browser-specific launch arguments.
        
        Handles:
        1. Browser profile flags (e.g., --profile-directory=Default)
        2. URL opening (if provided)
        3. Search query → URL construction
        4. Search engine detection (youtube → youtube URL template)
        
        CONSTRAINTS:
        - search_query wins over url (never both)
        - Only browsers accept url/search_query
        - URL encoding is strict (urllib.parse.quote)
        
        SAFETY RULE: Args are ONLY applied to executable targets.
        Protocol targets (spotify:, ms-settings:) do NOT accept CLI args.
        """
        # Only executable targets can have args
        if target.target_type != "executable":
            if url or search_query:
                logging.debug(f"Ignoring url/search_query for non-executable target: {target.value}")
            return target
        
        config = _get_app_config()
        browsers = config.get("browsers", {})
        search_config = config.get("search", {})
        
        # Check if this is a known browser
        for browser_name, browser_config in browsers.items():
            patterns = browser_config.get("executable_patterns", [])
            
            target_value_lower = target.value.lower() if target.value else ""
            app_name_lower = app_name.lower()
            
            is_browser = (
                app_name_lower == browser_name or
                any(p.lower() in target_value_lower for p in patterns)
            )
            
            if is_browser:
                # Start with browser's default args (profile)
                final_args = list(browser_config.get("default_args", []))
                
                # Construct URL from search_query or search_engine
                final_url = None
                
                # CONSTRAINT 1: search_query wins over url
                if search_query:
                    if url:
                        logging.info(f"search_query provided, ignoring explicit url")
                    
                    # Use specific search engine if provided, else default to google
                    engine = search_engine or "google"
                    engine_template = search_config.get("engines", {}).get(
                        engine,
                        "https://www.google.com/search?q={query}"
                    )
                    
                    # CONSTRAINT 3: Strict URL encoding
                    encoded_query = url_encode(search_query, safe='')
                    final_url = engine_template.format(query=encoded_query)
                    logging.info(f"Search '{search_query}' on {engine} -> {final_url}")
                    
                elif url:
                    # Ensure URL has protocol
                    final_url = url
                    if not final_url.startswith(('http://', 'https://')):
                        final_url = f"https://{final_url}"
                    logging.info(f"Opening URL: {final_url}")
                
                # Append URL if we have one
                if final_url:
                    final_args.append(final_url)
                
                if final_args:
                    logging.info(f"Browser args: {final_args}")
                    return LaunchTarget(
                        target_type=target.target_type,
                        value=target.value,
                        resolution_method=target.resolution_method,
                        details=target.details,
                        args=final_args
                    )
                
                return target
        
        # CONSTRAINT 2: Non-browser apps ignore url/search_query
        if url or search_query:
            logging.debug(f"Ignoring url/search_query for non-browser app: {app_name}")
        
        return target
    
    def _execute_target(self, target: LaunchTarget) -> tuple[bool, str | None]:
        """Execute the resolved launch target.
        
        SAFETY RULE: Args are ONLY passed to executable targets.
        Protocol and shell targets ignore args.
        
        Returns:
            (success, error_message)
        """
        try:
            if target.target_type == "protocol":
                # Protocol URIs do NOT accept CLI arguments
                if target.args:
                    logging.debug(f"Ignoring args for protocol target: {target.value}")
                os.startfile(target.value)
                logging.info(f"Launched protocol: {target.value}")
                return True, None
                
            elif target.target_type == "executable":
                # Build command with optional args
                cmd = [target.value]
                if target.args:
                    cmd.extend(target.args)
                    logging.info(f"Launching {target.value} with args: {target.args}")
                
                subprocess.Popen(
                    cmd,
                    shell=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                logging.info(f"Launched executable: {target.value}")
                return True, None
                
            elif target.target_type == "shell":
                # Shell targets (including AppsFolder) do NOT accept CLI arguments
                if target.args:
                    logging.debug(f"Ignoring args for shell target: {target.value}")
                os.startfile(target.value)
                logging.info(f"Launched via shell: {target.value}")
                return True, None
                
            else:
                return False, f"Unknown target type: {target.target_type}"
                
        except FileNotFoundError:
            error = (
                f"No valid launch method found for '{target.value}'.\n"
                f"Tried: protocol, App Paths registry, Start Menu, common install paths.\n"
                f"Resolution method: {target.resolution_method.value}"
            )
            logging.warning(error)
            return False, error
            
        except PermissionError:
            error = f"Permission denied launching '{target.value}'"
            logging.warning(error)
            return False, error
        except OSError as e:
            if "Access is denied" in str(e):
                error = f"Permission denied launching '{target.value}'"
            else:
                error = f"OS error launching '{target.value}': {e}"
            logging.warning(error)
            return False, error
            
        except Exception as e:
            error = f"Launch failed for '{target.value}': {e}"
            logging.error(error)
            return False, error
    
    def _wait_for_window(self, app_name: str, timeout_sec: float) -> Dict[str, Any]:
        """Wait for window after shell launch"""
        start_time = time.time()
        found_window = None
        
        while time.time() - start_time < timeout_sec:
            matches = find_windows(app_name=app_name)
            if matches:
                found_window = matches[0]
                break
            time.sleep(0.5)
        
        if found_window:
            handle = AppHandle.create(app_name, f"shell:{app_name}")
            handle.bind_window(
                hwnd=found_window["hwnd"],
                pid=found_window["pid"],
                title=found_window["title"]
            )
            HandleRegistry.register(handle)
            
            return {
                "status": "success",
                "launch_method": "shell",
                "window": {
                    "title": found_window["title"],
                    "hwnd": found_window["hwnd"]
                },
                "app_handle": handle.to_dict()
            }
        else:
            return {
                "status": "partial",
                "launch_method": "shell",
                "note": "Application launched but window not detected within timeout"
            }
