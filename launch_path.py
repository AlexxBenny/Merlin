"""Tool: system.apps.launch.path

PATH-based executable launch for CLI tools.
Uses shutil.which() to resolve, then subprocess.Popen() to execute.

Category: system
Risk Level: medium
Side Effects: changes_focus, changes_ui_state, launches_process

WHEN TO USE:
- CLI tools that need command-line arguments (python --version, git status)
- Tools that are in system PATH
- When you need to pass arguments to the executable

WHEN NOT TO USE:
- GUI applications without arguments (use system.apps.launch.shell instead)
- Apps not in PATH (will fail)
"""

import subprocess
import shutil
import time
import psutil
from typing import Dict, Any, Optional
from tools.base import Tool
from tools.system.apps.utils import find_windows
from tools.system.apps.app_handle import AppHandle, HandleRegistry


class LaunchAppPath(Tool):
    """Launch CLI tool via PATH resolution and subprocess"""
    
    @property
    def name(self) -> str:
        return "system.apps.launch.path"
    
    @property
    def description(self) -> str:
        return (
            "Launches a CLI tool with arguments using PATH resolution. "
            "Use for commands like 'python --version' or 'git status'. "
            "Executable must be in system PATH."
        )
    
    @property
    def risk_level(self) -> str:
        return "medium"
        
    @property
    def side_effects(self) -> list[str]:
        return ["changes_focus", "changes_ui_state", "resource_usage"]
        
    @property
    def stabilization_time_ms(self) -> int:
        return 1000
        
    @property
    def reversible(self) -> bool:
        return True

    @property
    def requires_visual_confirmation(self) -> bool:
        return False  # CLI tools often don't need visual confirmation

    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "executable": {
                    "type": "string",
                    "description": "Name of executable (e.g., 'python', 'git', 'npm')"
                },
                "args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Command line arguments"
                },
                "wait_for_completion": {
                    "type": "boolean",
                    "default": False,
                    "description": "Wait for process to complete?"
                },
                "timeout_ms": {
                    "type": "integer",
                    "default": 30000,
                    "description": "Timeout for wait_for_completion"
                }
            },
            "required": ["executable"]
        }
    
    def execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute via PATH resolution and subprocess"""
        if not self.validate_args(args):
            return {"status": "error", "error": "Invalid arguments"}
            
        executable = args["executable"]
        cmd_args = args.get("args", [])
        wait_complete = args.get("wait_for_completion", False)
        timeout_sec = args.get("timeout_ms", 30000) / 1000.0
        
        # Resolve via PATH
        resolved_path = self._resolve_path(executable)
        if not resolved_path:
            return {
                "status": "error",
                "error": f"'{executable}' not found in PATH",
                "error_type": "environment",
                "launch_method": "path"
            }
        
        # Build command
        full_cmd = [resolved_path] + cmd_args
        
        try:
            if wait_complete:
                # Run and wait for completion
                result = subprocess.run(
                    full_cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout_sec
                )
                return {
                    "status": "success",
                    "launch_method": "path",
                    "exit_code": result.returncode,
                    "stdout": result.stdout[:1000] if result.stdout else "",
                    "stderr": result.stderr[:1000] if result.stderr else ""
                }
            else:
                # Start process without waiting
                proc = subprocess.Popen(full_cmd, shell=False)
                
                handle = AppHandle.create(executable, " ".join(full_cmd))
                HandleRegistry.register(handle)
                
                return {
                    "status": "success",
                    "launch_method": "path",
                    "pid": proc.pid,
                    "resolved_path": resolved_path,
                    "app_handle": handle.to_dict()
                }
                
        except subprocess.TimeoutExpired:
            return {
                "status": "error",
                "error": f"Process timed out after {timeout_sec}s",
                "error_type": "tool",
                "launch_method": "path"
            }
        except FileNotFoundError:
            return {
                "status": "error",
                "error": f"Executable not found: {resolved_path}",
                "error_type": "environment",
                "launch_method": "path"
            }
        except Exception as e:
            return {
                "status": "error",
                "error": f"Launch failed: {str(e)}",
                "error_type": "tool",
                "launch_method": "path"
            }
    
    def _resolve_path(self, executable: str) -> Optional[str]:
        """Resolve executable via PATH"""
        path = shutil.which(executable)
        if path:
            return path
        
        # Try with .exe on Windows
        import sys
        if sys.platform == "win32" and not executable.lower().endswith('.exe'):
            path = shutil.which(f"{executable}.exe")
            if path:
                return path
        
        return None
