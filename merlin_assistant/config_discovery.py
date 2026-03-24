# merlin_assistant/config_discovery.py

"""
Config path discovery for MERLIN.

Resolves config directories and .env paths using a strict precedence chain:

  1. MERLIN_CONFIG_DIR env var  (explicit override)
  2. User config dir            (installed mode: platformdirs)
  3. CWD / config               (dev mode only: repo root detected)
  4. Package default_config     (bundled fallback templates)

Dev mode is detected by checking if CWD contains main.py + merlin.py + config/
(repo fingerprint). This prevents accidental CWD config loading when a user
runs `merlin` from a random directory like ~/Downloads.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _is_dev_mode() -> bool:
    """Detect if running from the MERLIN repo root.

    Checks for the presence of main.py, merlin.py, and config/ in CWD.
    All three must exist — this is the repo fingerprint.
    """
    cwd = Path.cwd()
    return (
        (cwd / "main.py").is_file()
        and (cwd / "merlin.py").is_file()
        and (cwd / "config").is_dir()
    )


def _get_user_config_dir() -> Path:
    """Get the platform-appropriate user config directory.

    Uses platformdirs if available, falls back to manual resolution.

    Linux:   ~/.config/merlin/
    macOS:   ~/Library/Application Support/merlin/
    Windows: %APPDATA%\\merlin\\
    """
    try:
        from platformdirs import user_config_dir
        return Path(user_config_dir("merlin", appauthor=False))
    except ImportError:
        # Fallback if platformdirs not installed
        if sys.platform == "win32":
            base = os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")
            return Path(base) / "merlin"
        elif sys.platform == "darwin":
            return Path.home() / "Library" / "Application Support" / "merlin"
        else:
            xdg = os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")
            return Path(xdg) / "merlin"


def _get_package_default_dir() -> Path:
    """Get the bundled default_config directory shipped with the package."""
    return Path(__file__).parent / "default_config"


def get_config_dir() -> Path:
    """Resolve the active config directory.

    Precedence:
      1. MERLIN_CONFIG_DIR env var
      2. User config dir (platformdirs) — if it exists
      3. CWD/config — only if dev mode detected
      4. Package default_config — bundled fallback

    Returns:
        Path to the config directory to use.
    """
    # 1. Explicit env var override
    env_override = os.environ.get("MERLIN_CONFIG_DIR")
    if env_override:
        p = Path(env_override)
        if p.is_dir():
            logger.info("Config: using MERLIN_CONFIG_DIR=%s", p)
            return p
        logger.warning("MERLIN_CONFIG_DIR=%s does not exist, falling through", p)

    # 2. User config dir (installed mode)
    user_dir = _get_user_config_dir() / "config"
    if user_dir.is_dir():
        logger.info("Config: using user config dir %s", user_dir)
        return user_dir

    # 3. CWD/config (dev mode only)
    if _is_dev_mode():
        cwd_config = Path.cwd() / "config"
        logger.info("Config: dev mode — using %s", cwd_config)
        return cwd_config

    # 4. Package defaults (last resort)
    pkg_dir = _get_package_default_dir()
    if pkg_dir.is_dir():
        logger.info("Config: using bundled defaults at %s", pkg_dir)
        return pkg_dir

    # Nothing found — return CWD/config and let caller handle the error
    logger.warning("Config: no config directory found anywhere")
    return Path.cwd() / "config"


def get_env_path() -> Path:
    """Resolve the .env file path.

    Precedence:
      1. MERLIN_ENV_FILE env var
      2. User config dir .env
      3. CWD/.env (dev mode only)

    Returns:
        Path to the .env file (may not exist yet).
    """
    # 1. Explicit override
    env_override = os.environ.get("MERLIN_ENV_FILE")
    if env_override:
        return Path(env_override)

    # 2. User config dir
    user_env = _get_user_config_dir() / ".env"
    if user_env.is_file():
        return user_env

    # 3. Dev mode
    if _is_dev_mode():
        return Path.cwd() / ".env"

    # Default to user config dir (even if doesn't exist yet)
    return user_env


def get_user_config_root() -> Path:
    """Get the root user config directory (for merlin init).

    This is the parent of config/ and .env.

    Linux:   ~/.config/merlin/
    macOS:   ~/Library/Application Support/merlin/
    Windows: %APPDATA%\\merlin\\
    """
    return _get_user_config_dir()
