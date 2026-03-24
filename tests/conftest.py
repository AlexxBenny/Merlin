# tests/conftest.py

"""
Shared pytest configuration and fixtures.

Markers:
  @pytest.mark.slow — tests requiring live infrastructure (Ollama, network).
                       Excluded by default. Run with: python -m pytest -m slow
  @pytest.mark.windows_only — tests requiring Windows OS.
                               Auto-skipped on non-Windows platforms.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path so tests can import all packages
# (cortex, world, ir, skills, etc.) without requiring pip install -e .
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: marks tests requiring live infrastructure — excluded by default",
    )
    config.addinivalue_line(
        "markers",
        "windows_only: marks tests requiring Windows OS — auto-skipped on other platforms",
    )


def pytest_collection_modifyitems(config, items):
    """Skip slow-marked and windows_only-marked tests unless appropriate."""
    # Skip slow tests unless explicitly requested via -m slow
    if config.getoption("-m") == "slow":
        pass  # User explicitly asked for slow tests
    else:
        skip_slow = pytest.mark.skip(reason="slow test — run with: python -m pytest -m slow")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

    # Skip windows_only tests on non-Windows platforms
    if sys.platform != "win32":
        skip_win = pytest.mark.skip(reason="Windows-only test — skipped on this platform")
        for item in items:
            if "windows_only" in item.keywords:
                item.add_marker(skip_win)

