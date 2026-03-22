# tests/conftest.py

"""
Shared pytest configuration and fixtures.

Markers:
  @pytest.mark.slow — tests requiring live infrastructure (Ollama, network).
                       Excluded by default. Run with: python -m pytest -m slow
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


def pytest_collection_modifyitems(config, items):
    """Skip slow-marked tests unless explicitly requested via -m slow."""
    if config.getoption("-m") == "slow":
        return  # User explicitly asked for slow tests
    skip_slow = pytest.mark.skip(reason="slow test — run with: python -m pytest -m slow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
