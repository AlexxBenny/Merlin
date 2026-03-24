# tests/test_application_registry.py

"""
Tests for ApplicationEntity, ApplicationRegistry, and ApplicationDiscoveryService.
"""

import pytest
from unittest.mock import MagicMock, patch

from infrastructure.application_registry import (
    ApplicationEntity,
    ApplicationRegistry,
    LaunchStrategy,
    ResolutionMethod,
    STRATEGY_DEFAULTS,
)


# ─────────────────────────────────────────────────────────────
# ApplicationEntity
# ─────────────────────────────────────────────────────────────

class TestApplicationEntity:
    """Tests for ApplicationEntity data model."""

    def test_basic_construction(self):
        entity = ApplicationEntity(
            app_id="notepad",
            display_names=["Notepad"],
            canonical_process_names=["notepad.exe"],
        )
        assert entity.app_id == "notepad"
        assert entity.display_names == ["Notepad"]
        assert entity.canonical_process_names == ["notepad.exe"]
        assert entity.launch_strategies == []
        assert entity.is_uwp is False

    def test_best_strategy_returns_highest_priority(self):
        entity = ApplicationEntity(
            app_id="chrome",
            launch_strategies=[
                LaunchStrategy(type="shell", value="chrome",
                               method=ResolutionMethod.SHELL_FALLBACK, priority=10),
                LaunchStrategy(type="executable", value="C:\\chrome.exe",
                               method=ResolutionMethod.APP_PATHS, priority=100),
                LaunchStrategy(type="protocol", value="http:",
                               method=ResolutionMethod.PROTOCOL, priority=80),
            ],
        )
        best = entity.best_strategy()
        assert best is not None
        assert best.type == "executable"
        assert best.priority == 100

    def test_best_strategy_returns_none_when_empty(self):
        entity = ApplicationEntity(app_id="empty")
        assert entity.best_strategy() is None

    def test_strategies_by_type(self):
        entity = ApplicationEntity(
            app_id="chrome",
            launch_strategies=[
                LaunchStrategy(type="executable", value="C:\\chrome.exe",
                               method=ResolutionMethod.APP_PATHS, priority=100),
                LaunchStrategy(type="protocol", value="http:",
                               method=ResolutionMethod.PROTOCOL, priority=80),
                LaunchStrategy(type="executable", value="D:\\chrome.exe",
                               method=ResolutionMethod.INSTALL_SEARCH, priority=90),
            ],
        )
        exes = entity.strategies_by_type("executable")
        assert len(exes) == 2
        protos = entity.strategies_by_type("protocol")
        assert len(protos) == 1

    def test_has_process_name_case_insensitive(self):
        entity = ApplicationEntity(
            app_id="chrome",
            canonical_process_names=["chrome.exe", "chrome_proxy.exe"],
        )
        assert entity.has_process_name("chrome.exe") is True
        assert entity.has_process_name("Chrome.exe") is True
        assert entity.has_process_name("CHROME.EXE") is True
        assert entity.has_process_name("chrome_proxy.exe") is True
        assert entity.has_process_name("firefox.exe") is False


# ─────────────────────────────────────────────────────────────
# ApplicationRegistry
# ─────────────────────────────────────────────────────────────

class TestApplicationRegistry:
    """Tests for ApplicationRegistry lookup and management."""

    @pytest.fixture
    def registry(self):
        reg = ApplicationRegistry()
        reg.register(ApplicationEntity(
            app_id="chrome",
            display_names=["Google Chrome", "Chrome"],
            canonical_process_names=["chrome.exe"],
            launch_strategies=[
                LaunchStrategy(type="executable", value="C:\\chrome.exe",
                               method=ResolutionMethod.APP_PATHS, priority=100),
            ],
        ))
        reg.register(ApplicationEntity(
            app_id="spotify",
            display_names=["Spotify"],
            canonical_process_names=["spotify.exe"],
            protocols=["spotify:"],
            launch_strategies=[
                LaunchStrategy(type="protocol", value="spotify:",
                               method=ResolutionMethod.PROTOCOL, priority=80),
            ],
            is_uwp=True,
        ))
        reg.register(ApplicationEntity(
            app_id="notepad",
            display_names=["Notepad"],
            canonical_process_names=["notepad.exe"],
            launch_strategies=[
                LaunchStrategy(type="executable", value="C:\\notepad.exe",
                               method=ResolutionMethod.CLI_PATH, priority=100),
            ],
        ))
        return reg

    def test_get_by_id(self, registry):
        entity = registry.get("chrome")
        assert entity is not None
        assert entity.app_id == "chrome"

    def test_get_unknown_returns_none(self, registry):
        assert registry.get("unknown_app") is None

    def test_lookup_by_display_name(self, registry):
        entity = registry.lookup_by_name("Google Chrome")
        assert entity is not None
        assert entity.app_id == "chrome"

    def test_lookup_by_name_case_insensitive(self, registry):
        entity = registry.lookup_by_name("google chrome")
        assert entity is not None
        assert entity.app_id == "chrome"

    def test_lookup_by_app_id_as_name(self, registry):
        """app_id itself is always indexed as a name."""
        entity = registry.lookup_by_name("spotify")
        assert entity is not None
        assert entity.app_id == "spotify"

    def test_lookup_by_process(self, registry):
        entity = registry.lookup_by_process("chrome.exe")
        assert entity is not None
        assert entity.app_id == "chrome"

    def test_lookup_by_process_case_insensitive(self, registry):
        entity = registry.lookup_by_process("Spotify.exe")
        assert entity is not None
        assert entity.app_id == "spotify"

    def test_all_ids(self, registry):
        ids = registry.all_ids()
        assert set(ids) == {"chrome", "spotify", "notepad"}

    def test_len(self, registry):
        assert len(registry) == 3

    def test_contains(self, registry):
        assert "chrome" in registry
        assert "unknown" not in registry

    def test_summary(self, registry):
        summary = registry.summary()
        assert summary["total_entities"] == 3
        assert summary["by_type"]["uwp"] == 1
        assert summary["by_type"]["desktop"] == 2

    def test_atomic_refresh(self, registry):
        """refresh() replaces ALL data atomically."""
        new_entities = [
            ApplicationEntity(
                app_id="vscode",
                display_names=["Visual Studio Code"],
                canonical_process_names=["Code.exe"],
            ),
        ]
        registry.refresh(new_entities)

        # Old entities gone
        assert registry.get("chrome") is None
        assert registry.get("spotify") is None

        # New entity present
        entity = registry.get("vscode")
        assert entity is not None
        assert entity.app_id == "vscode"
        assert len(registry) == 1

    def test_refresh_updates_name_index(self, registry):
        """refresh() rebuilds name index from scratch."""
        new_entities = [
            ApplicationEntity(
                app_id="firefox",
                display_names=["Mozilla Firefox", "Firefox"],
                canonical_process_names=["firefox.exe"],
            ),
        ]
        registry.refresh(new_entities)

        assert registry.lookup_by_name("Mozilla Firefox") is not None
        assert registry.lookup_by_name("Google Chrome") is None

    def test_refresh_updates_process_index(self, registry):
        """refresh() rebuilds process index from scratch."""
        new_entities = [
            ApplicationEntity(
                app_id="firefox",
                display_names=["Firefox"],
                canonical_process_names=["firefox.exe"],
            ),
        ]
        registry.refresh(new_entities)

        assert registry.lookup_by_process("firefox.exe") is not None
        assert registry.lookup_by_process("chrome.exe") is None


# ─────────────────────────────────────────────────────────────
# ApplicationDiscoveryService
# ─────────────────────────────────────────────────────────────

class TestApplicationDiscoveryService:
    """Tests for discovery service (mock-based, no real OS access)."""

    def test_cli_path_scan_finds_notepad(self):
        """shutil.which('notepad') should resolve on Windows."""
        import shutil
        exe = shutil.which("notepad")
        if exe is None:
            pytest.skip("notepad not in PATH on this system")

        from infrastructure.app_discovery import ApplicationDiscoveryService
        discovery = ApplicationDiscoveryService(capability_registry=None)
        builders = {}
        count = discovery._scan_cli_path(builders)

        assert count > 0
        assert "notepad" in builders
        builder = builders["notepad"]
        assert len(builder.strategies) > 0
        assert builder.strategies[0].type == "executable"

    def test_derive_app_id_normalization(self):
        from infrastructure.app_discovery import ApplicationDiscoveryService

        assert ApplicationDiscoveryService._derive_app_id("Chrome.exe") == "chrome"
        assert ApplicationDiscoveryService._derive_app_id("Notepad.lnk") == "notepad"
        assert ApplicationDiscoveryService._derive_app_id("  Spotify  ") == "spotify"
        assert ApplicationDiscoveryService._derive_app_id("VS Code") == "vs code"

    def test_entity_builder_deduplicates_strategies(self):
        from infrastructure.app_discovery import _EntityBuilder

        builder = _EntityBuilder("test")
        s1 = LaunchStrategy(type="executable", value="C:\\test.exe",
                            method=ResolutionMethod.APP_PATHS, priority=100)
        s2 = LaunchStrategy(type="executable", value="C:\\test.exe",
                            method=ResolutionMethod.APP_PATHS, priority=100)
        builder.add_strategy(s1)
        builder.add_strategy(s2)  # duplicate

        assert len(builder.strategies) == 1

    def test_entity_builder_merges_strategies(self):
        from infrastructure.app_discovery import _EntityBuilder

        builder = _EntityBuilder("chrome")
        builder.add_strategy(LaunchStrategy(
            type="executable", value="C:\\chrome.exe",
            method=ResolutionMethod.APP_PATHS, priority=100,
        ))
        builder.add_strategy(LaunchStrategy(
            type="protocol", value="http:",
            method=ResolutionMethod.PROTOCOL, priority=80,
        ))

        entity = builder.build()
        assert len(entity.launch_strategies) == 2
        # Sorted by priority descending
        assert entity.launch_strategies[0].priority == 100
        assert entity.launch_strategies[1].priority == 80

    @pytest.mark.windows_only
    def test_entity_builder_infers_process_name(self):
        from infrastructure.app_discovery import _EntityBuilder

        builder = _EntityBuilder("chrome")
        builder.add_executable("C:\\Program Files\\Google\\Chrome\\chrome.exe")
        entity = builder.build()

        assert "chrome.exe" in entity.canonical_process_names

    @pytest.mark.windows_only
    def test_entity_builder_default_process_name(self):
        """If no executables found, infer from app_id."""
        from infrastructure.app_discovery import _EntityBuilder

        builder = _EntityBuilder("spotify")
        entity = builder.build()

        assert "spotify.exe" in entity.canonical_process_names

    def test_entity_builder_capabilities_binding(self):
        from infrastructure.app_discovery import _EntityBuilder

        caps = MagicMock()
        caps.supports_typing = True

        builder = _EntityBuilder("notepad")
        entity = builder.build(capabilities=caps)

        assert entity.capabilities is caps
        assert entity.capabilities.supports_typing is True

    def test_discover_all_with_empty_cap_registry(self):
        """discover_all works with no capabilities."""
        from infrastructure.app_discovery import ApplicationDiscoveryService

        discovery = ApplicationDiscoveryService(capability_registry=None)

        # Patch all scan methods to return 0 entries (fast test)
        with patch.object(discovery, '_scan_protocols', return_value=0), \
             patch.object(discovery, '_scan_app_paths', return_value=0), \
             patch.object(discovery, '_scan_start_menu', return_value=0), \
             patch.object(discovery, '_scan_appsfolder', return_value=0), \
             patch.object(discovery, '_scan_install_locations', return_value=0), \
             patch.object(discovery, '_scan_cli_path', return_value=0):

            entities = discovery.discover_all()
            assert entities == []


# ─────────────────────────────────────────────────────────────
# Strategy defaults
# ─────────────────────────────────────────────────────────────

class TestStrategyDefaults:
    """Verify data-driven strategy priority assignments."""

    def test_executable_highest_priority(self):
        assert STRATEGY_DEFAULTS["executable"]["priority"] == 100

    def test_appsfolder_second_priority(self):
        assert STRATEGY_DEFAULTS["appsfolder"]["priority"] == 90

    def test_protocol_third_priority(self):
        assert STRATEGY_DEFAULTS["protocol"]["priority"] == 80

    def test_shell_lowest_priority(self):
        assert STRATEGY_DEFAULTS["shell"]["priority"] == 10


# ─────────────────────────────────────────────────────────────
# Hardening: duplicate strategy collapse
# ─────────────────────────────────────────────────────────────

class TestStrategyDeduplication:
    """Verify strategies are correctly deduplicated."""

    def test_same_exe_path_via_different_methods_collapsed(self):
        """Chrome found via App Paths AND Start Menu with same value → 1 strategy."""
        from infrastructure.app_discovery import _EntityBuilder

        builder = _EntityBuilder("chrome")
        builder.add_strategy(LaunchStrategy(
            type="executable", value="C:\\Program Files\\Google\\Chrome\\chrome.exe",
            method=ResolutionMethod.APP_PATHS, priority=100,
        ))
        # Same path, different discovery method
        builder.add_strategy(LaunchStrategy(
            type="executable", value="C:\\Program Files\\Google\\Chrome\\chrome.exe",
            method=ResolutionMethod.START_MENU, priority=100,
        ))

        assert len(builder.strategies) == 1

    def test_different_exe_paths_not_collapsed(self):
        """Same app found at different paths → separate strategies."""
        from infrastructure.app_discovery import _EntityBuilder

        builder = _EntityBuilder("notepad")
        builder.add_strategy(LaunchStrategy(
            type="executable", value="C:\\Windows\\notepad.exe",
            method=ResolutionMethod.CLI_PATH, priority=100,
        ))
        builder.add_strategy(LaunchStrategy(
            type="executable", value="C:\\Windows\\System32\\notepad.exe",
            method=ResolutionMethod.APP_PATHS, priority=100,
        ))

        assert len(builder.strategies) == 2

    def test_case_insensitive_exe_dedup_on_windows(self):
        """Same path with different casing → collapsed (Windows is case-insensitive)."""
        from infrastructure.app_discovery import _EntityBuilder

        builder = _EntityBuilder("chrome")
        builder.add_strategy(LaunchStrategy(
            type="executable", value="C:\\Program Files\\chrome.exe",
            method=ResolutionMethod.APP_PATHS, priority=100,
        ))
        builder.add_strategy(LaunchStrategy(
            type="executable", value="c:\\program files\\chrome.exe",
            method=ResolutionMethod.START_MENU, priority=100,
        ))

        assert len(builder.strategies) == 1

    def test_protocol_dedup_is_case_sensitive(self):
        """Protocol values are case-sensitive — no false collapse."""
        from infrastructure.app_discovery import _EntityBuilder

        builder = _EntityBuilder("test")
        builder.add_strategy(LaunchStrategy(
            type="protocol", value="spotify:",
            method=ResolutionMethod.PROTOCOL, priority=80,
        ))
        builder.add_strategy(LaunchStrategy(
            type="protocol", value="Spotify:",
            method=ResolutionMethod.PROTOCOL, priority=80,
        ))

        assert len(builder.strategies) == 2

    def test_different_types_not_collapsed(self):
        """Protocol and executable with same app never collapsed."""
        from infrastructure.app_discovery import _EntityBuilder

        builder = _EntityBuilder("spotify")
        builder.add_strategy(LaunchStrategy(
            type="protocol", value="spotify:",
            method=ResolutionMethod.PROTOCOL, priority=80,
        ))
        builder.add_strategy(LaunchStrategy(
            type="executable", value="C:\\spotify\\spotify.exe",
            method=ResolutionMethod.INSTALL_SEARCH, priority=100,
        ))

        assert len(builder.strategies) == 2


# ─────────────────────────────────────────────────────────────
# Hardening: canonical ID stability
# ─────────────────────────────────────────────────────────────

class TestCanonicalIdStability:
    """Verify _derive_app_id produces stable results for known variations."""

    def test_chrome_variations(self):
        from infrastructure.app_discovery import ApplicationDiscoveryService as ADS

        assert ADS._derive_app_id("Google Chrome") == "chrome"
        assert ADS._derive_app_id("chrome") == "chrome"
        assert ADS._derive_app_id("Chrome.exe") == "chrome"
        assert ADS._derive_app_id("CHROME.EXE") == "chrome"

    def test_firefox_variations(self):
        from infrastructure.app_discovery import ApplicationDiscoveryService as ADS

        assert ADS._derive_app_id("Mozilla Firefox") == "firefox"
        assert ADS._derive_app_id("firefox") == "firefox"
        assert ADS._derive_app_id("Firefox.exe") == "firefox"

    def test_vscode_variations(self):
        from infrastructure.app_discovery import ApplicationDiscoveryService as ADS

        assert ADS._derive_app_id("Visual Studio Code") == "vscode"
        assert ADS._derive_app_id("code") == "code"  # CLI name stays as-is

    def test_edge_variations(self):
        from infrastructure.app_discovery import ApplicationDiscoveryService as ADS

        assert ADS._derive_app_id("Microsoft Edge") == "msedge"

    def test_terminal_variations(self):
        from infrastructure.app_discovery import ApplicationDiscoveryService as ADS

        assert ADS._derive_app_id("Windows Terminal") == "windowsterminal"
        assert ADS._derive_app_id("Windows PowerShell") == "powershell"
        assert ADS._derive_app_id("Command Prompt") == "cmd"

    def test_simple_names_unchanged(self):
        from infrastructure.app_discovery import ApplicationDiscoveryService as ADS

        assert ADS._derive_app_id("spotify") == "spotify"
        assert ADS._derive_app_id("notepad") == "notepad"
        assert ADS._derive_app_id("discord") == "discord"

    def test_suffix_stripping(self):
        from infrastructure.app_discovery import ApplicationDiscoveryService as ADS

        assert ADS._derive_app_id("Notepad.lnk") == "notepad"
        assert ADS._derive_app_id("calc.exe") == "calc"
        assert ADS._derive_app_id("  Spotify  ") == "spotify"


# ─────────────────────────────────────────────────────────────
# Hardening: capability defaults
# ─────────────────────────────────────────────────────────────

class TestCapabilityDefaults:
    """Verify unknown apps get safe defaults (None capabilities)."""

    def test_unknown_app_gets_none_capabilities(self):
        from infrastructure.app_discovery import _EntityBuilder

        builder = _EntityBuilder("unknown_app")
        entity = builder.build(capabilities=None)

        assert entity.capabilities is None

    def test_known_app_gets_capabilities(self):
        from infrastructure.app_discovery import _EntityBuilder

        caps = MagicMock()
        caps.supports_typing = False
        caps.supports_copy = False

        builder = _EntityBuilder("spotify")
        entity = builder.build(capabilities=caps)

        assert entity.capabilities is caps
        assert entity.capabilities.supports_typing is False
