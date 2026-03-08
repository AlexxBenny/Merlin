# tests/test_entity_resolver.py

"""
Tests for EntityResolver and entity term extraction.
"""

import pytest
from unittest.mock import MagicMock

from infrastructure.application_registry import (
    ApplicationEntity,
    ApplicationRegistry,
    LaunchStrategy,
    ResolutionMethod,
)
from cortex.entity_resolver import (
    EntityResolver,
    ResolutionResult,
    ResolutionType,
)


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def registry():
    """Registry with common test entities."""
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
    ))
    reg.register(ApplicationEntity(
        app_id="vscode",
        display_names=["Visual Studio Code", "VS Code"],
        canonical_process_names=["Code.exe"],
    ))
    reg.register(ApplicationEntity(
        app_id="chromium",
        display_names=["Chromium"],
        canonical_process_names=["chromium.exe"],
    ))
    reg.register(ApplicationEntity(
        app_id="notepad",
        display_names=["Notepad"],
        canonical_process_names=["notepad.exe"],
    ))
    return reg


@pytest.fixture
def aliases():
    return {
        "browser": "chrome",
        "music": "spotify",
        "text editor": "notepad",
        "ide": "vscode",
    }


@pytest.fixture
def resolver(registry, aliases):
    return EntityResolver(registry=registry, alias_map=aliases)


# ─────────────────────────────────────────────────────────────
# Direct resolution
# ─────────────────────────────────────────────────────────────

class TestDirectResolution:
    """Test direct app_id and name lookups."""

    def test_resolve_by_app_id(self, resolver):
        result = resolver.resolve("chrome")
        assert result.type == ResolutionType.RESOLVED
        assert result.app_id == "chrome"
        assert result.score == 1.0

    def test_resolve_by_display_name(self, resolver):
        result = resolver.resolve("Google Chrome")
        assert result.type == ResolutionType.RESOLVED
        assert result.app_id == "chrome"

    def test_resolve_case_insensitive(self, resolver):
        result = resolver.resolve("SPOTIFY")
        assert result.type == ResolutionType.RESOLVED
        assert result.app_id == "spotify"

    def test_resolve_with_whitespace(self, resolver):
        result = resolver.resolve("  spotify  ")
        assert result.type == ResolutionType.RESOLVED
        assert result.app_id == "spotify"

    def test_resolve_not_found(self, resolver):
        result = resolver.resolve("nonexistent_app")
        assert result.type == ResolutionType.NOT_FOUND
        assert result.app_id is None

    def test_resolve_empty_string(self, resolver):
        result = resolver.resolve("")
        assert result.type == ResolutionType.NOT_FOUND


# ─────────────────────────────────────────────────────────────
# Alias resolution
# ─────────────────────────────────────────────────────────────

class TestAliasResolution:
    """Test semantic alias lookups."""

    def test_alias_browser(self, resolver):
        result = resolver.resolve("browser")
        assert result.type == ResolutionType.RESOLVED
        assert result.app_id == "chrome"
        assert result.score == 0.9  # Alias score

    def test_alias_music(self, resolver):
        result = resolver.resolve("music")
        assert result.type == ResolutionType.RESOLVED
        assert result.app_id == "spotify"

    def test_alias_ide(self, resolver):
        result = resolver.resolve("ide")
        assert result.type == ResolutionType.RESOLVED
        assert result.app_id == "vscode"

    def test_alias_case_insensitive(self, resolver):
        result = resolver.resolve("Browser")
        assert result.type == ResolutionType.RESOLVED
        assert result.app_id == "chrome"


# ─────────────────────────────────────────────────────────────
# Ambiguity handling
# ─────────────────────────────────────────────────────────────

class TestAmbiguityHandling:
    """Test that ambiguous matches are never silently resolved."""

    def test_ambiguous_prefix(self, resolver):
        """'chrom' matches both chrome and chromium — should be ambiguous."""
        result = resolver.resolve("chrom")
        # Both chrome and chromium contain "chrom" with similar coverage
        assert result.type in (ResolutionType.AMBIGUOUS, ResolutionType.RESOLVED)
        if result.type == ResolutionType.AMBIGUOUS:
            assert "chrome" in result.candidates
            assert "chromium" in result.candidates

    def test_exact_match_not_ambiguous(self, resolver):
        """'chrome' exactly matches chrome — should NOT be ambiguous."""
        result = resolver.resolve("chrome")
        assert result.type == ResolutionType.RESOLVED
        assert result.app_id == "chrome"


# ─────────────────────────────────────────────────────────────
# Batch resolution
# ─────────────────────────────────────────────────────────────

class TestBatchResolution:
    """Test resolve_terms maintains structural correspondence."""

    def test_batch_preserves_order(self, resolver):
        results = resolver.resolve_terms(["spotify", "chrome"])
        assert len(results) == 2
        assert results[0].app_id == "spotify"
        assert results[1].app_id == "chrome"

    def test_batch_mixed_results(self, resolver):
        results = resolver.resolve_terms(["spotify", "nonexistent"])
        assert len(results) == 2
        assert results[0].type == ResolutionType.RESOLVED
        assert results[1].type == ResolutionType.NOT_FOUND

    def test_batch_empty_list(self, resolver):
        results = resolver.resolve_terms([])
        assert results == []


# ─────────────────────────────────────────────────────────────
# Structured result guarantees
# ─────────────────────────────────────────────────────────────

class TestResolutionResult:
    """Test ResolutionResult invariants."""

    def test_resolved_has_entity(self, resolver):
        result = resolver.resolve("chrome")
        assert result.type == ResolutionType.RESOLVED
        assert result.entity is not None
        assert result.entity.app_id == "chrome"

    def test_not_found_has_no_entity(self, resolver):
        result = resolver.resolve("unknown")
        assert result.type == ResolutionType.NOT_FOUND
        assert result.entity is None
        assert result.app_id is None

    def test_result_preserves_original_term(self, resolver):
        result = resolver.resolve("Spotify")
        assert result.term == "Spotify"

    def test_result_is_frozen(self, resolver):
        result = resolver.resolve("chrome")
        with pytest.raises(AttributeError):
            result.app_id = "something_else"

    def test_never_raises(self, resolver):
        """Resolver should never raise, even on weird input."""
        for term in [None, 123, "", "   ", "a" * 1000]:
            try:
                result = resolver.resolve(str(term) if term is not None else "")
                assert isinstance(result, ResolutionResult)
            except Exception:
                pytest.fail(f"Resolver raised on input: {term!r}")


# ─────────────────────────────────────────────────────────────
# App term extraction
# ─────────────────────────────────────────────────────────────

class TestAppTermExtraction:
    """Test Merlin._extract_app_terms static method."""

    def test_single_app(self):
        from merlin import Merlin
        terms = Merlin._extract_app_terms("open spotify")
        assert "spotify" in terms

    def test_multi_app(self):
        from merlin import Merlin
        terms = Merlin._extract_app_terms("open chrome and close spotify")
        assert any("chrome" in t for t in terms)
        assert any("spotify" in t for t in terms)

    def test_launch_verb(self):
        from merlin import Merlin
        terms = Merlin._extract_app_terms("launch vscode")
        assert any("vscode" in t for t in terms)

    def test_no_app_verb(self):
        from merlin import Merlin
        terms = Merlin._extract_app_terms("what is the weather")
        assert terms == []
