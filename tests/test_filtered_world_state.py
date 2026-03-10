# tests/test_filtered_world_state.py

"""
Unit tests for FilteredWorldStateProvider — Phase 3A.

Tests the query-scoped view projection:
- Confidence-gated keyword detection (unique vs ambiguous)
- Domain detection from candidate skills
- Union of skill domains and keyword domains
- State section projection
- Always-include sections (time)
- Fallback to full state (no domain signal)
- False positive prevention
"""

import pytest
from world.state import (
    WorldState, SystemState, HardwareState, SessionState,
    ResourceState, MediaState, TimeState,
)
from world.snapshot import WorldSnapshot
from cortex.filtered_world_state_provider import (
    FilteredWorldStateProvider, _AMBIGUOUS_THRESHOLD,
)
from cortex.world_state_provider import SimpleWorldStateProvider


# ─────────────────────────────────────────────────────────────
# Fixture: a fully populated WorldState
# ─────────────────────────────────────────────────────────────

def _full_world_state() -> WorldState:
    """Build a WorldState with every section populated."""
    return WorldState(
        active_app="Chrome",
        active_window="Google - Chrome",
        cwd="C:\\Users\\alex\\Projects",
        media=MediaState(
            platform="Spotify",
            title="Test Song",
            artist="Test Artist",
            is_playing=True,
        ),
        system=SystemState(
            resources=ResourceState(
                cpu_percent=45.0,
                cpu_status="normal",
                memory_percent=60.0,
                memory_status="normal",
                disk_percent=70.0,
            ),
            hardware=HardwareState(
                battery_percent=85.0,
                battery_charging=False,
                battery_status="normal",
                brightness_percent=50,
                volume_percent=30,
                muted=False,
                nightlight_enabled=False,
            ),
            session=SessionState(
                foreground_app="Chrome",
                foreground_window="Google - Chrome",
                idle_seconds=10.0,
                open_apps=["Chrome", "VSCode", "Spotify"],
            ),
        ),
        time=TimeState(
            hour=14,
            minute=30,
            day_of_week="Tuesday",
            date="2026-02-18",
        ),
    )


def _snapshot(state: WorldState = None) -> WorldSnapshot:
    return WorldSnapshot.build(
        state=state or _full_world_state(),
        recent_events=[],
    )


# ─────────────────────────────────────────────────────────────
# Tests: Confidence-gated keyword detection
# ─────────────────────────────────────────────────────────────

class TestConfidenceGatedKeywords:
    """Verify that keyword detection uses unique/ambiguous tiers."""

    def test_unique_keyword_triggers_domain(self):
        """'music' is unique for media — one match is sufficient."""
        provider = FilteredWorldStateProvider()
        view = provider.build_schema(_snapshot(), query="play music")

        assert "media" in view
        assert view["media"]["platform"] == "Spotify"
        assert "time" in view

    def test_single_ambiguous_keyword_does_NOT_trigger(self):
        """'open' alone is ambiguous — must NOT trigger system domain.

        Prevents: 'open research paper' → system (false positive).
        """
        provider = FilteredWorldStateProvider()
        view = provider.build_schema(
            _snapshot(), query="open research paper",
        )

        # 'open' is ambiguous for system (1 hit, threshold=2)
        # No unique keywords match any domain
        # → no domain signal → full state (safe default)
        simple = SimpleWorldStateProvider()
        full = simple.build_schema(_snapshot())
        assert view == full

    def test_create_alone_does_NOT_trigger_fs(self):
        """'create' alone is ambiguous — 'create summary' must NOT filter.

        Prevents: 'create summary' → fs (false positive).
        """
        provider = FilteredWorldStateProvider()
        view = provider.build_schema(
            _snapshot(), query="create summary",
        )

        simple = SimpleWorldStateProvider()
        full = simple.build_schema(_snapshot())
        assert view == full

    def test_search_alone_does_NOT_trigger_browser(self):
        """'search' alone is ambiguous — 'search memory usage' → full state.

        Prevents: 'search memory usage' → browser (false positive).
        """
        provider = FilteredWorldStateProvider()
        view = provider.build_schema(
            _snapshot(), query="search memory usage",
        )

        # 'search' → ambiguous browser (1 hit, <2)
        # 'memory' → ambiguous system (1 hit, <2)
        # Neither meets threshold → full state
        simple = SimpleWorldStateProvider()
        full = simple.build_schema(_snapshot())
        assert view == full

    def test_two_ambiguous_keywords_DO_trigger(self):
        """Two ambiguous keywords from same domain → domain detected."""
        provider = FilteredWorldStateProvider()
        # 'open' + 'close' are both ambiguous for system
        view = provider.build_schema(
            _snapshot(), query="open chrome and close the app",
        )

        # 'chrome' → unique for browser → browser detected
        # 'open' + 'close' + 'app' = 3 ambiguous for system → system detected
        assert "system" in view

    def test_unique_keyword_overrides_ambiguous_threshold(self):
        """'folder' is unique for fs — even without 2 ambiguous matches."""
        provider = FilteredWorldStateProvider()
        view = provider.build_schema(_snapshot(), query="create folder")

        # 'folder' is unique for fs → detected immediately
        assert "cwd" in view
        assert view["cwd"] == "C:\\Users\\alex\\Projects"


# ─────────────────────────────────────────────────────────────
# Tests: Domain-specific unique keywords
# ─────────────────────────────────────────────────────────────

class TestUniqueKeywordDetection:
    """Verify unique keywords for each domain."""

    def test_media_unique_keywords(self):
        """Unique media keywords trigger media domain."""
        provider = FilteredWorldStateProvider()
        for kw in ["spotify", "music", "song", "track", "audio"]:
            view = provider.build_schema(_snapshot(), query=f"what {kw}")
            assert "media" in view, f"'{kw}' should trigger media domain"

    def test_system_unique_keywords(self):
        """Unique system keywords trigger system domain."""
        provider = FilteredWorldStateProvider()
        for kw in ["brightness", "volume", "mute", "battery"]:
            view = provider.build_schema(_snapshot(), query=f"check {kw}")
            assert "system" in view, f"'{kw}' should trigger system domain"
            assert "hardware" in view["system"]

    def test_fs_unique_keywords(self):
        """Unique fs keywords trigger fs domain."""
        provider = FilteredWorldStateProvider()
        for kw in ["folder", "directory"]:
            view = provider.build_schema(_snapshot(), query=f"list {kw}")
            assert "cwd" in view, f"'{kw}' should trigger fs domain"

    def test_browser_unique_keywords(self):
        """Unique browser keywords trigger browser domain."""
        provider = FilteredWorldStateProvider()
        for kw in ["chrome", "firefox", "browser", "tab", "website"]:
            view = provider.build_schema(_snapshot(), query=f"use {kw}")
            assert "system" in view, f"'{kw}' should trigger browser domain"
            assert "session" in view["system"]


# ─────────────────────────────────────────────────────────────
# Tests: Union of candidate skills and keywords
# ─────────────────────────────────────────────────────────────

class TestUnionDomainResolution:
    """Verify UNION behavior: domains_from_skills ∪ domains_from_keywords."""

    def test_skills_plus_keywords_union(self):
        """Skills provide system, keywords detect media → both in view."""
        provider = FilteredWorldStateProvider()
        view = provider.build_schema(
            _snapshot(),
            query="play music",
            candidate_skills={"system.set_volume"},
        )

        # Skills → system domain
        assert "system" in view
        assert "hardware" in view["system"]
        # Keywords → media domain (music is unique)
        assert "media" in view
        assert "time" in view

    def test_cross_domain_query_all_detected(self):
        """Complex cross-domain query detects all relevant domains."""
        provider = FilteredWorldStateProvider()
        view = provider.build_schema(
            _snapshot(),
            query="open chrome, search agentic AI, create folder, play music",
        )

        # 'chrome' → unique browser → system.session
        assert "system" in view
        assert "session" in view["system"]

        # 'folder' → unique fs → cwd
        assert "cwd" in view

        # 'music' → unique media → media + system.hardware
        assert "media" in view
        assert "hardware" in view["system"]

    def test_candidate_skills_extract_domains(self):
        """Skill names correctly map to domains."""
        provider = FilteredWorldStateProvider()
        view = provider.build_schema(
            _snapshot(),
            candidate_skills={"system.set_volume", "system.mute"},
        )

        assert "system" in view
        assert "hardware" in view["system"]
        assert "session" in view["system"]
        assert "time" in view
        assert "media" not in view

    def test_multiple_unique_domains_detected(self):
        """Query touching multiple unique keywords → all domains detected."""
        provider = FilteredWorldStateProvider()
        view = provider.build_schema(
            _snapshot(),
            query="mute the volume and play music",
        )

        # 'mute'+'volume' → unique for system
        assert "system" in view
        assert "hardware" in view["system"]
        # 'music' → unique for media
        assert "media" in view


# ─────────────────────────────────────────────────────────────
# Tests: Fallback behavior
# ─────────────────────────────────────────────────────────────

class TestFallbackBehavior:
    """Verify safe fallbacks when no domain signal is available."""

    def test_no_query_no_skills_returns_full_state(self):
        """No domain signal → full state dump (safe default)."""
        provider = FilteredWorldStateProvider()
        view = provider.build_schema(_snapshot())

        simple = SimpleWorldStateProvider()
        full = simple.build_schema(_snapshot())
        assert view == full

    def test_unknown_query_returns_full_state(self):
        """Unrecognized query → no keywords match → full state."""
        provider = FilteredWorldStateProvider()
        view = provider.build_schema(
            _snapshot(),
            query="tell me a joke",
        )

        simple = SimpleWorldStateProvider()
        full = simple.build_schema(_snapshot())
        assert view == full

    def test_weak_signal_returns_full_state(self):
        """Single ambiguous keyword → too weak → full state."""
        provider = FilteredWorldStateProvider()
        view = provider.build_schema(
            _snapshot(),
            query="play something interesting",
        )

        # 'play' is ambiguous for media (1 hit, <2)
        # No unique keyword matches → full state
        simple = SimpleWorldStateProvider()
        full = simple.build_schema(_snapshot())
        assert view == full


# ─────────────────────────────────────────────────────────────
# Tests: Always-include sections
# ─────────────────────────────────────────────────────────────

class TestAlwaysInclude:
    """Verify that time and system.session are always included."""

    def test_time_always_included(self):
        """Time section should always be in the view."""
        provider = FilteredWorldStateProvider()
        view = provider.build_schema(_snapshot(), query="create folder test")

        assert "time" in view
        assert view["time"]["hour"] == 14

    def test_session_always_included_for_media(self):
        """system.session must be visible for media queries.

        Without this, 'close spotify' planning fails because
        the LLM can't see tracked_apps → guesses app names →
        entity resolution fails.
        """
        provider = FilteredWorldStateProvider()
        view = provider.build_schema(_snapshot(), query="play music")

        assert "system" in view
        assert "session" in view["system"]

    def test_session_always_included_for_fs(self):
        """system.session is in ALWAYS_INCLUDE — present for all domains."""
        provider = FilteredWorldStateProvider()
        view = provider.build_schema(_snapshot(), query="create folder test")

        assert "system" in view
        assert "session" in view["system"]

    def test_time_missing_in_state(self):
        """If time is None in state, view doesn't include it (no crash)."""
        state = WorldState()  # time is None by default
        provider = FilteredWorldStateProvider()
        view = provider.build_schema(
            _snapshot(state),
            query="create folder test",
        )

        assert "time" not in view or view.get("time") is None


# ─────────────────────────────────────────────────────────────
# Tests: View projection mechanics
# ─────────────────────────────────────────────────────────────

class TestViewProjection:
    """Verify the projection logic correctly extracts nested paths."""

    def test_nested_path_projection(self):
        """'system.hardware' should include the entire hardware subtree."""
        provider = FilteredWorldStateProvider()
        view = provider.build_schema(_snapshot(), query="set volume to 50")

        hw = view["system"]["hardware"]
        assert "battery_percent" in hw
        assert "brightness_percent" in hw
        assert "volume_percent" in hw
        assert "muted" in hw
        assert "nightlight_enabled" in hw

    def test_view_is_smaller_than_full_state(self):
        """Filtered view should have fewer leaf keys than full state."""
        provider = FilteredWorldStateProvider()
        simple = SimpleWorldStateProvider()

        view = provider.build_schema(_snapshot(), query="play music")
        full = simple.build_schema(_snapshot())

        view_keys = FilteredWorldStateProvider._count_leaf_keys(view)
        full_keys = FilteredWorldStateProvider._count_leaf_keys(full)

        assert view_keys < full_keys, (
            f"Filtered view ({view_keys} keys) should be smaller "
            f"than full state ({full_keys} keys)"
        )


# ─────────────────────────────────────────────────────────────
# Tests: SimpleWorldStateProvider backward compatibility
# ─────────────────────────────────────────────────────────────

class TestSimpleWorldStateProviderCompat:
    """Verify SimpleWorldStateProvider accepts new signature params."""

    def test_accepts_query(self):
        simple = SimpleWorldStateProvider()
        full = simple.build_schema(_snapshot())
        with_query = simple.build_schema(_snapshot(), query="play music")
        assert full == with_query

    def test_accepts_candidate_skills(self):
        simple = SimpleWorldStateProvider()
        full = simple.build_schema(_snapshot())
        with_skills = simple.build_schema(
            _snapshot(), candidate_skills={"system.set_volume"},
        )
        assert full == with_skills


# ─────────────────────────────────────────────────────────────
# Tests: Custom config
# ─────────────────────────────────────────────────────────────

class TestCustomConfig:
    """Verify config-driven domain mapping."""

    def test_custom_domain_map(self):
        """Custom domain→state mapping is respected."""
        provider = FilteredWorldStateProvider(
            domain_state_map={"fs": ["cwd", "system.resources"]},
        )
        view = provider.build_schema(
            _snapshot(),
            query="create folder test",
        )

        assert "cwd" in view
        assert "system" in view
        assert "resources" in view["system"]
        assert "hardware" not in view.get("system", {})

    def test_custom_always_include(self):
        """Custom always_include is respected."""
        provider = FilteredWorldStateProvider(
            always_include=["time", "active_app"],
        )
        view = provider.build_schema(
            _snapshot(),
            query="play music",
        )

        assert "time" in view
        assert "active_app" in view
