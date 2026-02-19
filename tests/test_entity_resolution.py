# tests/test_entity_resolution.py

"""
Tests for entity_registry integration with WorldResolver.

Validates:
- Ordinal resolution from entity_registry (Priority 1.5)
- Entity registry survives domain switches
- entity_registry passed to resolver correctly
- Resolution source is correctly tagged
"""

import pytest

from conversation.frame import ConversationFrame, EntityRecord
from world.resolver import WorldResolver, QueryContext, ResolvedReference


class TestEntityRegistryResolution:
    """Validate ordinal resolution from entity_registry."""

    def test_ordinal_from_entity_registry(self):
        """'the second result' resolves from entity_registry when no visible_lists."""
        registry = {
            "results": EntityRecord(
                type="list",
                value=[
                    {"title": "GenAI Overview", "url": "http://example.com/1"},
                    {"title": "GenAI Tutorial", "url": "http://example.com/2"},
                    {"title": "GenAI Deep Dive", "url": "http://example.com/3"},
                ],
                source_mission="m_1",
            ),
        }

        ctx = WorldResolver.resolve(
            user_text="play the second video",
            visible_lists={},  # empty — simulates domain switch
            entity_registry=registry,
        )
        assert ctx.has_referential_language
        assert ctx.is_resolved
        assert len(ctx.resolved_references) == 1
        ref = ctx.resolved_references[0]
        assert ref.ordinal == "second"
        assert ref.index == 1
        assert ref.resolved_value["title"] == "GenAI Tutorial"
        assert ref.resolution_source == "entity_registry"

    def test_visible_lists_take_priority(self):
        """visible_lists (Priority 1) resolve before entity_registry (Priority 1.5)."""
        visible_lists = {
            "n_0.results": [
                {"title": "Live Result A"},
                {"title": "Live Result B"},
            ],
        }
        registry = {
            "results": EntityRecord(
                type="list",
                value=[
                    {"title": "Old Result A"},
                    {"title": "Old Result B"},
                ],
                source_mission="m_old",
            ),
        }

        ctx = WorldResolver.resolve(
            user_text="play the first video",
            visible_lists=visible_lists,
            entity_registry=registry,
        )
        assert ctx.is_resolved
        # Should resolve from visible_lists, not entity_registry
        ref = ctx.resolved_references[0]
        assert ref.resolved_value["title"] == "Live Result A"
        assert ref.resolution_source == "ordinal"  # from visible_lists

    def test_no_registry_no_resolution(self):
        """Without entity_registry or visible_lists, ordinal fails cleanly."""
        ctx = WorldResolver.resolve(
            user_text="play the second video",
            visible_lists={},
            entity_registry={},
        )
        assert ctx.has_referential_language
        assert not ctx.is_resolved

    def test_out_of_range_fails_cleanly(self):
        """Ordinal beyond registry list length doesn't resolve."""
        registry = {
            "results": EntityRecord(
                type="list",
                value=[{"title": "only_one"}],
                source_mission="m_1",
            ),
        }
        ctx = WorldResolver.resolve(
            user_text="play the fifth video",
            visible_lists={},
            entity_registry=registry,
        )
        assert ctx.has_referential_language
        assert not ctx.is_resolved

    def test_non_list_entities_skipped(self):
        """entity_registry entries with non-list values are skipped for ordinals."""
        registry = {
            "folder": EntityRecord(
                type="path",
                value="/Desktop/hello",
                source_mission="m_1",
            ),
        }
        ctx = WorldResolver.resolve(
            user_text="play the second video",
            visible_lists={},
            entity_registry=registry,
        )
        assert ctx.has_referential_language
        assert not ctx.is_resolved  # scalar entity can't resolve ordinals


class TestRegistrySurvivesDomainSwitch:
    """Validate the core use case: entities persist across domain switches."""

    def test_search_then_create_then_play(self):
        """
        Scenario:
        1. User searches YouTube → results stored in entity_registry
        2. User creates a folder → domain switches, visible_lists change
        3. User says 'play the second video' → must resolve from registry
        """
        frame = ConversationFrame()

        # Turn 1: YouTube search results
        search_results = [
            {"title": "AI Overview", "url": "http://yt.com/1"},
            {"title": "AI Tutorial", "url": "http://yt.com/2"},
            {"title": "AI Deep Dive", "url": "http://yt.com/3"},
        ]
        frame.register_entity("results", search_results, "list", "m_search")
        frame.active_domain = "browser"
        frame.active_entity = "search results for 'AI'"

        # Turn 2: Create folder (domain switch — new visible_lists)
        frame.register_entity("path", "/Desktop/research", "path", "m_folder")
        frame.active_domain = "fs"
        frame.active_entity = "folder 'research'"

        # Turn 3: 'play the second video' — visible_lists empty (last mission was fs)
        ctx = WorldResolver.resolve(
            user_text="play the second video",
            visible_lists={},  # fs mission produced no lists
            active_entity=frame.active_entity,
            entity_registry=frame.entity_registry,
        )

        assert ctx.is_resolved
        ref = ctx.resolved_references[0]
        assert ref.ordinal == "second"
        assert ref.resolved_value["title"] == "AI Tutorial"
        assert ref.resolution_source == "entity_registry"

    def test_registry_list_key_prefix(self):
        """Resolved reference list_key includes 'entity_registry.' prefix."""
        registry = {
            "files": EntityRecord(
                type="list",
                value=["a.txt", "b.txt"],
                source_mission="m_1",
            ),
        }
        ctx = WorldResolver.resolve(
            user_text="open the first file",
            visible_lists={},
            entity_registry=registry,
        )
        assert ctx.is_resolved
        assert ctx.resolved_references[0].list_key == "entity_registry.files"
