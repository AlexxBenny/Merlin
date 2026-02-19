# tests/test_world_resolver.py

"""
Tests for Phase 1D: WorldResolver Integration.

Validates:
- "second video" resolves when list exists
- Same phrase fails cleanly when list doesn't exist
- Resolver never called post-compilation (architectural invariant)
- Referential language detection
- Entity reference resolution
- QueryContext annotation semantics
"""

import pytest

from world.resolver import (
    WorldResolver,
    QueryContext,
    ResolvedReference,
    ReferenceResolutionError,
)
from conversation.frame import ConversationFrame
from conversation.outcome import MissionOutcome


class TestReferentialLanguageDetection:
    """Validate detection of referential language patterns."""

    def test_ordinal_detected(self):
        assert WorldResolver.detect_referential_language("play the second video")

    def test_entity_reference_detected(self):
        assert WorldResolver.detect_referential_language("open that folder")

    def test_pronoun_detected(self):
        assert WorldResolver.detect_referential_language("delete it")

    def test_no_referential_language(self):
        assert not WorldResolver.detect_referential_language(
            "create a folder named hello"
        )

    def test_case_insensitive(self):
        assert WorldResolver.detect_referential_language("play the FIRST video")

    def test_multiple_patterns(self):
        assert WorldResolver.detect_referential_language(
            "play the second one from that list"
        )


class TestOrdinalResolution:
    """Validate ordinal resolution from visible_lists."""

    def test_second_video_resolves(self):
        """Core test: 'second video' resolves when list exists."""
        visible_lists = {
            "n_0.results": [
                {"title": "Python tutorial", "url": "http://example.com/1"},
                {"title": "JS tutorial", "url": "http://example.com/2"},
                {"title": "Rust tutorial", "url": "http://example.com/3"},
            ]
        }
        ctx = WorldResolver.resolve("play the second video", visible_lists)
        assert ctx.has_referential_language
        assert ctx.is_resolved
        assert len(ctx.resolved_references) == 1
        ref = ctx.resolved_references[0]
        assert ref.ordinal == "second"
        assert ref.index == 1
        assert ref.resolved_value["title"] == "JS tutorial"

    def test_first_item_resolves(self):
        visible_lists = {"n_0.files": ["file_a.txt", "file_b.txt"]}
        ctx = WorldResolver.resolve("open the first item", visible_lists)
        assert ctx.is_resolved
        assert ctx.resolved_references[0].resolved_value == "file_a.txt"

    def test_ordinal_fails_cleanly_no_list(self):
        """Same phrase fails cleanly when no list exists."""
        ctx = WorldResolver.resolve("play the second video", {})
        assert ctx.has_referential_language
        assert not ctx.is_resolved
        assert ctx.resolved_references == []

    def test_ordinal_out_of_range_fails_cleanly(self):
        visible_lists = {"n_0.results": [{"title": "only_one"}]}
        ctx = WorldResolver.resolve("play the fifth video", visible_lists)
        assert ctx.has_referential_language
        assert not ctx.is_resolved

    def test_multiple_ordinals_resolved(self):
        visible_lists = {
            "n_0.results": [
                {"title": "a"},
                {"title": "b"},
                {"title": "c"},
            ]
        }
        ctx = WorldResolver.resolve(
            "compare the first and third results", visible_lists
        )
        assert len(ctx.resolved_references) == 2
        assert ctx.resolved_references[0].ordinal == "first"
        assert ctx.resolved_references[1].ordinal == "third"


class TestEntityResolution:
    """Validate entity reference resolution."""

    def test_that_folder_resolves(self):
        ctx = WorldResolver.resolve(
            "open that folder",
            visible_lists={},
            active_entity="folder 'hello'",
        )
        assert ctx.is_resolved
        assert ctx.resolved_references[0].entity_hint == "folder 'hello'"

    def test_the_file_resolves(self):
        ctx = WorldResolver.resolve(
            "delete the file",
            visible_lists={},
            active_entity="report.pdf",
        )
        assert ctx.is_resolved
        assert ctx.resolved_references[0].entity_hint == "report.pdf"

    def test_no_entity_no_resolution(self):
        ctx = WorldResolver.resolve(
            "open that folder",
            visible_lists={},
            active_entity=None,
        )
        assert ctx.has_referential_language
        assert not ctx.is_resolved


class TestQueryContext:
    """Validate QueryContext semantics."""

    def test_original_text_preserved(self):
        ctx = WorldResolver.resolve("hello world", {})
        assert ctx.original_text == "hello world"

    def test_non_referential_returns_clean(self):
        ctx = WorldResolver.resolve("create a folder named hello", {})
        assert not ctx.has_referential_language
        assert not ctx.is_resolved
        assert ctx.resolved_references == []

    def test_is_resolved_property(self):
        ctx = QueryContext(original_text="test")
        assert not ctx.is_resolved

        ctx.resolved_references.append(
            ResolvedReference(ordinal="first", index=0)
        )
        assert ctx.is_resolved


class TestResolveOrdinalFromLists:
    """Test direct programmatic ordinal resolution."""

    def test_valid_ordinal(self):
        lists = {"results": [{"title": "a"}, {"title": "b"}]}
        key, value = WorldResolver.resolve_ordinal_from_lists("second", lists)
        assert key == "results"
        assert value["title"] == "b"

    def test_unsupported_ordinal_raises(self):
        with pytest.raises(ReferenceResolutionError, match="Unsupported"):
            WorldResolver.resolve_ordinal_from_lists("eleventh", {"r": []})

    def test_empty_lists_raises(self):
        with pytest.raises(ReferenceResolutionError, match="No visible"):
            WorldResolver.resolve_ordinal_from_lists("first", {})

    def test_out_of_range_raises(self):
        lists = {"results": [{"title": "only_one"}]}
        with pytest.raises(ReferenceResolutionError, match="out of range"):
            WorldResolver.resolve_ordinal_from_lists("third", lists)

    def test_case_insensitive(self):
        lists = {"results": [{"title": "a"}]}
        key, value = WorldResolver.resolve_ordinal_from_lists("First", lists)
        assert value["title"] == "a"


class TestResolverNeverMutatesPlan:
    """Architectural invariant: resolver runs before cortex, never post."""

    def test_resolver_outputs_annotations_only(self):
        """Resolver produces QueryContext, not modified text or plans."""
        ctx = WorldResolver.resolve(
            "play the second video",
            {"results": [{"id": 1}, {"id": 2}]},
        )
        # Output is QueryContext, not modified text
        assert isinstance(ctx, QueryContext)
        assert ctx.original_text == "play the second video"
        # Original text is unchanged
        assert "second" in ctx.original_text
