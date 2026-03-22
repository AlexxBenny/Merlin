# tests/test_file_index.py

"""
Tests for FileIndex — lazy-built file index with incremental updates.

Covers:
- Build from LocationConfig anchors
- Search: exact/starts-with/substring matching + ranking
- Incremental add/remove
- Invalidation and rebuild
- Bounded: max_depth and max_files limits
"""

import tempfile
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from world.file_index import FileIndex
from world.file_ref import FileRef, _generate_ref_id


# ── Test helpers ──────────────────────────────────────────────

def _make_location_config(anchors: dict):
    """Create a mock LocationConfig with all_anchors()."""
    mock = MagicMock()
    mock.all_anchors.return_value = {k: str(v) for k, v in anchors.items()}
    mock.resolve.side_effect = lambda anchor: Path(anchors[anchor])
    return mock


def _create_test_tree(base_dir):
    """Create a test file tree under base_dir.

    Structure:
        base_dir/
            resume.pdf
            report.docx
            photo.jpg
            projects/
                myapp/
                    main.py
                    config.yaml
    """
    Path(base_dir, "resume.pdf").write_text("pdf content")
    Path(base_dir, "report.docx").write_text("docx content")
    Path(base_dir, "photo.jpg").write_text("jpg content")
    Path(base_dir, "projects").mkdir()
    Path(base_dir, "projects", "myapp").mkdir()
    Path(base_dir, "projects", "myapp", "main.py").write_text("print('hi')")
    Path(base_dir, "projects", "myapp", "config.yaml").write_text("key: val")


# ── Build tests ──────────────────────────────────────────────

class TestFileIndexBuild:

    def test_build_indexes_files(self, tmp_path):
        _create_test_tree(tmp_path)
        loc = _make_location_config({"DESKTOP": tmp_path})

        idx = FileIndex()
        idx.build(loc, max_depth=5)

        assert idx.is_built
        assert idx.total_files == 5  # 3 at root + 2 in projects/myapp

    def test_build_respects_max_depth(self, tmp_path):
        _create_test_tree(tmp_path)
        loc = _make_location_config({"DESKTOP": tmp_path})

        idx = FileIndex()
        idx.build(loc, max_depth=1)

        # max_depth=1 → only root files, not projects/myapp/*
        assert idx.total_files == 3

    def test_build_handles_missing_anchor(self, tmp_path):
        loc = _make_location_config({"MISSING": tmp_path / "nonexistent"})

        idx = FileIndex()
        idx.build(loc)

        assert idx.is_built
        assert idx.total_files == 0

    def test_build_skips_hidden_directories(self, tmp_path):
        Path(tmp_path, ".hidden").mkdir()
        Path(tmp_path, ".hidden", "secret.txt").write_text("secret")
        Path(tmp_path, "visible.txt").write_text("visible")
        loc = _make_location_config({"ROOT": tmp_path})

        idx = FileIndex()
        idx.build(loc, max_depth=5)

        assert idx.total_files == 1  # only visible.txt


# ── Search tests ──────────────────────────────────────────────

class TestFileIndexSearch:

    def test_exact_match_highest_confidence(self, tmp_path):
        _create_test_tree(tmp_path)
        loc = _make_location_config({"DESKTOP": tmp_path})

        idx = FileIndex()
        idx.build(loc)

        results = idx.search("resume.pdf")
        assert len(results) == 1
        assert results[0].confidence == 1.0
        assert results[0].name == "resume.pdf"

    def test_substring_match(self, tmp_path):
        _create_test_tree(tmp_path)
        loc = _make_location_config({"DESKTOP": tmp_path})

        idx = FileIndex()
        idx.build(loc)

        # "esume" is a substring of "resume.pdf" but NOT starts-with
        results = idx.search("esume")
        assert len(results) == 1
        assert results[0].name == "resume.pdf"
        assert results[0].confidence == 0.5  # substring

    def test_starts_with_match(self, tmp_path):
        _create_test_tree(tmp_path)
        loc = _make_location_config({"DESKTOP": tmp_path})

        idx = FileIndex()
        idx.build(loc)

        results = idx.search("report")
        assert len(results) == 1
        assert results[0].name == "report.docx"
        assert results[0].confidence == 0.8  # starts-with

    def test_case_insensitive(self, tmp_path):
        _create_test_tree(tmp_path)
        loc = _make_location_config({"DESKTOP": tmp_path})

        idx = FileIndex()
        idx.build(loc)

        results = idx.search("RESUME.PDF")
        assert len(results) == 1

    def test_no_match_returns_empty(self, tmp_path):
        _create_test_tree(tmp_path)
        loc = _make_location_config({"DESKTOP": tmp_path})

        idx = FileIndex()
        idx.build(loc)

        results = idx.search("nonexistent.xyz")
        assert len(results) == 0

    def test_lazy_build_on_first_search(self, tmp_path):
        _create_test_tree(tmp_path)
        loc = _make_location_config({"DESKTOP": tmp_path})

        idx = FileIndex()
        assert not idx.is_built

        results = idx.search("resume", location_config=loc)
        assert idx.is_built
        assert len(results) == 1

    def test_max_results_limit(self, tmp_path):
        # Create many files
        for i in range(60):
            Path(tmp_path, f"file_{i:03d}.txt").write_text(f"content {i}")
        loc = _make_location_config({"ROOT": tmp_path})

        idx = FileIndex()
        idx.build(loc)

        results = idx.search("file", max_results=10)
        assert len(results) == 10

    def test_results_ranked_by_confidence(self, tmp_path):
        Path(tmp_path, "report.txt").write_text("a")
        Path(tmp_path, "annual_report.txt").write_text("b")
        loc = _make_location_config({"ROOT": tmp_path})

        idx = FileIndex()
        idx.build(loc)

        results = idx.search("report")
        # "report.txt" is starts-with (0.8), "annual_report.txt" is substring (0.5)
        assert results[0].name == "report.txt"
        assert results[0].confidence > results[1].confidence


# ── Incremental update tests ──────────────────────────────────

class TestFileIndexIncrementalUpdates:

    def test_add_ref(self, tmp_path):
        idx = FileIndex()
        idx._built = True  # Pretend built

        ref = FileRef(
            ref_id=_generate_ref_id(),
            name="new_file.txt",
            anchor="DESKTOP",
            relative_path="new_file.txt",
            size_bytes=100,
        )
        idx.add_ref(ref)

        results = idx.search("new_file")
        assert len(results) == 1
        assert results[0].name == "new_file.txt"

    def test_remove_ref(self, tmp_path):
        idx = FileIndex()
        idx._built = True

        ref = FileRef(
            ref_id="test_ref_123",
            name="to_remove.txt",
            anchor="DESKTOP",
            relative_path="to_remove.txt",
        )
        idx.add_ref(ref)
        assert len(idx.search("to_remove")) == 1

        idx.remove_ref("test_ref_123")
        assert len(idx.search("to_remove")) == 0

    def test_invalidate_clears_index(self, tmp_path):
        _create_test_tree(tmp_path)
        loc = _make_location_config({"DESKTOP": tmp_path})

        idx = FileIndex()
        idx.build(loc)
        assert idx.is_built
        assert idx.total_files > 0

        idx.invalidate()
        assert not idx.is_built
        assert idx.total_files == 0


# ── FileRef tests ──────────────────────────────────────────────

class TestFileRef:

    def test_resolve_path(self, tmp_path):
        loc = _make_location_config({"DESKTOP": tmp_path})
        ref = FileRef(
            ref_id="test_ref",
            name="resume.pdf",
            anchor="DESKTOP",
            relative_path="docs/resume.pdf",
        )

        result = ref.resolve(loc)
        assert result == tmp_path / "docs" / "resume.pdf"

    def test_to_output_dict(self):
        ref = FileRef(
            ref_id="test_ref",
            name="resume.pdf",
            anchor="DESKTOP",
            relative_path="resume.pdf",
            size_bytes=1024,
            confidence=0.95,
        )

        d = ref.to_output_dict()
        assert d["ref_id"] == "test_ref"
        assert d["name"] == "resume.pdf"
        assert d["anchor"] == "DESKTOP"
        assert d["size_bytes"] == 1024
        assert d["confidence"] == 0.95
