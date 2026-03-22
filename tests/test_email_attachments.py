# tests/test_email_attachments.py

"""
Tests for DraftMessageSkill attachment validation.

Covers:
- FileRef → path resolution via location_config
- Existence, permissions, size, MIME type, duplicate checks
- Integration with EmailClient.create_draft(attachments=...)
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from skills.email.draft_message import DraftMessageSkill


# ── Test helpers ──────────────────────────────────────────────

def _make_deps(tmp_path=None):
    """Create mock dependencies for DraftMessageSkill."""
    llm = MagicMock()
    llm.complete.return_value = "SUBJECT: Test\nBODY:\nTest body"

    client = MagicMock()
    client.create_draft.return_value = {
        "id": "draft_test_123",
        "body": "Test body",
    }

    loc = MagicMock()
    if tmp_path:
        loc.resolve.return_value = Path(tmp_path)
    else:
        loc.resolve.return_value = Path(".")

    return llm, client, loc


# ── Validation tests ──────────────────────────────────────────

class TestAttachmentValidation:

    def test_valid_attachment(self, tmp_path):
        # Create a test file
        test_file = tmp_path / "resume.pdf"
        test_file.write_text("pdf content")

        llm, client, loc = _make_deps(tmp_path)
        skill = DraftMessageSkill(
            content_llm=llm,
            email_client=client,
            location_config=loc,
        )

        refs = [{"anchor": "DESKTOP", "relative_path": "resume.pdf", "name": "resume.pdf"}]
        result = skill._validate_attachments(refs)

        assert len(result) == 1
        assert result[0]["name"] == "resume.pdf"
        assert result[0]["path"] == str(tmp_path / "resume.pdf")
        assert "mime_type" in result[0]

    def test_missing_file_raises(self, tmp_path):
        llm, client, loc = _make_deps(tmp_path)
        skill = DraftMessageSkill(
            content_llm=llm,
            email_client=client,
            location_config=loc,
        )

        refs = [{"anchor": "DESKTOP", "relative_path": "nonexistent.pdf", "name": "x"}]

        with pytest.raises(FileNotFoundError, match="not found"):
            skill._validate_attachments(refs)

    def test_oversized_file_raises(self, tmp_path):
        # Create a file that exceeds the limit
        test_file = tmp_path / "huge.bin"
        # Write 26 MB (exceeds 25 MB limit)
        with open(test_file, "wb") as f:
            f.write(b"x" * (26 * 1024 * 1024))

        llm, client, loc = _make_deps(tmp_path)
        skill = DraftMessageSkill(
            content_llm=llm,
            email_client=client,
            location_config=loc,
        )

        refs = [{"anchor": "DESKTOP", "relative_path": "huge.bin", "name": "huge.bin"}]

        with pytest.raises(ValueError, match="too large"):
            skill._validate_attachments(refs)

    def test_duplicate_name_raises(self, tmp_path):
        test_file = tmp_path / "doc.pdf"
        test_file.write_text("content")

        llm, client, loc = _make_deps(tmp_path)
        skill = DraftMessageSkill(
            content_llm=llm,
            email_client=client,
            location_config=loc,
        )

        refs = [
            {"anchor": "DESKTOP", "relative_path": "doc.pdf", "name": "doc.pdf"},
            {"anchor": "DESKTOP", "relative_path": "doc.pdf", "name": "doc.pdf"},
        ]

        with pytest.raises(ValueError, match="Duplicate"):
            skill._validate_attachments(refs)

    def test_mime_type_detection(self, tmp_path):
        pdf = tmp_path / "test.pdf"
        pdf.write_text("pdf")
        txt = tmp_path / "test.txt"
        txt.write_text("text")

        llm, client, loc = _make_deps(tmp_path)
        skill = DraftMessageSkill(
            content_llm=llm,
            email_client=client,
            location_config=loc,
        )

        refs = [
            {"anchor": "DESKTOP", "relative_path": "test.pdf", "name": "test.pdf"},
            {"anchor": "DESKTOP", "relative_path": "test.txt", "name": "test.txt"},
        ]
        result = skill._validate_attachments(refs)

        assert result[0]["mime_type"] == "application/pdf"
        assert result[1]["mime_type"] == "text/plain"

    def test_empty_attachments_returns_empty(self, tmp_path):
        llm, client, loc = _make_deps(tmp_path)
        skill = DraftMessageSkill(
            content_llm=llm,
            email_client=client,
            location_config=loc,
        )

        result = skill._validate_attachments([])
        assert result == []


# ── Integration test ──────────────────────────────────────────

class TestAttachmentIntegration:

    def test_attachments_passed_to_email_client(self, tmp_path):
        test_file = tmp_path / "report.pdf"
        test_file.write_text("report content")

        llm, client, loc = _make_deps(tmp_path)
        skill = DraftMessageSkill(
            content_llm=llm,
            email_client=client,
            location_config=loc,
        )

        timeline = MagicMock()
        inputs = {
            "prompt": "Send the report",
            "recipient": "bob@example.com",
            "attachments": [
                {"anchor": "DESKTOP", "relative_path": "report.pdf", "name": "report.pdf"},
            ],
        }

        result = skill.execute(inputs, timeline)

        # EmailClient.create_draft should have been called with attachments
        call_kwargs = client.create_draft.call_args
        assert call_kwargs is not None
        attachments = call_kwargs.kwargs.get("attachments") or call_kwargs[1].get("attachments")
        assert attachments is not None
        assert len(attachments) == 1
        assert attachments[0]["name"] == "report.pdf"

    def test_no_attachments_works(self, tmp_path):
        llm, client, loc = _make_deps(tmp_path)
        skill = DraftMessageSkill(
            content_llm=llm,
            email_client=client,
            location_config=loc,
        )

        timeline = MagicMock()
        inputs = {
            "prompt": "Say hello",
            "recipient": "bob@example.com",
        }

        result = skill.execute(inputs, timeline)

        # Should not raise
        assert result.outputs["draft_id"] == "draft_test_123"
