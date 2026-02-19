# tests/test_json_extraction.py

"""
Tests for Phase 4A: Robust JSON Extraction.

Validates:
- Clean JSON passes through
- Markdown fences stripped
- Preamble text before JSON handled
- Trailing commentary after JSON handled
- Braces inside string literals handled correctly
- Escaped quotes inside strings handled
- Nested objects handled
- Empty/garbage input raises ParseError
- Unclosed blocks raise ParseError
"""

import json
import pytest

from cortex.json_extraction import extract_json_block
from errors import ParseError


class TestCleanJSON:
    """Valid JSON should pass through unchanged (minus whitespace)."""

    def test_simple_object(self):
        raw = '{"nodes": [{"id": "n1", "skill": "fs.create_folder"}]}'
        result = extract_json_block(raw)
        parsed = json.loads(result)
        assert parsed["nodes"][0]["id"] == "n1"

    def test_multiline_object(self):
        raw = """{
  "nodes": [
    {"id": "n1", "skill": "fs.create_folder"}
  ]
}"""
        result = extract_json_block(raw)
        parsed = json.loads(result)
        assert len(parsed["nodes"]) == 1


class TestMarkdownFences:
    """JSON wrapped in markdown fences."""

    def test_json_fence(self):
        raw = '```json\n{"id": "n1"}\n```'
        result = extract_json_block(raw)
        assert json.loads(result)["id"] == "n1"

    def test_plain_fence(self):
        raw = '```\n{"id": "n1"}\n```'
        result = extract_json_block(raw)
        assert json.loads(result)["id"] == "n1"

    def test_uppercase_fence(self):
        raw = '```JSON\n{"id": "n1"}\n```'
        result = extract_json_block(raw)
        assert json.loads(result)["id"] == "n1"


class TestPreambleAndTrailing:
    """JSON preceded/followed by natural language."""

    def test_preamble_text(self):
        raw = 'Here is the mission plan:\n{"id": "n1"}'
        result = extract_json_block(raw)
        assert json.loads(result)["id"] == "n1"

    def test_trailing_commentary(self):
        raw = '{"id": "n1"}\nThis is the plan I generated.'
        result = extract_json_block(raw)
        assert json.loads(result)["id"] == "n1"

    def test_preamble_and_trailing(self):
        raw = 'Sure! Here you go:\n{"id": "n1"}\nLet me know if you need changes.'
        result = extract_json_block(raw)
        assert json.loads(result)["id"] == "n1"

    def test_preamble_with_markdown_fence(self):
        raw = 'Here is the plan:\n```json\n{"id": "n1"}\n```\nDone.'
        result = extract_json_block(raw)
        assert json.loads(result)["id"] == "n1"


class TestStringAwareBraceCounting:
    """Braces inside string literals must not confuse the parser."""

    def test_braces_in_string_value(self):
        raw = '{"description": "Use {braces} inside string"}'
        result = extract_json_block(raw)
        parsed = json.loads(result)
        assert parsed["description"] == "Use {braces} inside string"

    def test_escaped_quotes_in_string(self):
        raw = '{"msg": "He said \\"hello\\" to me"}'
        result = extract_json_block(raw)
        parsed = json.loads(result)
        assert "hello" in parsed["msg"]

    def test_nested_braces_in_string(self):
        raw = '{"code": "if (x) { return {a: 1}; }", "id": "n1"}'
        result = extract_json_block(raw)
        parsed = json.loads(result)
        assert parsed["id"] == "n1"

    def test_deeply_nested_objects(self):
        raw = '{"a": {"b": {"c": {"d": "value"}}}}'
        result = extract_json_block(raw)
        parsed = json.loads(result)
        assert parsed["a"]["b"]["c"]["d"] == "value"


class TestMultipleJSONBlocks:
    """When multiple JSON objects exist, take the first."""

    def test_takes_first_block(self):
        raw = '{"first": true}\n{"second": true}'
        result = extract_json_block(raw)
        parsed = json.loads(result)
        assert parsed.get("first") is True
        assert "second" not in parsed


class TestErrorCases:
    """Invalid inputs raise ParseError."""

    def test_empty_string(self):
        with pytest.raises(ParseError, match="Empty response"):
            extract_json_block("")

    def test_whitespace_only(self):
        with pytest.raises(ParseError, match="Empty response"):
            extract_json_block("   \n\t  ")

    def test_no_json_at_all(self):
        with pytest.raises(ParseError, match="No JSON object found"):
            extract_json_block("This is just text with no JSON")

    def test_unclosed_block(self):
        with pytest.raises(ParseError, match="Unclosed JSON structure"):
            extract_json_block('{"nodes": [')

    def test_only_closing_brace(self):
        """Stray closing brace should not crash."""
        with pytest.raises(ParseError, match="No JSON object found"):
            extract_json_block("some text } more text")

    def test_array_not_treated_as_object(self):
        """Top-level arrays are not extracted (we only want objects)."""
        with pytest.raises(ParseError, match="No JSON object found"):
            extract_json_block('[1, 2, 3]')
