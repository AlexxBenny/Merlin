# cortex/json_extraction.py

"""
Defensive JSON block extraction from LLM output.

LLMs often wrap valid JSON in markdown fences, preamble text,
or trailing commentary. This module extracts the first valid
JSON object or array block.

String-aware brace/bracket counting:
- Braces/brackets inside quoted strings are ignored.
- Escaped quotes inside strings are handled.
- Partial/unclosed blocks raise ParseError.
- If ambiguous or no JSON found → ParseError (never malformed fragment).
"""

from errors import ParseError


def extract_json_block(text: str) -> str:
    """Extract the first top-level JSON object from text.

    Handles:
    - Raw JSON
    - Markdown ```json fences
    - Preamble text before JSON
    - Trailing commentary after JSON
    - Nested braces in values
    - Braces inside string literals (string-aware)

    Returns:
        Clean JSON string (the first {...} block).

    Raises:
        ParseError: if no valid JSON block is found.
    """
    if not text or not text.strip():
        raise ParseError("Empty response from LLM")

    stripped = _strip_markdown_fences(text)

    block = _find_json_object(stripped)
    if block is None:
        raise ParseError(
            f"No JSON object found in LLM response "
            f"(first 200 chars: {text[:200]!r})"
        )

    return block


def extract_json_array(text: str) -> str:
    """Extract the first top-level JSON array from text.

    Handles:
    - Raw JSON arrays
    - Markdown ```json fences
    - Preamble text before JSON
    - Trailing commentary after JSON
    - Nested arrays/objects in elements
    - Brackets/braces inside string literals (string-aware)

    Returns:
        Clean JSON string (the first [...] block).

    Raises:
        ParseError: if no valid JSON array is found.
    """
    if not text or not text.strip():
        raise ParseError("Empty response from LLM")

    stripped = _strip_markdown_fences(text)

    block = _find_json_array(stripped)
    if block is None:
        raise ParseError(
            f"No JSON array found in LLM response "
            f"(first 200 chars: {text[:200]!r})"
        )

    return block


def _strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences wrapping JSON.

    Handles:
    - ```json ... ```
    - ``` ... ```
    - ```JSON ... ```
    """
    stripped = text.strip()

    # Check for opening fence
    if stripped.startswith("```"):
        # Find end of opening fence line
        first_newline = stripped.find("\n")
        if first_newline == -1:
            return stripped  # Just a fence line, no content

        # Remove opening fence line
        inner = stripped[first_newline + 1:]

        # Remove closing fence
        if inner.rstrip().endswith("```"):
            last_fence = inner.rfind("```")
            inner = inner[:last_fence]

        return inner.strip()

    return stripped


def _find_json_object(text: str) -> str | None:
    """Find the first top-level JSON object {...}."""
    return _find_json_structure(text, '{', '}')


def _find_json_array(text: str) -> str | None:
    """Find the first top-level JSON array [...]."""
    return _find_json_structure(text, '[', ']')


def _find_json_structure(text: str, open_char: str, close_char: str) -> str | None:
    """Find the first top-level JSON structure using string-aware delimiter counting.

    Works for both objects ({...}) and arrays ([...]) depending on the
    open_char/close_char pair provided.

    Respects:
    - String literals (delimiters inside "..." are ignored)
    - Escaped quotes inside strings (\\")

    Returns the structure substring or None if not found.
    """
    start = None
    depth = 0
    in_string = False
    i = 0

    while i < len(text):
        char = text[i]

        if in_string:
            if char == "\\" and i + 1 < len(text):
                # Skip escaped character entirely
                i += 2
                continue
            if char == '"':
                in_string = False
            i += 1
            continue

        # Not in string
        if char == '"':
            if start is not None:
                # Only track strings inside the JSON block
                in_string = True
            i += 1
            continue

        if char == open_char:
            if depth == 0:
                start = i
            depth += 1
            i += 1
            continue

        if char == close_char:
            depth -= 1
            if depth == 0 and start is not None:
                return text[start:i + 1]
            if depth < 0:
                # Stray closing delimiter — reset
                depth = 0
                start = None
            i += 1
            continue

        i += 1

    # If we opened a block but never closed it
    if start is not None and depth > 0:
        raise ParseError(
            f"Unclosed JSON structure starting at position {start}"
        )

    return None
