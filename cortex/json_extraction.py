# cortex/json_extraction.py

"""
Defensive JSON block extraction from LLM output.

LLMs often wrap valid JSON in markdown fences, preamble text,
or trailing commentary. This module extracts the first valid
JSON object block.

String-aware brace counting:
- Braces inside quoted strings are ignored.
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

    # Step 1: Strip markdown fences if present
    stripped = _strip_markdown_fences(text)

    # Step 2: Find and extract the first top-level {...} block
    block = _find_json_object(stripped)
    if block is None:
        raise ParseError(
            f"No JSON object found in LLM response "
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
    """Find the first top-level JSON object using string-aware brace counting.

    Respects:
    - String literals (braces inside "..." are ignored)
    - Escaped quotes inside strings (\\\")
    - Does NOT attempt to parse arrays as top-level (only objects)

    Returns the object substring or None if not found.
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

        if char == '{':
            if depth == 0:
                start = i
            depth += 1
            i += 1
            continue

        if char == '}':
            depth -= 1
            if depth == 0 and start is not None:
                return text[start:i + 1]
            if depth < 0:
                # Stray closing brace — reset
                depth = 0
                start = None
            i += 1
            continue

        i += 1

    # If we opened a block but never closed it
    if start is not None and depth > 0:
        raise ParseError(
            f"Unclosed JSON object starting at position {start}"
        )

    return None
