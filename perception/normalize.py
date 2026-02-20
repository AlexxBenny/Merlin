# perception/normalize.py

"""
Deterministic text normalization for reflex matching.

Single source of truth for input normalization.
Called by BrainCore.route() and ReflexEngine.try_match().

NEVER used for LLM prompts, logs, or audit trail —
only for pattern matching against reflex templates.

Rules (order matters):
1. Strip leading/trailing whitespace
2. Lowercase
3. Strip trailing sentence punctuation (.!?)
4. Collapse internal whitespace
5. Strip leading speech filler words
"""

import re

# Leading fillers common in voice transcription.
# Conservative — only unambiguous, semantically empty prefixes.
# Allows optional comma/punctuation after filler (e.g., "Um, mute")
_FILLER_PREFIX = re.compile(
    r"^(?:um|uh|erm|hmm|hey|okay|ok|so|well|like|ah)[,;]?\s+",
    re.IGNORECASE,
)

# Trailing sentence punctuation (Whisper always adds these)
_TRAILING_PUNCT = re.compile(r"[.!?]+$")

# Multiple whitespace → single space
_MULTI_SPACE = re.compile(r"\s+")


def normalize_for_matching(text: str) -> str:
    """
    Normalize text for deterministic reflex matching.

    Strips trailing punctuation, collapses whitespace,
    removes leading speech fillers. Preserves internal
    punctuation (e.g., "2.5", "Dr. Smith").

    Returns empty string for empty/whitespace-only input.
    """
    if not text:
        return ""

    result = text.strip().lower()

    # Strip leading speech fillers
    result = _FILLER_PREFIX.sub("", result)

    # Strip trailing sentence punctuation
    result = _TRAILING_PUNCT.sub("", result)

    # Collapse internal whitespace
    result = _MULTI_SPACE.sub(" ", result).strip()

    return result
