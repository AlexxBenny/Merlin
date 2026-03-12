# brain/ordinal_resolver.py

"""
Ordinal Entity Resolution — browser entity reference detection.

Detects ordinal/positional references in user queries when a browser
session with entities is active. Resolves them to browser.click actions.

Examples resolved:
    "open the first one"       → browser.click(entity_index=1)
    "click the second result"  → browser.click(entity_index=2)
    "select the third link"    → browser.click(entity_index=3)
    "open the 5th item"        → browser.click(entity_index=5)

NOT resolved (require full LLM reasoning):
    "open that one"            → needs anaphora resolution (future)
    "the avengers one"         → needs entity text matching (future)
    "the next one"             → needs context tracking (future)

Design rules:
    - Zero LLM. Pure regex + ordinal lookup.
    - Only fires when browser has entities AND ordinal detected.
    - Returns None if no ordinal detected → caller falls through to normal flow.
    - Entity index must exist in top_entities → prevents clicking non-existent items.
"""

import re
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Ordinal word → integer mapping
# ─────────────────────────────────────────────────────────────

_ORDINAL_WORDS: Dict[str, int] = {
    "first": 1, "1st": 1,
    "second": 2, "2nd": 2,
    "third": 3, "3rd": 3,
    "fourth": 4, "4th": 4,
    "fifth": 5, "5th": 5,
    "sixth": 6, "6th": 6,
    "seventh": 7, "7th": 7,
    "eighth": 8, "8th": 8,
    "ninth": 9, "9th": 9,
    "tenth": 10, "10th": 10,
}

# Pattern: "the Nth" or ordinal words, optionally followed by
# entity-like nouns (result, link, site, one, item, video, etc.)
_ORDINAL_PATTERN = re.compile(
    r"\b(?:the\s+)?"
    r"(?:"
    r"(?P<word>" + "|".join(re.escape(w) for w in _ORDINAL_WORDS) + r")"
    r"|"
    r"(?P<num>\d{1,2})(?:st|nd|rd|th)"
    r")"
    r"(?:\s+(?:one|result|link|site|item|video|option|entry|page|button))?"
    r"\b",
    re.IGNORECASE,
)

# Action verbs that confirm entity interaction intent
_ENTITY_ACTION_VERBS = frozenset({
    "open", "click", "select", "choose", "pick",
    "go", "visit", "view", "watch", "play",
})


def detect_ordinal_entity_reference(
    query: str,
    top_entities: List[Dict[str, Any]],
) -> Optional[Tuple[int, str]]:
    """Detect ordinal entity reference in a query.

    Args:
        query: User query text (raw, not normalized).
        top_entities: List of {index, type, text} dicts from BrowserWorldState.

    Returns:
        (entity_index, matched_text) if ordinal detected and entity exists.
        None if no ordinal found or entity index doesn't exist.
    """
    if not top_entities:
        return None

    text = query.lower().strip()

    # Check for action verbs (must have at least one to confirm intent)
    tokens = set(text.split())
    has_action_verb = bool(tokens & _ENTITY_ACTION_VERBS)
    if not has_action_verb:
        return None

    # Search for ordinal pattern
    match = _ORDINAL_PATTERN.search(text)
    if not match:
        return None

    # Extract numeric index
    if match.group("word"):
        index = _ORDINAL_WORDS[match.group("word").lower()]
    elif match.group("num"):
        index = int(match.group("num"))
    else:
        return None

    # Validate index exists in entities
    valid_indices = {e["index"] for e in top_entities}
    if index not in valid_indices:
        logger.debug(
            "Ordinal resolver: index %d not in entity list (valid: %s)",
            index, sorted(valid_indices),
        )
        return None

    logger.info(
        "Ordinal resolver: '%s' → entity_index=%d",
        query[:50], index,
    )
    return (index, match.group(0))
