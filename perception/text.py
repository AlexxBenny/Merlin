# perception/text.py

"""
TextPerception — Converts raw text input into Percepts.

This is the simplest perception channel: stdin → Percept.
Speech and vision channels will follow the same pattern.
"""

import time
from brain.core import Percept


class TextPerception:
    """
    Produces Percepts from text strings.

    Does NOT read from stdin itself — the caller provides the text.
    This keeps the perception layer pure and testable.
    """

    def perceive(self, text: str) -> Percept:
        """
        Convert raw text into a Percept.

        Confidence is always 1.0 for typed text
        (no ambiguity in what the user typed).
        """
        return Percept(
            modality="text",
            payload=text.strip(),
            confidence=1.0,
            timestamp=time.time(),
        )
