# cortex/synonyms.py

"""
Global Synonym Dictionary — Centralized, not per-skill.

Expands user tokens before intent matching.
Central dictionary = no duplication, no inconsistency, no combinatorial drift.

Two dictionaries:
- VERB_SYNONYMS:  maps synonym → canonical verb (many-to-one)
- NOUN_SYNONYMS:  maps synonym → canonical noun (many-to-one)

Canonical forms should match the verbs/keywords declared in SkillContract.
"""

from typing import Dict


# Maps synonym verbs → canonical verbs that appear in SkillContract.intent_verbs
VERB_SYNONYMS: Dict[str, str] = {
    # set/adjust family
    "adjust": "set",
    "change": "set",
    "modify": "set",
    "put": "set",
    "make": "set",
    "crank": "set",
    "raise": "set",
    "lower": "set",
    "increase": "set",
    "decrease": "set",

    # open/launch family
    "launch": "open",
    "run": "open",
    "execute": "open",

    # close/quit family
    "quit": "close",
    "exit": "close",
    "kill": "close",
    "end": "close",
    "shut": "close",
    "terminate": "close",

    # play/resume family
    "resume": "play",
    "begin": "play",
    "unpause": "play",

    # pause/stop family
    "halt": "pause",
    "stop": "pause",

    # mute family
    "silence": "mute",
    "hush": "mute",

    # unmute family
    "unsilence": "unmute",

    # navigation
    "forward": "next",
    "skip": "next",
    "advance": "next",
    "back": "previous",
    "prev": "previous",
    "rewind": "previous",

    # toggle
    "enable": "toggle",
    "disable": "toggle",
    "flip": "toggle",

    # create family
    "make": "create",
    "new": "create",
    "add": "create",
    "mkdir": "create",

    # focus family
    "bring": "focus",
    "activate": "focus",

    # list/show family
    "show": "list",
    "display": "list",
    "what": "list",
}


# Maps synonym nouns → canonical nouns that appear in SkillContract.intent_keywords
NOUN_SYNONYMS: Dict[str, str] = {
    # volume
    "sound": "volume",
    "loudness": "volume",
    "speaker": "volume",
    "speakers": "volume",
    "audio": "volume",

    # brightness
    "light": "brightness",
    "screen": "brightness",
    "display": "brightness",

    # music/media
    "song": "music",
    "track": "music",
    "media": "music",
    "playback": "music",
    "tune": "music",
    "playlist": "music",

    # app
    "application": "app",
    "program": "app",
    "window": "app",
    "process": "app",

    # folder
    "directory": "folder",
    "dir": "folder",

    # night light
    "night light": "nightlight",
    "blue light": "nightlight",
    "night mode": "nightlight",
}


def expand_token(token: str) -> str:
    """Expand a single token through both synonym dictionaries.

    Returns canonical form if found, original otherwise.
    """
    lower = token.lower()
    if lower in VERB_SYNONYMS:
        return VERB_SYNONYMS[lower]
    if lower in NOUN_SYNONYMS:
        return NOUN_SYNONYMS[lower]
    return lower
