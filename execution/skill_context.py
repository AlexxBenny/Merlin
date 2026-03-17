# execution/skill_context.py

"""
SkillContext — Minimal, typed cross-cutting execution context.

Frozen. Built once per mission execution, NOT per startup.
Passed to all skills as an optional parameter.

Contains ONLY data that spans skill boundaries:
- user: Typed identity profile (from UserKnowledgeStore.get_user_profile())
- time: Current datetime at mission start

Everything else lives in its authoritative source:
- WorldSnapshot for system state (apps, media, etc.)
- WorldTimeline for events
- UserKnowledgeStore for full memory access (DI-injected separately)

Access rules:
- LLM prompt formatting → use UserKnowledgeStore.format_profile_for_prompt()
- Lightweight identity access in logic → use context.user
- Do NOT mix these — one is for prompts, one is for code paths.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class UserProfile:
    """Typed user identity — populated from UserKnowledgeStore.get_user_profile().

    All fields are Optional because memory accumulates over time.
    A fresh install has None for everything. As the user interacts,
    fields fill in without requiring code changes.
    """
    name: Optional[str] = None
    email: Optional[str] = None
    timezone: Optional[str] = None
    location: Optional[str] = None
    job_title: Optional[str] = None
    company: Optional[str] = None
    language: Optional[str] = None

    @classmethod
    def from_profile_dict(cls, profile: Dict[str, Any]) -> "UserProfile":
        """Build from UserKnowledgeStore.get_user_profile() dict.

        Only picks keys matching field names. Extra keys
        (prefixed prefs/traits/rels) are ignored — those are
        for prompt formatting, not typed access.
        """
        import dataclasses
        field_names = {f.name for f in dataclasses.fields(cls)}
        kwargs = {k: v for k, v in profile.items() if k in field_names}
        return cls(**kwargs)


@dataclass(frozen=True)
class SkillContext:
    """Minimal cross-cutting context for skill execution.

    Built per-mission (not once at startup) to keep time fresh.
    Frozen — immutable during execution.
    """
    user: UserProfile
    time: datetime
