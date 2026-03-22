# providers/whatsapp/contact_resolver.py

"""
ContactResolver — Resolves human names to WhatsApp JIDs.

Multi-stage resolution pipeline using UserKnowledgeStore.relationships:
    1. EXACT: relationships["{name}_phone"]
    2. ALIAS: _CANONICAL_KEYS[name] → canonical → relationships["{canonical}_phone"]
    3. FUZZY: Scan all *_phone entries, fuzzy match key prefix against name
    4. Result:
       - 0 candidates → ContactNotFoundError
       - 1 candidate  → resolved JID
       - 2+ candidates → ContactAmbiguousError (triggers ask-back)

Follows the same pattern as EntityResolver (app names → app_id)
and PreferenceResolver (preference queries → values).
Uses MERLIN's existing ask-back mechanism (SkillResult status="no_op")
with zero changes to merlin.py, executor, or orchestrator.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from memory.user_knowledge import UserKnowledgeStore, normalize_key

logger = logging.getLogger(__name__)


class ContactNotFoundError(Exception):
    """No phone number found for the given contact name.

    Carries user-facing message for the ask-back flow.
    """

    def __init__(self, contact_name: str, message: str):
        self.contact_name = contact_name
        self.user_message = message
        super().__init__(message)


class ContactAmbiguousError(Exception):
    """Multiple phone numbers match the given contact name.

    Carries options for the ask-back flow.
    """

    def __init__(
        self,
        contact_name: str,
        message: str,
        candidates: List[Dict[str, str]],
    ):
        self.contact_name = contact_name
        self.user_message = message
        self.candidates = candidates
        super().__init__(message)


# Common name aliases for contact resolution.
# Separate from UserKnowledgeStore._CANONICAL_KEYS (which handles
# preference/fact aliases like "sound" → "volume").
# These are WhatsApp-specific name mappings.
_CONTACT_ALIASES: Dict[str, str] = {
    "mother": "mom",
    "mum": "mom",
    "mommy": "mom",
    "amma": "mom",
    "ma": "mom",
    "father": "dad",
    "papa": "dad",
    "daddy": "dad",
    "appa": "dad",
    "pa": "dad",
    "brother": "bro",
    "sister": "sis",
    "husband": "hubby",
    "wife": "wifey",
    "grandmother": "grandma",
    "grandfather": "grandpa",
}


class ContactResolver:
    """Resolves human names to WhatsApp phone numbers via UserKnowledgeStore.

    Contact data is stored in the relationships domain as:
        mom_phone = "919876543210"
        dad_phone = "919876543211"
        alex_work_phone = "14155551234"

    Usage:
        resolver = ContactResolver(user_knowledge)
        phone, name = resolver.resolve("mom")
        # → ("919876543210", "mom")
    """

    def __init__(self, user_knowledge: UserKnowledgeStore):
        self._knowledge = user_knowledge

    def resolve(self, contact: str) -> Tuple[str, str]:
        """Resolve a contact name or phone number to (phone, display_name).

        Args:
            contact: Human name ("Mom"), alias ("amma"), or
                     raw phone number ("919876543210").

        Returns:
            Tuple of (phone_number, display_name).

        Raises:
            ContactNotFoundError: No number found (triggers ask-back).
            ContactAmbiguousError: Multiple matches (triggers ask-back).
        """
        # ── Direct phone number (digits only, 7-15 chars) ──
        stripped = contact.strip().replace("+", "").replace(" ", "").replace("-", "")
        if stripped.isdigit() and 7 <= len(stripped) <= 15:
            logger.info(
                "[CONTACT] Direct phone number: %s", stripped,
            )
            return stripped, contact.strip()

        # ── Stage 1: Exact match ──
        name = contact.strip().lower().replace(" ", "_")
        phone_key = f"{name}_phone"
        phone = self._knowledge.query(phone_key)
        if phone:
            logger.info(
                "[CONTACT] Exact match: %s → %s", name, phone,
            )
            return str(phone), name.replace("_", " ").title()

        # ── Stage 2: Alias resolution ──
        alias_name = _CONTACT_ALIASES.get(name)
        if alias_name:
            phone_key = f"{alias_name}_phone"
            phone = self._knowledge.query(phone_key)
            if phone:
                logger.info(
                    "[CONTACT] Alias match: %s → %s → %s",
                    name, alias_name, phone,
                )
                return str(phone), alias_name.replace("_", " ").title()

        # ── Stage 3: Fuzzy scan ──
        candidates = self._fuzzy_scan(name)

        if len(candidates) == 1:
            match = candidates[0]
            logger.info(
                "[CONTACT] Fuzzy single match: %s → %s (%s)",
                name, match["name"], match["phone"],
            )
            return match["phone"], match["name"]

        if len(candidates) > 1:
            # Build options list for ask-back
            options = [
                f"  {i+1}. {c['name']} ({c['phone']})"
                for i, c in enumerate(candidates[:5])
            ]
            message = (
                f"Multiple contacts match '{contact}':\n"
                + "\n".join(options)
                + "\nWhich one did you mean?"
            )
            raise ContactAmbiguousError(
                contact_name=contact,
                message=message,
                candidates=candidates[:5],
            )

        # ── Stage 4: Not found ──
        message = (
            f"I don't have a phone number for '{contact}'. "
            f"What is their WhatsApp number?"
        )
        raise ContactNotFoundError(
            contact_name=contact,
            message=message,
        )

    def _fuzzy_scan(self, name: str) -> List[Dict[str, str]]:
        """Scan all *_phone entries for fuzzy name matches.

        Looks at every relationship key ending in _phone and checks
        if the prefix fuzzy-matches the target name.
        """
        candidates = []
        # Access the raw relationships dict
        relationships = getattr(
            self._knowledge, "_relationships", {},
        )

        for key, entry in relationships.items():
            if not key.endswith("_phone"):
                continue
            # Extract the name prefix (e.g., "mom" from "mom_phone")
            prefix = key[:-6]  # strip "_phone"
            if not prefix:
                continue

            phone = str(entry.value)

            # Fuzzy check: substring match in either direction
            if name in prefix or prefix in name:
                display = prefix.replace("_", " ").title()
                candidates.append({
                    "name": display,
                    "phone": phone,
                    "key": key,
                })
            # Also check word overlap for multi-word names
            # e.g., "alex" matches "alex_work_phone"
            elif any(
                part == name or name in part or part in name
                for part in prefix.split("_")
            ):
                display = prefix.replace("_", " ").title()
                candidates.append({
                    "name": display,
                    "phone": phone,
                    "key": key,
                })

        return candidates
