# providers/whatsapp/contact_resolver.py

"""
ContactResolver — Hybrid resolver: deterministic-first, discovery-second.

Resolution priority (deterministic → discovery → fallback → learning):

    Stage 1  DETERMINISTIC (UserKnowledgeStore)
             relationships["{name}_phone"] or alias → canonical → phone
             ✓ Explicit, auditable, user-controlled

    Stage 2  DISCOVERY (Neonize contact store)
             client.get_all_contacts() + fuzzy name matching
             ✓ Uses synced WhatsApp contacts from linked device
             ✓ 0 matches → falls through
             ✓ 1 strong match → use (and optionally learn)
             ✓ 2+ matches → ask-back

    Stage 3  RAW PHONE (direct passthrough)
             Digits-only input → use as-is
             ✓ Already handled first (before stages 1-2)

    Stage 4  LEARNING (optional, after successful send)
             Cache successful neonize resolution → UserKnowledgeStore
             ✓ Future calls for same name become deterministic

This preserves MERLIN's deterministic-first design:
- A mapped contact NEVER goes through fuzzy matching
- Neonize is a *discovery* layer, not a source of truth
- Ambiguity ALWAYS triggers ask-back (no silent wrong sends)
"""

import logging
import re
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


# ── WhatsApp-specific nickname aliases ──
# Maps colloquial names to canonical relationship keys.
# These are checked BEFORE neonize discovery (Stage 1, alias sub-step).
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

# Characters to strip when normalizing contact names for comparison
_STRIP_RE = re.compile(r"[^\w\s]", re.UNICODE)


class ContactResolver:
    """Hybrid contact resolver: deterministic-first, discovery-second.

    Stage 1: UserKnowledgeStore  (explicit mappings — deterministic)
    Stage 2: Neonize contacts    (synced from phone — discovery/fuzzy)
    Stage 3: Raw phone number    (digits-only input — passthrough)

    Plus optional learning: cache neonize hits into UserKnowledgeStore.

    Args:
        user_knowledge: MERLIN's UserKnowledgeStore instance.
        connection_manager: Optional. WhatsAppConnectionManager instance.
                           If None or not connected, Stage 2 is skipped.
    """

    def __init__(
        self,
        user_knowledge: UserKnowledgeStore,
        connection_manager: Any = None,
    ):
        self._knowledge = user_knowledge
        self._conn_manager = connection_manager

    def resolve(self, contact: str) -> Tuple[str, str]:
        """Resolve a contact name or phone number to (phone, display_name).

        Args:
            contact: Human name ("Mom"), alias ("amma"), neonize-stored
                     contact name ("Alex"), or raw phone ("919876543210").

        Returns:
            Tuple of (phone_number, display_name).

        Raises:
            ContactNotFoundError: No number found (triggers ask-back).
            ContactAmbiguousError: Multiple matches (triggers ask-back).
        """
        # ── Stage 0: Direct phone number (digits-only, 7-15 chars) ──
        stripped = (
            contact.strip()
            .replace("+", "")
            .replace(" ", "")
            .replace("-", "")
        )
        if stripped.isdigit() and 7 <= len(stripped) <= 15:
            logger.info("[CONTACT] Direct phone number: %s", stripped)
            return stripped, contact.strip()

        # ── Stage 1: DETERMINISTIC — UserKnowledgeStore ──
        result = self._resolve_from_knowledge(contact)
        if result is not None:
            return result

        # ── Stage 2: DISCOVERY — Neonize contact store ──
        result = self._resolve_from_neonize(contact)
        if result is not None:
            return result

        # ── Stage 3: NOT FOUND — trigger ask-back ──
        message = (
            f"I don't have a phone number for '{contact}'. "
            f"What is their WhatsApp number?"
        )
        raise ContactNotFoundError(
            contact_name=contact,
            message=message,
        )

    # ─────────────────────────────────────────────────────────
    # Stage 1: UserKnowledgeStore (deterministic)
    # ─────────────────────────────────────────────────────────

    def _resolve_from_knowledge(
        self, contact: str,
    ) -> Optional[Tuple[str, str]]:
        """Try to resolve from explicit UserKnowledgeStore mappings.

        Checks:
            1. Exact: relationships["{name}_phone"]
            2. Alias: _CONTACT_ALIASES[name] → canonical → phone
            3. Fuzzy scan of all *_phone keys in relationships
        """
        name = contact.strip().lower().replace(" ", "_")

        # 1a. Exact key lookup
        phone_key = f"{name}_phone"
        phone = self._knowledge.query(phone_key)
        if phone:
            logger.info("[CONTACT] Stage1 exact: %s → %s", name, phone)
            return str(phone), name.replace("_", " ").title()

        # 1b. Alias resolution
        alias_name = _CONTACT_ALIASES.get(name)
        if alias_name:
            phone_key = f"{alias_name}_phone"
            phone = self._knowledge.query(phone_key)
            if phone:
                logger.info(
                    "[CONTACT] Stage1 alias: %s → %s → %s",
                    name, alias_name, phone,
                )
                return str(phone), alias_name.replace("_", " ").title()

        # 1c. Fuzzy scan of *_phone keys
        candidates = self._fuzzy_scan_knowledge(name)
        if len(candidates) == 1:
            match = candidates[0]
            logger.info(
                "[CONTACT] Stage1 fuzzy: %s → %s (%s)",
                name, match["name"], match["phone"],
            )
            return match["phone"], match["name"]

        if len(candidates) > 1:
            self._raise_ambiguous(contact, candidates)

        # Not found in knowledge store — fall through to Stage 2
        return None

    def _fuzzy_scan_knowledge(
        self, name: str,
    ) -> List[Dict[str, str]]:
        """Scan all *_phone entries in UserKnowledgeStore for fuzzy matches."""
        candidates = []
        relationships = getattr(self._knowledge, "_relationships", {})

        for key, entry in relationships.items():
            if not key.endswith("_phone"):
                continue
            prefix = key[:-6]  # strip "_phone"
            if not prefix:
                continue

            phone = str(entry.value)

            # Substring match in either direction
            if name in prefix or prefix in name:
                display = prefix.replace("_", " ").title()
                candidates.append({
                    "name": display,
                    "phone": phone,
                    "key": key,
                })
            # Word overlap for multi-word names
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

    # ─────────────────────────────────────────────────────────
    # Stage 2: Neonize contact store (discovery)
    # ─────────────────────────────────────────────────────────

    def _resolve_from_neonize(
        self, contact: str,
    ) -> Optional[Tuple[str, str]]:
        """Try to discover the contact from the neonize-synced contact store.

        Uses client.contact.get_all_contacts() which returns WhatsApp-synced
        contacts from the linked device's app state as a list of Contact
        protobuf objects, each with .JID and .Info fields.

        Returns None if neonize is unavailable or no matches found.
        Raises ContactAmbiguousError if multiple matches found.
        """
        if self._conn_manager is None:
            logger.debug(
                "[CONTACT] Stage2 skipped — no connection manager",
            )
            return None

        # Lazily get the connected neonize client
        try:
            neonize_client = self._conn_manager.get_client()
        except RuntimeError:
            logger.debug(
                "[CONTACT] Stage2 skipped — WhatsApp not connected",
            )
            return None

        try:
            # Contacts are on client.contact (ContactStore), not client
            all_contacts = neonize_client.contact.get_all_contacts()
        except Exception as e:
            logger.warning(
                "[CONTACT] Stage2 failed — get_all_contacts error: %s", e,
            )
            return None

        if not all_contacts:
            return None

        # Normalize the search name
        search = self._normalize_name(contact)
        candidates = []

        for contact_entry in all_contacts:
            # Each entry is a Contact protobuf with .JID and .Info fields
            # .Info has: PushName, FullName, FirstName, BusinessName
            jid = contact_entry.JID
            contact_info = contact_entry.Info
            names = self._extract_names(contact_info)

            for raw_name in names:
                normalized = self._normalize_name(raw_name)
                if not normalized:
                    continue

                # Exact normalized match (strongest signal)
                if normalized == search:
                    candidates.append({
                        "name": raw_name,
                        "phone": self._jid_to_phone(jid),
                        "jid": str(jid),
                        "match_type": "exact",
                    })
                # First name match (e.g., "Alex" matches "Alex Kumar")
                elif (
                    normalized.startswith(search + " ")
                    or search == normalized.split()[0]
                ):
                    candidates.append({
                        "name": raw_name,
                        "phone": self._jid_to_phone(jid),
                        "jid": str(jid),
                        "match_type": "first_name",
                    })

        if not candidates:
            return None

        # Deduplicate by phone number (same person, different name fields)
        seen_phones = set()
        unique = []
        for c in candidates:
            if c["phone"] not in seen_phones:
                seen_phones.add(c["phone"])
                unique.append(c)
        candidates = unique

        # Prefer exact matches over first-name matches
        exact = [c for c in candidates if c["match_type"] == "exact"]
        if len(exact) == 1:
            match = exact[0]
            logger.info(
                "[CONTACT] Stage2 exact neonize match: %s → %s (%s)",
                contact, match["name"], match["phone"],
            )
            return match["phone"], match["name"]

        if len(candidates) == 1:
            match = candidates[0]
            logger.info(
                "[CONTACT] Stage2 single neonize match: %s → %s (%s)",
                contact, match["name"], match["phone"],
            )
            return match["phone"], match["name"]

        if len(candidates) > 1:
            self._raise_ambiguous(contact, candidates)

        return None

    # ─────────────────────────────────────────────────────────
    # Stage 4: Learning (called externally after successful send)
    # ─────────────────────────────────────────────────────────

    def learn_contact(
        self, name: str, phone: str, source: str = "neonize",
    ) -> None:
        """Cache a successful resolution into UserKnowledgeStore.

        Called by WhatsAppClient after a successful send, so future
        queries for the same name resolve deterministically (Stage 1).

        Args:
            name: The original contact name used ("Alex").
            phone: The resolved phone number ("14155551234").
            source: Where the resolution came from ("neonize", "user").
        """
        canonical = name.strip().lower().replace(" ", "_")
        phone_key = f"{canonical}_phone"

        # Only learn if not already mapped (don't overwrite explicit mappings)
        existing = self._knowledge.query(phone_key)
        if existing is not None:
            logger.debug(
                "[CONTACT] Skip learning — %s already mapped to %s",
                phone_key, existing,
            )
            return

        try:
            self._knowledge.set_relationship(phone_key, phone)
            logger.info(
                "[CONTACT] Learned: %s → %s (source=%s)",
                phone_key, phone, source,
            )
        except Exception as e:
            logger.warning(
                "[CONTACT] Failed to learn %s → %s: %s",
                phone_key, phone, e,
            )

    # ─────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _normalize_name(name: str) -> str:
        """Normalize a contact name for comparison.

        Strips emoji, punctuation, extra whitespace, and lowercases.
        "Alex 🔥" → "alex"
        "+91 98765 43210" → "91 98765 43210"
        "Office Alex" → "office alex"
        """
        # Strip non-word, non-space chars (emoji, punctuation)
        cleaned = _STRIP_RE.sub("", name)
        # Collapse whitespace and lowercase
        return " ".join(cleaned.split()).lower().strip()

    @staticmethod
    def _jid_to_phone(jid) -> str:
        """Extract phone number from a neonize JID protobuf.

        Neonize JIDs are Neonize_pb2.JID protobuf objects with fields:
        - User: phone number string (e.g., "918714066264")
        - Server: server string (e.g., "s.whatsapp.net")

        Falls back to str parsing for non-protobuf JIDs.
        """
        # Protobuf JID — access .User field directly
        user = getattr(jid, "User", None)
        if user:
            return str(user)
        # Fallback: string JID format "phone@server"
        jid_str = str(jid)
        if "@" in jid_str:
            return jid_str.split("@")[0]
        return jid_str

    @staticmethod
    def _extract_names(contact_info) -> List[str]:
        """Extract all possible name strings from a neonize ContactInfo.

        ContactInfo is a protobuf object with fields:
        - PushName (WhatsApp display name set by the contact)
        - FullName (contact book full name)
        - FirstName (contact book first name)
        - BusinessName (business profile name)

        Returns a list of non-empty name strings.
        """
        name_fields = [
            "PushName", "FullName", "FirstName", "BusinessName",
        ]
        names = []
        for field in name_fields:
            value = getattr(contact_info, field, None)
            if value and isinstance(value, str) and value.strip():
                names.append(value.strip())
        return names

    def _raise_ambiguous(
        self, contact: str, candidates: List[Dict[str, str]],
    ) -> None:
        """Raise ContactAmbiguousError with formatted options."""
        display_candidates = candidates[:5]
        options = [
            f"  {i+1}. {c['name']} ({c['phone']})"
            for i, c in enumerate(display_candidates)
        ]
        message = (
            f"Multiple contacts match '{contact}':\n"
            + "\n".join(options)
            + "\nWhich one did you mean?"
        )
        raise ContactAmbiguousError(
            contact_name=contact,
            message=message,
            candidates=display_candidates,
        )
