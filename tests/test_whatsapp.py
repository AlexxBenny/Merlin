# tests/test_whatsapp.py

"""
Tests for WhatsApp integration components.

Tests:
- ContactResolver: exact, alias, fuzzy, ambiguous, not-found
- WhatsAppRateLimiter: acquire, depletion, refill, thread-safety
- WhatsApp skills: contract validation, no_op on ambiguity, send success/failure
- CommunicationChannel / ChannelMessage: dataclass integrity
"""

import time
import threading
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from providers.communication.base import CommunicationChannel, ChannelMessage
from providers.whatsapp.rate_limiter import (
    WhatsAppRateLimiter,
    RateLimitExceeded,
)
from providers.whatsapp.contact_resolver import (
    ContactResolver,
    ContactNotFoundError,
    ContactAmbiguousError,
)


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

class FakeKnowledgeStore:
    """Minimal mock of UserKnowledgeStore for contact tests."""

    def __init__(self, relationships: Dict[str, Any]):
        from dataclasses import dataclass

        @dataclass
        class FakeEntry:
            value: Any

        self._relationships = {
            k: FakeEntry(value=v) for k, v in relationships.items()
        }

    def query(self, key: str) -> Optional[Any]:
        """Simulate UserKnowledgeStore.query (cross-domain search)."""
        from memory.user_knowledge import normalize_key
        canonical = normalize_key(key)
        entry = self._relationships.get(canonical)
        return entry.value if entry else None


@pytest.fixture
def knowledge_store():
    """A fake knowledge store with sample contacts."""
    return FakeKnowledgeStore({
        "mom_phone": "919876543210",
        "dad_phone": "919876543211",
        "alex_work_phone": "14155551234",
        "alex_personal_phone": "14155555678",
    })


@pytest.fixture
def resolver(knowledge_store):
    return ContactResolver(knowledge_store)


@pytest.fixture
def rate_limiter():
    return WhatsAppRateLimiter(max_messages=3, window_seconds=2)


# ─────────────────────────────────────────────────────────────
# ChannelMessage tests
# ─────────────────────────────────────────────────────────────

class TestChannelMessage:
    def test_channel_message_is_frozen(self):
        msg = ChannelMessage(
            id="wa-test",
            channel="whatsapp",
            recipient_id="1234@s.whatsapp.net",
            contact_name="Test",
            direction="outbound",
            content_type="text",
            content="hello",
            status="sent",
            timestamp=time.time(),
        )
        assert msg.status == "sent"
        with pytest.raises(AttributeError):
            msg.status = "failed"  # type: ignore — frozen

    def test_channel_message_defaults(self):
        msg = ChannelMessage(
            id="wa-test",
            channel="whatsapp",
            recipient_id="1234",
            contact_name="Test",
            direction="outbound",
            content_type="text",
            content="hi",
            status="sent",
            timestamp=1.0,
        )
        assert msg.metadata == {}
        assert msg.error is None


# ─────────────────────────────────────────────────────────────
# ContactResolver tests
# ─────────────────────────────────────────────────────────────

class TestContactResolver:
    def test_exact_match(self, resolver):
        """Direct name lookup: mom → mom_phone."""
        phone, name = resolver.resolve("mom")
        assert phone == "919876543210"
        assert name == "Mom"

    def test_exact_match_case_insensitive(self, resolver):
        """Case-insensitive: Mom, MOM, mom all resolve."""
        phone, _ = resolver.resolve("Mom")
        assert phone == "919876543210"
        phone2, _ = resolver.resolve("MOM")
        assert phone2 == "919876543210"

    def test_exact_match_dad(self, resolver):
        phone, name = resolver.resolve("dad")
        assert phone == "919876543211"
        assert name == "Dad"

    def test_alias_resolution(self, resolver):
        """Alias: amma → mom → mom_phone."""
        phone, name = resolver.resolve("amma")
        assert phone == "919876543210"
        # Display name should be the canonical (mom)
        assert name == "Mom"

    def test_alias_mother(self, resolver):
        """Alias: mother → mom → mom_phone."""
        phone, _ = resolver.resolve("mother")
        assert phone == "919876543210"

    def test_alias_papa(self, resolver):
        """Alias: papa → dad → dad_phone."""
        phone, _ = resolver.resolve("papa")
        assert phone == "919876543211"

    def test_direct_phone_number(self, resolver):
        """Raw phone number passes through without resolution."""
        phone, name = resolver.resolve("919876543210")
        assert phone == "919876543210"
        assert name == "919876543210"

    def test_direct_phone_with_plus(self, resolver):
        phone, _ = resolver.resolve("+919876543210")
        assert phone == "919876543210"

    def test_direct_phone_with_dashes(self, resolver):
        phone, _ = resolver.resolve("1-415-555-1234")
        assert phone == "14155551234"

    def test_fuzzy_single_match(self, knowledge_store):
        """Fuzzy match when only one alex_*_phone exists."""
        store = FakeKnowledgeStore({"alex_phone": "12345"})
        resolver = ContactResolver(store)
        phone, name = resolver.resolve("alex")
        assert phone == "12345"

    def test_ambiguous_multiple_matches(self, resolver):
        """Multiple alex_*_phone entries → ContactAmbiguousError."""
        with pytest.raises(ContactAmbiguousError) as exc_info:
            resolver.resolve("alex")

        err = exc_info.value
        assert "alex" in err.user_message.lower()
        assert len(err.candidates) == 2  # alex_work + alex_personal

    def test_ambiguous_error_has_options(self, resolver):
        """ContactAmbiguousError carries structured candidates."""
        with pytest.raises(ContactAmbiguousError) as exc_info:
            resolver.resolve("alex")

        candidates = exc_info.value.candidates
        phones = [c["phone"] for c in candidates]
        assert "14155551234" in phones
        assert "14155555678" in phones

    def test_not_found(self, resolver):
        """Unknown contact → ContactNotFoundError with user message."""
        with pytest.raises(ContactNotFoundError) as exc_info:
            resolver.resolve("unknown_person")

        err = exc_info.value
        assert "unknown_person" in err.user_message
        assert "phone number" in err.user_message.lower()


# ─────────────────────────────────────────────────────────────
# RateLimiter tests
# ─────────────────────────────────────────────────────────────

class TestRateLimiter:
    def test_acquire_within_limit(self, rate_limiter):
        """Acquire up to max_messages should succeed."""
        assert rate_limiter.acquire() is True
        assert rate_limiter.acquire() is True
        assert rate_limiter.acquire() is True

    def test_acquire_exceeds_limit(self, rate_limiter):
        """Fourth acquire should fail (limit=3)."""
        rate_limiter.acquire()
        rate_limiter.acquire()
        rate_limiter.acquire()
        assert rate_limiter.acquire() is False

    def test_remaining_count(self, rate_limiter):
        assert rate_limiter.remaining == 3
        rate_limiter.acquire()
        assert rate_limiter.remaining == 2
        rate_limiter.acquire()
        assert rate_limiter.remaining == 1
        rate_limiter.acquire()
        assert rate_limiter.remaining == 0

    def test_acquire_or_raise(self, rate_limiter):
        """acquire_or_raise should raise RateLimitExceeded."""
        rate_limiter.acquire()
        rate_limiter.acquire()
        rate_limiter.acquire()
        with pytest.raises(RateLimitExceeded) as exc_info:
            rate_limiter.acquire_or_raise()
        assert exc_info.value.retry_after >= 0

    def test_refill_after_window(self):
        """After window expires, tokens should refill."""
        limiter = WhatsAppRateLimiter(max_messages=2, window_seconds=0.5)
        limiter.acquire()
        limiter.acquire()
        assert limiter.acquire() is False  # Depleted

        time.sleep(0.6)  # Wait for window to expire
        assert limiter.acquire() is True  # Refilled

    def test_thread_safety(self, rate_limiter):
        """Concurrent acquires should not exceed limit."""
        results = []

        def worker():
            r = rate_limiter.acquire()
            results.append(r)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Exactly 3 should succeed (limit=3)
        assert sum(results) == 3


# ─────────────────────────────────────────────────────────────
# WhatsApp Skills tests (with mocked client)
# ─────────────────────────────────────────────────────────────

class TestWhatsAppSendMessageSkill:
    def _make_skill(self, client_mock):
        from skills.whatsapp.send_message import WhatsAppSendMessageSkill
        return WhatsAppSendMessageSkill(whatsapp_client=client_mock)

    def test_contract_registered(self):
        """Contract has expected name and risk level."""
        from skills.whatsapp.send_message import WhatsAppSendMessageSkill
        c = WhatsAppSendMessageSkill.contract
        assert c.name == "whatsapp.send_message"
        assert c.risk_level == "destructive"
        assert "contact" in c.inputs

    def test_successful_send(self):
        """Successful send returns SkillResult with status info."""
        mock_client = MagicMock()
        mock_client.send_text.return_value = ChannelMessage(
            id="wa-test1",
            channel="whatsapp",
            recipient_id="919876543210",
            contact_name="Mom",
            direction="outbound",
            content_type="text",
            content="hello",
            status="sent",
            timestamp=time.time(),
        )

        skill = self._make_skill(mock_client)
        result = skill.execute(
            {"contact": "mom", "message_text": "hello"},
            world=MagicMock(),
        )

        assert result.status != "no_op"
        assert "Mom" in result.outputs["send_status"]

    def test_ambiguous_contact_returns_no_op(self):
        """Ambiguous contact triggers ask-back via no_op."""
        mock_client = MagicMock()
        mock_client.send_text.side_effect = ContactAmbiguousError(
            contact_name="alex",
            message="Multiple contacts match",
            candidates=[
                {"name": "Alex Work", "phone": "111"},
                {"name": "Alex Personal", "phone": "222"},
            ],
        )

        skill = self._make_skill(mock_client)
        result = skill.execute(
            {"contact": "alex", "message_text": "yo"},
            world=MagicMock(),
        )

        assert result.status == "no_op"
        assert result.metadata["reason"] == "ambiguous_input"
        assert "Multiple contacts" in result.metadata["message"]
        assert len(result.metadata["options"]) == 2

    def test_contact_not_found_returns_no_op(self):
        """Unknown contact triggers ask-back via no_op."""
        mock_client = MagicMock()
        mock_client.send_text.side_effect = ContactNotFoundError(
            contact_name="unknown",
            message="I don't have a phone number for 'unknown'.",
        )

        skill = self._make_skill(mock_client)
        result = skill.execute(
            {"contact": "unknown", "message_text": "test"},
            world=MagicMock(),
        )

        assert result.status == "no_op"
        assert result.metadata["reason"] == "ambiguous_input"
        assert result.metadata["options"] == []

    def test_send_failure_returns_error(self):
        """Provider failure returns error in outputs (not no_op)."""
        mock_client = MagicMock()
        mock_client.send_text.return_value = ChannelMessage(
            id="wa-fail",
            channel="whatsapp",
            recipient_id="123",
            contact_name="Test",
            direction="outbound",
            content_type="text",
            content="test",
            status="failed",
            timestamp=time.time(),
            error="Connection lost",
        )

        skill = self._make_skill(mock_client)
        result = skill.execute(
            {"contact": "test", "message_text": "test"},
            world=MagicMock(),
        )

        assert "Failed" in result.outputs["send_status"]
        assert "Connection lost" in result.outputs["send_status"]


class TestWhatsAppSendFileSkill:
    def test_contract_registered(self):
        """Contract has expected name and inputs."""
        from skills.whatsapp.send_file import WhatsAppSendFileSkill
        c = WhatsAppSendFileSkill.contract
        assert c.name == "whatsapp.send_file"
        assert c.risk_level == "destructive"
        assert "file_path" in c.inputs
        assert "contact" in c.inputs
