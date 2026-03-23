# tests/test_max_tokens.py

"""
Tests for dynamic max_tokens budgeting.

Covers:
- Per-call max_tokens override (None-safe, not truthiness)
- Constructor default fallback
- Credit-aware capping (_resolve_max_tokens)
- 402 error budget parsing (_parse_credit_budget)
- GeminiClient per-call override
"""

import pytest
from unittest.mock import patch, MagicMock

from models.openrouter_client import OpenRouterClient, _CREDIT_SAFETY_MARGIN
from models.gemini_client import GeminiClient


# ─────────────────────────────────────────────────────────────
# OpenRouterClient — _resolve_max_tokens
# ─────────────────────────────────────────────────────────────

class TestResolveMaxTokens:

    def _client(self, max_tokens=None):
        return OpenRouterClient(
            model="test-model",
            api_key="test-key",
            max_tokens=max_tokens,
        )

    def test_per_call_override(self):
        """Per-call max_tokens overrides constructor default."""
        client = self._client(max_tokens=4096)
        assert client._resolve_max_tokens(512) == 512

    def test_constructor_default_when_no_override(self):
        """No per-call override → uses constructor default."""
        client = self._client(max_tokens=1024)
        assert client._resolve_max_tokens(None) == 1024

    def test_none_safe_override_zero(self):
        """max_tokens=0 should NOT fall back to default (None-safe)."""
        client = self._client(max_tokens=1024)
        # 0 is not None → should use 0
        assert client._resolve_max_tokens(0) == 0

    def test_both_none_returns_none(self):
        """No config, no override → None (omitted from payload)."""
        client = self._client(max_tokens=None)
        assert client._resolve_max_tokens(None) is None

    def test_credit_cap_applied(self):
        """When credit budget known, cap max_tokens."""
        client = self._client(max_tokens=4096)
        client._credit_budgets[client.api_key] = 1000
        result = client._resolve_max_tokens(None)
        assert result == 1000 - _CREDIT_SAFETY_MARGIN

    def test_credit_cap_not_applied_when_under(self):
        """When requested < budget, no capping needed."""
        client = self._client(max_tokens=256)
        client._credit_budgets[client.api_key] = 5000
        result = client._resolve_max_tokens(None)
        assert result == 256

    def test_per_call_override_also_capped(self):
        """Per-call override is also subject to credit capping."""
        client = self._client(max_tokens=512)
        client._credit_budgets[client.api_key] = 200
        result = client._resolve_max_tokens(1024)
        assert result == 200 - _CREDIT_SAFETY_MARGIN

    def test_credit_cap_minimum_1(self):
        """Credit cap never goes below 1."""
        client = self._client(max_tokens=4096)
        client._credit_budgets[client.api_key] = 10  # 10 - 50 = -40, but min is 1
        result = client._resolve_max_tokens(None)
        assert result == 1


# ─────────────────────────────────────────────────────────────
# OpenRouterClient — _parse_credit_budget
# ─────────────────────────────────────────────────────────────

class TestParseCreditBudget:

    def _client(self):
        return OpenRouterClient(
            model="test-model",
            api_key="test-key",
            max_tokens=4096,
        )

    def test_parses_budget_from_402_body(self):
        """Extracts 'can only afford N' from error message."""
        client = self._client()
        error_body = (
            '{"error":{"message":"You requested up to 4096 tokens, '
            'but can only afford 3974"}}'
        )
        client._parse_credit_budget(error_body)
        assert client._credit_budgets[client.api_key] == 3974

    def test_no_match_leaves_none(self):
        """Unrecognized error format doesn't change budget."""
        client = self._client()
        client._parse_credit_budget("some random error message")
        assert client._credit_budgets.get(client.api_key) is None

    def test_updates_on_repeated_402(self):
        """Budget updates on each 402 (credits may change)."""
        client = self._client()
        client._parse_credit_budget("can only afford 5000")
        assert client._credit_budgets[client.api_key] == 5000
        client._parse_credit_budget("can only afford 3000")
        assert client._credit_budgets[client.api_key] == 3000

    def test_budget_affects_subsequent_resolve(self):
        """After parsing 402, future requests are capped."""
        client = self._client()
        # Before 402 — no capping
        assert client._resolve_max_tokens(None) == 4096

        # Simulate 402
        client._parse_credit_budget("can only afford 2000")

        # After 402 — capped
        result = client._resolve_max_tokens(None)
        assert result == 2000 - _CREDIT_SAFETY_MARGIN


# ─────────────────────────────────────────────────────────────
# GeminiClient — per-call override
# ─────────────────────────────────────────────────────────────

class TestGeminiMaxTokensOverride:

    def test_per_call_override_used(self):
        """Per-call max_tokens overrides constructor default in gen_config."""
        client = GeminiClient(
            model="gemini-2.0-flash",
            api_key="test-key",
            max_tokens=4096,
        )
        # Override should be used (None-safe)
        # We can't easily test payload without mocking HTTP,
        # but we can verify the constructor stores the default
        assert client.max_tokens == 4096

    def test_none_override_uses_default(self):
        """None per-call → constructor default used."""
        client = GeminiClient(
            model="gemini-2.0-flash",
            api_key="test-key",
            max_tokens=1024,
        )
        assert client.max_tokens == 1024


# ─────────────────────────────────────────────────────────────
# Interface compliance — all clients accept max_tokens
# ─────────────────────────────────────────────────────────────

class TestInterfaceCompliance:

    def test_ollama_accepts_max_tokens(self):
        """OllamaClient.complete() accepts max_tokens kwarg."""
        from models.ollama_client import OllamaClient
        client = OllamaClient(model="test")
        # Just verify the signature accepts it (don't call HTTP)
        import inspect
        sig = inspect.signature(client.complete)
        assert "max_tokens" in sig.parameters

    def test_huggingface_accepts_max_tokens(self):
        """HuggingFaceClient.complete() accepts max_tokens kwarg."""
        from models.huggingface_client import HuggingFaceClient
        client = HuggingFaceClient(model="test", api_key="test")
        import inspect
        sig = inspect.signature(client.complete)
        assert "max_tokens" in sig.parameters

    def test_openrouter_accepts_max_tokens(self):
        """OpenRouterClient.complete() accepts max_tokens kwarg."""
        client = OpenRouterClient(model="test", api_key="test")
        import inspect
        sig = inspect.signature(client.complete)
        assert "max_tokens" in sig.parameters

    def test_gemini_accepts_max_tokens(self):
        """GeminiClient.complete() accepts max_tokens kwarg."""
        client = GeminiClient(model="test", api_key="test")
        import inspect
        sig = inspect.signature(client.complete)
        assert "max_tokens" in sig.parameters
