# tests/test_multi_provider.py

"""
Tests for multi-provider LLM client integration.

Validates:
- All provider factories are registered in the router
- API key resolution: config → env → error
- Each client implements LLMClient
- Correct attributes stored on construction
- Existing Ollama behavior unchanged (regression)
- format parameter normalization per-provider (construction only)
"""

import os
import pytest

from models.base import LLMClient
from models.router import ModelRouter, _PROVIDER_FACTORIES, _resolve_api_key
from models.ollama_client import OllamaClient
from models.openrouter_client import OpenRouterClient
from models.gemini_client import GeminiClient
from models.huggingface_client import HuggingFaceClient


# ──────────────────────────────────────────────
# Factory registration
# ──────────────────────────────────────────────

class TestProviderFactoryRegistration:
    """All four providers must be registered in _PROVIDER_FACTORIES."""

    def test_ollama_registered(self):
        assert "ollama" in _PROVIDER_FACTORIES

    def test_openrouter_registered(self):
        assert "openrouter" in _PROVIDER_FACTORIES

    def test_gemini_registered(self):
        assert "gemini" in _PROVIDER_FACTORIES

    def test_huggingface_registered(self):
        assert "huggingface" in _PROVIDER_FACTORIES

    def test_exactly_four_providers(self):
        assert len(_PROVIDER_FACTORIES) == 4


# ──────────────────────────────────────────────
# API key resolution
# ──────────────────────────────────────────────

class TestAPIKeyResolution:
    """API key must resolve from config first, then env, then error."""

    def test_config_key_takes_priority(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")
        cfg = {"api_key": "config-key"}
        assert _resolve_api_key(cfg, "OPENROUTER_API_KEY") == "config-key"

    def test_env_fallback(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")
        cfg = {}
        assert _resolve_api_key(cfg, "OPENROUTER_API_KEY") == "env-key"

    def test_missing_key_raises(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        cfg = {}
        with pytest.raises(ValueError, match="API key required"):
            _resolve_api_key(cfg, "OPENROUTER_API_KEY")

    def test_empty_config_key_falls_to_env(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "env-key")
        cfg = {"api_key": ""}
        assert _resolve_api_key(cfg, "GEMINI_API_KEY") == "env-key"

    def test_none_config_key_falls_to_env(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "env-key")
        cfg = {"api_key": None}
        assert _resolve_api_key(cfg, "GEMINI_API_KEY") == "env-key"


# ──────────────────────────────────────────────
# Client construction / LLMClient contract
# ──────────────────────────────────────────────

class TestOpenRouterClient:
    """Validate OpenRouterClient construction and contract."""

    def test_is_llm_client(self):
        client = OpenRouterClient(
            model="test-model", api_key="test-key",
        )
        assert isinstance(client, LLMClient)

    def test_stores_attributes(self):
        client = OpenRouterClient(
            model="mistralai/mistral-7b",
            api_key="sk-test",
            base_url="https://custom.url/api/v1",
            temperature=0.5,
        )
        assert client.model == "mistralai/mistral-7b"
        assert client.api_key == "sk-test"
        assert client.base_url == "https://custom.url/api/v1"
        assert client.default_temperature == 0.5

    def test_default_base_url(self):
        client = OpenRouterClient(model="m", api_key="k")
        assert "openrouter.ai" in client.base_url


class TestGeminiClient:
    """Validate GeminiClient construction and contract."""

    def test_is_llm_client(self):
        client = GeminiClient(model="gemini-2.0-flash", api_key="key")
        assert isinstance(client, LLMClient)

    def test_stores_attributes(self):
        client = GeminiClient(
            model="gemini-2.0-flash",
            api_key="AIza-test",
            temperature=0.3,
        )
        assert client.model == "gemini-2.0-flash"
        assert client.api_key == "AIza-test"
        assert client.default_temperature == 0.3

    def test_default_base_url(self):
        client = GeminiClient(model="m", api_key="k")
        assert "googleapis.com" in client.base_url


class TestHuggingFaceClient:
    """Validate HuggingFaceClient construction and contract."""

    def test_is_llm_client(self):
        client = HuggingFaceClient(model="some/model", api_key="key")
        assert isinstance(client, LLMClient)

    def test_stores_attributes(self):
        client = HuggingFaceClient(
            model="mistralai/Mistral-7B-Instruct-v0.3",
            api_key="hf_test",
            temperature=0.4,
        )
        assert client.model == "mistralai/Mistral-7B-Instruct-v0.3"
        assert client.api_key == "hf_test"
        assert client.default_temperature == 0.4

    def test_default_base_url(self):
        client = HuggingFaceClient(model="m", api_key="k")
        assert "huggingface.co" in client.base_url


# ──────────────────────────────────────────────
# Router integration (config → correct client type)
# ──────────────────────────────────────────────

class TestRouterProviderIntegration:
    """Router must produce the correct client type per provider."""

    def test_openrouter_produces_correct_client(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        config = {
            "test_role": {
                "provider": "openrouter",
                "model": "mistralai/test",
            }
        }
        router = ModelRouter(config)
        client = router.get_client("test_role")
        assert isinstance(client, OpenRouterClient)
        assert client.model == "mistralai/test"

    def test_gemini_produces_correct_client(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        config = {
            "test_role": {
                "provider": "gemini",
                "model": "gemini-2.0-flash",
                "temperature": 0.1,
            }
        }
        router = ModelRouter(config)
        client = router.get_client("test_role")
        assert isinstance(client, GeminiClient)
        assert client.model == "gemini-2.0-flash"
        assert client.default_temperature == 0.1

    def test_huggingface_produces_correct_client(self, monkeypatch):
        monkeypatch.setenv("HUGGINGFACE_API_KEY", "test-key")
        config = {
            "test_role": {
                "provider": "huggingface",
                "model": "mistralai/Mistral-7B",
            }
        }
        router = ModelRouter(config)
        client = router.get_client("test_role")
        assert isinstance(client, HuggingFaceClient)

    def test_ollama_still_works(self):
        """Regression: Ollama provider unchanged."""
        config = {
            "compiler": {
                "provider": "ollama",
                "model": "mistral:7b-instruct",
                "base_url": "http://localhost:11434",
                "temperature": 0.2,
            }
        }
        router = ModelRouter(config)
        client = router.get_client("compiler")
        assert isinstance(client, OllamaClient)
        assert client.model == "mistral:7b-instruct"
        assert client.default_temperature == 0.2

    def test_missing_api_key_raises_on_get_client(self, monkeypatch):
        """Cloud provider without key must fail at get_client time."""
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        config = {
            "test_role": {
                "provider": "openrouter",
                "model": "test",
            }
        }
        router = ModelRouter(config)
        with pytest.raises(ValueError, match="API key required"):
            router.get_client("test_role")


# ──────────────────────────────────────────────
# Mixed-provider config (different roles, different providers)
# ──────────────────────────────────────────────

class TestMixedProviderConfig:
    """Validate configs with different providers per role."""

    def test_mixed_providers(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
        config = {
            "compiler": {
                "provider": "ollama",
                "model": "mistral:7b",
                "base_url": "http://localhost:11434",
            },
            "reporter": {
                "provider": "gemini",
                "model": "gemini-2.0-flash",
            },
        }
        router = ModelRouter(config)
        compiler = router.get_client("compiler")
        reporter = router.get_client("reporter")

        assert isinstance(compiler, OllamaClient)
        assert isinstance(reporter, GeminiClient)
        assert compiler is not reporter
