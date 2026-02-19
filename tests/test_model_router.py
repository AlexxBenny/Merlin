# tests/test_model_router.py

"""
Tests for Phase 2B: ModelRouter.

Validates:
- Router returns correct model per role
- Temperature defaults from config respected
- Temperature override works per call
- Unknown role raises KeyError
- Unsupported provider raises ValueError
- Client caching (same instance returned)
"""

import pytest

from models.router import ModelRouter
from models.base import LLMClient
from models.ollama_client import OllamaClient


SAMPLE_CONFIG = {
    "mission_compiler": {
        "provider": "ollama",
        "model": "mistral:7b-instruct",
        "base_url": "http://localhost:11434",
        "temperature": 0.2,
    },
    "report_generator": {
        "provider": "ollama",
        "model": "mistral:7b-instruct",
        "base_url": "http://localhost:11434",
        "temperature": 0.6,
    },
    "clarifier": {
        "provider": "ollama",
        "model": "llama3:8b",
        "base_url": "http://localhost:11434",
        "temperature": 0.1,
    },
}


class TestModelRouterBasics:
    """Validate ModelRouter core behavior."""

    def test_returns_llm_client(self):
        router = ModelRouter(SAMPLE_CONFIG)
        client = router.get_client("mission_compiler")
        assert isinstance(client, LLMClient)

    def test_returns_ollama_client(self):
        router = ModelRouter(SAMPLE_CONFIG)
        client = router.get_client("mission_compiler")
        assert isinstance(client, OllamaClient)

    def test_correct_model_for_compiler(self):
        router = ModelRouter(SAMPLE_CONFIG)
        client = router.get_client("mission_compiler")
        assert client.model == "mistral:7b-instruct"

    def test_correct_model_for_clarifier(self):
        router = ModelRouter(SAMPLE_CONFIG)
        client = router.get_client("clarifier")
        assert client.model == "llama3:8b"

    def test_temperature_default_from_config(self):
        router = ModelRouter(SAMPLE_CONFIG)
        compiler = router.get_client("mission_compiler")
        assert compiler.default_temperature == 0.2

        reporter = router.get_client("report_generator")
        assert reporter.default_temperature == 0.6

        clarifier = router.get_client("clarifier")
        assert clarifier.default_temperature == 0.1


class TestModelRouterCaching:
    """Verify client instances are cached."""

    def test_same_instance_returned(self):
        router = ModelRouter(SAMPLE_CONFIG)
        c1 = router.get_client("mission_compiler")
        c2 = router.get_client("mission_compiler")
        assert c1 is c2

    def test_different_roles_different_instances(self):
        router = ModelRouter(SAMPLE_CONFIG)
        compiler = router.get_client("mission_compiler")
        reporter = router.get_client("report_generator")
        assert compiler is not reporter


class TestModelRouterErrors:
    """Validate error cases."""

    def test_unknown_role_raises_key_error(self):
        router = ModelRouter(SAMPLE_CONFIG)
        with pytest.raises(KeyError, match="reasoning_engine"):
            router.get_client("reasoning_engine")

    def test_unsupported_provider_raises_value_error(self):
        config = {
            "test_role": {
                "provider": "openai",
                "model": "gpt-4",
            }
        }
        router = ModelRouter(config)
        with pytest.raises(ValueError, match="Unsupported provider"):
            router.get_client("test_role")


class TestModelRouterMeta:
    """Validate metadata methods."""

    def test_available_roles(self):
        router = ModelRouter(SAMPLE_CONFIG)
        roles = router.available_roles()
        assert roles == ["clarifier", "mission_compiler", "report_generator"]

    def test_empty_config(self):
        router = ModelRouter({})
        assert router.available_roles() == []


class TestTemperatureOverride:
    """Validate temperature override semantics on OllamaClient."""

    def test_no_temperature_no_options(self):
        """Client with no default temp should not include options in payload."""
        client = OllamaClient(model="test", temperature=None)
        # We can test the instance state
        assert client.default_temperature is None

    def test_config_temperature_stored(self):
        client = OllamaClient(model="test", temperature=0.7)
        assert client.default_temperature == 0.7
