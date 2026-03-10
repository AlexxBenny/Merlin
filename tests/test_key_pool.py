# tests/test_key_pool.py

"""
Tests for API key rotation system.

Covers:
- Round-robin cycling
- 3-tier resolution (role-specific → shared → singular)
- Backward compatibility with singular API_KEY
- Pool size tracking
- 429 retry with key rotation in clients
- reset_pools for test isolation
"""

import os
import unittest
from unittest.mock import MagicMock, patch

from models.key_pool import (
    _parse_keys,
    get_role_keys,
    resolve_api_key,
    pool_size,
    reset_pools,
)


class TestParseKeys(unittest.TestCase):
    """Parse comma-separated key strings."""

    def test_single_key(self):
        assert _parse_keys("key1") == ["key1"]

    def test_multiple_keys(self):
        assert _parse_keys("key1,key2,key3") == ["key1", "key2", "key3"]

    def test_strips_whitespace(self):
        assert _parse_keys(" key1 , key2 ") == ["key1", "key2"]

    def test_empty_string(self):
        assert _parse_keys("") == []

    def test_none(self):
        assert _parse_keys(None) == []

    def test_trailing_comma(self):
        assert _parse_keys("key1,key2,") == ["key1", "key2"]


class TestGetRoleKeys(unittest.TestCase):
    """3-tier key resolution."""

    def setUp(self):
        reset_pools()

    @patch.dict(os.environ, {
        "OPENROUTER_MISSION_COMPILER_API_KEYS": "role1,role2",
        "OPENROUTER_API_KEYS": "shared1",
        "OPENROUTER_API_KEY": "single",
    })
    def test_role_specific_takes_priority(self):
        keys = get_role_keys("openrouter", "mission_compiler")
        assert keys == ["role1", "role2"]

    @patch.dict(os.environ, {
        "OPENROUTER_API_KEYS": "shared1,shared2",
        "OPENROUTER_API_KEY": "single",
    }, clear=True)
    def test_shared_fallback(self):
        keys = get_role_keys("openrouter", "content_generator")
        assert keys == ["shared1", "shared2"]

    @patch.dict(os.environ, {
        "OPENROUTER_API_KEY": "single-key",
    }, clear=True)
    def test_singular_fallback(self):
        keys = get_role_keys("openrouter", "content_generator")
        assert keys == ["single-key"]

    @patch.dict(os.environ, {}, clear=True)
    def test_no_keys_raises(self):
        with self.assertRaises(ValueError) as ctx:
            get_role_keys("openrouter", "mission_compiler")
        assert "No API keys" in str(ctx.exception)

    @patch.dict(os.environ, {
        "GEMINI_MISSION_COMPILER_API_KEYS": "gkey1,gkey2",
    }, clear=True)
    def test_gemini_role_specific(self):
        keys = get_role_keys("gemini", "mission_compiler")
        assert keys == ["gkey1", "gkey2"]


class TestRoundRobin(unittest.TestCase):
    """Round-robin key cycling."""

    def setUp(self):
        reset_pools()

    @patch.dict(os.environ, {
        "OPENROUTER_API_KEYS": "a,b,c",
    }, clear=True)
    def test_cycles_through_keys(self):
        keys = [resolve_api_key("openrouter", "test") for _ in range(7)]
        assert keys == ["a", "b", "c", "a", "b", "c", "a"]

    @patch.dict(os.environ, {
        "OPENROUTER_API_KEY": "only",
    }, clear=True)
    def test_single_key_always_returns_same(self):
        keys = [resolve_api_key("openrouter", "test") for _ in range(3)]
        assert keys == ["only", "only", "only"]

    @patch.dict(os.environ, {
        "OPENROUTER_X_API_KEYS": "x1,x2",
        "OPENROUTER_Y_API_KEYS": "y1,y2,y3",
    }, clear=True)
    def test_roles_have_independent_cycles(self):
        x1 = resolve_api_key("openrouter", "x")
        y1 = resolve_api_key("openrouter", "y")
        x2 = resolve_api_key("openrouter", "x")
        y2 = resolve_api_key("openrouter", "y")
        assert x1 == "x1"
        assert y1 == "y1"
        assert x2 == "x2"
        assert y2 == "y2"


class TestPoolSize(unittest.TestCase):

    def setUp(self):
        reset_pools()

    @patch.dict(os.environ, {
        "OPENROUTER_API_KEYS": "a,b,c",
    }, clear=True)
    def test_pool_size(self):
        assert pool_size("openrouter", "test") == 3

    @patch.dict(os.environ, {
        "OPENROUTER_API_KEY": "single",
    }, clear=True)
    def test_pool_size_single(self):
        assert pool_size("openrouter", "test") == 1


class TestRouterKeyIntegration(unittest.TestCase):
    """Router creates clients with correct key resolution."""

    def setUp(self):
        reset_pools()

    @patch.dict(os.environ, {
        "OPENROUTER_API_KEY": "test-key",
    }, clear=True)
    def test_router_uses_env_key(self):
        from models.router import ModelRouter
        config = {
            "test_role": {
                "provider": "openrouter",
                "model": "test-model",
            }
        }
        router = ModelRouter(config)
        client = router.get_client("test_role")
        assert client.api_key == "test-key"

    def test_router_uses_inline_key(self):
        reset_pools()
        from models.router import ModelRouter
        config = {
            "test_role": {
                "provider": "openrouter",
                "model": "test-model",
                "api_key": "inline-key",
            }
        }
        router = ModelRouter(config)
        client = router.get_client("test_role")
        assert client.api_key == "inline-key"


class TestRetryOn429(unittest.TestCase):
    """429 retry with key rotation."""

    def setUp(self):
        reset_pools()

    @patch.dict(os.environ, {
        "OPENROUTER_API_KEYS": "key1,key2,key3",
    }, clear=True)
    def test_openrouter_retries_on_429(self):
        from models.openrouter_client import OpenRouterClient

        client = OpenRouterClient(
            model="test",
            api_key="key1",
            _pool_provider="openrouter",
            _pool_role="test",
        )

        call_count = 0
        keys_used = []

        original_do_request = client._do_request

        def mock_do_request(prompt, temp, fmt, timeout=None, max_tokens=None):
            nonlocal call_count
            call_count += 1
            keys_used.append(client.api_key)
            if call_count < 3:
                raise RuntimeError("OpenRouter API error 429: rate limited")
            return "success"

        client._do_request = mock_do_request
        result = client.complete("test prompt")

        assert result == "success"
        assert call_count == 3
        # First attempt uses initial key, subsequent rotate
        assert len(keys_used) == 3

    @patch.dict(os.environ, {
        "OPENROUTER_API_KEY": "only-key",
    }, clear=True)
    def test_single_key_no_retry(self):
        from models.openrouter_client import OpenRouterClient

        client = OpenRouterClient(
            model="test",
            api_key="only-key",
            # No pool metadata → no retry
        )

        client._do_request = MagicMock(
            side_effect=RuntimeError("OpenRouter API error 429")
        )

        with self.assertRaises(RuntimeError):
            client.complete("test")

        # Only one attempt (no pool → no retry)
        client._do_request.assert_called_once()


if __name__ == "__main__":
    unittest.main()
