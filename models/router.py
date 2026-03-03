# models/router.py

"""
ModelRouter — role-based LLM client lookup.

Reads config/models.yaml and produces one LLMClient per role.
Supports: ollama, openrouter, gemini, huggingface.

Design rules:
- Config is the SINGLE SOURCE OF TRUTH for model selection.
- No retries, no fallback.
- Temperature defaults come from config; callers can override per-call.
- API key resolution: per-role config → key_pool rotation → error.
"""

import logging
import os
from typing import Any, Dict, Optional

from models.base import LLMClient
from models.ollama_client import OllamaClient
from models.openrouter_client import OpenRouterClient
from models.gemini_client import GeminiClient
from models.huggingface_client import HuggingFaceClient
from models.key_pool import resolve_api_key, pool_size


logger = logging.getLogger(__name__)


def _resolve_api_key_for_role(
    cfg: Dict[str, Any], provider: str, role: str,
) -> str:
    """Resolve API key: config inline → key_pool rotation → error.

    Priority:
      1. Per-role api_key in models.yaml (highest)
      2. Key pool rotation (env vars, round-robin)
    """
    # 1. Inline key in config (highest priority)
    inline_key = cfg.get("api_key")
    if inline_key:
        return inline_key

    # 2. Key pool (env-based, round-robin)
    return resolve_api_key(provider, role)


def _build_client(
    cfg: Dict[str, Any], provider: str, role: str,
) -> LLMClient:
    """Build an LLMClient for a given provider + role config.

    The role is passed to key resolution so each role can use
    a different API key (or rotate across multiple keys).
    """
    if provider == "ollama":
        return OllamaClient(
            model=cfg.get("model", "llama3"),
            base_url=cfg.get("base_url", "http://localhost:11434"),
            timeout=cfg.get("timeout", 120.0),
            temperature=cfg.get("temperature"),
        )

    if provider == "openrouter":
        key = _resolve_api_key_for_role(cfg, "openrouter", role)
        num_keys = 1
        if not cfg.get("api_key"):
            # Using pool — track size for retry bounds
            num_keys = pool_size("openrouter", role)
        return OpenRouterClient(
            model=cfg["model"],
            api_key=key,
            base_url=cfg.get("base_url", "https://openrouter.ai/api/v1"),
            timeout=cfg.get("timeout", 120.0),
            temperature=cfg.get("temperature"),
            max_tokens=cfg.get("max_tokens"),
            _pool_provider="openrouter" if num_keys > 1 else None,
            _pool_role=role if num_keys > 1 else None,
        )

    if provider == "gemini":
        key = _resolve_api_key_for_role(cfg, "gemini", role)
        num_keys = 1
        if not cfg.get("api_key"):
            num_keys = pool_size("gemini", role)
        return GeminiClient(
            model=cfg["model"],
            api_key=key,
            base_url=cfg.get(
                "base_url",
                "https://generativelanguage.googleapis.com/v1beta",
            ),
            timeout=cfg.get("timeout", 120.0),
            temperature=cfg.get("temperature"),
            max_tokens=cfg.get("max_tokens"),
            _pool_provider="gemini" if num_keys > 1 else None,
            _pool_role=role if num_keys > 1 else None,
        )

    if provider == "huggingface":
        key = _resolve_api_key_for_role(cfg, "huggingface", role)
        return HuggingFaceClient(
            model=cfg["model"],
            api_key=key,
            base_url=cfg.get(
                "base_url", "https://api-inference.huggingface.co",
            ),
            timeout=cfg.get("timeout", 120.0),
            temperature=cfg.get("temperature"),
        )

    raise ValueError(
        f"Unsupported provider '{provider}'. "
        f"Supported: ollama, openrouter, gemini, huggingface"
    )


class ModelRouter:
    """
    Role → LLMClient router.

    Usage:
        router = ModelRouter(models_config)
        compiler = router.get_client("mission_compiler")
        reporter = router.get_client("report_generator")
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: The full models.yaml dict. Top-level keys are role names
                    (e.g. 'mission_compiler', 'report_generator', 'clarifier').
        """
        self._config = config
        self._clients: Dict[str, LLMClient] = {}

    def get_client(self, role: str) -> LLMClient:
        """
        Get (or lazily create) the LLM client for a given role.

        Args:
            role: One of the top-level keys in models.yaml.

        Returns:
            An LLMClient configured for the role.

        Raises:
            KeyError: if the role is not defined in config.
            ValueError: if the provider is unsupported.
        """
        if role in self._clients:
            return self._clients[role]

        if role not in self._config:
            raise KeyError(
                f"No model configuration for role '{role}'. "
                f"Available roles: {sorted(self._config.keys())}"
            )

        role_cfg = self._config[role]
        provider = role_cfg.get("provider", "ollama")

        client = _build_client(role_cfg, provider, role)
        self._clients[role] = client

        logger.info(
            "ModelRouter: created %s client for role '%s' (model=%s, temp=%s)",
            provider,
            role,
            role_cfg.get("model", "?"),
            role_cfg.get("temperature", "default"),
        )

        return client

    def available_roles(self) -> list:
        """Return all configured role names."""
        return sorted(self._config.keys())
