# models/router.py

"""
ModelRouter — role-based LLM client lookup.

Reads config/models.yaml and produces one LLMClient per role.
Supports: ollama, openrouter, gemini, huggingface.

Design rules:
- Config is the SINGLE SOURCE OF TRUTH for model selection.
- No retries, no fallback.
- Temperature defaults come from config; callers can override per-call.
- API key resolution: config.api_key → env var → error.
"""

import logging
import os
from typing import Any, Dict, Optional

from models.base import LLMClient
from models.ollama_client import OllamaClient
from models.openrouter_client import OpenRouterClient
from models.gemini_client import GeminiClient
from models.huggingface_client import HuggingFaceClient


logger = logging.getLogger(__name__)


def _resolve_api_key(cfg: Dict[str, Any], env_var: str) -> str:
    """Resolve API key: config → env → error."""
    key = cfg.get("api_key") or os.environ.get(env_var)
    if not key:
        raise ValueError(
            f"API key required. Set 'api_key' in models.yaml "
            f"or {env_var} environment variable."
        )
    return key


# Supported providers
_PROVIDER_FACTORIES = {
    "ollama": lambda cfg: OllamaClient(
        model=cfg.get("model", "llama3"),
        base_url=cfg.get("base_url", "http://localhost:11434"),
        timeout=cfg.get("timeout", 120.0),
        temperature=cfg.get("temperature"),
    ),
    "openrouter": lambda cfg: OpenRouterClient(
        model=cfg["model"],
        api_key=_resolve_api_key(cfg, "OPENROUTER_API_KEY"),
        base_url=cfg.get("base_url", "https://openrouter.ai/api/v1"),
        timeout=cfg.get("timeout", 120.0),
        temperature=cfg.get("temperature"),
        max_tokens=cfg.get("max_tokens"),
    ),
    "gemini": lambda cfg: GeminiClient(
        model=cfg["model"],
        api_key=_resolve_api_key(cfg, "GEMINI_API_KEY"),
        base_url=cfg.get(
            "base_url",
            "https://generativelanguage.googleapis.com/v1beta",
        ),
        timeout=cfg.get("timeout", 120.0),
        temperature=cfg.get("temperature"),
        max_tokens=cfg.get("max_tokens"),
    ),
    "huggingface": lambda cfg: HuggingFaceClient(
        model=cfg["model"],
        api_key=_resolve_api_key(cfg, "HUGGINGFACE_API_KEY"),
        base_url=cfg.get(
            "base_url", "https://api-inference.huggingface.co",
        ),
        timeout=cfg.get("timeout", 120.0),
        temperature=cfg.get("temperature"),
    ),
}


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

        factory = _PROVIDER_FACTORIES.get(provider)
        if factory is None:
            raise ValueError(
                f"Unsupported provider '{provider}' for role '{role}'. "
                f"Supported: {sorted(_PROVIDER_FACTORIES.keys())}"
            )

        client = factory(role_cfg)
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
