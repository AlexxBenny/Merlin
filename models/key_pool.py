# models/key_pool.py

"""
API Key Pool — Round-robin key rotation per role.

Designed for single-user burst patterns:
- Short burst (5-8 LLM calls in 2-3 seconds)
- Then idle
- Then another burst

Resolution order (most-specific → least-specific):
  1. {PROVIDER}_{ROLE}_API_KEYS   (role-specific, plural, comma-separated)
  2. {PROVIDER}_API_KEYS          (shared, plural, comma-separated)
  3. {PROVIDER}_API_KEY           (singular fallback, backward compat)

Examples:
  OPENROUTER_MISSION_COMPILER_API_KEYS=key1,key2,key3
  OPENROUTER_API_KEYS=shared1,shared2
  OPENROUTER_API_KEY=single-key

No YAML keys. No state persistence. No tracking.
Just in-memory itertools.cycle per pool.
"""

import logging
import os
from itertools import cycle
from typing import Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)


# Provider → env var prefix mapping
_PROVIDER_ENV_PREFIX = {
    "openrouter": "OPENROUTER",
    "gemini": "GEMINI",
    "huggingface": "HUGGINGFACE",
}

# In-memory key cycles — keyed by "provider:role"
_key_cycles: Dict[str, Iterator[str]] = {}
_key_counts: Dict[str, int] = {}  # For retry bounds


def _parse_keys(value: Optional[str]) -> List[str]:
    """Parse comma-separated key string into list."""
    if not value:
        return []
    return [k.strip() for k in value.split(",") if k.strip()]


def get_role_keys(provider: str, role: str) -> List[str]:
    """Get all available API keys for a provider+role combination.

    Resolution order:
      1. {PROVIDER}_{ROLE}_API_KEYS  (role-specific plural)
      2. {PROVIDER}_API_KEYS         (shared plural)
      3. {PROVIDER}_API_KEY          (singular fallback)

    Returns:
        List of API keys (at least one).

    Raises:
        ValueError: if no keys found anywhere.
    """
    prefix = _PROVIDER_ENV_PREFIX.get(provider, provider.upper())
    role_upper = role.upper()

    # 1. Role-specific plural
    keys = _parse_keys(os.environ.get(f"{prefix}_{role_upper}_API_KEYS"))
    if keys:
        logger.debug(
            "KeyPool: %s/%s using %d role-specific keys",
            provider, role, len(keys),
        )
        return keys

    # 2. Shared plural
    keys = _parse_keys(os.environ.get(f"{prefix}_API_KEYS"))
    if keys:
        logger.debug(
            "KeyPool: %s/%s using %d shared keys",
            provider, role, len(keys),
        )
        return keys

    # 3. Singular fallback (backward compat)
    single = os.environ.get(f"{prefix}_API_KEY")
    if single and single.strip():
        logger.debug(
            "KeyPool: %s/%s using singular fallback key",
            provider, role,
        )
        return [single.strip()]

    raise ValueError(
        f"No API keys for {provider}/{role}. "
        f"Set {prefix}_{role_upper}_API_KEYS, "
        f"{prefix}_API_KEYS, or {prefix}_API_KEY."
    )


def resolve_api_key(provider: str, role: str) -> str:
    """Get next API key via round-robin rotation.

    First call → key1, second → key2, third → key3, fourth → key1, ...

    Thread-safe enough for single-user (GIL protects next()).
    """
    pool_id = f"{provider}:{role}"

    if pool_id not in _key_cycles:
        keys = get_role_keys(provider, role)
        _key_cycles[pool_id] = cycle(keys)
        _key_counts[pool_id] = len(keys)

    return next(_key_cycles[pool_id])


def pool_size(provider: str, role: str) -> int:
    """Number of keys in the pool for retry bounds."""
    pool_id = f"{provider}:{role}"
    if pool_id not in _key_counts:
        keys = get_role_keys(provider, role)
        _key_counts[pool_id] = len(keys)
    return _key_counts[pool_id]


def reset_pools() -> None:
    """Clear all cached cycles. For testing only."""
    _key_cycles.clear()
    _key_counts.clear()
