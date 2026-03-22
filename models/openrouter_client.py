# models/openrouter_client.py

"""
OpenRouterClient — LLM interface via OpenRouter REST API.

OpenRouter exposes an OpenAI-compatible chat completions endpoint.
This client uses raw urllib — no external dependencies.

Design rules:
- This is a DUMB pipe. It sends prompts, returns text.
- No retries, no fallback, no caching.
- API key resolved at construction time.
- format="json" → response_format={"type": "json_object"}
- If pool metadata provided, 429 errors trigger key rotation + retry.
- Credit-aware: tracks last known credit budget from 402 errors and
  auto-caps max_tokens to avoid future rejections.
"""

import json
import logging
import re
from typing import Any, Dict, Optional, Union
from urllib import request, error

from models.base import LLMClient

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

# Safety margin subtracted from credit budget to avoid edge-case rejections
_CREDIT_SAFETY_MARGIN = 50


class OpenRouterClient(LLMClient):
    """
    OpenRouter REST API client (OpenAI-compatible).

    Usage:
        client = OpenRouterClient(
            model="mistralai/mistral-7b-instruct",
            api_key="sk-or-...",
        )
        response = client.complete("What is 2+2?")
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 60.0,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        _pool_provider: Optional[str] = None,
        _pool_role: Optional[str] = None,
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.default_temperature = temperature
        self.max_tokens = max_tokens
        # Pool metadata for 429 retry with key rotation
        self._pool_provider = _pool_provider
        self._pool_role = _pool_role
        # Credit-aware budget tracking
        self._credit_budget: Optional[int] = None

    def complete(
        self,
        prompt: str,
        *,
        temperature: Optional[float] = None,
        format: Optional[Union[str, Dict[str, Any]]] = None,
        timeout: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Send a prompt to OpenRouter and return the response text.

        Failover policy:
          - Primary key stays stable across calls.
          - On retryable failure (429, 402, 5xx, timeout) → rotate to
            next key and retry.
          - If all keys fail → bubble the last error.

        Args:
            prompt: The prompt string.
            temperature: Optional per-call override.
            format: Optional output format constraint.
                    - "json": enables JSON mode via response_format
                    - dict: JSON schema (mapped to response_format)
                    - None: unconstrained text output
            max_tokens: Optional per-call max output tokens override.
                        If None, use the constructor default.

        Raises:
            ConnectionError: if OpenRouter is unreachable
            RuntimeError: if the API returns an error
        """
        # Determine retry budget from pool
        max_attempts = 1
        if self._pool_provider and self._pool_role:
            from models.key_pool import pool_size
            max_attempts = pool_size(self._pool_provider, self._pool_role)

        last_error = None
        for attempt in range(max_attempts):
            # Failover: rotate key after a failed attempt
            if self._pool_provider and self._pool_role and attempt > 0:
                from models.key_pool import resolve_api_key
                failed_key = self.api_key
                self.api_key = resolve_api_key(
                    self._pool_provider, self._pool_role,
                )
                logger.info(
                    "[KEY_FAILOVER] Key …%s failed, rotated to …%s "
                    "(attempt %d/%d)",
                    failed_key[-4:] if failed_key else "????",
                    self.api_key[-4:] if self.api_key else "????",
                    attempt + 1, max_attempts,
                )

            try:
                return self._do_request(
                    prompt, temperature, format, timeout, max_tokens,
                )
            except (RuntimeError, ConnectionError) as e:
                msg = str(e)
                retryable = any(
                    code in msg
                    for code in ("429", "402", "500", "502", "503")
                ) or "timed out" in msg
                if retryable and attempt < max_attempts - 1:
                    last_error = e
                    continue
                raise

        raise last_error or RuntimeError("All keys exhausted.")

    def _resolve_max_tokens(
        self, per_call: Optional[int],
    ) -> Optional[int]:
        """Resolve effective max_tokens with credit-aware capping.

        Priority:
          1. per_call override (if not None)
          2. constructor default (self.max_tokens)

        Then capped against last known credit budget (if available).
        """
        requested = (
            per_call if per_call is not None
            else self.max_tokens
        )
        if requested is None:
            return None

        # Cap against credit budget if known
        if self._credit_budget is not None:
            cap = max(self._credit_budget - _CREDIT_SAFETY_MARGIN, 1)
            if requested > cap:
                logger.info(
                    "[TOKEN_BUDGET] Capping max_tokens %d → %d "
                    "(credit budget=%d, margin=%d)",
                    requested, cap,
                    self._credit_budget, _CREDIT_SAFETY_MARGIN,
                )
                return cap

        return requested

    def _parse_credit_budget(self, error_body: str) -> None:
        """Extract credit budget from 402 error body.

        Looks for: "can only afford N" in the error message.
        Stores the value for future request capping.
        """
        match = re.search(r"can only afford (\d+)", error_body)
        if match:
            budget = int(match.group(1))
            self._credit_budget = budget
            logger.info(
                "[TOKEN_BUDGET] Learned credit budget: %d tokens", budget,
            )

    def _do_request(
        self,
        prompt: str,
        temperature: Optional[float],
        format: Optional[Union[str, Dict[str, Any]]],
        timeout: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Execute a single HTTP request to OpenRouter."""
        url = f"{self.base_url}/chat/completions"

        effective_temp = (
            temperature if temperature is not None
            else self.default_temperature
        )

        payload_dict: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if effective_temp is not None:
            payload_dict["temperature"] = effective_temp

        effective_max = self._resolve_max_tokens(max_tokens)
        if effective_max is not None:
            payload_dict["max_tokens"] = effective_max

        # Normalize format parameter to OpenAI response_format
        if format is not None:
            if format == "json":
                payload_dict["response_format"] = {"type": "json_object"}
            elif isinstance(format, dict):
                payload_dict["response_format"] = {
                    "type": "json_schema",
                    "json_schema": format,
                }

        payload = json.dumps(payload_dict).encode("utf-8")

        req = request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        effective_timeout = timeout if timeout is not None else self.timeout
        try:
            with request.urlopen(req, timeout=effective_timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                choices = body.get("choices", [])
                if not choices:
                    return ""
                return choices[0].get("message", {}).get("content", "")

        except error.HTTPError as e:
            body_text = ""
            try:
                body_text = e.read().decode("utf-8", errors="replace")
            except Exception:
                pass

            # Extract credit budget from 402 errors
            if e.code == 402:
                self._parse_credit_budget(body_text)

            key_hint = self.api_key[-4:] if self.api_key else "????"
            logger.error(
                "OpenRouter API error %d (key …%s): %s",
                e.code, key_hint, body_text[:500],
            )
            raise RuntimeError(
                f"OpenRouter API error {e.code}: {body_text[:200]}"
            ) from e

        except error.URLError as e:
            key_hint = self.api_key[-4:] if self.api_key else "????"
            logger.error(
                "OpenRouter unreachable at %s (key …%s): %s",
                self.base_url, key_hint, e,
            )
            raise ConnectionError(
                f"Cannot reach OpenRouter at {self.base_url}"
            ) from e

        except (TimeoutError, OSError) as e:
            key_hint = self.api_key[-4:] if self.api_key else "????"
            logger.error(
                "OpenRouter timed out at %s (key …%s): %s",
                self.base_url, key_hint, e,
            )
            raise ConnectionError(
                f"OpenRouter timed out at {self.base_url} "
                f"(timeout={self.timeout}s)"
            ) from e

        except Exception as e:
            key_hint = self.api_key[-4:] if self.api_key else "????"
            logger.error("OpenRouter API error (key …%s): %s", key_hint, e)
            raise RuntimeError(f"OpenRouter API error: {e}") from e

    def is_available(self) -> bool:
        """Check if OpenRouter API is reachable with valid key."""
        try:
            url = f"{self.base_url}/models"
            req = request.Request(
                url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                method="GET",
            )
            with request.urlopen(req, timeout=15) as resp:
                return resp.status == 200
        except Exception:
            return False
