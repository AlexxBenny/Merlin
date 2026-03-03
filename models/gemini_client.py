# models/gemini_client.py

"""
GeminiClient — LLM interface via Google Gemini REST API.

Uses the generativelanguage.googleapis.com endpoint directly.
No google-generativeai SDK — raw urllib for determinism.

Design rules:
- This is a DUMB pipe. It sends prompts, returns text.
- No retries, no fallback, no caching.
- API key resolved at construction time.
- format="json" → generationConfig.responseMimeType = "application/json"
- If pool metadata provided, 429 errors trigger key rotation + retry.
"""

import json
import logging
from typing import Any, Dict, Optional, Union
from urllib import request, error
from urllib.parse import urlencode

from models.base import LLMClient

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


class GeminiClient(LLMClient):
    """
    Google Gemini REST API client.

    Usage:
        client = GeminiClient(
            model="gemini-2.0-flash",
            api_key="AIza...",
        )
        response = client.complete("What is 2+2?")
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 120.0,
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

    def complete(
        self,
        prompt: str,
        *,
        temperature: Optional[float] = None,
        format: Optional[Union[str, Dict[str, Any]]] = None,
    ) -> str:
        """
        Send a prompt to Gemini and return the response text.

        Args:
            prompt: The prompt string.
            temperature: Optional per-call override.
            format: Optional output format constraint.
                    - "json": enables JSON mode via responseMimeType
                    - dict: JSON schema via responseSchema
                    - None: unconstrained text output

        Raises:
            ConnectionError: if Gemini is unreachable
            RuntimeError: if the API returns an error
        """
        # Determine retry budget from pool
        max_attempts = 1
        if self._pool_provider and self._pool_role:
            from models.key_pool import pool_size
            max_attempts = pool_size(self._pool_provider, self._pool_role)

        last_error = None
        for attempt in range(max_attempts):
            # Resolve current key (rotates on each call if pool exists)
            if self._pool_provider and self._pool_role and attempt > 0:
                from models.key_pool import resolve_api_key
                self.api_key = resolve_api_key(
                    self._pool_provider, self._pool_role,
                )
                logger.info(
                    "[KEY_ROTATION] Gemini 429 → rotated to next key "
                    "(attempt %d/%d)", attempt + 1, max_attempts,
                )

            try:
                return self._do_request(prompt, temperature, format)
            except RuntimeError as e:
                if "429" in str(e) and attempt < max_attempts - 1:
                    last_error = e
                    continue
                raise

        raise last_error or RuntimeError("All keys rate limited.")

    def _do_request(
        self,
        prompt: str,
        temperature: Optional[float],
        format: Optional[Union[str, Dict[str, Any]]],
    ) -> str:
        """Execute a single HTTP request to Gemini."""
        url = (
            f"{self.base_url}/models/{self.model}:generateContent"
            f"?key={self.api_key}"
        )

        effective_temp = (
            temperature if temperature is not None
            else self.default_temperature
        )

        payload_dict: Dict[str, Any] = {
            "contents": [
                {"parts": [{"text": prompt}]}
            ],
        }

        # Build generationConfig
        gen_config: Dict[str, Any] = {}
        if self.max_tokens is not None:
            gen_config["maxOutputTokens"] = self.max_tokens
        if effective_temp is not None:
            gen_config["temperature"] = effective_temp

        # Normalize format parameter to Gemini generationConfig
        if format is not None:
            if format == "json":
                gen_config["responseMimeType"] = "application/json"
            elif isinstance(format, dict):
                gen_config["responseMimeType"] = "application/json"
                gen_config["responseSchema"] = format

        if gen_config:
            payload_dict["generationConfig"] = gen_config

        payload = json.dumps(payload_dict).encode("utf-8")

        req = request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                candidates = body.get("candidates", [])
                if not candidates:
                    return ""
                content = candidates[0].get("content", {})
                parts = content.get("parts", [])
                if not parts:
                    return ""
                return parts[0].get("text", "")

        except error.HTTPError as e:
            body_text = ""
            try:
                body_text = e.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            logger.error(
                "Gemini API error %d: %s", e.code, body_text[:500]
            )
            raise RuntimeError(
                f"Gemini API error {e.code}: {body_text[:200]}"
            ) from e

        except error.URLError as e:
            logger.error(
                "Gemini unreachable at %s: %s", self.base_url, e
            )
            raise ConnectionError(
                f"Cannot reach Gemini at {self.base_url}"
            ) from e

        except (TimeoutError, OSError) as e:
            logger.error(
                "Gemini timed out at %s: %s", self.base_url, e
            )
            raise ConnectionError(
                f"Gemini timed out at {self.base_url} "
                f"(timeout={self.timeout}s)"
            ) from e

        except Exception as e:
            logger.error("Gemini API error: %s", e)
            raise RuntimeError(f"Gemini API error: {e}") from e

    def is_available(self) -> bool:
        """Check if Gemini API is reachable with valid key."""
        try:
            url = (
                f"{self.base_url}/models/{self.model}"
                f"?key={self.api_key}"
            )
            req = request.Request(url, method="GET")
            with request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except Exception:
            return False
