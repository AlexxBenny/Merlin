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
"""

import json
import logging
from typing import Any, Dict, Optional, Union
from urllib import request, error

from models.base import LLMClient

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


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
        timeout: float = 120.0,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.default_temperature = temperature
        self.max_tokens = max_tokens

    def complete(
        self,
        prompt: str,
        *,
        temperature: Optional[float] = None,
        format: Optional[Union[str, Dict[str, Any]]] = None,
    ) -> str:
        """
        Send a prompt to OpenRouter and return the response text.

        Args:
            prompt: The prompt string.
            temperature: Optional per-call override.
            format: Optional output format constraint.
                    - "json": enables JSON mode via response_format
                    - dict: JSON schema (mapped to response_format)
                    - None: unconstrained text output

        Raises:
            ConnectionError: if OpenRouter is unreachable
            RuntimeError: if the API returns an error
        """
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
        if self.max_tokens is not None:
            payload_dict["max_tokens"] = self.max_tokens

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

        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
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
            logger.error(
                "OpenRouter API error %d: %s", e.code, body_text[:500]
            )
            raise RuntimeError(
                f"OpenRouter API error {e.code}: {body_text[:200]}"
            ) from e

        except error.URLError as e:
            logger.error(
                "OpenRouter unreachable at %s: %s", self.base_url, e
            )
            raise ConnectionError(
                f"Cannot reach OpenRouter at {self.base_url}"
            ) from e

        except (TimeoutError, OSError) as e:
            logger.error(
                "OpenRouter timed out at %s: %s", self.base_url, e
            )
            raise ConnectionError(
                f"OpenRouter timed out at {self.base_url} "
                f"(timeout={self.timeout}s)"
            ) from e

        except Exception as e:
            logger.error("OpenRouter API error: %s", e)
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
            with request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except Exception:
            return False
