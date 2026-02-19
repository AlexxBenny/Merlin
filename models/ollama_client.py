# models/ollama_client.py

"""
OllamaClient — LLM interface via Ollama REST API.

Design rules:
- This is a DUMB pipe. It sends prompts, returns text.
- No retries, no fallback, no caching (those belong elsewhere).
- Configuration comes from config/models.yaml.
"""

import json
import logging
from typing import Optional
from urllib import request, error

from models.base import LLMClient

logger = logging.getLogger(__name__)

# Default Ollama endpoint
DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3"


class OllamaClient(LLMClient):
    """
    Minimal Ollama REST API client.

    Usage:
        client = OllamaClient(model="llama3")
        response = client.complete("What is 2+2?")
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 120.0,
        temperature: Optional[float] = None,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.default_temperature = temperature

    def complete(
        self,
        prompt: str,
        *,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Send a prompt to Ollama and return the response text.

        Args:
            prompt: The prompt string.
            temperature: Optional per-call override. Falls back to
                         default_temperature if not provided.

        Raises:
            ConnectionError: if Ollama is unreachable
            RuntimeError: if the API returns an error
        """

        url = f"{self.base_url}/api/generate"

        effective_temp = temperature if temperature is not None else self.default_temperature

        payload_dict = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        if effective_temp is not None:
            payload_dict["options"] = {"temperature": effective_temp}

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
                return body.get("response", "")

        except error.URLError as e:
            logger.error("Ollama unreachable at %s: %s", self.base_url, e)
            raise ConnectionError(
                f"Cannot reach Ollama at {self.base_url}"
            ) from e

        except (TimeoutError, OSError) as e:
            logger.error("Ollama timed out at %s: %s", self.base_url, e)
            raise ConnectionError(
                f"Ollama timed out at {self.base_url} (timeout={self.timeout}s)"
            ) from e

        except Exception as e:
            logger.error("Ollama API error: %s", e)
            raise RuntimeError(f"Ollama API error: {e}") from e

    def is_available(self) -> bool:
        """Check if the Ollama server is reachable."""
        try:
            url = f"{self.base_url}/api/tags"
            req = request.Request(url, method="GET")
            with request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except Exception:
            return False
