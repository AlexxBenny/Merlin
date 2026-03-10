# models/huggingface_client.py

"""
HuggingFaceClient — LLM interface via Hugging Face Inference API.

Uses the api-inference.huggingface.co endpoint directly.
No huggingface_hub SDK — raw urllib for determinism.

Design rules:
- This is a DUMB pipe. It sends prompts, returns text.
- No retries, no fallback, no caching.
- API key resolved at construction time.
- format="json" → prompt instruction only (HF has no strict JSON mode).
  The caller's prompt must enforce structure.
"""

import json
import logging
from typing import Any, Dict, Optional, Union
from urllib import request, error

from models.base import LLMClient

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api-inference.huggingface.co"


class HuggingFaceClient(LLMClient):
    """
    Hugging Face Inference API client.

    Usage:
        client = HuggingFaceClient(
            model="mistralai/Mistral-7B-Instruct-v0.3",
            api_key="hf_...",
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
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.default_temperature = temperature

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
        Send a prompt to Hugging Face and return the response text.

        Args:
            prompt: The prompt string.
            temperature: Optional per-call override.
            format: Optional output format constraint.
                    - "json": appended instruction to prompt (HF has no
                      native JSON mode — prompt engineering only)
                    - dict: ignored (no schema enforcement on HF)
                    - None: unconstrained text output

        Raises:
            ConnectionError: if HF API is unreachable
            RuntimeError: if the API returns an error
        """
        url = f"{self.base_url}/models/{self.model}"

        effective_temp = (
            temperature if temperature is not None
            else self.default_temperature
        )

        # HF Inference API has no strict JSON mode.
        # Best-effort: prepend instruction when format="json" requested.
        effective_prompt = prompt
        if format == "json":
            effective_prompt = (
                "You must respond with ONLY valid JSON. "
                "No markdown, no commentary.\n\n" + prompt
            )

        parameters: Dict[str, Any] = {
            "return_full_text": False,
        }
        if effective_temp is not None:
            parameters["temperature"] = effective_temp

        payload_dict: Dict[str, Any] = {
            "inputs": effective_prompt,
            "parameters": parameters,
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

                # HF returns a list of generation results
                if isinstance(body, list) and body:
                    return body[0].get("generated_text", "")
                elif isinstance(body, dict):
                    # Some models return a dict with generated_text
                    return body.get("generated_text", "")
                return ""

        except error.HTTPError as e:
            body_text = ""
            try:
                body_text = e.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            logger.error(
                "HuggingFace API error %d: %s", e.code, body_text[:500]
            )
            raise RuntimeError(
                f"HuggingFace API error {e.code}: {body_text[:200]}"
            ) from e

        except error.URLError as e:
            logger.error(
                "HuggingFace unreachable at %s: %s", self.base_url, e
            )
            raise ConnectionError(
                f"Cannot reach HuggingFace at {self.base_url}"
            ) from e

        except (TimeoutError, OSError) as e:
            logger.error(
                "HuggingFace timed out at %s: %s", self.base_url, e
            )
            raise ConnectionError(
                f"HuggingFace timed out at {self.base_url} "
                f"(timeout={self.timeout}s)"
            ) from e

        except Exception as e:
            logger.error("HuggingFace API error: %s", e)
            raise RuntimeError(f"HuggingFace API error: {e}") from e

    def is_available(self) -> bool:
        """Check if HF model is available for inference."""
        try:
            url = f"{self.base_url}/models/{self.model}"
            req = request.Request(
                url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                method="GET",
            )
            with request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except Exception:
            return False
