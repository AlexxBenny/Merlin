# models/base.py

"""
LLMClient — Abstract base for all LLM providers.

Design rules:
- This is a contract, not an implementation.
- Every provider (Ollama, Gemini, OpenRouter, etc.) must implement this.
- complete() accepts optional temperature override per call.
- No retries, no fallback (those belong in higher layers).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union


class LLMClient(ABC):
    """Abstract LLM client interface.

    All LLM providers MUST inherit from this.
    MissionCortex, ReportBuilder, and any other consumer
    should type-hint against LLMClient, not a concrete class.
    """

    @abstractmethod
    def complete(
        self,
        prompt: str,
        *,
        temperature: Optional[float] = None,
        format: Optional[Union[str, Dict[str, Any]]] = None,
        timeout: Optional[float] = None,
    ) -> str:
        """Send a prompt and return the response text.

        Args:
            prompt: The prompt string.
            temperature: Optional per-call temperature override.
                         If None, use the client's default temperature.
            format: Optional output format constraint.
                    - "json": basic JSON mode
                    - dict: JSON schema for structured output (Ollama v0.5+)
                    - None: unconstrained text output
            timeout: Optional per-call timeout in seconds.
                     If None, use the client's default timeout.

        Returns:
            The LLM's response as a string.

        Raises:
            ConnectionError: if the LLM is unreachable.
            RuntimeError: if the API returns an error.
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the LLM provider is reachable."""
        ...
