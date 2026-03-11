"""Minimal LLM client for Memoria — only used by reflect/reflection engine.

Wraps OpenAI-compatible API. If no LLM is configured, reflect gracefully degrades.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_client: Any = None


def get_llm_client() -> Any:
    """Get or create a minimal LLM client. Returns None if not configured."""
    global _client
    if _client is not None:
        return _client

    api_key = os.environ.get("MEMORIA_LLM_API_KEY", "")
    if not api_key:
        return None

    try:
        import openai
        base_url = os.environ.get("MEMORIA_LLM_BASE_URL")
        model = os.environ.get("MEMORIA_LLM_MODEL", "gpt-4o-mini")
        _client = MinimalLLMClient(api_key=api_key, base_url=base_url, model=model)
        return _client
    except ImportError:
        logger.warning("openai package not installed — reflect unavailable")
        return None


class MinimalLLMClient:
    """Thin wrapper around OpenAI chat completions."""

    def __init__(self, api_key: str, base_url: str | None = None, model: str = "gpt-4o-mini"):
        import openai
        self._client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
        resp = self._client.chat.completions.create(
            model=kwargs.get("model", self.model),
            messages=messages,
            temperature=kwargs.get("temperature", 0.3),
        )
        return resp.choices[0].message.content or ""
