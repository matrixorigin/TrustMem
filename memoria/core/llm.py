"""Minimal LLM client for Memoria — only used by reflect/entity extraction.

Wraps OpenAI-compatible API. If no LLM is configured, reflect gracefully degrades.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_client: Any = None


def get_llm_client() -> Any:
    """Get or create a minimal LLM client. Returns None if not configured."""
    global _client
    if _client is not None:
        return _client

    from memoria.config import get_settings

    s = get_settings()
    if not s.llm_api_key:
        return None

    try:
        _client = MinimalLLMClient(
            api_key=s.llm_api_key,
            base_url=s.llm_base_url,
            model=s.llm_model,
        )
        return _client
    except ImportError:
        logger.warning("openai package not installed — reflect unavailable")
        return None


class MinimalLLMClient:
    """Thin wrapper around OpenAI chat completions."""

    def __init__(
        self, api_key: str, base_url: str | None = None, model: str = "gpt-4o-mini"
    ):
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

    def chat_with_tools(
        self,
        messages: list[dict[str, str]],
        tools: list | None = None,
        tool_choice: str | None = None,
        **kwargs,
    ) -> dict:
        """Chat with optional tool calling support."""
        resp = self._client.chat.completions.create(
            model=kwargs.get("model", self.model),
            messages=messages,
            temperature=kwargs.get("temperature", 0.3),
            tools=tools or [],
            tool_choice=tool_choice or "auto",
        )
        return {
            "content": resp.choices[0].message.content or "",
            "tool_calls": resp.choices[0].message.tool_calls or [],
        }
