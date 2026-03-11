"""Tests for TypedObserver truncation logic in _extract_via_llm."""
from __future__ import annotations

from unittest.mock import MagicMock


class TestObserverTruncation:
    """Verify _extract_via_llm respects message and char limits."""

    def _make_observer(self, mock_llm=None):
        from memoria.core.memory.tabular.typed_observer import TypedObserver
        return TypedObserver(
            store=MagicMock(), llm_client=mock_llm or MagicMock(),
            embed_fn=None, db_factory=None,
        )

    def test_truncates_to_max_messages(self):
        obs = self._make_observer()
        obs.llm = MagicMock()
        obs.llm.chat_with_tools.return_value = {"content": "[]"}

        msgs = [{"role": "user", "content": f"msg-{i}"} for i in range(50)]
        obs._extract_via_llm(msgs)

        call_args = obs.llm.chat_with_tools.call_args
        user_content = call_args[1]["messages"][1]["content"]
        # Only last 20 messages should appear (each is "[user]: msg-XX")
        assert "msg-49" in user_content
        assert "msg-30" in user_content
        assert "msg-29" not in user_content

    def test_truncates_to_max_chars(self):
        obs = self._make_observer()
        obs.llm = MagicMock()
        obs.llm.chat_with_tools.return_value = {"content": "[]"}

        # 5 messages with 2000 chars each → 10000 chars total, should be cut to 6000
        msgs = [{"role": "user", "content": "x" * 2000} for _ in range(5)]
        obs._extract_via_llm(msgs)

        call_args = obs.llm.chat_with_tools.call_args
        user_content = call_args[1]["messages"][1]["content"]
        assert len(user_content) <= 6000

    def test_per_message_content_capped_at_500(self):
        obs = self._make_observer()
        obs.llm = MagicMock()
        obs.llm.chat_with_tools.return_value = {"content": "[]"}

        msgs = [{"role": "user", "content": "A" * 1000}]
        obs._extract_via_llm(msgs)

        call_args = obs.llm.chat_with_tools.call_args
        user_content = call_args[1]["messages"][1]["content"]
        # "[user]: " prefix + 500 chars max
        assert len(user_content) <= 510

    def test_task_hint_passed(self):
        obs = self._make_observer()
        obs.llm = MagicMock()
        obs.llm.chat_with_tools.return_value = {"content": "[]"}

        obs._extract_via_llm([{"role": "user", "content": "hi"}])

        call_args = obs.llm.chat_with_tools.call_args
        assert call_args[1]["task_hint"] == "memory_extraction"
