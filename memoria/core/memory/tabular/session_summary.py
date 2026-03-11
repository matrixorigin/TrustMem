"""Session Summarizer — incremental and full session summaries.

Incremental summaries are session-scoped (session_id set).
Full summaries are cross-session (session_id=NULL) and supersede incrementals.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from memoria.core.memory.config import MemoryGovernanceConfig, DEFAULT_CONFIG
from memoria.core.memory.tabular.store import MemoryStore
from memoria.core.memory.types import Memory, MemoryType, TrustTier, _utcnow

logger = logging.getLogger(__name__)


_SESSION_SUMMARY_TAG = "[session_summary]"
_INCREMENTAL_TAG = "[session_summary:incremental]"


class SessionSummarizer:
    """Generate incremental and full session summaries."""

    def __init__(
        self,
        store: MemoryStore,
        llm_client: Any = None,
        embed_fn: Any = None,
        config: Optional[MemoryGovernanceConfig] = None,
    ):
        self.store = store
        self.llm = llm_client
        self.embed_fn = embed_fn
        self.config = config or DEFAULT_CONFIG
        self._incremental_ids: dict[str, list[str]] = {}  # session_id -> [memory_ids]
        self._last_summary_idx: dict[str, int] = {}  # session_id -> last summarized msg index
        self._last_summary_time: dict[str, datetime] = {}  # session_id -> last summary time

    def check_and_summarize(
        self,
        user_id: str,
        session_id: str,
        messages: list[dict[str, Any]],
        turn_count: int,
        session_start: datetime,
    ) -> Optional[Memory]:
        """Check thresholds and generate incremental summary if needed."""
        threshold = self.config.session_summary_turn_threshold
        if threshold <= 0:
            return None

        # Check turn threshold (every N turns)
        if turn_count > 0 and turn_count % threshold == 0:
            return self._generate_incremental(user_id, session_id, messages)

        # Check time threshold — trigger if enough time since last summary (or session start)
        last_time = self._last_summary_time.get(session_id, session_start)
        hours_since = (_utcnow() - last_time).total_seconds() / 3600.0
        if hours_since >= self.config.session_summary_time_threshold_hours:
            return self._generate_incremental(user_id, session_id, messages)

        return None

    def generate_full_summary(
        self,
        user_id: str,
        session_id: str,
        messages: list[dict[str, Any]],
    ) -> Optional[Memory]:
        """Generate full session summary on close. Supersedes incrementals."""
        if not messages:
            return None

        content = self._summarize(messages, full=True)
        if not content:
            return None

        mem = Memory(
            memory_id=uuid.uuid4().hex,
            user_id=user_id,
            memory_type=MemoryType.SEMANTIC,
            content=f"{_SESSION_SUMMARY_TAG} {content}",
            initial_confidence=0.8,
            trust_tier=TrustTier.T3_INFERRED,
            session_id=None,  # Cross-session
            observed_at=_utcnow(),
        )
        if self.embed_fn:
            try:
                mem.embedding = self.embed_fn(content)
            except Exception:
                pass

        mem = self.store.create(mem)

        # Supersede incrementals
        for mid in self._incremental_ids.get(session_id, []):
            try:
                self.store.deactivate(mid)
            except Exception:
                pass
        self._incremental_ids.pop(session_id, None)

        return mem

    def _generate_incremental(
        self, user_id: str, session_id: str, messages: list[dict[str, Any]],
    ) -> Optional[Memory]:
        # Only summarize messages since last summary
        start_idx = self._last_summary_idx.get(session_id, 0)
        new_messages = messages[start_idx:]
        if not new_messages:
            return None

        content = self._summarize(new_messages, full=False)
        if not content:
            return None

        mem = Memory(
            memory_id=uuid.uuid4().hex,
            user_id=user_id,
            memory_type=MemoryType.SEMANTIC,
            content=f"{_INCREMENTAL_TAG} {content}",
            initial_confidence=0.7,
            trust_tier=TrustTier.T3_INFERRED,
            session_id=session_id,  # Session-scoped
            observed_at=_utcnow(),
        )
        if self.embed_fn:
            try:
                mem.embedding = self.embed_fn(content)
            except Exception:
                pass

        mem = self.store.create(mem)
        self._incremental_ids.setdefault(session_id, []).append(mem.memory_id)
        self._last_summary_idx[session_id] = len(messages)
        self._last_summary_time[session_id] = _utcnow()
        return mem

    def _summarize(self, messages: list[dict[str, Any]], full: bool) -> Optional[str]:
        """Summarize messages via LLM or fallback to truncation."""
        texts = [m.get("content", "") for m in messages if m.get("content")]
        if not texts:
            return None

        concat = "\n".join(texts)

        if self.llm and len(concat) > 100:
            try:
                prompt = "Summarize this conversation in 2-3 sentences." if full else \
                         "Summarize the recent part of this conversation in 1-2 sentences."
                result = self.llm.chat_with_tools(messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": concat[:4000]},
                ], task_hint="session_summary")
                return result.get("content") or concat[:500]
            except Exception:
                pass

        return concat[:500]
