"""Tiered memory loader for PromptAssembler §4.

Moved from core/memory/ to core/context/ — this is a context-layer consumption
strategy, not a memory-internal component. See memory-architecture.md §11.

L0: Profile (always loaded, ~200 tokens)
L1: Query-aware retrieval (per-turn, ~800 tokens)

Consumes memory through MemoryService (Protocol-based interface).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from memoria.core.memory.tabular.metrics import MemoryMetrics
from memoria.core.memory.types import MemoryType

if TYPE_CHECKING:
    from memoria.core.memory.tabular.explain import RetrievalStats

logger = logging.getLogger(__name__)


@dataclass
class TieredLoaderStats:
    l0_loaded: bool = False
    l0_tokens: int = 0
    l0_ms: float = 0.0
    l1_loaded: bool = False
    l1_count: int = 0
    l1_tokens: int = 0
    l1_ms: float = 0.0
    l1_error: bool = False
    retrieval: RetrievalStats | None = None
    total_ms: float = 0.0


class TieredMemoryLoader:
    """Load L0 (profile) + L1 (query-relevant) memories for prompt §4.

    Uses MemoryService as the sole interface to the memory module.
    """

    def __init__(self, memory_service: Any, metrics: MemoryMetrics | None = None):
        self._svc = memory_service
        self._metrics = metrics or MemoryMetrics()

    def load_l0(self, user_id: str) -> str:
        try:
            return self._svc.get_profile(user_id) or ""
        except Exception as e:
            logger.debug("L0 load failed: %s", e)
            self._metrics.increment("tiered_loader_l0_errors")
            return ""

    def load_l1(
        self,
        user_id: str,
        session_id: str,
        query: str,
        query_embedding: list[float] | None = None,
        task_hint: str | None = None,
        limit: int = 10,
        explain: bool = False,
    ) -> tuple[str, RetrievalStats | None]:
        try:
            memories, stats = self._svc.retrieve(
                user_id=user_id,
                query=query,
                session_id=session_id,
                query_embedding=query_embedding,
                memory_types=[MemoryType.SEMANTIC, MemoryType.PROCEDURAL],
                top_k=limit,
                task_hint=task_hint,
                explain=explain,
            )
            if not memories:
                return "", stats
            lines = ["Relevant Memories:"]
            for m in memories:
                lines.append(f"- [{m.memory_type.value}] {m.content}")
            return "\n".join(lines), stats
        except Exception as e:
            logger.debug("L1 load failed: %s", e)
            self._metrics.increment("tiered_loader_l1_errors")
            return "", None

    def build_section(
        self,
        user_id: str,
        session_id: str,
        query: str,
        query_embedding: list[float] | None = None,
        task_hint: str | None = None,
        explain: bool = False,
    ) -> tuple[str, TieredLoaderStats | None]:
        start = time.time() if explain else 0
        stats = TieredLoaderStats() if explain else None
        parts = []

        l0_start = time.time() if explain else 0
        l0 = self.load_l0(user_id)
        if l0:
            parts.append(l0)
        if stats:
            stats.l0_loaded = bool(l0)
            stats.l0_tokens = len(l0.split()) if l0 else 0
            stats.l0_ms = (time.time() - l0_start) * 1000

        l1_start = time.time() if explain else 0
        l1, retrieval_stats = self.load_l1(user_id, session_id, query, query_embedding, task_hint, explain=explain)
        if l1:
            parts.append(l1)
        if stats:
            stats.l1_loaded = bool(l1)
            stats.l1_count = len(l1.split("\n")) - 1 if l1 else 0
            stats.l1_tokens = len(l1.split()) if l1 else 0
            stats.l1_ms = (time.time() - l1_start) * 1000
            stats.l1_error = not l1 and retrieval_stats is None
            stats.retrieval = retrieval_stats
            stats.total_ms = (time.time() - start) * 1000

        return "\n\n".join(parts), stats

    def invalidate_profile(self, user_id: str) -> None:
        self._svc.invalidate_profile(user_id)
