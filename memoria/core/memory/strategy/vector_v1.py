"""VectorRetrievalStrategy — cosine/L2 + keyword hybrid retrieval on mem_memories.

Thin adapter over the existing MemoryRetriever. No index tables needed.

See docs/design/memory/backend-management.md §3.3
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from memoria.core.memory.types import Memory, MemoryType, RetrievalWeights

if TYPE_CHECKING:
    from memoria.core.db_consumer import DbFactory
    from memoria.core.memory.config import MemoryGovernanceConfig
    from memoria.core.memory.tabular.metrics import MemoryMetrics


class VectorRetrievalStrategy:
    """vector:v1 — hybrid retrieval directly on mem_memories table.

    No auxiliary index tables. Reads canonical storage directly.
    """

    def __init__(
        self,
        db_factory: DbFactory,
        *,
        params: dict[str, Any] | None = None,
        config: MemoryGovernanceConfig | None = None,
        metrics: MemoryMetrics | None = None,
    ) -> None:
        from memoria.core.memory.tabular.retriever import MemoryRetriever

        self._retriever = MemoryRetriever(
            db_factory, config=config, metrics=metrics,
        )

    @property
    def strategy_key(self) -> str:
        return "vector:v1"

    def retrieve(
        self,
        user_id: str,
        query: str,
        query_embedding: list[float] | None = None,
        *,
        top_k: int = 10,
        task_type: str | None = None,
        session_id: str = "",
        memory_types: list[MemoryType] | None = None,
        weights: RetrievalWeights | None = None,
        include_cross_session: bool = True,
        explain: bool = False,
    ) -> tuple[list[Memory], Any]:
        """Retrieve via hybrid vector + keyword scoring."""
        return self._retriever.retrieve(
            user_id=user_id,
            query_text=query,
            session_id=session_id,
            query_embedding=query_embedding,
            memory_types=memory_types,
            limit=top_k,
            task_hint=task_type,
            weights=weights,
            include_cross_session=include_cross_session,
            explain=explain,
        )
