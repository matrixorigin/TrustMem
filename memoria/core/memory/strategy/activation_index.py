"""ActivationIndexManager — maintains graph nodes/edges for activation:v1.

Extracted from GraphMemoryService: graph builder + consolidator + opinion evolution.

See docs/design/memory/backend-management.md §3.2
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from sqlalchemy.exc import SQLAlchemyError

from memoria.core.memory.graph.consolidation import GraphConsolidator
from memoria.core.memory.graph.graph_builder import GraphBuilder
from memoria.core.memory.graph.graph_store import GraphStore
from memoria.core.memory.strategy.protocol import BackfillResult

if TYPE_CHECKING:
    from memoria.core.db_consumer import DbFactory
    from memoria.core.memory.config import MemoryGovernanceConfig
    from memoria.core.memory.types import Memory

logger = logging.getLogger(__name__)

_RECOVERABLE = (SQLAlchemyError, OSError, ConnectionError, TimeoutError)


class ActivationIndexManager:
    """Maintains graph_nodes/edges index tables for activation:v1.

    Called by CanonicalStorage after store/observe_turn.
    """

    def __init__(
        self,
        db_factory: DbFactory,
        *,
        params: dict[str, Any] | None = None,
        config: MemoryGovernanceConfig | None = None,
        embed_fn: Any | None = None,
    ) -> None:
        self._db_factory = db_factory
        if config is None:
            from memoria.core.memory.config import DEFAULT_CONFIG

            config = DEFAULT_CONFIG
        self._config = config
        self._store = GraphStore(db_factory)
        self._builder = GraphBuilder(self._store, config=config, embed_fn=embed_fn)
        self._consolidator = GraphConsolidator(db_factory, config=config)
        self._pending_sync: list[str] = []

    def on_memories_stored(
        self,
        user_id: str,
        memories: list[Memory],
        *,
        session_id: str | None = None,
    ) -> None:
        """Ingest memories into graph and run opinion evolution."""
        try:
            created = self._builder.ingest(
                user_id,
                memories,
                [],
                session_id=session_id,
            )
            self._run_opinion_evolution(user_id, created)
        except _RECOVERABLE:
            logger.warning(
                "Graph ingest failed for %d memories, queued for retry",
                len(memories),
                exc_info=True,
            )
            self._pending_sync.extend(m.memory_id for m in memories)

    def on_governance(self, user_id: str) -> None:
        """Run graph consolidation + drain pending syncs."""
        pending = self._pending_sync[:]
        self._pending_sync.clear()
        if pending:
            logger.info("Draining %d pending graph syncs", len(pending))

        try:
            self._consolidator.consolidate(user_id)
        except _RECOVERABLE as e:
            logger.warning("Graph consolidation failed: %s", e)

    def backfill_needed(self, user_id: str) -> bool:
        """Check if graph index needs building for this user."""
        return not self._store.has_min_nodes(user_id, 1)

    def backfill(self, user_id: str) -> BackfillResult:
        """Build graph index from canonical mem_memories table.

        Idempotent — skips memories that already have graph nodes.
        """
        from memoria.core.memory.tabular.store import MemoryStore

        result = BackfillResult()
        store = MemoryStore(self._db_factory)
        memories = store.list_active(user_id, load_embedding=True)

        for mem in memories:
            existing = self._store.get_node_by_memory_id(mem.memory_id)
            if existing:
                result.skipped += 1
                continue
            try:
                self._builder.ingest(
                    user_id,
                    [mem],
                    [],
                    session_id=mem.session_id,
                )
                result.processed += 1
            except _RECOVERABLE as e:
                result.errors.append(f"{mem.memory_id}: {e}")

        return result

    def drop_index(self, user_id: str) -> None:
        """Remove all graph nodes/edges for this user."""
        self._store.delete_user_data(user_id)

    def _run_opinion_evolution(
        self,
        user_id: str,
        created_nodes: list[Any],
    ) -> None:
        """Run opinion evolution for newly created nodes."""
        from memoria.core.memory.graph.opinion import evolve_opinions

        for node in created_nodes:
            if not node.embedding:
                continue
            try:
                evolve_opinions(self._store, node.node_id, user_id, self._config)
            except _RECOVERABLE:
                logger.warning(
                    "Opinion evolution failed for node %s",
                    node.node_id,
                    exc_info=True,
                )

    @property
    def pending_sync_count(self) -> int:
        return len(self._pending_sync)

    def get_reflection_candidates(
        self,
        user_id: str,
        *,
        since_hours: int = 24,
    ) -> list | None:
        """Return graph-based reflection candidates, or None for fallback."""
        from memoria.core.memory.graph.candidates import GraphCandidateProvider

        try:
            provider = GraphCandidateProvider(self._db_factory, config=self._config)
            candidates = provider.get_reflection_candidates(
                user_id,
                since_hours=since_hours,
            )
            return candidates if candidates else None
        except _RECOVERABLE:
            logger.warning("Graph candidate selection failed", exc_info=True)
            return None

    def get_graph_stats(self, user_id: str) -> dict[str, int]:
        """Graph node count for this user."""
        return {"total_nodes": self._store.count_user_nodes(user_id)}

    def consolidate(self, user_id: str) -> Any:
        """Run graph consolidation directly (for testing/admin)."""
        return self._consolidator.consolidate(user_id)
