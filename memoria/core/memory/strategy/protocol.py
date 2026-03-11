"""Protocols for pluggable retrieval strategies and index managers.

See docs/design/memory/backend-management.md §3.1, §3.2
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from memoria.core.memory.interfaces import ReflectionCandidate
    from memoria.core.memory.types import Memory


@dataclass
class BackfillResult:
    """Result of an index backfill operation."""

    processed: int = 0
    skipped: int = 0
    errors: list[str] = field(default_factory=list)


@runtime_checkable
class RetrievalStrategy(Protocol):
    """Pluggable retrieval strategy. Only responsible for retrieve()."""

    @property
    def strategy_key(self) -> str:
        """Unique key: 'vector:v1', 'activation:v1', etc."""
        ...

    def retrieve(
        self,
        user_id: str,
        query: str,
        query_embedding: list[float] | None = None,
        *,
        top_k: int = 10,
        task_type: str | None = None,
    ) -> list[Memory]:
        """Retrieve memories ranked by this strategy's algorithm."""
        ...


@runtime_checkable
class IndexManager(Protocol):
    """Maintains auxiliary index tables for a retrieval strategy."""

    def on_memories_stored(
        self,
        user_id: str,
        memories: list[Memory],
        *,
        session_id: str | None = None,
    ) -> None:
        """Called after canonical storage writes. Update index."""
        ...

    def on_governance(self, user_id: str) -> None:
        """Called during governance. Maintain index health."""
        ...

    def backfill_needed(self, user_id: str) -> bool:
        """Check if backfill is needed for this user."""
        ...

    def backfill(self, user_id: str) -> BackfillResult:
        """Backfill index from canonical storage. Must be idempotent."""
        ...

    def drop_index(self, user_id: str) -> None:
        """Remove this user's data from index tables."""
        ...

    def get_reflection_candidates(
        self, user_id: str, *, since_hours: int = 24,
    ) -> list[ReflectionCandidate] | None:
        """Return reflection candidates from index, or None to use canonical fallback."""
        ...
