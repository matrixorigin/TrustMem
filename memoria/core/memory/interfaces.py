"""Protocol-based interfaces for the memory module.

External consumers should depend on these Protocols, not on internal classes.
This enables refactoring memory internals without breaking consumers.

See docs/design/memory-architecture.md §11 "Module Independence".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from memoria.core.memory.types import (
        Memory,
        MemoryType,
        RetrievalWeights,
        TrustTier,
    )


@runtime_checkable
class MemoryReader(Protocol):
    """Read-path interface for memory retrieval."""

    def retrieve(
        self,
        user_id: str,
        query: str,
        *,
        session_id: str = "",
        query_embedding: list[float] | None = None,
        memory_types: list[MemoryType] | None = None,
        top_k: int = 10,
        task_hint: str | None = None,
        weights: RetrievalWeights | None = None,
        include_cross_session: bool = True,
    ) -> list[Memory]:
        """Retrieve memories ranked by hybrid relevance."""
        ...

    def get_profile(self, user_id: str) -> str | None:
        """Get synthesized user profile string (~200 tokens)."""
        ...


@runtime_checkable
class MemoryWriter(Protocol):
    """Write-path interface for memory persistence."""

    def store(
        self,
        user_id: str,
        content: str,
        *,
        memory_type: MemoryType,
        source_event_ids: list[str] | None = None,
        initial_confidence: float = 0.75,
        # No default here — TrustTier lives in core.memory.types which is
        # behind TYPE_CHECKING.  Concrete implementations (MemoryService)
        # provide the real default (T3_INFERRED).
        trust_tier: TrustTier | None = None,
        session_id: str | None = None,
    ) -> Memory:
        """Store a single memory with contradiction detection."""
        ...

    def observe_turn(
        self,
        user_id: str,
        messages: list[dict[str, Any]],
        *,
        source_event_ids: list[str] | None = None,
    ) -> list[Memory]:
        """Extract and persist memories from a conversation turn."""
        ...


class MemoryAdmin(Protocol):
    """Admin interface for governance and health."""

    def run_governance(self, user_id: str) -> GovernanceReport:
        """Run all governance cycles for a user."""
        ...

    def health_check(self, user_id: str) -> HealthReport:
        """Get memory health analytics."""
        ...


@dataclass
class ReflectionCandidate:
    """A candidate cluster for reflection synthesis.

    Produced by backend-specific CandidateProviders, consumed by the
    shared ReflectionEngine.
    """

    memories: list[Memory]
    signal: str  # "semantic_cluster" | "contradiction" | "summary_recurrence"
    importance_score: float = 0.0  # pre-computed by backend-specific scorer
    session_ids: list[str] = field(default_factory=list)


class CandidateProvider(Protocol):
    """Each backend implements this to feed the shared ReflectionEngine."""

    def get_reflection_candidates(
        self,
        user_id: str,
        *,
        since_hours: int = 24,
    ) -> list[ReflectionCandidate]:
        """Return candidate clusters for reflection."""
        ...


@dataclass
class GovernanceReport:
    """Result of a governance cycle."""

    cleaned_tool_results: int = 0
    archived_working: int = 0
    cleaned_stale: int = 0
    quarantined: int = 0
    scenes_created: int = 0
    cleaned_branches: int = 0
    cleaned_snapshots: int = 0
    compressed_redundant: int = 0
    pollution_detected: bool = False
    errors: list[str] | None = None
    total_ms: float = 0.0
    # Observability (incremental governance)
    users_processed: int = 0
    users_skipped_no_changes: int = 0
    input_memories: int = 0
    reflection_candidates_found: int = 0
    reflection_candidates_synthesized: int = 0
    reflection_candidates_skipped_low_importance: int = 0


@dataclass
class HealthReport:
    """Memory health analytics."""

    total: int = 0
    active: int = 0
    inactive: int = 0
    avg_content_size: float = 0.0
    per_type_stats: dict[str, Any] | None = None
    pollution: dict[str, Any] | None = None
