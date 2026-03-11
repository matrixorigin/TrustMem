"""TabularCandidateProvider — three-signal candidate selection for reflection.

Signal 1: Semantic clustering via DB-side cosine_similarity (cross-session)
Signal 2: Contradiction pairs via supersede chain
Signal 3: Session summary recurrence via DB-side cosine_similarity

See docs/design/memory/tabular-memory.md "Candidate Selection (Tabular-Specific)"
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sqlalchemy import and_, text
from sqlalchemy.orm import aliased

from memoria.core.memory.models.memory import MemoryRecord
from memoria.core.db_consumer import DbConsumer
from memoria.core.memory.config import DEFAULT_CONFIG, MemoryGovernanceConfig
from memoria.core.memory.interfaces import ReflectionCandidate
from memoria.core.memory.reflection.importance import score_candidate
from memoria.core.memory.types import Memory, MemoryType, TrustTier, _utcnow

if TYPE_CHECKING:
    from datetime import datetime

    from sqlalchemy.orm import Session as DbSession

    from memoria.core.db_consumer import DbFactory

logger = logging.getLogger(__name__)

# Columns needed for Signal 2 (no embedding — saves ~1.5KB/row)
_LIGHT_COLS = [
    MemoryRecord.memory_id, MemoryRecord.user_id, MemoryRecord.content,
    MemoryRecord.memory_type, MemoryRecord.session_id, MemoryRecord.observed_at,
    MemoryRecord.initial_confidence, MemoryRecord.trust_tier,
    MemoryRecord.superseded_by, MemoryRecord.is_active,
    MemoryRecord.source_event_ids, MemoryRecord.created_at, MemoryRecord.updated_at,
]


class TabularCandidateProvider(DbConsumer):
    """Provide reflection candidates from the tabular (flat-table) backend.

    Implements CandidateProvider protocol.
    Uses DB-side cosine_similarity for clustering — no embedding data leaves the DB.
    """

    def __init__(self, db_factory: DbFactory, config: MemoryGovernanceConfig | None = None) -> None:
        super().__init__(db_factory)
        self._config = config or DEFAULT_CONFIG

    def get_reflection_candidates(
        self, user_id: str, *, since_hours: int = 24,
    ) -> list[ReflectionCandidate]:
        """Collect candidates from all three signals in a single DB session."""
        candidates: list[ReflectionCandidate] = []
        cutoff = _hours_ago(since_hours)

        with self._db() as db:
            for name, fn in [
                ("semantic_clusters", lambda: self._signal_semantic_clusters(db, user_id, cutoff)),
                ("contradiction_pairs", lambda: self._signal_contradiction_pairs(db, user_id, cutoff)),
                ("summary_recurrence", lambda: self._signal_summary_recurrence(db, user_id)),
            ]:
                try:
                    candidates.extend(fn())
                except Exception as e:
                    logger.warning(
                        "Signal %s failed for user=%s since=%s: %s",
                        name, user_id, cutoff, e, exc_info=True,
                    )

        return candidates

    # ── Signal 1: Semantic clustering (DB-side) ───────────────────────

    def _signal_semantic_clusters(
        self, db: DbSession, user_id: str, cutoff: datetime,
    ) -> list[ReflectionCandidate]:
        """Find cross-session clusters via DB-side cosine_similarity self-join.

        Returns similar pairs from DB, then merges into clusters via union-find.
        Embedding data never leaves the database.
        """
        A = aliased(MemoryRecord, name="a")
        B = aliased(MemoryRecord, name="b")

        # DB-side: find all similar cross-session pairs
        pairs = (
            db.query(A.memory_id, A.session_id, B.memory_id, B.session_id)
            .join(B, and_(
                A.user_id == B.user_id,
                A.memory_id < B.memory_id,  # deduplicate
            ))
            .filter(
                A.user_id == user_id,
                A.is_active == 1,
                A.memory_type.in_(["semantic", "procedural"]),
                A.observed_at > cutoff,
                A.embedding.isnot(None),
                B.is_active == 1,
                B.memory_type.in_(["semantic", "procedural"]),
                B.observed_at > cutoff,
                B.embedding.isnot(None),
                A.session_id != B.session_id,
                text("cosine_similarity(a.embedding, b.embedding) >= :threshold").bindparams(threshold=self._config.cluster_similarity_threshold),
            )
            .limit(5000)
            .all()
        )

        if not pairs:
            return []

        # Union-find to merge pairs into clusters
        parent: dict[str, str] = {}
        session_map: dict[str, str | None] = {}

        def find(x: str) -> str:
            while parent.get(x, x) != x:
                parent[x] = parent.get(parent[x], parent[x])
                x = parent[x]
            return x

        def union(x: str, y: str) -> None:
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[rx] = ry

        for a_mid, a_sid, b_mid, b_sid in pairs:
            session_map[a_mid] = a_sid
            session_map[b_mid] = b_sid
            union(a_mid, b_mid)

        # Group by cluster root
        clusters: dict[str, set[str]] = {}
        for mid in session_map:
            root = find(mid)
            clusters.setdefault(root, set()).add(mid)

        # Filter: need >= self._config.min_cross_session_count distinct sessions
        candidates = []
        for mids in clusters.values():
            session_ids = list({session_map[m] for m in mids if session_map[m]})
            if len(session_ids) >= self._config.min_cross_session_count:
                # Fetch full Memory objects for qualifying clusters only
                rows = (
                    db.query(MemoryRecord)
                    .filter(MemoryRecord.memory_id.in_(list(mids)))
                    .all()
                )
                mems = [_to_domain(r) for r in rows]
                c = ReflectionCandidate(
                    memories=mems,
                    signal="semantic_cluster",
                    session_ids=session_ids,
                )
                c.importance_score = score_candidate(c)
                candidates.append(c)
        return candidates

    # ── Signal 2: Contradiction pairs ─────────────────────────────────

    def _signal_contradiction_pairs(
        self, db: DbSession, user_id: str, cutoff: datetime,
    ) -> list[ReflectionCandidate]:
        """Find memories that superseded each other recently.

        Uses light columns (no embedding) — contradiction pairs don't need vectors.
        """
        OldMem = aliased(MemoryRecord)
        NewMem = aliased(MemoryRecord)

        old_cols = [getattr(OldMem, c.key) for c in _LIGHT_COLS]
        new_cols = [getattr(NewMem, c.key) for c in _LIGHT_COLS]

        rows = (
            db.query(*old_cols, *new_cols)
            .join(NewMem, OldMem.superseded_by == NewMem.memory_id)
            .filter(
                OldMem.user_id == user_id,
                NewMem.user_id == user_id,
                NewMem.observed_at > cutoff,
            )
            .all()
        )

        n = len(_LIGHT_COLS)
        candidates = []
        for row in rows:
            old_mem = _row_tuple_to_memory(row[:n])
            new_mem = _row_tuple_to_memory(row[n:])
            session_ids = list({
                s for s in [old_mem.session_id, new_mem.session_id] if s
            })
            candidates.append(ReflectionCandidate(
                memories=[old_mem, new_mem],
                signal="contradiction",
                importance_score=score_candidate(ReflectionCandidate(
                    memories=[old_mem, new_mem],
                    signal="contradiction",
                    session_ids=session_ids,
                )),
                session_ids=session_ids,
            ))
        return candidates

    # ── Signal 3: Session summary recurrence (DB-side) ────────────────

    def _signal_summary_recurrence(
        self, db: DbSession, user_id: str,
    ) -> list[ReflectionCandidate]:
        """Find recurring themes across session summaries (7-day window).

        Uses DB-side cosine_similarity to find similar summary pairs,
        then union-find to form clusters of size >= self._config.min_summary_recurrence.
        """
        cutoff_7d = _hours_ago(self._config.summary_recurrence_window_days * 24)

        A = aliased(MemoryRecord, name="sa")
        B = aliased(MemoryRecord, name="sb")

        pairs = (
            db.query(A.memory_id, B.memory_id)
            .join(B, and_(
                A.user_id == B.user_id,
                A.memory_id < B.memory_id,
            ))
            .filter(
                A.user_id == user_id,
                A.memory_type == "semantic",
                A.session_id.is_(None),
                A.is_active == 1,
                A.observed_at > cutoff_7d,
                A.embedding.isnot(None),
                B.memory_type == "semantic",
                B.session_id.is_(None),
                B.is_active == 1,
                B.observed_at > cutoff_7d,
                B.embedding.isnot(None),
                text("cosine_similarity(sa.embedding, sb.embedding) >= :threshold").bindparams(threshold=self._config.cluster_similarity_threshold),
            )
            .limit(5000)
            .all()
        )

        if not pairs:
            return []

        # Union-find
        parent: dict[str, str] = {}

        def find(x: str) -> str:
            while parent.get(x, x) != x:
                parent[x] = parent.get(parent[x], parent[x])
                x = parent[x]
            return x

        def union(x: str, y: str) -> None:
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[rx] = ry

        all_mids: set[str] = set()
        for a_mid, b_mid in pairs:
            all_mids.add(a_mid)
            all_mids.add(b_mid)
            union(a_mid, b_mid)

        clusters: dict[str, set[str]] = {}
        for mid in all_mids:
            root = find(mid)
            clusters.setdefault(root, set()).add(mid)

        candidates = []
        for mids in clusters.values():
            if len(mids) >= self._config.min_summary_recurrence:
                rows = (
                    db.query(MemoryRecord)
                    .filter(MemoryRecord.memory_id.in_(list(mids)))
                    .all()
                )
                candidates.append(ReflectionCandidate(
                    memories=[_to_domain(r) for r in rows],
                    signal="summary_recurrence",
                    importance_score=score_candidate(ReflectionCandidate(
                        memories=[_to_domain(r) for r in rows],
                        signal="summary_recurrence",
                        session_ids=[],
                    )),
                    session_ids=[],
                ))
        return candidates


# ── Helpers ───────────────────────────────────────────────────────────


def _hours_ago(hours: int) -> datetime:
    from datetime import timedelta
    return _utcnow() - timedelta(hours=hours)


def _to_domain(row: MemoryRecord) -> Memory:
    """Convert ORM model row to Memory dataclass."""
    return Memory(
        memory_id=row.memory_id,
        user_id=row.user_id,
        memory_type=MemoryType(row.memory_type),
        content=row.content,
        initial_confidence=row.initial_confidence,
        embedding=row.embedding,
        source_event_ids=row.source_event_ids or [],
        superseded_by=row.superseded_by,
        is_active=bool(row.is_active),
        session_id=row.session_id,
        observed_at=row.observed_at,
        created_at=row.created_at,
        updated_at=row.updated_at,
        trust_tier=TrustTier(row.trust_tier) if row.trust_tier else TrustTier.T3_INFERRED,
    )


def _row_tuple_to_memory(vals: tuple) -> Memory:
    """Convert a column-tuple (from light query) to Memory. No embedding."""
    (mid, uid, content, mtype, sid, observed, conf, tier,
     superseded, active, src_ids, created, updated) = vals
    return Memory(
        memory_id=mid, user_id=uid, content=content,
        memory_type=MemoryType(mtype),
        session_id=sid, observed_at=observed,
        initial_confidence=conf,
        trust_tier=TrustTier(tier) if tier else TrustTier.T3_INFERRED,
        embedding=None,
        source_event_ids=src_ids or [],
        superseded_by=superseded,
        is_active=bool(active),
        created_at=created, updated_at=updated,
    )


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors. Kept for unit tests."""
    import math
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
