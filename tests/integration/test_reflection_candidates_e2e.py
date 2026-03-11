"""Integration tests for TabularCandidateProvider — real DB, all 3 signals.

Verifies:
- Signal 1: semantic clustering across sessions
- Signal 2: contradiction pairs via supersede chain
- Signal 3: session summary recurrence
- Edge cases: no data, single session, no embeddings
"""

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from tests.conftest import TEST_EMBEDDING_DIM
from memoria.core.memory.models.memory import MemoryRecord
from memoria.core.memory.tabular.candidates import TabularCandidateProvider, _cosine_similarity


def _utcnow():
    return datetime.now(timezone.utc)


def _make_memory(
    db, user_id: str, content: str, *,
    memory_type: str = "semantic",
    session_id: str | None = "s1",
    embedding: list[float] | None = None,
    superseded_by: str | None = None,
    is_active: int = 1,
    observed_at: datetime | None = None,
) -> MemoryRecord:
    """Insert a memory record and return it."""
    mid = uuid.uuid4().hex
    row = MemoryRecord(
        memory_id=mid,
        user_id=user_id,
        session_id=session_id,
        memory_type=memory_type,
        content=content,
        initial_confidence=0.75,
        trust_tier="T3",
        embedding=embedding,
        source_event_ids=[],
        superseded_by=superseded_by,
        is_active=is_active,
        observed_at=observed_at or _utcnow(),
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


# ── Signal 1: Semantic Clustering ─────────────────────────────────────


class TestSignalSemanticClusters:

    def test_cross_session_cluster_found(self, db, db_factory):
        """Two similar memories in different sessions → 1 candidate."""
        uid = f"u-{uuid.uuid4().hex[:8]}"
        emb = [1.0] * TEST_EMBEDDING_DIM
        _make_memory(db, uid, "Use ruff for linting", session_id="s1", embedding=emb)
        _make_memory(db, uid, "Use ruff for linting", session_id="s2", embedding=emb)

        provider = TabularCandidateProvider(db_factory)
        candidates = provider.get_reflection_candidates(uid, since_hours=1)

        semantic = [c for c in candidates if c.signal == "semantic_cluster"]
        assert len(semantic) == 1
        assert len(semantic[0].memories) == 2
        assert set(semantic[0].session_ids) == {"s1", "s2"}

    def test_same_session_not_clustered(self, db, db_factory):
        """Two similar memories in same session → no candidate."""
        uid = f"u-{uuid.uuid4().hex[:8]}"
        emb = [1.0] * TEST_EMBEDDING_DIM
        _make_memory(db, uid, "Use ruff", session_id="s1", embedding=emb)
        _make_memory(db, uid, "Use ruff", session_id="s1", embedding=emb)

        provider = TabularCandidateProvider(db_factory)
        candidates = provider.get_reflection_candidates(uid, since_hours=1)

        semantic = [c for c in candidates if c.signal == "semantic_cluster"]
        assert len(semantic) == 0

    def test_dissimilar_memories_not_clustered(self, db, db_factory):
        """Two orthogonal embeddings → no cluster."""
        uid = f"u-{uuid.uuid4().hex[:8]}"
        emb_a = [1.0] + [0.0] * (TEST_EMBEDDING_DIM - 1)
        emb_b = [0.0] + [1.0] + [0.0] * (TEST_EMBEDDING_DIM - 2)
        _make_memory(db, uid, "topic A", session_id="s1", embedding=emb_a)
        _make_memory(db, uid, "topic B", session_id="s2", embedding=emb_b)

        provider = TabularCandidateProvider(db_factory)
        candidates = provider.get_reflection_candidates(uid, since_hours=1)

        semantic = [c for c in candidates if c.signal == "semantic_cluster"]
        assert len(semantic) == 0

    def test_old_memories_excluded(self, db, db_factory):
        """Memories older than since_hours are excluded."""
        uid = f"u-{uuid.uuid4().hex[:8]}"
        emb = [1.0] * TEST_EMBEDDING_DIM
        old_time = _utcnow() - timedelta(hours=48)
        _make_memory(db, uid, "old", session_id="s1", embedding=emb, observed_at=old_time)
        _make_memory(db, uid, "old", session_id="s2", embedding=emb, observed_at=old_time)

        provider = TabularCandidateProvider(db_factory)
        candidates = provider.get_reflection_candidates(uid, since_hours=24)

        semantic = [c for c in candidates if c.signal == "semantic_cluster"]
        assert len(semantic) == 0

    def test_no_embedding_excluded(self, db, db_factory):
        """Memories without embeddings are excluded."""
        uid = f"u-{uuid.uuid4().hex[:8]}"
        _make_memory(db, uid, "no emb", session_id="s1", embedding=None)
        _make_memory(db, uid, "no emb", session_id="s2", embedding=None)

        provider = TabularCandidateProvider(db_factory)
        candidates = provider.get_reflection_candidates(uid, since_hours=1)

        semantic = [c for c in candidates if c.signal == "semantic_cluster"]
        assert len(semantic) == 0

    def test_memory_fields_populated(self, db, db_factory):
        """Verify all Memory fields are correctly populated from DB."""
        uid = f"u-{uuid.uuid4().hex[:8]}"
        emb = [0.5] * TEST_EMBEDDING_DIM
        _make_memory(db, uid, "content A", session_id="s1", embedding=emb,
                     memory_type="procedural")
        _make_memory(db, uid, "content A", session_id="s2", embedding=emb,
                     memory_type="procedural")

        provider = TabularCandidateProvider(db_factory)
        candidates = provider.get_reflection_candidates(uid, since_hours=1)

        semantic = [c for c in candidates if c.signal == "semantic_cluster"]
        assert len(semantic) == 1
        mem = semantic[0].memories[0]
        assert mem.user_id == uid
        assert mem.memory_type.value == "procedural"
        assert mem.content == "content A"
        assert mem.initial_confidence == 0.75
        assert mem.trust_tier.value == "T3"
        assert mem.is_active is True
        assert mem.observed_at is not None
        assert mem.embedding is not None


# ── Signal 2: Contradiction Pairs ─────────────────────────────────────


class TestSignalContradictionPairs:

    def test_supersede_chain_found(self, db, db_factory):
        """Old memory superseded by new → 1 contradiction candidate."""
        uid = f"u-{uuid.uuid4().hex[:8]}"
        new_row = _make_memory(db, uid, "new belief", session_id="s2")
        _make_memory(
            db, uid, "old belief", session_id="s1",
            superseded_by=new_row.memory_id, is_active=0,
        )

        provider = TabularCandidateProvider(db_factory)
        candidates = provider.get_reflection_candidates(uid, since_hours=1)

        contras = [c for c in candidates if c.signal == "contradiction"]
        assert len(contras) == 1
        assert len(contras[0].memories) == 2
        assert contras[0].importance_score > 0.3  # contradiction signal scores high
        # Verify old and new are both present
        contents = {m.content for m in contras[0].memories}
        assert "old belief" in contents
        assert "new belief" in contents

    def test_old_supersede_excluded(self, db, db_factory):
        """Supersede where new memory is old → excluded."""
        uid = f"u-{uuid.uuid4().hex[:8]}"
        old_time = _utcnow() - timedelta(hours=48)
        new_row = _make_memory(db, uid, "new", session_id="s2", observed_at=old_time)
        _make_memory(
            db, uid, "old", session_id="s1",
            superseded_by=new_row.memory_id, is_active=0,
            observed_at=old_time - timedelta(hours=1),
        )

        provider = TabularCandidateProvider(db_factory)
        candidates = provider.get_reflection_candidates(uid, since_hours=24)

        contras = [c for c in candidates if c.signal == "contradiction"]
        assert len(contras) == 0


# ── Signal 3: Summary Recurrence ──────────────────────────────────────


class TestSignalSummaryRecurrence:

    def test_recurring_summaries_found(self, db, db_factory):
        """3+ similar cross-session summaries → 1 recurrence candidate."""
        uid = f"u-{uuid.uuid4().hex[:8]}"
        emb = [0.7] * TEST_EMBEDDING_DIM
        for i in range(4):
            _make_memory(
                db, uid, f"User prefers concise output v{i}",
                memory_type="semantic", session_id=None, embedding=emb,
            )

        provider = TabularCandidateProvider(db_factory)
        candidates = provider.get_reflection_candidates(uid, since_hours=24)

        recur = [c for c in candidates if c.signal == "summary_recurrence"]
        assert len(recur) == 1
        assert len(recur[0].memories) == 4
        assert recur[0].importance_score > 0  # pre-computed by score_candidate

    def test_too_few_summaries_excluded(self, db, db_factory):
        """Only 2 summaries → below threshold, no candidate."""
        uid = f"u-{uuid.uuid4().hex[:8]}"
        emb = [0.7] * TEST_EMBEDDING_DIM
        for i in range(2):
            _make_memory(
                db, uid, f"summary {i}",
                memory_type="semantic", session_id=None, embedding=emb,
            )

        provider = TabularCandidateProvider(db_factory)
        candidates = provider.get_reflection_candidates(uid, since_hours=24)

        recur = [c for c in candidates if c.signal == "summary_recurrence"]
        assert len(recur) == 0


# ── Clustering Helper ─────────────────────────────────────────────────


class TestCosineAndClustering:

    def test_cosine_identical(self):
        assert abs(_cosine_similarity([1, 0], [1, 0]) - 1.0) < 0.001

    def test_cosine_orthogonal(self):
        assert abs(_cosine_similarity([1, 0], [0, 1])) < 0.001

    def test_cosine_zero_vector(self):
        assert _cosine_similarity([0, 0], [1, 0]) == 0.0


# ── Cross-User Isolation ──────────────────────────────────────────────


class TestUserIsolation:

    def test_other_user_memories_not_included(self, db, db_factory):
        """Memories from other users must not appear in candidates."""
        uid_a = f"u-{uuid.uuid4().hex[:8]}"
        uid_b = f"u-{uuid.uuid4().hex[:8]}"
        emb = [1.0] * TEST_EMBEDDING_DIM
        _make_memory(db, uid_a, "A's memory", session_id="s1", embedding=emb)
        _make_memory(db, uid_b, "B's memory", session_id="s2", embedding=emb)

        provider = TabularCandidateProvider(db_factory)
        candidates = provider.get_reflection_candidates(uid_a, since_hours=1)

        # uid_a has only 1 memory → no cross-session cluster
        for c in candidates:
            for m in c.memories:
                assert m.user_id == uid_a
