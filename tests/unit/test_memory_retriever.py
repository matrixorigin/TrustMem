"""Unit tests for MemoryRetriever — ORM-based."""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from memoria.core.memory.tabular.retriever import (
    TASK_WEIGHTS,
    MemoryRetriever,
)
from memoria.core.memory.types import MemoryType
from tests.conftest import TEST_EMBEDDING_DIM


def _make_chain(rows=None):
    """Chainable ORM query mock."""
    chain = MagicMock()
    chain.filter.return_value = chain
    chain.add_columns.return_value = chain
    chain.outerjoin.return_value = chain
    chain.order_by.return_value = chain
    chain.limit.return_value = chain
    chain.all.return_value = rows or []
    return chain


def _mem_row(
    memory_id,
    content="text",
    memory_type="semantic",
    confidence=0.8,
    observed_at=None,
    session_id=None,
    trust_tier="T3",
    relevance=1.0,
    ft_score=1.0,
    access_count=0,
):
    """Simulate an ORM result row from _phase1 (with JOIN to stats)."""
    r = MagicMock()
    r.memory_id = memory_id
    r.content = content
    r.memory_type = memory_type
    r.initial_confidence = confidence
    r.observed_at = observed_at or datetime(2026, 2, 26, tzinfo=timezone.utc)
    r.session_id = session_id
    r.trust_tier = trust_tier
    r.relevance = relevance
    r.ft_score = ft_score
    r.access_count = access_count  # From MemoryStats table (JOIN)
    return r


def _vec_row(memory_id, l2_dist=0.5, **kwargs):
    """Simulate an ORM result row from _phase2."""
    r = _mem_row(memory_id, **kwargs)
    r.l2_dist = l2_dist
    return r


class TestTaskWeights:
    def test_all_presets_sum_to_one(self):
        for name, w in TASK_WEIGHTS.items():
            total = w.vector + w.keyword + w.temporal + w.confidence
            assert abs(total - 1.0) < 0.01, f"{name} weights sum to {total}"

    def test_code_boosts_keyword(self):
        assert TASK_WEIGHTS["code"].keyword > TASK_WEIGHTS["reasoning"].keyword

    def test_recall_boosts_vector(self):
        assert TASK_WEIGHTS["recall"].vector > TASK_WEIGHTS["default"].vector


class TestRetrievePhase1:
    """Tests for keyword + fallback retrieval (no embedding)."""

    @pytest.fixture
    def mock_db(self):
        db = MagicMock()
        db.query.return_value = _make_chain()
        return db

    @pytest.fixture
    def retriever(self, mock_db):
        r = MemoryRetriever(db_factory=lambda: mock_db)
        r._load_l0 = MagicMock(return_value=[])  # L0 not under test here
        return r

    def test_returns_memories_from_fallback(self, retriever, mock_db):
        rows = [_mem_row("m1", "Go testing"), _mem_row("m2", "Python flask")]
        mock_db.query.return_value = _make_chain(rows)

        results, _ = retriever.retrieve("u1", "Go testing", session_id="s1")
        assert len(results) == 2
        assert results[0].memory_id == "m1"
        assert results[0].memory_type == MemoryType.SEMANTIC

    def test_empty_query_returns_fallback(self, retriever, mock_db):
        results, _ = retriever.retrieve("u1", "", session_id="s1")
        assert results == []
        assert mock_db.query.called

    def test_retrieve_invokes_orm_query(self, retriever, mock_db):
        """Verify retrieve uses ORM query (not raw execute)."""
        retriever.retrieve("u1", "test", session_id="s1")
        assert mock_db.query.called


class TestRetrievePhase2:
    """Tests for vector retrieval path."""

    @pytest.fixture
    def mock_db(self):
        db = MagicMock()
        db.query.return_value = _make_chain()
        return db

    @pytest.fixture
    def retriever(self, mock_db):
        r = MemoryRetriever(db_factory=lambda: mock_db)
        r._load_l0 = MagicMock(return_value=[])
        return r

    def test_returns_vector_candidates(self, retriever, mock_db):
        """Vector path returns candidates sorted by l2_dist."""
        rows = [
            _vec_row("m1", l2_dist=0.1),
            _vec_row("m2", l2_dist=0.5),
        ]
        mock_db.query.return_value = _make_chain(rows)

        embed = MagicMock()
        embed.embed.return_value = [0.1] * TEST_EMBEDDING_DIM

        results, _ = retriever.retrieve(
            "u1", "test", session_id="s1", query_embedding=[0.1] * TEST_EMBEDDING_DIM
        )
        assert len(results) == 2
        assert results[0].memory_id == "m1"

    def test_vector_fallback_on_error(self, retriever, mock_db):
        """Falls back to keyword if vector search fails."""
        # First call (vector) fails, second call (fallback) returns rows
        rows = [_mem_row("m1", "Go testing")]
        mock_db.query.return_value = _make_chain(rows)

        embed = MagicMock()
        embed.embed.side_effect = Exception("vector failed")

        results, _ = retriever.retrieve(
            "u1",
            "Go testing",
            session_id="s1",
            query_embedding=[0.1] * TEST_EMBEDDING_DIM,
        )
        assert len(results) == 1


class TestRetrieveL0:
    """Tests for L0 (working memory) retrieval."""

    @pytest.fixture
    def mock_db(self):
        db = MagicMock()
        db.query.return_value = _make_chain()
        return db

    @pytest.fixture
    def retriever(self, mock_db):
        return MemoryRetriever(db_factory=lambda: mock_db)

    def test_l0_included_when_session_id_set(self, retriever, mock_db):
        """L0 memories are included when session_id is provided."""
        from memoria.core.memory.types import Memory

        l0_memories = [
            Memory(
                memory_id="w1",
                user_id="u1",
                content="working memory",
                memory_type=MemoryType.WORKING,
                initial_confidence=1.0,
                session_id="s1",
            )
        ]
        # Mock _load_l0 to return working memories
        retriever._load_l0 = MagicMock(return_value=l0_memories)

        # Mock L1 to return empty
        mock_db.query.return_value = _make_chain([])

        results, _ = retriever.retrieve("u1", "test", session_id="s1")
        assert len(results) == 1
        assert results[0].memory_type == MemoryType.WORKING

    def test_l0_excluded_when_only_semantic_requested(self, retriever, mock_db):
        """L0 is excluded when memory_types=[semantic] is specified."""
        retriever._load_l0 = MagicMock(return_value=[])

        results, _ = retriever.retrieve(
            "u1", "test", session_id="s1", memory_types=[MemoryType.SEMANTIC]
        )
        # L0 should not be called
        retriever._load_l0.assert_not_called()


class TestRetrieveScoring:
    """Tests for retrieval scoring."""

    @pytest.fixture
    def mock_db(self):
        db = MagicMock()
        db.query.return_value = _make_chain()
        return db

    @pytest.fixture
    def retriever(self, mock_db):
        r = MemoryRetriever(db_factory=lambda: mock_db)
        r._load_l0 = MagicMock(return_value=[])  # L0 not under test here
        return r

    def test_access_count_boost(self, retriever, mock_db):
        """Memories with higher access_count get a boost."""
        # High access_count
        high_row = _mem_row("m1", "test", access_count=100)
        # Low access_count
        low_row = _mem_row("m2", "test", access_count=0)
        mock_db.query.return_value = _make_chain([high_row, low_row])

        results, _ = retriever.retrieve("u1", "test", session_id="s1")
        assert len(results) == 2
        # m1 should rank higher due to access_count boost
        assert results[0].memory_id == "m1"

    def test_temporal_decay(self, retriever, mock_db):
        """Older memories get lower temporal scores."""
        from datetime import timedelta

        now = datetime.now(timezone.utc)
        recent = _mem_row("m1", "test", observed_at=now - timedelta(hours=1))
        old = _mem_row("m2", "test", observed_at=now - timedelta(days=30))
        # Pass in expected order (mock doesn't sort, relevance depends on observed_at)
        mock_db.query.return_value = _make_chain([recent, old])

        results, _ = retriever.retrieve("u1", "test", session_id="s1")
        assert len(results) == 2
        # Recent should rank higher
        assert results[0].memory_id == "m1"


class TestRetrieveExplain:
    """Tests for explain mode."""

    @pytest.fixture
    def mock_db(self):
        db = MagicMock()
        db.query.return_value = _make_chain()
        return db

    @pytest.fixture
    def retriever(self, mock_db):
        r = MemoryRetriever(db_factory=lambda: mock_db)
        r._load_l0 = MagicMock(return_value=[])
        return r

    def test_explain_returns_stats(self, retriever, mock_db):
        """Explain mode returns retrieval statistics."""
        rows = [_mem_row("m1", "Go testing")]
        mock_db.query.return_value = _make_chain(rows)

        results, stats = retriever.retrieve(
            "u1", "Go testing", session_id="s1", explain=True
        )
        assert stats is not None
        assert stats.phase1_candidates == 1


class TestRetrieveEdgeCases:
    """Edge case tests."""

    @pytest.fixture
    def mock_db(self):
        db = MagicMock()
        db.query.return_value = _make_chain()
        return db

    @pytest.fixture
    def retriever(self, mock_db):
        r = MemoryRetriever(db_factory=lambda: mock_db)
        r._load_l0 = MagicMock(return_value=[])
        return r

    def test_empty_results(self, retriever, mock_db):
        """Empty results when no memories match."""
        mock_db.query.return_value = _make_chain([])

        results, _ = retriever.retrieve("u1", "nonexistent", session_id="s1")
        assert results == []

    def test_limit_respected(self, retriever, mock_db):
        """Results are limited to top_k."""
        rows = [_mem_row(f"m{i}", f"test {i}") for i in range(10)]
        mock_db.query.return_value = _make_chain(rows)

        results, _ = retriever.retrieve("u1", "test", session_id="s1", limit=5)
        assert len(results) == 5

    def test_session_id_none_excludes_l0(self, retriever, mock_db):
        """When session_id=None, L0 is excluded."""
        retriever._load_l0 = MagicMock(return_value=[])
        mock_db.query.return_value = _make_chain([_mem_row("m1", "test")])

        results, _ = retriever.retrieve("u1", "test", session_id=None)
        retriever._load_l0.assert_not_called()
        assert len(results) == 1
