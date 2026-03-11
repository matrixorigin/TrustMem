"""Unit tests for MemoryRetriever — ORM-based."""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from memoria.core.memory.tabular.retriever import TASK_WEIGHTS, MemoryRetriever
from memoria.core.memory.types import MemoryType
from tests.conftest import TEST_EMBEDDING_DIM


def _make_chain(rows=None):
    """Chainable ORM query mock."""
    chain = MagicMock()
    chain.filter.return_value = chain
    chain.add_columns.return_value = chain
    chain.order_by.return_value = chain
    chain.limit.return_value = chain
    chain.all.return_value = rows or []
    return chain


def _mem_row(memory_id, content="text", memory_type="semantic",
             confidence=0.8, observed_at=None, session_id=None,
             trust_tier="T3", relevance=1.0, ft_score=1.0):
    """Simulate an ORM result row from _phase1."""
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
        return MemoryRetriever(db_factory=lambda: mock_db)

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
        return MemoryRetriever(db_factory=lambda: mock_db)

    def test_vector_path_invoked_with_embedding(self, retriever, mock_db):
        """When query_embedding is provided, phase2 should run."""
        retriever.retrieve("u1", "test", session_id="s1", query_embedding=[0.1] * TEST_EMBEDDING_DIM)
        # At least 2 query calls: phase1 fallback + phase2 vector
        assert mock_db.query.call_count >= 2

    def test_vector_failure_graceful(self, retriever, mock_db):
        """Vector search failure should not crash — falls back to phase1 results."""
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                return _make_chain([_mem_row("m1")])  # phase1
            raise RuntimeError("vector down")

        mock_db.query.side_effect = side_effect
        results, _ = retriever.retrieve("u1", "test", session_id="s1", query_embedding=[0.1] * TEST_EMBEDDING_DIM)
        assert len(results) >= 1


class TestRetrieveExplain:
    """Tests for explain mode stats."""

    @pytest.fixture
    def mock_db(self):
        db = MagicMock()
        db.query.return_value = _make_chain()
        return db

    @pytest.fixture
    def retriever(self, mock_db):
        return MemoryRetriever(db_factory=lambda: mock_db)

    def test_explain_returns_stats(self, retriever, mock_db):
        _, stats = retriever.retrieve("u1", "test", session_id="s1", explain=True)
        assert stats is not None
        assert stats.total_ms >= 0

    def test_no_explain_returns_none(self, retriever, mock_db):
        _, stats = retriever.retrieve("u1", "test", session_id="s1", explain=False)
        assert stats is None

    def test_explain_candidate_scores_populated(self, retriever, mock_db):
        """explain=True with hybrid merge should populate per-candidate score breakdown."""
        rows_p1 = [_mem_row("m1", "Go testing"), _mem_row("m2", "Python flask")]
        rows_p2 = [_vec_row("m1", l2_dist=0.3), _vec_row("m3", l2_dist=0.8)]

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                return _make_chain(rows_p1)
            return _make_chain(rows_p2)

        mock_db.query.side_effect = side_effect

        memories, stats = retriever.retrieve(
            "u1", "Go testing", session_id="s1",
            query_embedding=[0.1] * TEST_EMBEDDING_DIM, explain=True,
        )
        assert stats is not None
        assert len(stats.candidate_scores) == len(memories)

        # Verify score breakdown fields are present and ordered by rank
        for i, cs in enumerate(stats.candidate_scores):
            assert cs.rank == i + 1
            assert cs.memory_id in {"m1", "m2", "m3"}
            assert cs.final_score > 0
            # All 4 dimension scores should be non-negative
            assert cs.vector_score >= 0
            assert cs.keyword_score >= 0
            assert cs.temporal_score >= 0
            assert cs.confidence_score >= 0

        # Scores should be descending
        scores = [cs.final_score for cs in stats.candidate_scores]
        assert scores == sorted(scores, reverse=True)

    def test_explain_no_candidate_scores_without_explain(self, retriever, mock_db):
        """explain=False should not populate candidate_scores."""
        rows = [_mem_row("m1")]
        mock_db.query.return_value = _make_chain(rows)

        _, stats = retriever.retrieve("u1", "test", session_id="s1", explain=False)
        assert stats is None


# ---------------------------------------------------------------------------
# BM25 normalization: score/(score+1) saturating transform
# ---------------------------------------------------------------------------

class TestBM25Normalization:
    """Verify the saturating transform used for keyword_score."""

    @pytest.fixture
    def retriever(self):
        return MemoryRetriever(db_factory=MagicMock(), metrics=MagicMock())

    def _make_candidate(self, keyword_score: float):
        from memoria.core.memory.tabular.retriever import _Candidate
        return _Candidate(
            memory_id="m1", content="x", memory_type="preference",
            initial_confidence=0.9, observed_at=datetime.now(timezone.utc),
            session_id="s1", keyword_score=keyword_score,
        )

    @pytest.mark.parametrize("raw,expected_approx", [
        (0.0, 0.0),
        (1.0, 0.5),
        (9.0, 0.9),
        (999.0, 0.999),
        (-1.0, 0.0),   # negative clamped to 0
    ])
    def test_bm25_score_normalization(self, retriever, raw, expected_approx):
        from memoria.core.memory.tabular.retriever import RetrievalWeights
        w = RetrievalWeights(vector=0, keyword=1, temporal=0, confidence=0)
        c = self._make_candidate(raw)
        final, _, kw, _, _ = retriever._score_candidate(c, w, datetime.now(timezone.utc).timestamp())
        assert abs(kw - expected_approx) < 0.01, f"raw={raw} → kw={kw}, expected≈{expected_approx}"
        assert final == pytest.approx(kw)  # keyword weight=1, others=0
