"""Unit tests for graph activation and retrieval (Phase 2)."""

from unittest.mock import MagicMock

import pytest

from memoria.core.memory.graph.activation import (
    SIGMOID_THETA,
    SpreadingActivation,
    _sigmoid,
)
from memoria.core.memory.graph.retriever import (
    ActivationRetriever,
    _effective_confidence,
)
from memoria.core.memory.graph.types import Edge, GraphNodeData, NodeType


class TestSigmoid:
    def test_at_threshold(self):
        assert _sigmoid(SIGMOID_THETA) == pytest.approx(0.5)

    def test_high_input(self):
        assert _sigmoid(1.0) > 0.95

    def test_low_input(self):
        assert _sigmoid(-1.0) < 0.05

    def test_clamp_extremes(self):
        assert _sigmoid(100.0) == 1.0
        assert _sigmoid(-100.0) == 0.0


class TestSpreadingActivation:
    def _make_store(self):
        store = MagicMock()
        store.get_edges_bidirectional.return_value = ({}, {})
        store.get_edges_for_nodes.return_value = {}
        store.filter_retrievable_node_ids.side_effect = lambda _user_id, node_ids: set(
            node_ids
        )
        return store

    def test_no_anchors_no_activation(self):
        store = self._make_store()
        sa = SpreadingActivation(store)
        sa.propagate()
        assert sa.get_activated() == {}

    def test_anchor_activates_self(self):
        store = self._make_store()
        sa = SpreadingActivation(store)
        sa.set_anchors({"a": 0.8})
        sa.propagate()
        result = sa.get_activated()
        assert "a" in result
        assert result["a"] > 0

    def test_activation_spreads_to_neighbors(self):
        store = self._make_store()

        def mock_bidir(node_ids):
            incoming = {nid: [] for nid in node_ids}
            outgoing = {nid: [] for nid in node_ids}
            if "b" in node_ids:
                incoming["b"] = [Edge("a", "association", 1.0)]
            if "a" in node_ids:
                outgoing["a"] = [Edge("b", "association", 1.0)]
            return incoming, outgoing

        store.get_edges_bidirectional.side_effect = mock_bidir
        store.get_edges_for_nodes.side_effect = lambda ids: {
            nid: [Edge("b", "association", 1.0)] if nid == "a" else [] for nid in ids
        }

        sa = SpreadingActivation(store)
        sa.set_anchors({"a": 0.8})
        sa.propagate()
        result = sa.get_activated()
        assert "b" in result

    def test_empty_graph(self):
        store = self._make_store()
        sa = SpreadingActivation(store)
        sa.set_anchors({})
        sa.propagate()
        assert sa.get_activated() == {}

    def test_filter_node_ids_prunes_activation_between_steps(self):
        store = self._make_store()

        def mock_bidir(node_ids):
            incoming = {nid: [] for nid in node_ids}
            outgoing = {nid: [] for nid in node_ids}
            if "a" in node_ids:
                outgoing["a"] = [Edge("b", "association", 1.0)]
            if "b" in node_ids:
                outgoing["b"] = [Edge("c", "association", 1.0)]
                incoming["b"] = [Edge("a", "association", 1.0)]
            if "c" in node_ids:
                incoming["c"] = [Edge("b", "association", 1.0)]
            return incoming, outgoing

        store.get_edges_bidirectional.side_effect = mock_bidir
        store.get_edges_for_nodes.side_effect = lambda ids: {
            nid: [Edge("x", "association", 1.0)] if nid in {"a", "b"} else []
            for nid in ids
        }

        sa = SpreadingActivation(
            store,
            filter_node_ids=lambda node_ids: {nid for nid in node_ids if nid != "b"},
        )
        sa.set_anchors({"a": 0.8})
        sa.propagate(iterations=2)

        result = sa.get_activated()
        assert "b" not in result
        assert "c" not in result


class TestEffectiveConfidence:
    def test_no_created_at(self):
        node = GraphNodeData(
            node_id="n1",
            user_id="u1",
            node_type=NodeType.SEMANTIC,
            content="test",
            confidence=0.8,
        )
        assert _effective_confidence(node) == 0.8

    def test_decay_over_time(self):
        from datetime import datetime, timedelta, timezone

        old = datetime.now(timezone.utc) - timedelta(days=60)
        node = GraphNodeData(
            node_id="n1",
            user_id="u1",
            node_type=NodeType.SEMANTIC,
            content="test",
            confidence=0.8,
            trust_tier="T3",
            created_at=old.isoformat(),
        )
        decayed = _effective_confidence(node)
        assert decayed < 0.8
        assert decayed > 0


class TestActivationRetriever:
    def _make_retriever(self):
        store = MagicMock()
        store.has_min_nodes.return_value = True
        store.find_similar_with_scores.return_value = []
        store.get_edges_bidirectional.return_value = ({}, {})
        store.get_edges_for_nodes.return_value = {}
        store.get_nodes_by_ids.return_value = []
        store.filter_retrievable_node_ids.side_effect = lambda _user_id, node_ids: set(
            node_ids
        )
        store.filter_retrievable_nodes.side_effect = lambda _user_id, nodes: nodes
        return ActivationRetriever(store), store

    def test_fallback_when_graph_too_small(self):
        retriever, store = self._make_retriever()
        store.has_min_nodes.return_value = False
        result = retriever.retrieve("u1", "query", [0.1] * 10)
        assert result == []
        store.find_similar_with_scores.assert_not_called()

    def test_fallback_when_no_embedding(self):
        retriever, _ = self._make_retriever()
        result = retriever.retrieve("u1", "query")
        assert result == []

    def test_returns_scored_nodes(self):
        retriever, store = self._make_retriever()
        node = GraphNodeData(
            node_id="n1",
            user_id="u1",
            node_type=NodeType.SEMANTIC,
            content="test",
            confidence=0.8,
            importance=0.5,
        )
        store.find_similar_with_scores.return_value = [(node, 0.9)]
        store.get_nodes_by_ids.return_value = [node]

        result = retriever.retrieve("u1", "query", [0.1] * 10)
        assert len(result) == 1
        assert result[0][0].node_id == "n1"
        assert result[0][1] > 0

    def test_conflict_penalty_applied(self):
        retriever, store = self._make_retriever()
        node = GraphNodeData(
            node_id="n1",
            user_id="u1",
            node_type=NodeType.SEMANTIC,
            content="test",
            confidence=0.8,
            importance=0.5,
            conflicts_with="n2",
            conflict_resolution="superseded",
        )
        store.find_similar_with_scores.return_value = [(node, 0.9)]
        store.get_nodes_by_ids.return_value = [node]

        result = retriever.retrieve("u1", "query", [0.1] * 10)
        assert len(result) == 1
        # Superseded penalty = 0.5
        assert result[0][1] < 0.5


class TestGraphServiceRetrieve:
    def test_activation_result_converted_to_memories(self):
        from memoria.core.memory.graph.service import GraphMemoryService

        svc = GraphMemoryService(lambda: MagicMock())
        svc._tabular = MagicMock()

        mock_retriever = MagicMock()
        node = GraphNodeData(
            node_id="n1",
            user_id="u1",
            node_type=NodeType.SEMANTIC,
            content="test",
            confidence=0.8,
            memory_id="mem1",
            session_id="s1",
            trust_tier="T3",
        )
        mock_retriever.retrieve.return_value = [(node, 0.9)]
        svc._activation_retriever = mock_retriever

        result = svc.retrieve("u1", "query", query_embedding=[0.1] * 10)
        assert len(result) == 1
        assert result[0].content == "test"

    def test_fallback_to_tabular_on_activation_failure(self):
        from sqlalchemy.exc import OperationalError
        from memoria.core.memory.graph.service import GraphMemoryService

        svc = GraphMemoryService(lambda: MagicMock())
        svc._tabular = MagicMock()
        svc._tabular.retrieve.return_value = ["tabular_mem"]

        mock_retriever = MagicMock()
        mock_retriever.retrieve.side_effect = OperationalError(
            "db", {}, Exception("conn lost")
        )
        svc._activation_retriever = mock_retriever

        result = svc.retrieve("u1", "query", query_embedding=[0.1] * 10)
        assert result == ["tabular_mem"]

    def test_fallback_when_activation_returns_empty(self):
        from memoria.core.memory.graph.service import GraphMemoryService

        svc = GraphMemoryService(lambda: MagicMock())
        svc._tabular = MagicMock()
        svc._tabular.retrieve.return_value = ["tabular_mem"]

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = []
        svc._activation_retriever = mock_retriever

        result = svc.retrieve("u1", "query", query_embedding=[0.1] * 10)
        assert result == ["tabular_mem"]
