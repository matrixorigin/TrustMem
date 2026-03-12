from unittest.mock import MagicMock

from memoria.core.memory.graph.retriever import ActivationRetriever
from memoria.core.memory.graph.types import GraphNodeData, NodeType


class TestActivationRetrieverConsistency:
    def test_filters_semantic_nodes_without_active_backing_memory(self):
        store = MagicMock()
        store.has_min_nodes.return_value = True

        stale_node = GraphNodeData(
            node_id="n-stale",
            user_id="u1",
            node_type=NodeType.SEMANTIC,
            content="stale memory",
            memory_id="mem-stale",
            confidence=0.9,
            importance=0.5,
            trust_tier="T2",
        )
        live_node = GraphNodeData(
            node_id="n-live",
            user_id="u1",
            node_type=NodeType.SEMANTIC,
            content="live memory",
            memory_id="mem-live",
            confidence=0.9,
            importance=0.5,
            trust_tier="T2",
        )
        scene_node = GraphNodeData(
            node_id="n-scene",
            user_id="u1",
            node_type=NodeType.SCENE,
            content="scene summary",
            confidence=0.8,
            importance=0.9,
            trust_tier="T3",
        )

        store.find_similar_with_scores.return_value = [
            (stale_node, 0.95),
            (live_node, 0.9),
        ]
        store.get_edges_bidirectional.return_value = ({}, {})
        store.get_edges_for_nodes.return_value = {}
        store.get_nodes_by_ids.return_value = [stale_node, live_node, scene_node]
        store.filter_retrievable_node_ids.side_effect = lambda _user_id, node_ids: {
            nid for nid in node_ids if nid != "n-stale"
        }
        store.filter_retrievable_nodes.return_value = [live_node, scene_node]

        retriever = ActivationRetriever(store)
        result = retriever.retrieve("u1", "query", [0.1] * 10, top_k=5)

        store.filter_retrievable_node_ids.assert_called()
        store.filter_retrievable_nodes.assert_called_once()
        returned = {node.node_id for node, _ in result}
        assert "n-stale" not in returned
        assert "n-live" in returned
        assert "n-scene" in returned
