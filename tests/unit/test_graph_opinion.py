"""Tests for graph opinion evolution — §4.5."""

from unittest.mock import MagicMock, patch

import pytest

from memoria.core.memory.config import DEFAULT_CONFIG
from memoria.core.memory.graph.opinion import (
    OPINION_ACTIVATION_THRESHOLD,
    OPINION_ITERATIONS,
    OpinionEvolutionResult,
    evolve_opinions,
)
from memoria.core.memory.graph.types import GraphNodeData, NodeType


def _node(
    node_id: str = "n1", node_type: NodeType = NodeType.SEMANTIC,
    confidence: float = 0.5, trust_tier: str = "T4",
    embedding: list[float] | None = None, is_active: bool = True,
) -> GraphNodeData:
    return GraphNodeData(
        node_id=node_id, user_id="u1", node_type=node_type,
        content="test content", embedding=embedding or [0.1] * 8,
        confidence=confidence, trust_tier=trust_tier, is_active=is_active,
    )


class TestEvolveOpinions:

    def _mock_store(self) -> MagicMock:
        store = MagicMock()
        store.get_nodes_by_ids.return_value = []
        store.get_pair_similarity.return_value = 0.9
        store.update_confidence.return_value = None
        store.deactivate_node.return_value = None
        return store

    @patch("memoria.core.memory.graph.opinion.SpreadingActivation")
    def test_no_activated_nodes_returns_empty(self, MockSA):
        store = self._mock_store()
        sa_instance = MockSA.return_value
        sa_instance.get_activated.return_value = {}

        result = evolve_opinions(store, "new1", "u1")

        assert result.scenes_evaluated == 0
        sa_instance.set_anchors.assert_called_once_with({"new1": 1.0})
        sa_instance.propagate.assert_called_once_with(iterations=OPINION_ITERATIONS)

    @patch("memoria.core.memory.graph.opinion.SpreadingActivation")
    def test_no_scene_nodes_in_activated(self, MockSA):
        store = self._mock_store()
        sa_instance = MockSA.return_value
        sa_instance.get_activated.return_value = {"new1": 1.0, "sem1": 0.5}
        store.get_nodes_by_ids.return_value = [_node("sem1", NodeType.SEMANTIC)]

        result = evolve_opinions(store, "new1", "u1")

        assert result.scenes_evaluated == 0

    @patch("memoria.core.memory.graph.opinion.SpreadingActivation")
    def test_supporting_evidence_updates_confidence(self, MockSA):
        store = self._mock_store()
        sa_instance = MockSA.return_value
        scene = _node("scene1", NodeType.SCENE, confidence=0.5, trust_tier="T4")
        sa_instance.get_activated.return_value = {"new1": 1.0, "scene1": 0.4}
        store.get_nodes_by_ids.return_value = [scene]
        store.get_pair_similarity.return_value = 0.9

        result = evolve_opinions(store, "new1", "u1")

        assert result.scenes_evaluated == 1
        assert result.supporting == 1
        store.update_confidence.assert_called_once_with(
            "scene1", 0.5 + DEFAULT_CONFIG.opinion_supporting_delta,
        )

    @patch("memoria.core.memory.graph.opinion.SpreadingActivation")
    def test_contradicting_evidence_decreases_confidence(self, MockSA):
        store = self._mock_store()
        sa_instance = MockSA.return_value
        scene = _node("scene1", NodeType.SCENE, confidence=0.5, trust_tier="T4")
        sa_instance.get_activated.return_value = {"new1": 1.0, "scene1": 0.4}
        store.get_nodes_by_ids.return_value = [scene]
        store.get_pair_similarity.return_value = 0.1

        result = evolve_opinions(store, "new1", "u1")

        assert result.contradicting == 1
        store.update_confidence.assert_called_once_with(
            "scene1", 0.5 + DEFAULT_CONFIG.opinion_contradicting_delta,
        )

    @patch("memoria.core.memory.graph.opinion.SpreadingActivation")
    def test_neutral_evidence_no_db_update(self, MockSA):
        store = self._mock_store()
        sa_instance = MockSA.return_value
        scene = _node("scene1", NodeType.SCENE, confidence=0.5, trust_tier="T4")
        sa_instance.get_activated.return_value = {"new1": 1.0, "scene1": 0.4}
        store.get_nodes_by_ids.return_value = [scene]
        store.get_pair_similarity.return_value = 0.5

        result = evolve_opinions(store, "new1", "u1")

        assert result.neutral == 1
        store.update_confidence.assert_not_called()
        store.deactivate_node.assert_not_called()

    @patch("memoria.core.memory.graph.opinion.SpreadingActivation")
    def test_quarantine_on_low_confidence(self, MockSA):
        store = self._mock_store()
        sa_instance = MockSA.return_value
        scene = _node("scene1", NodeType.SCENE, confidence=0.15, trust_tier="T4")
        sa_instance.get_activated.return_value = {"new1": 1.0, "scene1": 0.4}
        store.get_nodes_by_ids.return_value = [scene]
        store.get_pair_similarity.return_value = 0.1

        result = evolve_opinions(store, "new1", "u1")

        assert result.quarantined == 1
        store.deactivate_node.assert_called_once_with("scene1")

    @patch("memoria.core.memory.graph.opinion.SpreadingActivation")
    def test_high_confidence_does_not_promote_instantly(self, MockSA):
        """Opinion evolution does NOT promote — that's consolidation's job (§4.7)."""
        store = self._mock_store()
        sa_instance = MockSA.return_value
        threshold = DEFAULT_CONFIG.opinion_t4_to_t3_confidence
        delta = DEFAULT_CONFIG.opinion_supporting_delta
        scene = _node("scene1", NodeType.SCENE, confidence=threshold - delta + 0.001, trust_tier="T4")
        sa_instance.get_activated.return_value = {"new1": 1.0, "scene1": 0.4}
        store.get_nodes_by_ids.return_value = [scene]
        store.get_pair_similarity.return_value = 0.9

        result = evolve_opinions(store, "new1", "u1")

        assert result.supporting == 1
        # Only confidence updated, NOT tier — promotion requires age gate in consolidation
        store.update_confidence.assert_called_once()
        store.update_confidence_and_tier.assert_not_called()

    @patch("memoria.core.memory.graph.opinion.SpreadingActivation")
    def test_inactive_scene_skipped(self, MockSA):
        store = self._mock_store()
        sa_instance = MockSA.return_value
        scene = _node("scene1", NodeType.SCENE, is_active=False)
        sa_instance.get_activated.return_value = {"new1": 1.0, "scene1": 0.4}
        store.get_nodes_by_ids.return_value = [scene]

        result = evolve_opinions(store, "new1", "u1")

        assert result.scenes_evaluated == 0

    @patch("memoria.core.memory.graph.opinion.SpreadingActivation")
    def test_no_embedding_pair_skipped(self, MockSA):
        store = self._mock_store()
        sa_instance = MockSA.return_value
        scene = _node("scene1", NodeType.SCENE)
        sa_instance.get_activated.return_value = {"new1": 1.0, "scene1": 0.4}
        store.get_nodes_by_ids.return_value = [scene]
        store.get_pair_similarity.return_value = None

        result = evolve_opinions(store, "new1", "u1")

        assert result.scenes_evaluated == 0

    @patch("memoria.core.memory.graph.opinion.SpreadingActivation")
    def test_multiple_scenes_evaluated(self, MockSA):
        store = self._mock_store()
        sa_instance = MockSA.return_value
        scenes = [
            _node("s1", NodeType.SCENE, confidence=0.5, trust_tier="T4"),
            _node("s2", NodeType.SCENE, confidence=0.6, trust_tier="T3"),
        ]
        sa_instance.get_activated.return_value = {"new1": 1.0, "s1": 0.4, "s2": 0.35}
        store.get_nodes_by_ids.return_value = scenes
        store.get_pair_similarity.return_value = 0.9

        result = evolve_opinions(store, "new1", "u1")

        assert result.scenes_evaluated == 2
        assert result.supporting == 2
        assert store.update_confidence.call_count == 2
