"""GraphCandidateProvider — graph-based reflection candidate selection.

Uses spreading activation (DB-backed) to find high-activation subgraphs,
then groups into ReflectionCandidate clusters.

See docs/design/memory/graph-memory.md §4.3
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from memoria.core.memory.graph.activation import SpreadingActivation
from memoria.core.memory.graph.graph_store import GraphStore
from memoria.core.memory.graph.types import GraphNodeData, NodeType
from memoria.core.memory.interfaces import ReflectionCandidate
from memoria.core.memory.reflection.importance import score_candidate
from memoria.core.memory.types import Memory, MemoryType, TrustTier

if TYPE_CHECKING:
    from memoria.core.db_consumer import DbFactory
    from memoria.core.memory.config import MemoryGovernanceConfig

logger = logging.getLogger(__name__)

CLUSTER_ACTIVATION_THRESHOLD = 0.3
MIN_CLUSTER_SIZE = 3
MIN_SESSIONS = 2


class GraphCandidateProvider:
    def __init__(
        self, db_factory: DbFactory,
        config: MemoryGovernanceConfig | None = None,
    ) -> None:
        self._store = GraphStore(db_factory)
        if config is None:
            from memoria.core.memory.config import DEFAULT_CONFIG
            config = DEFAULT_CONFIG
        self._config = config

    def get_reflection_candidates(
        self, user_id: str, *, since_hours: int = 24,
    ) -> list[ReflectionCandidate]:
        node_count = self._store.count_user_nodes(user_id)
        if node_count < MIN_CLUSTER_SIZE:
            return []

        # Use recent episodic nodes as anchors — lightweight metadata-only query
        recent_cols = self._store.get_user_nodes(
            user_id, node_type=NodeType.EPISODIC, active_only=True,
            load_embedding=False,
        )
        anchors_list = sorted(recent_cols, key=lambda n: n.node_id, reverse=True)[:20]
        if not anchors_list:
            return []

        # DB-backed activation — no full graph load
        sa = SpreadingActivation(self._store)
        sa.set_anchors({n.node_id: 0.8 for n in anchors_list})
        sa.propagate()
        activated = sa.get_activated(min_activation=CLUSTER_ACTIVATION_THRESHOLD)

        if not activated:
            return []

        # Load only activated nodes
        activated_nodes = self._store.get_nodes_by_ids(list(activated.keys()))
        semantic_activated = [
            n for n in activated_nodes
            if n.node_type in (NodeType.SEMANTIC, NodeType.SCENE)
        ]
        if len(semantic_activated) < MIN_CLUSTER_SIZE:
            return []

        # Connected components via edge table
        clusters = self._find_connected_components(semantic_activated)

        candidates: list[ReflectionCandidate] = []
        for cluster in clusters:
            if len(cluster) < MIN_CLUSTER_SIZE:
                continue
            session_ids = list({n.session_id for n in cluster if n.session_id})
            if len(session_ids) < MIN_SESSIONS:
                continue
            memories = [self._node_to_memory(n) for n in cluster]
            has_conflict = any(n.conflicts_with for n in cluster)
            avg_activation = sum(
                activated.get(n.node_id, 0.0) for n in cluster
            ) / len(cluster)
            c = ReflectionCandidate(
                memories=memories,
                signal="contradiction" if has_conflict else "semantic_cluster",
                session_ids=session_ids,
            )
            c.importance_score = score_candidate(c, activation_energy=avg_activation)
            candidates.append(c)
        return candidates

    def _find_connected_components(
        self, nodes: list[GraphNodeData],
    ) -> list[list[GraphNodeData]]:
        """Connected components using edge table (not in-memory edges)."""
        node_set = {n.node_id for n in nodes}
        node_map = {n.node_id: n for n in nodes}

        # Batch fetch all edges between these nodes
        all_out = self._store.get_edges_for_nodes(node_set)
        adjacency: dict[str, set[str]] = {nid: set() for nid in node_set}
        for nid, edges in all_out.items():
            for e in edges:
                if e.target_id in node_set:
                    adjacency[nid].add(e.target_id)
                    adjacency[e.target_id].add(nid)

        visited: set[str] = set()
        components: list[list[GraphNodeData]] = []
        for nid in node_set:
            if nid in visited:
                continue
            component: list[GraphNodeData] = []
            queue = [nid]
            while queue:
                cur = queue.pop(0)
                if cur in visited:
                    continue
                visited.add(cur)
                component.append(node_map[cur])
                for neighbor in adjacency.get(cur, set()):
                    if neighbor not in visited:
                        queue.append(neighbor)
            if component:
                components.append(component)
        return components

    @staticmethod
    def _node_to_memory(node: GraphNodeData) -> Memory:
        try:
            tier = TrustTier(node.trust_tier)
        except ValueError:
            tier = TrustTier.T3_INFERRED
        return Memory(
            memory_id=node.memory_id or node.node_id,
            user_id=node.user_id,
            memory_type=MemoryType.SEMANTIC,
            content=node.content,
            initial_confidence=node.confidence,
            embedding=node.embedding,
            session_id=node.session_id,
            trust_tier=tier,
        )
