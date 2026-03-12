"""Spreading Activation engine — DB-backed iterative expansion.

Each propagation round fetches only the edges needed from DB,
instead of loading the entire graph into Python.

See docs/design/memory/graph-memory.md §3.2
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable

from memoria.core.memory.graph.types import EDGE_TYPE_MULTIPLIER, Edge

if TYPE_CHECKING:
    from memoria.core.memory.graph.graph_store import GraphStore

# ── Algorithm hyperparameters ─────────────────────────────────────────

DECAY_RATE = 0.5  # δ: retention decay per iteration
SPREADING_FACTOR = 0.8  # S: how much activation spreads to neighbors
INHIBITION_BETA = 0.15  # β: lateral inhibition strength
INHIBITION_TOP_M = 7  # M: number of top nodes for inhibition
SIGMOID_GAMMA = 5.0  # γ: sigmoid steepness
SIGMOID_THETA = 0.1  # θ: sigmoid threshold
NUM_ITERATIONS = 3

# §13.1 Task-type edge boosts — applied on top of EDGE_TYPE_MULTIPLIER
TASK_EDGE_BOOST: dict[str, dict[str, float]] = {
    "code_review": {"causal": 1.5, "temporal": 0.5, "association": 1.0},
    "debugging": {"causal": 2.0, "temporal": 1.5, "association": 0.5},
    "planning": {"association": 1.2, "causal": 1.0, "temporal": 0.8},
}


def _sigmoid(x: float) -> float:
    z = SIGMOID_GAMMA * (x - SIGMOID_THETA)
    if z < -20:
        return 0.0
    if z > 20:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))


def _edge_weight(edge: Edge, task_boost: dict[str, float] | None = None) -> float:
    # edge.weight carries per-edge quality (e.g. entity_link: regex=0.8, llm=0.9, manual=1.0).
    # EDGE_TYPE_MULTIPLIER is the base multiplier per edge type.
    base = edge.weight * EDGE_TYPE_MULTIPLIER.get(edge.edge_type, 1.0)
    if task_boost:
        base *= task_boost.get(edge.edge_type, 1.0)
    return base


class SpreadingActivation:
    """DB-backed spreading activation.

    Each iteration:
    1. Fetch incoming edges for all active nodes (single DB query)
    2. Compute raw activation from neighbors
    3. Lateral inhibition + sigmoid

    Total DB queries = NUM_ITERATIONS (3), regardless of graph size.
    """

    def __init__(
        self,
        store: GraphStore,
        *,
        task_type: str | None = None,
        filter_node_ids: Callable[[set[str]], set[str]] | None = None,
    ) -> None:
        self._store = store
        self._activation: dict[str, float] = {}
        self._out_degree: dict[str, int] = {}
        self._task_boost = TASK_EDGE_BOOST.get(task_type, None) if task_type else None
        self._filter_node_ids = filter_node_ids

    def set_anchors(self, anchors: dict[str, float]) -> None:
        self._activation = dict(anchors)

    def propagate(self, iterations: int = NUM_ITERATIONS) -> None:
        if not self._activation:
            return
        for _ in range(iterations):
            self._propagation_step()

    def get_activated(self, *, min_activation: float = 0.05) -> dict[str, float]:
        return {nid: a for nid, a in self._activation.items() if a >= min_activation}

    def _propagation_step(self) -> None:
        """One iteration: fetch edges → spread → inhibit → sigmoid.

        DB queries per iteration: 1 (bidirectional) + 0-1 (fan-out cache miss).
        """
        active_ids = set(self._activation.keys())
        if not active_ids:
            return

        # Single DB query: incoming + outgoing for all active nodes
        incoming, outgoing = self._store.get_edges_bidirectional(active_ids)

        # Collect contributor IDs for fan-out normalization
        contributor_ids: set[str] = set()
        for edges in incoming.values():
            for e in edges:
                contributor_ids.add(e.target_id)

        # Fan-out cache: only fetch uncached contributors (0-1 DB query)
        uncached = contributor_ids - set(self._out_degree.keys())
        if uncached:
            out_edges = self._store.get_edges_for_nodes(uncached)
            for nid, edges in out_edges.items():
                self._out_degree[nid] = max(len(edges), 1)
        # Also cache fan-out for active nodes from the outgoing we already have
        for nid, edges in outgoing.items():
            if nid not in self._out_degree:
                self._out_degree[nid] = max(len(edges), 1)

        # Compute raw activation
        raw: dict[str, float] = {}

        for nid in active_ids:
            retention = (1 - DECAY_RATE) * self._activation.get(nid, 0.0)
            spread = 0.0
            for edge in incoming.get(nid, []):
                neighbor_id = edge.target_id
                neighbor_act = self._activation.get(neighbor_id, 0.0)
                if neighbor_act <= 0:
                    continue
                fan = self._out_degree.get(neighbor_id, 1)
                spread += (
                    SPREADING_FACTOR
                    * _edge_weight(edge, self._task_boost)
                    * neighbor_act
                    / fan
                )
            raw[nid] = retention + spread

        # Activate newly reached neighbors (from outgoing, already fetched)
        for nid in active_ids:
            for edge in outgoing.get(nid, []):
                tid = edge.target_id
                if tid not in raw:
                    neighbor_act = self._activation.get(nid, 0.0)
                    if neighbor_act > 0:
                        fan = self._out_degree.get(nid, 1)
                        spread_val = (
                            SPREADING_FACTOR
                            * _edge_weight(edge, self._task_boost)
                            * neighbor_act
                            / fan
                        )
                        raw[tid] = spread_val

        # Lateral inhibition
        inhibited = self._lateral_inhibition(raw)

        # Sigmoid + update
        self._activation = {}
        for nid, val in inhibited.items():
            s = _sigmoid(val)
            if s > 0.01:
                self._activation[nid] = s
        if self._filter_node_ids and self._activation:
            allowed = self._filter_node_ids(set(self._activation))
            self._activation = {
                nid: score for nid, score in self._activation.items() if nid in allowed
            }

    @staticmethod
    def _lateral_inhibition(raw: dict[str, float]) -> dict[str, float]:
        if not raw:
            return raw
        sorted_items = sorted(raw.items(), key=lambda x: x[1], reverse=True)
        top_m_values = [v for _, v in sorted_items[:INHIBITION_TOP_M]]
        result: dict[str, float] = {}
        for nid, val in raw.items():
            inhibition = sum(
                INHIBITION_BETA * (top_val - val)
                for top_val in top_m_values
                if top_val > val
            )
            result[nid] = max(0.0, val - inhibition)
        return result
