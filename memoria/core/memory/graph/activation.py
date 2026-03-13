"""Spreading Activation engine — DB-backed iterative expansion.

Each propagation round fetches only the edges needed from DB,
instead of loading the entire graph into Python.

See docs/design/memory/graph-memory.md §3.2
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from memoria.core.memory.graph.types import EDGE_TYPE_MULTIPLIER, Edge

if TYPE_CHECKING:
    from memoria.core.memory.config import MemoryGovernanceConfig
    from memoria.core.memory.graph.graph_store import GraphStore

# ── Default hyperparameters (overridden by config when provided) ──────

DECAY_RATE = 0.5
SPREADING_FACTOR = 0.8
INHIBITION_BETA = 0.15
INHIBITION_TOP_M = 7
SIGMOID_GAMMA = 5.0
SIGMOID_THETA = 0.1
NUM_ITERATIONS = 3

# §13.1 Task-type edge boosts — applied on top of EDGE_TYPE_MULTIPLIER
TASK_EDGE_BOOST: dict[str, dict[str, float]] = {
    "code_review": {"causal": 1.5, "temporal": 0.5, "association": 1.0},
    "debugging": {"causal": 2.0, "temporal": 1.5, "association": 0.5},
    "planning": {"association": 1.2, "causal": 1.0, "temporal": 0.8},
}


def _edge_weight(
    edge: Edge,
    task_boost: dict[str, float] | None = None,
    type_override: dict[str, float] | None = None,
) -> float:
    mult = EDGE_TYPE_MULTIPLIER.get(edge.edge_type, 1.0)
    if type_override and edge.edge_type in type_override:
        mult = type_override[edge.edge_type]
    base = edge.weight * mult
    if task_boost:
        base *= task_boost.get(edge.edge_type, 1.0)
    return base


class SpreadingActivation:
    """DB-backed spreading activation with configurable hyperparameters."""

    def __init__(
        self,
        store: GraphStore,
        *,
        task_type: str | None = None,
        config: MemoryGovernanceConfig | None = None,
    ) -> None:
        self._store = store
        self._activation: dict[str, float] = {}
        self._out_degree: dict[str, int] = {}
        self._task_boost = TASK_EDGE_BOOST.get(task_type, None) if task_type else None
        # Load params from config or use module defaults
        if config is not None:
            self._decay = config.activation_decay_rate
            self._spread = config.activation_spreading_factor
            self._beta = config.activation_inhibition_beta
            self._top_m = config.activation_inhibition_top_m
            self._gamma = config.activation_sigmoid_gamma
            self._theta = config.activation_sigmoid_theta
            self._type_override: dict[str, float] = {
                "entity_link": config.activation_entity_link_multiplier,
            }
        else:
            self._decay = DECAY_RATE
            self._spread = SPREADING_FACTOR
            self._beta = INHIBITION_BETA
            self._top_m = INHIBITION_TOP_M
            self._gamma = SIGMOID_GAMMA
            self._theta = SIGMOID_THETA
            self._type_override = {}

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
        """One iteration: fetch edges → spread → inhibit → sigmoid."""
        active_ids = set(self._activation.keys())
        if not active_ids:
            return

        incoming, outgoing = self._store.get_edges_bidirectional(active_ids)

        contributor_ids: set[str] = set()
        for edges in incoming.values():
            for e in edges:
                contributor_ids.add(e.target_id)

        uncached = contributor_ids - set(self._out_degree.keys())
        if uncached:
            out_edges = self._store.get_edges_for_nodes(uncached)
            for nid, edges in out_edges.items():
                self._out_degree[nid] = max(len(edges), 1)
        for nid, edges in outgoing.items():
            if nid not in self._out_degree:
                self._out_degree[nid] = max(len(edges), 1)

        raw: dict[str, float] = {}

        for nid in active_ids:
            retention = (1 - self._decay) * self._activation.get(nid, 0.0)
            spread = 0.0
            for edge in incoming.get(nid, []):
                neighbor_id = edge.target_id
                neighbor_act = self._activation.get(neighbor_id, 0.0)
                if neighbor_act <= 0:
                    continue
                fan = self._out_degree.get(neighbor_id, 1)
                spread += (
                    self._spread
                    * _edge_weight(edge, self._task_boost, self._type_override)
                    * neighbor_act
                    / fan
                )
            raw[nid] = retention + spread

        for nid in active_ids:
            for edge in outgoing.get(nid, []):
                tid = edge.target_id
                if tid not in raw:
                    neighbor_act = self._activation.get(nid, 0.0)
                    if neighbor_act > 0:
                        fan = self._out_degree.get(nid, 1)
                        spread_val = (
                            self._spread
                            * _edge_weight(edge, self._task_boost, self._type_override)
                            * neighbor_act
                            / fan
                        )
                        raw[tid] = spread_val

        inhibited = self._lateral_inhibition(raw)

        self._activation = {}
        for nid, val in inhibited.items():
            s = self._sigmoid(val)
            if s > 0.01:
                self._activation[nid] = s

    def _sigmoid(self, x: float) -> float:
        z = self._gamma * (x - self._theta)
        if z < -20:
            return 0.0
        if z > 20:
            return 1.0
        return 1.0 / (1.0 + math.exp(-z))

    def _lateral_inhibition(self, raw: dict[str, float]) -> dict[str, float]:
        if not raw:
            return raw
        top_m = min(self._top_m, max(len(raw) // 3, 1))
        sorted_items = sorted(raw.items(), key=lambda x: x[1], reverse=True)
        top_m_values = [v for _, v in sorted_items[:top_m]]
        result: dict[str, float] = {}
        for nid, val in raw.items():
            inhibition = sum(
                self._beta * (top_val - val)
                for top_val in top_m_values
                if top_val > val
            )
            result[nid] = max(0.0, val - inhibition)
        return result
