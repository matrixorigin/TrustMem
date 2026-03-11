"""Graph Opinion Evolution — §4.5 evidence-based scene confidence updates.

On each new episodic event:
1. Run lightweight activation (1 iteration) anchored on the new node
2. Find activated scene nodes (activation > threshold)
3. Compute cosine similarity between new event and each scene (DB-side)
4. Delegate to OpinionEvolver for confidence delta + tier logic
5. Write updates back to DB (confidence, trust_tier, is_active)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from memoria.core.memory.graph.activation import SpreadingActivation
from memoria.core.memory.graph.types import NodeType
from memoria.core.memory.reflection.opinion import OpinionEvolver, OpinionUpdate
from memoria.core.memory.types import Memory, MemoryType, TrustTier

if TYPE_CHECKING:
    from memoria.core.memory.config import MemoryGovernanceConfig
    from memoria.core.memory.graph.graph_store import GraphStore

logger = logging.getLogger(__name__)

OPINION_ACTIVATION_THRESHOLD = 0.3
OPINION_ITERATIONS = 1  # lightweight — single propagation round


@dataclass
class OpinionEvolutionResult:
    """Summary of one opinion evolution pass."""

    scenes_evaluated: int = 0
    supporting: int = 0
    contradicting: int = 0
    neutral: int = 0
    quarantined: int = 0
    updates: list[OpinionUpdate] = field(default_factory=list)


def evolve_opinions(
    store: GraphStore,
    new_node_id: str,
    user_id: str,
    config: MemoryGovernanceConfig | None = None,
) -> OpinionEvolutionResult:
    """Run opinion evolution for a single new node against activated scenes.

    Args:
        store: graph store for DB access.
        new_node_id: the newly ingested node (episodic or semantic).
        user_id: owner.
        config: governance config (uses defaults if None).
    """
    result = OpinionEvolutionResult()

    # 1. Lightweight activation anchored on the new node
    sa = SpreadingActivation(store)
    sa.set_anchors({new_node_id: 1.0})
    sa.propagate(iterations=OPINION_ITERATIONS)
    activated = sa.get_activated(min_activation=OPINION_ACTIVATION_THRESHOLD)

    if not activated:
        return result

    # 2. Filter to scene nodes only
    activated_ids = set(activated.keys()) - {new_node_id}
    if not activated_ids:
        return result

    scene_nodes = [
        n for n in store.get_nodes_by_ids(list(activated_ids))
        if n.node_type == NodeType.SCENE and n.is_active
    ]
    if not scene_nodes:
        return result

    # 3. Compute similarity + evaluate evidence for each scene
    evolver = OpinionEvolver(config)

    for scene in scene_nodes:
        sim = store.get_pair_similarity(new_node_id, scene.node_id)
        if sim is None:
            continue  # one or both lack embeddings

        # Convert to Memory for OpinionEvolver interface
        scene_mem = Memory(
            memory_id=scene.node_id,
            user_id=scene.user_id,
            memory_type=MemoryType.SEMANTIC,
            content=scene.content,
            initial_confidence=scene.confidence,
            trust_tier=TrustTier(scene.trust_tier) if scene.trust_tier in {t.value for t in TrustTier} else TrustTier.T3_INFERRED,
        )

        update = evolver.evaluate_evidence(sim, scene_mem)
        result.scenes_evaluated += 1
        result.updates.append(update)

        if update.evidence_type == "neutral":
            result.neutral += 1
            continue

        if update.evidence_type == "supporting":
            result.supporting += 1
        else:
            result.contradicting += 1

        # 4. Write back to DB — confidence only; promotion is consolidation's job (§4.7)
        if update.quarantined:
            store.deactivate_node(scene.node_id)
            result.quarantined += 1
            logger.info("Scene %s quarantined (confidence %.2f)", scene.node_id, update.new_confidence)
        elif update.new_confidence != update.old_confidence:
            store.update_confidence(scene.node_id, update.new_confidence)

    return result
