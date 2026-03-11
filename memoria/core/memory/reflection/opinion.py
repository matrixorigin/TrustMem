"""OpinionEvolver — evidence-based confidence updates for scene/reflection memories.

When new evidence arrives, existing reflection-produced memories (scenes) have
their confidence adjusted: supporting evidence increases confidence, contradicting
evidence decreases it. Trust tier promotion follows confidence thresholds.

See docs/design/memory/graph-memory.md §4.5
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from memoria.core.memory.config import MemoryGovernanceConfig
    from memoria.core.memory.types import Memory

logger = logging.getLogger(__name__)


@dataclass
class OpinionUpdate:
    """Result of an opinion evolution step."""

    memory_id: str
    old_confidence: float
    new_confidence: float
    evidence_type: str  # "supporting" | "contradicting" | "neutral"
    promoted: bool = False
    quarantined: bool = False


class OpinionEvolver:
    """Evolve confidence of reflection-produced memories based on new evidence."""

    def __init__(self, config: MemoryGovernanceConfig | None = None) -> None:
        from memoria.core.memory.config import DEFAULT_CONFIG
        c = config or DEFAULT_CONFIG
        self._supporting_delta = c.opinion_supporting_delta
        self._contradicting_delta = c.opinion_contradicting_delta
        self._confidence_cap = c.opinion_confidence_cap
        self._supporting_threshold = c.opinion_supporting_threshold
        self._contradicting_threshold = c.opinion_contradicting_threshold
        self._quarantine_threshold = c.opinion_quarantine_threshold
        self._t4_to_t3_confidence = c.opinion_t4_to_t3_confidence

    def evaluate_evidence(
        self,
        similarity: float,
        scene: Memory,
    ) -> OpinionUpdate:
        """Determine how new evidence affects a scene memory's confidence.

        Args:
            similarity: cosine similarity between new event and scene content.
            scene: the existing scene/reflection memory.

        Returns:
            OpinionUpdate with new confidence and any tier changes.
        """
        old_conf = scene.initial_confidence

        if similarity >= self._supporting_threshold:
            evidence_type = "supporting"
            new_conf = min(old_conf + self._supporting_delta, self._confidence_cap)
        elif similarity <= self._contradicting_threshold:
            evidence_type = "contradicting"
            new_conf = max(old_conf + self._contradicting_delta, 0.0)
        else:
            return OpinionUpdate(
                memory_id=scene.memory_id,
                old_confidence=old_conf,
                new_confidence=old_conf,
                evidence_type="neutral",
            )

        quarantined = new_conf < self._quarantine_threshold
        promoted = (
            not quarantined
            and new_conf >= self._t4_to_t3_confidence
            and scene.trust_tier.value == "T4"
        )

        return OpinionUpdate(
            memory_id=scene.memory_id,
            old_confidence=old_conf,
            new_confidence=new_conf,
            evidence_type=evidence_type,
            promoted=promoted,
            quarantined=quarantined,
        )
