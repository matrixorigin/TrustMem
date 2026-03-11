"""Tests for OpinionEvolver — evidence-based confidence updates."""

import pytest

from memoria.core.memory.config import DEFAULT_CONFIG
from memoria.core.memory.reflection.opinion import (
    OpinionEvolver,
    OpinionUpdate,
)
from memoria.core.memory.types import Memory, MemoryType, TrustTier

# Use config defaults for test assertions
SUPPORTING_DELTA = DEFAULT_CONFIG.opinion_supporting_delta
CONTRADICTING_DELTA = DEFAULT_CONFIG.opinion_contradicting_delta
CONFIDENCE_CAP = DEFAULT_CONFIG.opinion_confidence_cap
SUPPORTING_THRESHOLD = DEFAULT_CONFIG.opinion_supporting_threshold
CONTRADICTING_THRESHOLD = DEFAULT_CONFIG.opinion_contradicting_threshold
QUARANTINE_THRESHOLD = DEFAULT_CONFIG.opinion_quarantine_threshold
T4_TO_T3_CONFIDENCE = DEFAULT_CONFIG.opinion_t4_to_t3_confidence


def _scene(confidence: float = 0.5, tier: TrustTier = TrustTier.T4_UNVERIFIED) -> Memory:
    return Memory(
        memory_id="scene-1", user_id="u1", memory_type=MemoryType.SEMANTIC,
        content="User prefers verbose errors", initial_confidence=confidence,
        trust_tier=tier,
    )


class TestOpinionEvolver:
    def setup_method(self):
        self.evolver = OpinionEvolver()

    def test_supporting_evidence_increases_confidence(self):
        scene = _scene(0.5)
        result = self.evolver.evaluate_evidence(0.9, scene)

        assert result.evidence_type == "supporting"
        assert result.new_confidence == 0.5 + SUPPORTING_DELTA
        assert result.old_confidence == 0.5
        assert not result.quarantined
        assert not result.promoted

    def test_contradicting_evidence_decreases_confidence(self):
        scene = _scene(0.5)
        result = self.evolver.evaluate_evidence(0.1, scene)

        assert result.evidence_type == "contradicting"
        assert result.new_confidence == 0.5 + CONTRADICTING_DELTA  # 0.4
        assert not result.quarantined

    def test_neutral_evidence_no_change(self):
        scene = _scene(0.5)
        result = self.evolver.evaluate_evidence(0.5, scene)

        assert result.evidence_type == "neutral"
        assert result.new_confidence == result.old_confidence

    def test_confidence_capped_at_max(self):
        scene = _scene(CONFIDENCE_CAP - 0.01)
        result = self.evolver.evaluate_evidence(0.9, scene)

        assert result.new_confidence == CONFIDENCE_CAP

    def test_confidence_floored_at_zero(self):
        scene = _scene(0.05)
        result = self.evolver.evaluate_evidence(0.1, scene)

        assert result.new_confidence == 0.0
        assert result.quarantined  # below QUARANTINE_THRESHOLD

    def test_quarantine_below_threshold(self):
        scene = _scene(QUARANTINE_THRESHOLD - 0.01)
        result = self.evolver.evaluate_evidence(0.1, scene)

        assert result.quarantined

    def test_t4_promoted_to_t3_at_high_confidence(self):
        scene = _scene(T4_TO_T3_CONFIDENCE - SUPPORTING_DELTA + 0.001, TrustTier.T4_UNVERIFIED)
        result = self.evolver.evaluate_evidence(0.9, scene)

        assert result.promoted
        assert result.new_confidence >= T4_TO_T3_CONFIDENCE

    def test_t3_not_promoted_even_at_high_confidence(self):
        """Only T4 can be auto-promoted to T3."""
        scene = _scene(0.85, TrustTier.T3_INFERRED)
        result = self.evolver.evaluate_evidence(0.9, scene)

        assert not result.promoted

    def test_boundary_supporting_threshold(self):
        scene = _scene(0.5)
        # Exactly at threshold
        result = self.evolver.evaluate_evidence(SUPPORTING_THRESHOLD, scene)
        assert result.evidence_type == "supporting"

        # Just below threshold
        result = self.evolver.evaluate_evidence(SUPPORTING_THRESHOLD - 0.01, scene)
        assert result.evidence_type == "neutral"

    def test_boundary_contradicting_threshold(self):
        scene = _scene(0.5)
        # Exactly at threshold
        result = self.evolver.evaluate_evidence(CONTRADICTING_THRESHOLD, scene)
        assert result.evidence_type == "contradicting"

        # Just above threshold
        result = self.evolver.evaluate_evidence(CONTRADICTING_THRESHOLD + 0.01, scene)
        assert result.evidence_type == "neutral"

    def test_memory_id_preserved(self):
        scene = _scene(0.5)
        result = self.evolver.evaluate_evidence(0.9, scene)
        assert result.memory_id == "scene-1"
