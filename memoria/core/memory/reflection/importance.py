"""Importance scoring — multi-signal heuristic for reflection candidates.

Computes importance from 4 graph signals. No LLM calls.
Each backend calls this (or its own scorer) and sets candidate.importance_score
before passing to ReflectionEngine.

See docs/design/memory/graph-memory.md §4.4, §13.3
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from memoria.core.memory.interfaces import ReflectionCandidate

# Default weights (sum = 1.0)
W_CENTRALITY = 0.25
W_CROSS_SESSION = 0.25
W_CONTRADICTION = 0.30
W_RECURRENCE = 0.20

# §13.3 Task-type-specific weight overrides
TASK_IMPORTANCE_WEIGHTS: dict[str, dict[str, float]] = {
    "debugging":   {"contradiction": 0.45, "recurrence": 0.25, "centrality": 0.15, "cross_session": 0.15},
    "code_review": {"cross_session": 0.35, "centrality": 0.30, "recurrence": 0.20, "contradiction": 0.15},
    "planning":    {"cross_session": 0.30, "centrality": 0.30, "recurrence": 0.25, "contradiction": 0.15},
}

# Thresholds
IMMEDIATE_THRESHOLD = 0.7  # event-triggered reflection
DAILY_THRESHOLD = 0.5      # queued for daily reflection


def _get_weights(task_type: str | None) -> tuple[float, float, float, float]:
    """Return (centrality, cross_session, contradiction, recurrence) weights."""
    if task_type and task_type in TASK_IMPORTANCE_WEIGHTS:
        w = TASK_IMPORTANCE_WEIGHTS[task_type]
        return w["centrality"], w["cross_session"], w["contradiction"], w["recurrence"]
    return W_CENTRALITY, W_CROSS_SESSION, W_CONTRADICTION, W_RECURRENCE


def score_candidate(
    candidate: ReflectionCandidate,
    activation_energy: float = 0.0,
    task_type: str | None = None,
) -> float:
    """Score a reflection candidate. Returns 0.0-1.0.

    Args:
        candidate: the candidate cluster
        activation_energy: avg activation of cluster nodes (graph-specific)
        task_type: optional task type for weight adjustment (§13.3)
    """
    w_cen, w_cs, w_con, w_rec = _get_weights(task_type)

    centrality = min(activation_energy, 1.0) if activation_energy > 0 else min(len(candidate.memories) / 5.0, 1.0)
    cross_session = min(len(set(candidate.session_ids)) / 3.0, 1.0)

    if candidate.signal == "contradiction":
        contradiction = 1.0
    elif any(
        getattr(m, "initial_confidence", 1.0) < 0.5
        for m in candidate.memories
    ):
        contradiction = 0.7
    else:
        contradiction = 0.0

    recurrence = min(len(candidate.memories) / 5.0, 1.0)

    return (
        w_cen * centrality
        + w_cs * cross_session
        + w_con * contradiction
        + w_rec * recurrence
    )
