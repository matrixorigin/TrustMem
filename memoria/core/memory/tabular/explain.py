"""EXPLAIN ANALYZE for memory operations.

Like database EXPLAIN ANALYZE, provides execution stats for debugging,
testing, and observability. Stats flow bottom-up through the call chain.

Usage:
    # Component level
    memories, stats = retriever.retrieve(..., explain=True)
    
    # API level
    POST /chat {"explain": true}
    -> {"response": ..., "explain": {"retrieval": {...}, "pipeline": {...}}}
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class CandidateScore:
    """Per-candidate scoring breakdown — answers 'why is this memory ranked here?'"""

    memory_id: str
    final_score: float
    vector_score: float
    keyword_score: float
    temporal_score: float
    confidence_score: float
    rank: int  # 1-based position in final ranking


@dataclass
class RetrievalStats:
    """Phase-level stats for hybrid retrieval."""
    
    # Phase 1: keyword/fallback
    keyword_attempted: bool = False
    keyword_hit: bool = False
    keyword_error: Optional[str] = None
    phase1_candidates: int = 0
    
    # Phase 2: vector
    vector_attempted: bool = False
    vector_hit: bool = False
    vector_error: Optional[str] = None
    phase2_candidates: int = 0
    
    # Phase 3: merge
    merged_candidates: int = 0
    final_count: int = 0
    candidate_scores: list[CandidateScore] = field(default_factory=list)
    
    # Timing (ms)
    phase1_ms: float = 0.0
    phase2_ms: float = 0.0
    merge_ms: float = 0.0
    total_ms: float = 0.0


@dataclass
class ContradictionStats:
    """Stats for contradiction detection."""
    
    checked: bool = False
    found: bool = False
    superseded_id: Optional[str] = None
    error: Optional[str] = None
    query_ms: float = 0.0


@dataclass
class ObserverStats:
    """Stats for memory observation (extraction + storage)."""
    
    memories_extracted: int = 0
    memories_stored: int = 0
    memories_superseded: int = 0
    contradiction: Optional[ContradictionStats] = None
    total_ms: float = 0.0


@dataclass
class SandboxStats:
    """Stats for sandbox validation."""
    
    enabled: bool = False
    branch_name: Optional[str] = None
    validated: bool = False
    rolled_back: bool = False
    error: Optional[str] = None
    total_ms: float = 0.0


@dataclass
class GovernanceStats:
    """Stats for governance operations."""
    
    decay_triggered: bool = False
    memories_decayed: int = 0
    quarantine_triggered: bool = False
    memories_quarantined: int = 0
    compression_triggered: bool = False
    memories_compressed: int = 0
    total_ms: float = 0.0


@dataclass
class PipelineStats:
    """Stats for the full memory pipeline (observe → sandbox → governance)."""
    
    observer: Optional[ObserverStats] = None
    sandbox: Optional[SandboxStats] = None
    governance: Optional[GovernanceStats] = None
    total_ms: float = 0.0


@dataclass 
class MemoryStats:
    """Top-level stats aggregating all memory operations in a request."""
    
    retrieval: Optional[RetrievalStats] = None
    pipeline: Optional[PipelineStats] = None
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict, omitting None values."""
        def _clean(obj):
            if obj is None:
                return None
            if hasattr(obj, '__dataclass_fields__'):
                return {k: _clean(v) for k, v in asdict(obj).items() if v is not None}
            return obj
        return _clean(self)


@dataclass
class ExplainResult:
    """Complete EXPLAIN ANALYZE output for a chat request."""
    
    memory: Optional[MemoryStats] = None
    # Future: llm, skill_selection, etc.
    total_ms: float = 0.0
    
    def to_dict(self) -> dict:
        result = {}
        if self.memory:
            result["memory"] = self.memory.to_dict()
        result["total_ms"] = self.total_ms
        return result
