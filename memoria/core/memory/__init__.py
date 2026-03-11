"""Memory subsystem — typed memory with tiered retrieval.

Public Interface (for external consumers):
- create_memory_service: factory function (preferred)
- MemoryService: unified facade (canonical storage + pluggable retrieval)
- MemoryReader, MemoryWriter, MemoryAdmin: Protocol interfaces
- GovernanceReport, HealthReport: result types
- Memory, MemoryType, TrustTier, RetrievalWeights: shared types

See docs/design/memory/backend-management.md
"""

# ── Public interface ──────────────────────────────────────────────────
from memoria.core.memory.config import DEFAULT_CONFIG, MemoryGovernanceConfig
from memoria.core.memory.experiment import (
    ExperimentConflictError,
    ExperimentLimitError,
    MemoryExperimentManager,
)
from memoria.core.memory.factory import create_memory_service, set_user_strategy, switch_user_strategy
from memoria.core.memory.interfaces import (
    CandidateProvider,
    GovernanceReport,
    HealthReport,
    MemoryAdmin,
    MemoryReader,
    MemoryWriter,
    ReflectionCandidate,
)
from memoria.core.memory.service import MemoryService
from memoria.core.memory.types import (
    TRUST_TIER_INITIAL_CONFIDENCE,
    Memory,
    MemoryType,
    RetrievalWeights,
    TrustTier,
    trust_tier_defaults,
)

__all__ = [
    "DEFAULT_CONFIG",
    "TRUST_TIER_INITIAL_CONFIDENCE",
    "CandidateProvider",
    "ExperimentConflictError",
    "ExperimentLimitError",
    "GovernanceReport",
    "HealthReport",
    "Memory",
    "MemoryAdmin",
    "MemoryExperimentManager",
    "MemoryGovernanceConfig",
    "MemoryReader",
    "MemoryService",
    "MemoryType",
    "MemoryWriter",
    "ReflectionCandidate",
    "RetrievalWeights",
    "TrustTier",
    "create_memory_service",
    "set_user_strategy",
    "switch_user_strategy",
    "trust_tier_defaults",
]
