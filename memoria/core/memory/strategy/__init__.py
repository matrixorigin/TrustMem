"""Retrieval strategy subsystem — pluggable retrieval for memory."""

from memoria.core.memory.strategy.protocol import IndexManager, RetrievalStrategy
from memoria.core.memory.strategy.registry import StrategyDescriptor, StrategyRegistry

__all__ = [
    "IndexManager",
    "RetrievalStrategy",
    "StrategyDescriptor",
    "StrategyRegistry",
]
