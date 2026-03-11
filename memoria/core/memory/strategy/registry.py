"""Strategy registry — maps strategy keys to implementations.

See docs/design/memory/backend-management.md §3.4
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from memoria.core.memory.strategy.protocol import IndexManager, RetrievalStrategy

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StrategyDescriptor:
    """Identifies a retrieval strategy + version + params."""

    strategy_type: str  # "vector" | "activation" | ...
    version: str  # "v1", "v2"
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.strategy_type}:{self.version}"

    @classmethod
    def parse(cls, key: str, params: dict[str, Any] | None = None) -> StrategyDescriptor:
        """Parse 'vector:v1' into StrategyDescriptor."""
        parts = key.split(":", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid strategy key '{key}', expected 'type:version'")
        return cls(strategy_type=parts[0], version=parts[1], params=params or {})


@dataclass
class _RegistryEntry:
    strategy_factory: Any  # Callable that creates RetrievalStrategy
    index_manager_factory: Any | None = None  # Callable that creates IndexManager


class StrategyRegistry:
    """Registry of available retrieval strategies."""

    def __init__(self) -> None:
        self._entries: dict[str, _RegistryEntry] = {}

    def register(
        self,
        key: str,
        strategy_factory: Any,
        index_manager_factory: Any | None = None,
    ) -> None:
        """Register a strategy factory.

        Args:
            key: Strategy key like 'vector:v1'.
            strategy_factory: Callable(**deps) -> RetrievalStrategy.
            index_manager_factory: Optional Callable(**deps) -> IndexManager.
        """
        self._entries[key] = _RegistryEntry(
            strategy_factory=strategy_factory,
            index_manager_factory=index_manager_factory,
        )
        logger.info("Registered strategy: %s", key)

    def create_strategy(
        self, descriptor: StrategyDescriptor, **deps: Any,
    ) -> RetrievalStrategy:
        """Create a retrieval strategy instance."""
        entry = self._entries.get(descriptor.key)
        if entry is None:
            raise ValueError(
                f"Unknown strategy '{descriptor.key}'. "
                f"Available: {list(self._entries.keys())}"
            )
        return entry.strategy_factory(params=descriptor.params, **deps)

    def create_index_manager(
        self, descriptor: StrategyDescriptor, **deps: Any,
    ) -> IndexManager | None:
        """Create an index manager, or None if strategy needs no index."""
        entry = self._entries.get(descriptor.key)
        if entry is None or entry.index_manager_factory is None:
            return None
        return entry.index_manager_factory(params=descriptor.params, **deps)

    def list_available(self) -> list[str]:
        """List registered strategy keys."""
        return list(self._entries.keys())
