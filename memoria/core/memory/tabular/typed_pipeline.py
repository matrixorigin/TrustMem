"""Typed memory pipeline: TypedObserver → Sandbox → Persist.

Pipeline phases:
  Phase 1: Observer extracts candidate memories (NOT yet persisted)
  Phase 2: Sandbox validates candidates in a zero-copy branch (optional)
  Phase 3: Persist validated memories (rejected candidates are discarded)

Reflector removed — no episodic→semantic promotion (episodic type eliminated).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from memoria.core.db_consumer import DbFactory
from memoria.core.memory.config import MemoryGovernanceConfig, DEFAULT_CONFIG
from memoria.core.memory.tabular.explain import ObserverStats, SandboxStats, PipelineStats
from memoria.core.memory.tabular.metrics import MemoryMetrics
from memoria.core.memory.tabular.store import MemoryStore
from memoria.core.memory.tabular.typed_observer import TypedObserver
from memoria.core.memory.tabular.sandbox import MemorySandbox
from memoria.core.memory.tabular.profile import ProfileManager
from memoria.core.memory.types import Memory

logger = logging.getLogger(__name__)


@dataclass
class TypedPipelineResult:
    memories_extracted: int = 0
    memories_validated: int = 0
    memories_rejected: int = 0
    profile_changed: bool = False
    errors: list[str] = field(default_factory=list)
    stats: Optional[PipelineStats] = None


def run_typed_memory_pipeline(
    db_factory: DbFactory,
    user_id: str,
    messages: list[dict[str, Any]],
    source_event_ids: Optional[list[str]] = None,
    llm_client: Any = None,
    embed_fn: Any = None,
    config: Optional[MemoryGovernanceConfig] = None,
    query_for_sandbox: Optional[str] = None,
    explain: bool = False,
    metrics: Optional[MemoryMetrics] = None,
) -> TypedPipelineResult:
    """Run typed memory pipeline: extract → validate → persist."""
    if config is None:
        config = DEFAULT_CONFIG

    _metrics = metrics or MemoryMetrics()
    start = time.time() if explain else 0
    result = TypedPipelineResult()
    if explain:
        result.stats = PipelineStats()

    store = MemoryStore(db_factory, metrics=_metrics)
    profile_mgr = ProfileManager(store)

    # Phase 1: Observer — extract candidate memories (NOT persisted yet)
    observer_start = time.time() if explain else 0
    candidates: list[Memory] = []
    observer_stats: Optional[ObserverStats] = None
    try:
        observer = TypedObserver(
            store=store,
            llm_client=llm_client,
            embed_fn=embed_fn,
            contradiction_threshold=config.contradiction_similarity_threshold,
            db_factory=db_factory,
            metrics=_metrics,
        )
        candidates = observer.extract_candidates(user_id, messages, source_event_ids)
        result.memories_extracted = len(candidates)
        if explain:
            observer_stats = ObserverStats(memories_extracted=len(candidates))
    except Exception as e:
        logger.error("Typed pipeline observer failed: %s", e)
        result.errors.append(f"observer: {e}")
        return result

    if not candidates:
        if result.stats:
            result.stats.total_ms = (time.time() - start) * 1000
        return result

    # Phase 2: Sandbox validation (optional)
    sandbox_stats: Optional[SandboxStats] = None
    validated = candidates
    if query_for_sandbox:
        validated = []
        try:
            sandbox = MemorySandbox(db_factory)
            needs_validation = []
            for mem in candidates:
                if mem.memory_type.value in config.sandbox_enabled_types:
                    needs_validation.append(mem)
                else:
                    validated.append(mem)

            if needs_validation:
                passed, sandbox_stats = sandbox.validate_memories(
                    user_id=user_id,
                    new_memories=needs_validation,
                    query_text=query_for_sandbox,
                    query_embedding=needs_validation[0].embedding,
                    explain=explain,
                )
                if passed:
                    validated.extend(needs_validation)
                    result.memories_validated = len(needs_validation)
                else:
                    result.memories_rejected = len(needs_validation)
                    if sandbox_stats:
                        sandbox_stats.rolled_back = True
            else:
                validated = candidates
        except Exception as e:
            logger.warning("Sandbox validation failed, accepting all: %s", e)
            _metrics.increment("sandbox_validation_errors")
            if explain and sandbox_stats is None:
                sandbox_stats = SandboxStats(enabled=True, error=str(e))
            validated = candidates

    # Phase 3: Persist validated memories (with contradiction check)
    persisted: list[Memory] = []
    for mem in validated:
        try:
            stored, c_stats = observer.persist_with_contradiction_check(mem, explain)
            persisted.append(stored)
            if observer_stats and c_stats and c_stats.found:
                observer_stats.memories_superseded += 1
                if observer_stats.contradiction is None:
                    observer_stats.contradiction = c_stats
        except Exception as e:
            logger.warning("Failed to persist memory: %s", e)

    if observer_stats:
        observer_stats.memories_stored = len(persisted)
        observer_stats.total_ms = (time.time() - observer_start) * 1000

    result.profile_changed = profile_mgr.update_from_memories(user_id, persisted)

    if result.stats:
        result.stats.observer = observer_stats
        result.stats.sandbox = sandbox_stats
        result.stats.total_ms = (time.time() - start) * 1000

    return result
