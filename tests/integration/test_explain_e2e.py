"""End-to-end EXPLAIN ANALYZE test — verifies explain flows through entire call chain."""

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from memoria.core.memory.tabular.store import MemoryStore
from memoria.core.memory.tabular.retriever import MemoryRetriever
from memoria.core.memory.tabular.typed_observer import TypedObserver
from memoria.core.memory.tabular.typed_pipeline import run_typed_memory_pipeline
from memoria.core.tiered_loader import TieredMemoryLoader
from memoria.core.memory.tabular.service import MemoryService
from memoria.core.memory.types import Memory, MemoryType

EMBEDDING_DIM = 384  # fixed for integration tests


def _uid():
    return f"explain_test_{uuid4().hex}"


def _embed(text: str) -> list[float]:
    """Deterministic embedding for testing."""
    return [hash(text) % 100 / 100.0] * EMBEDDING_DIM


class TestExplainE2E:
    """End-to-end EXPLAIN ANALYZE verification."""

    @pytest.fixture
    def db_factory(self):
        """Real DB factory for integration tests."""
        from tests.integration.conftest import _get_session_local

        SessionLocal = _get_session_local()
        return SessionLocal

    @pytest.fixture
    def cleanup_memories(self, db_factory):
        """Cleanup created memories after test."""
        memory_ids = []
        yield memory_ids
        if memory_ids:
            from sqlalchemy import text

            db = db_factory()
            try:
                db.execute(
                    text("DELETE FROM mem_memories WHERE memory_id IN :ids"),
                    {"ids": tuple(memory_ids)},
                )
                db.commit()
            finally:
                db.close()

    def test_retriever_explain_shows_all_phases(self, db_factory, cleanup_memories):
        """Retriever explain shows keyword, vector, and merge phases."""
        store = MemoryStore(db_factory)
        retriever = MemoryRetriever(db_factory)
        user_id = _uid()
        session_id = f"sess_{uuid4().hex}"

        # Create test memory with embedding
        mem = Memory(
            memory_id=f"exp_{uuid4().hex}",
            user_id=user_id,
            memory_type=MemoryType.SEMANTIC,
            content="User prefers Python for data analysis",
            initial_confidence=0.9,
            embedding=_embed("Python data analysis"),
            observed_at=datetime.now(timezone.utc),
        )
        cleanup_memories.append(mem.memory_id)
        store.create(mem)

        # Retrieve with explain=True
        results, stats = retriever.retrieve(
            user_id=user_id,
            query_text="Python",
            session_id=session_id,
            query_embedding=_embed("Python"),
            explain=True,
        )

        # Verify stats structure
        assert stats is not None
        print("\n=== RETRIEVER EXPLAIN OUTPUT ===")
        print("Phase 1 (keyword/fallback):")
        print(f"  - attempted: {stats.keyword_attempted}")
        print(f"  - hit: {stats.keyword_hit}")
        print(f"  - error: {stats.keyword_error}")
        print(f"  - candidates: {stats.phase1_candidates}")
        print(f"  - time: {stats.phase1_ms:.2f}ms")
        print("Phase 2 (vector):")
        print(f"  - attempted: {stats.vector_attempted}")
        print(f"  - hit: {stats.vector_hit}")
        print(f"  - error: {stats.vector_error}")
        print(f"  - candidates: {stats.phase2_candidates}")
        print(f"  - time: {stats.phase2_ms:.2f}ms")
        print("Phase 3 (merge):")
        print(f"  - merged candidates: {stats.merged_candidates}")
        print(f"  - final count: {stats.final_count}")
        print(f"  - time: {stats.merge_ms:.2f}ms")
        print(f"Total: {stats.total_ms:.2f}ms")

        # Assertions
        assert stats.total_ms > 0
        assert stats.vector_attempted  # Should have tried vector search

    def test_observer_explain_shows_contradiction_check(
        self, db_factory, cleanup_memories
    ):
        """Observer explain shows extraction and contradiction detection."""
        store = MemoryStore(db_factory)
        user_id = _uid()

        # Create existing memory that might contradict
        old_mem = Memory(
            memory_id=f"old_{uuid4().hex}",
            user_id=user_id,
            memory_type=MemoryType.PROFILE,
            content="User prefers tabs",
            initial_confidence=0.8,
            embedding=[0.5] * EMBEDDING_DIM,
            observed_at=datetime.now(timezone.utc),
        )
        cleanup_memories.append(old_mem.memory_id)
        store.create(old_mem)

        # Observer with real DB
        observer = TypedObserver(
            store=store,
            llm_client=None,  # Skip LLM extraction
            embed_fn=lambda x: [0.5] * EMBEDDING_DIM,  # Same embedding → contradiction
            db_factory=db_factory,
        )

        # Write contradicting memory with explain
        new_mem, stats = observer.observe_explicit(
            user_id=user_id,
            content="User prefers spaces",
            memory_type=MemoryType.PROFILE,
            initial_confidence=0.9,
            explain=True,
        )
        cleanup_memories.append(new_mem.memory_id)

        print("\n=== OBSERVER EXPLAIN OUTPUT ===")
        print("Contradiction check:")
        print(f"  - checked: {stats.checked}")
        print(f"  - found: {stats.found}")
        print(f"  - superseded_id: {stats.superseded_id}")
        print(f"  - error: {stats.error}")
        print(f"  - query_ms: {stats.query_ms:.2f}ms")

        # Should have found contradiction
        assert stats.checked
        assert stats.found
        assert stats.superseded_id == old_mem.memory_id

    def test_pipeline_explain_aggregates_all_stats(self, db_factory, cleanup_memories):
        """Pipeline explain aggregates observer, sandbox, and governance stats."""
        user_id = _uid()

        # Mock LLM to return memories
        mock_llm = MagicMock()
        mock_llm.chat_with_tools.return_value = {
            "content": json.dumps(
                [
                    {"type": "profile", "content": "User likes Go", "confidence": 0.9},
                    {
                        "type": "semantic",
                        "content": "Discussed testing",
                        "confidence": 0.7,
                    },
                ]
            )
        }

        result = run_typed_memory_pipeline(
            db_factory=db_factory,
            user_id=user_id,
            messages=[{"role": "user", "content": "I like Go and testing"}],
            llm_client=mock_llm,
            embed_fn=_embed,
            query_for_sandbox="Go testing",
            explain=True,
        )

        # Cleanup created memories
        from sqlalchemy import text

        db = db_factory()
        try:
            rows = db.execute(
                text("SELECT memory_id FROM mem_memories WHERE user_id = :uid"),
                {"uid": user_id},
            ).fetchall()
            for row in rows:
                cleanup_memories.append(row.memory_id)
        finally:
            db.close()

        print("\n=== PIPELINE EXPLAIN OUTPUT ===")
        print("Result:")
        print(f"  - memories_extracted: {result.memories_extracted}")
        print(f"  - memories_validated: {result.memories_validated}")
        print(f"  - memories_rejected: {result.memories_rejected}")

        if result.stats:
            print("Stats:")
            print(f"  - total_ms: {result.stats.total_ms:.2f}ms")
            if result.stats.observer:
                print("  Observer:")
                print(
                    f"    - memories_extracted: {result.stats.observer.memories_extracted}"
                )
                print(f"    - memories_stored: {result.stats.observer.memories_stored}")
                print(
                    f"    - memories_superseded: {result.stats.observer.memories_superseded}"
                )
                print(f"    - total_ms: {result.stats.observer.total_ms:.2f}ms")
            if result.stats.sandbox:
                print("  Sandbox:")
                print(f"    - enabled: {result.stats.sandbox.enabled}")
                print(f"    - validated: {result.stats.sandbox.validated}")
                print(f"    - error: {result.stats.sandbox.error}")
                print(f"    - total_ms: {result.stats.sandbox.total_ms:.2f}ms")

        assert result.stats is not None
        assert result.stats.total_ms > 0

    def test_tiered_loader_explain_shows_l0_l1(self, db_factory, cleanup_memories):
        """TieredLoader explain shows L0 and L1 loading stats."""
        store = MemoryStore(db_factory)
        loader = TieredMemoryLoader(MemoryService(db_factory))
        user_id = _uid()
        session_id = f"sess_{uuid4().hex}"

        # Create profile memory (L0)
        profile_mem = Memory(
            memory_id=f"prof_{uuid4().hex}",
            user_id=user_id,
            memory_type=MemoryType.PROFILE,
            content="User is a senior Python developer",
            initial_confidence=0.95,
            observed_at=datetime.now(timezone.utc),
        )
        cleanup_memories.append(profile_mem.memory_id)
        store.create(profile_mem)

        # Create episodic memory (L1)
        episodic_mem = Memory(
            memory_id=f"ep_{uuid4().hex}",
            user_id=user_id,
            memory_type=MemoryType.SEMANTIC,
            content="Discussed async patterns yesterday",
            initial_confidence=0.8,
            embedding=_embed("async patterns"),
            observed_at=datetime.now(timezone.utc),
        )
        cleanup_memories.append(episodic_mem.memory_id)
        store.create(episodic_mem)

        # Build section with explain
        section, stats = loader.build_section(
            user_id=user_id,
            session_id=session_id,
            query="async programming",
            query_embedding=_embed("async programming"),
            explain=True,
        )

        print("\n=== TIERED LOADER EXPLAIN OUTPUT ===")
        print("L0 (profile):")
        print(f"  - loaded: {stats.l0_loaded}")
        print(f"  - tokens: {stats.l0_tokens}")
        print(f"  - time: {stats.l0_ms:.2f}ms")
        print("L1 (query-relevant):")
        print(f"  - loaded: {stats.l1_loaded}")
        print(f"  - count: {stats.l1_count}")
        print(f"  - tokens: {stats.l1_tokens}")
        print(f"  - time: {stats.l1_ms:.2f}ms")
        if stats.retrieval:
            print("  Retrieval details:")
            print(f"    - phase1_candidates: {stats.retrieval.phase1_candidates}")
            print(f"    - phase2_candidates: {stats.retrieval.phase2_candidates}")
            print(f"    - final_count: {stats.retrieval.final_count}")
        print(f"Total: {stats.total_ms:.2f}ms")

        assert stats is not None
        assert stats.total_ms > 0

    def test_explain_verifies_no_silent_fallback(self, db_factory, cleanup_memories):
        """Use explain to verify vector search actually ran (not silent fallback)."""
        store = MemoryStore(db_factory)
        retriever = MemoryRetriever(db_factory)
        user_id = _uid()
        session_id = f"sess_{uuid4().hex}"

        # Create memory with embedding
        mem = Memory(
            memory_id=f"vec_{uuid4().hex}",
            user_id=user_id,
            memory_type=MemoryType.SEMANTIC,
            content="Vector search test memory",
            initial_confidence=0.9,
            embedding=[0.1] * EMBEDDING_DIM,
            observed_at=datetime.now(timezone.utc),
        )
        cleanup_memories.append(mem.memory_id)
        store.create(mem)

        # Retrieve with embedding and explain
        results, stats = retriever.retrieve(
            user_id=user_id,
            query_text="vector search",
            session_id=session_id,
            query_embedding=[0.1] * EMBEDDING_DIM,
            explain=True,
        )

        print("\n=== FALLBACK VERIFICATION ===")
        print(f"Vector search attempted: {stats.vector_attempted}")
        print(f"Vector search hit: {stats.vector_hit}")
        print(f"Vector search error: {stats.vector_error}")

        # CRITICAL: This is how we verify no silent fallback
        assert stats.vector_attempted, "Vector search should have been attempted"
        assert stats.vector_error is None, (
            f"Vector search should not have errors: {stats.vector_error}"
        )
        # If vector_hit is False but no error, it means no results (not fallback)
