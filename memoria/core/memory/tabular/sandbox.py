"""MemorySandbox — write-ahead validation using MO zero-copy branch.

Validates new memories in an isolated branch before committing to main table.
All SQL uses parameterized queries — no f-string interpolation of user data.

NOTE: This module uses raw SQL instead of ORM because branch tables are
dynamically created via ``data branch create table`` DDL — they have no
static ORM model binding.  Table names are internally generated (uuid hex),
never user input.

Supports explain=True for EXPLAIN ANALYZE style execution stats.
"""

from __future__ import annotations

import logging
import time
from memoria.core.utils.id_generator import generate_prefixed_id
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import text

from memoria.core.db_consumer import DbConsumer, DbFactory
from memoria.core.memory.tabular.explain import SandboxStats
from memoria.core.memory.tabular.metrics import MemoryMetrics
from memoria.core.memory.types import Memory, _utcnow

logger = logging.getLogger(__name__)


class MemorySandbox(DbConsumer):
    """Validate memories in a branch before committing."""

    def __init__(self, db_factory: DbFactory, db_name: str = "dev_agent", metrics: Optional[MemoryMetrics] = None):
        super().__init__(db_factory)
        self.db_name = db_name
        self._metrics = metrics or MemoryMetrics()

    def validate_memories(
        self,
        user_id: str,
        new_memories: list[Memory],
        query_text: str,
        query_embedding: Optional[list[float]] = None,
        explain: bool = False,
    ) -> tuple[bool, Optional[SandboxStats]]:
        """Validate that new memories improve retrieval quality.

        Flow:
        1. Create branch table (zero-copy)
        2. Insert new memories into branch
        3. Compare retrieval quality: branch vs main
        4. Drop branch (always, regardless of result)

        Returns:
            (improved, stats) — stats is None when explain=False.
        """
        start = time.time() if explain else 0
        stats = SandboxStats(enabled=True) if explain else None
        
        if not new_memories:
            if stats:
                stats.validated = True
                stats.total_ms = 0
            return True, stats

        branch_name = generate_prefixed_id("memories_sandbox")
        if stats:
            stats.branch_name = branch_name

        try:
            self._create_branch(branch_name)
            self._insert_to_branch(branch_name, new_memories)

            score_main = self._retrieval_score(
                "mem_memories", user_id, query_text, query_embedding
            )
            score_branch = self._retrieval_score(
                branch_name, user_id, query_text, query_embedding
            )

            improved = score_branch >= score_main
            logger.debug(
                "Sandbox validation: main=%.3f branch=%.3f improved=%s",
                score_main, score_branch, improved,
            )
            if stats:
                stats.validated = True
                stats.total_ms = (time.time() - start) * 1000
            return improved, stats

        except Exception as e:
            logger.warning("Sandbox validation failed: %s", e)
            self._metrics.increment("sandbox_validation_errors")
            if stats:
                stats.error = str(e)
                stats.total_ms = (time.time() - start) * 1000
            return True, stats  # Fail open: allow write if validation errors

        finally:
            self._drop_branch(branch_name)

    def _create_branch(self, branch_name: str) -> None:
        # branch_name is internally generated (uuid hex), not user input — safe for DDL.
        with self._db() as db:
            db.execute(text(
                f"data branch create table {branch_name} from mem_memories"
            ))
            db.commit()

    def _insert_to_branch(self, branch_name: str, memories: list[Memory]) -> None:
        # branch_name is internally generated (uuid hex), safe for DDL.
        # All user-controlled values go through parameterized :placeholders.
        with self._db() as db:
            for m in memories:
                # Embedding: serialize to string for MO vector literal, or NULL.
                # This is a numeric array we control (from embed_fn), not user text.
                if m.embedding:
                    vec_literal = "[" + ",".join(str(v) for v in m.embedding) + "]"
                else:
                    vec_literal = None

                source_ids = str(m.source_event_ids).replace("'", '"') if m.source_event_ids else "[]"
                now = m.observed_at or _utcnow()

                if vec_literal:
                    db.execute(text(f"""
                        INSERT INTO {branch_name}
                        (memory_id, user_id, memory_type, content, initial_confidence,
                         embedding, source_event_ids, is_active, observed_at, created_at)
                        VALUES (:mid, :uid, :mtype, :content, :conf,
                                :vec, :sources, 1, :obs_at, :created_at)
                    """), {
                        "mid": m.memory_id,
                        "uid": m.user_id,
                        "mtype": m.memory_type.value,
                        "content": m.content,
                        "conf": m.initial_confidence,
                        "vec": vec_literal,
                        "sources": source_ids,
                        "obs_at": now,
                        "created_at": now,
                    })
                else:
                    db.execute(text(f"""
                        INSERT INTO {branch_name}
                        (memory_id, user_id, memory_type, content, initial_confidence,
                         source_event_ids, is_active, observed_at, created_at)
                        VALUES (:mid, :uid, :mtype, :content, :conf,
                                :sources, 1, :obs_at, :created_at)
                    """), {
                        "mid": m.memory_id,
                        "uid": m.user_id,
                        "mtype": m.memory_type.value,
                        "content": m.content,
                        "conf": m.initial_confidence,
                        "sources": source_ids,
                        "obs_at": now,
                        "created_at": now,
                    })
            db.commit()

    def _retrieval_score(
        self,
        table_name: str,
        user_id: str,
        query_text: str,
        query_embedding: Optional[list[float]],
    ) -> float:
        """Compute aggregate retrieval score for top-5 results.

        table_name is internally generated (branch_name or literal "mem_memories") — safe for DDL.
        """
        with self._db() as db:
            if query_embedding:
                # Vector literal from embed_fn (numeric array), passed as parameter.
                vec_str = "[" + ",".join(str(v) for v in query_embedding) + "]"
                rows = db.execute(text(f"""
                    SELECT (1.0 / (1.0 + L2_DISTANCE(embedding, :vec))) AS sim
                    FROM {table_name}
                    WHERE user_id = :uid AND is_active = 1
                    ORDER BY sim DESC LIMIT 5
                """), {"uid": user_id, "vec": vec_str}).fetchall()
            else:
                rows = db.execute(text(f"""
                    SELECT confidence AS sim
                    FROM {table_name}
                    WHERE user_id = :uid AND is_active = 1
                    ORDER BY confidence DESC LIMIT 5
                """), {"uid": user_id}).fetchall()

            if not rows:
                return 0.0
            return sum(r.sim for r in rows) / len(rows)

    def _drop_branch(self, branch_name: str) -> None:
        try:
            with self._db() as db:
                db.execute(text(
                    f"data branch delete table {self.db_name}.{branch_name}"
                ))
                db.commit()
        except Exception as e:
            logger.warning("Failed to drop branch %s: %s", branch_name, e)
