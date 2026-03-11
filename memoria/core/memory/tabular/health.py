"""MemoryHealth — pollution detection, stats, cleanup."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sqlalchemy import text

if TYPE_CHECKING:
    from datetime import datetime

from memoria.core.db_consumer import DbConsumer, DbFactory

logger = logging.getLogger(__name__)


class MemoryHealth(DbConsumer):
    """Memory health analytics and pollution detection."""

    def __init__(
        self,
        db_factory: DbFactory,
        db_name: str = "dev_agent",
        pollution_threshold: float = 0.3,
    ):
        super().__init__(db_factory)
        self.db_name = db_name
        self.pollution_threshold = pollution_threshold

    def analyze(self, user_id: str) -> dict:
        """Get per-type stats: count, avg_confidence, contradiction_rate, staleness."""
        with self._db() as db:
            rows = db.execute(text("""
                SELECT
                    memory_type,
                    COUNT(*) as total,
                    AVG(initial_confidence) as avg_confidence,
                    COUNT(CASE WHEN superseded_by IS NOT NULL THEN 1 END) as superseded,
                    AVG(TIMESTAMPDIFF(HOUR, observed_at, NOW())) as avg_staleness_hours
                FROM mem_memories
                WHERE user_id = :uid
                GROUP BY memory_type
            """), {"uid": user_id}).fetchall()

        stats = {}
        for r in rows:
            contradiction_rate = r.superseded / r.total if r.total > 0 else 0
            stats[r.memory_type] = {
                "total": r.total,
                "avg_confidence": float(r.avg_confidence or 0),
                "contradiction_rate": contradiction_rate,
                "avg_staleness_hours": float(r.avg_staleness_hours or 0),
            }
        return stats

    def detect_pollution(self, user_id: str, since_timestamp: datetime) -> dict:
        """Detect pollution by checking supersede/delete ratio since timestamp."""
        try:
            with self._db() as db:
                # Count changes since timestamp
                result = db.execute(text("""
                    SELECT
                        COUNT(*) as total_changes,
                        COUNT(CASE WHEN superseded_by IS NOT NULL THEN 1 END) as supersedes
                    FROM mem_memories
                    WHERE user_id = :uid AND updated_at >= :ts
                """), {"uid": user_id, "ts": since_timestamp}).fetchone()

            total = result.total_changes or 0
            supersedes = result.supersedes or 0
            ratio = supersedes / total if total > 0 else 0
            is_polluted = ratio > self.pollution_threshold

            return {
                "is_polluted": is_polluted,
                "total_changes": total,
                "supersedes": supersedes,
                "ratio": ratio,
                "threshold": self.pollution_threshold,
            }
        except Exception as e:
            logger.warning("Pollution detection failed: %s", e)
            return {"is_polluted": False, "error": str(e)}

    def suggest_rollback_target(self, user_id: str) -> str | None:
        """Find the most likely bad memory (low confidence, recent, caused supersedes)."""
        with self._db() as db:
            row = db.execute(text("""
                SELECT memory_id
                FROM mem_memories
                WHERE user_id = :uid
                  AND is_active = 1
                  AND initial_confidence < 0.5
                ORDER BY observed_at DESC
                LIMIT 1
            """), {"uid": user_id}).fetchone()
        return row.memory_id if row else None

    def cleanup_snapshots(self, keep_last_n: int = 5) -> int:
        """Drop old milestone snapshots, keep last N."""
        with self._db() as db:
            rows = db.execute(text("""
                SELECT sname FROM mo_catalog.mo_snapshots
                WHERE prefix_eq(sname, 'mem_milestone_')
                ORDER BY ts DESC
            """)).fetchall()

        if len(rows) <= keep_last_n:
            return 0

        to_drop = [r.sname for r in rows[keep_last_n:]]
        dropped = 0

        # Use autocommit for DDL
        with self._db() as db:
            raw_conn = db.connection().connection
            raw_conn.autocommit(True)
            cursor = raw_conn.cursor()
            try:
                for name in to_drop:
                    try:
                        cursor.execute(f"drop snapshot {name}")
                        dropped += 1
                    except Exception as e:
                        logger.warning("Failed to drop snapshot %s: %s", name, e)
            finally:
                cursor.close()
                raw_conn.autocommit(False)

        logger.info("Cleaned up %d old snapshots", dropped)
        return dropped

    def cleanup_orphan_branches(self) -> int:
        """Clean up sandbox branches that were not properly dropped."""
        with self._db() as db:
            rows = db.execute(text("""
                SELECT table_name FROM information_schema.tables
                WHERE table_name LIKE 'memories_sandbox_%'
            """)).fetchall()

        if not rows:
            return 0

        cleaned = 0
        with self._db() as db:
            for r in rows:
                try:
                    db.execute(text(
                        f"data branch delete table {self.db_name}.{r.table_name}"
                    ))
                    db.commit()
                    cleaned += 1
                    logger.info("Cleaned orphan branch: %s", r.table_name)
                except Exception as e:
                    logger.warning("Failed to clean branch %s: %s", r.table_name, e)

        return cleaned

    def estimate_capacity(self, user_id: str) -> dict:
        """Estimate memory capacity and IVF index performance headroom.

        Provides data-driven answers to:
        - How many memories does this user have?
        - At what scale does IVF-flat degrade? (rule of thumb: >100K vectors)
        - Is partitioning needed?

        IVF-flat performance notes (MatrixOne):
        - Optimal: <50K vectors per index partition
        - Acceptable: 50K-200K (query time grows ~linearly)
        - Degraded: >200K (consider partitioning by user_id or time bucket)

        Returns a dict with current counts, growth projection, and recommendations.
        """
        _ivf_optimal = 50_000
        _ivf_degraded = 200_000

        with self._db() as db:
            # Per-user active vector count
            user_row = db.execute(text("""
                SELECT
                    COUNT(*) as total_active,
                    COUNT(CASE WHEN embedding IS NOT NULL THEN 1 END) as with_embedding,
                    MIN(observed_at) as oldest_active,
                    MAX(observed_at) as newest_active
                FROM mem_memories
                WHERE user_id = :uid AND is_active = 1
            """), {"uid": user_id}).fetchone()

            # Global vector count (for index-level assessment)
            global_row = db.execute(text("""
                SELECT COUNT(*) as global_total
                FROM mem_memories
                WHERE is_active = 1 AND embedding IS NOT NULL
            """)).fetchone()

            # 30-day growth rate for this user
            growth_row = db.execute(text("""
                SELECT COUNT(*) as added_30d
                FROM mem_memories
                WHERE user_id = :uid
                  AND observed_at >= NOW() - INTERVAL 30 DAY
            """), {"uid": user_id}).fetchone()

        total_active = user_row.total_active or 0
        with_embedding = user_row.with_embedding or 0
        global_total = global_row.global_total or 0
        added_30d = growth_row.added_30d or 0
        monthly_rate = added_30d
        days_to_ivf_optimal = (
            int((_ivf_optimal - global_total) / (monthly_rate / 30))
            if monthly_rate > 0 and global_total < _ivf_optimal
            else None
        )

        recommendation = "ok"
        if global_total > _ivf_degraded:
            recommendation = "partition_required"
        elif global_total > _ivf_optimal:
            recommendation = "monitor_query_latency"

        return {
            "user_active_memories": total_active,
            "user_with_embedding": with_embedding,
            "global_vector_count": global_total,
            "monthly_growth_rate": monthly_rate,
            "days_to_ivf_optimal_threshold": days_to_ivf_optimal,
            "ivf_thresholds": {"optimal": _ivf_optimal, "degraded": _ivf_degraded},
            "recommendation": recommendation,
            "partition_hint": "user_id_hash" if global_total > _ivf_degraded else None,
        }

    def get_storage_stats(self, user_id: str) -> dict:
        """Get storage statistics for monitoring."""
        with self._db() as db:
            row = db.execute(text("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN is_active = 1 THEN 1 ELSE 0 END) as active,
                    AVG(LENGTH(content)) as avg_content_size,
                    MIN(observed_at) as oldest,
                    MAX(observed_at) as newest
                FROM mem_memories
                WHERE user_id = :uid
            """), {"uid": user_id}).fetchone()

        return {
            "total": row.total or 0,
            "active": row.active or 0,
            "inactive": (row.total or 0) - (row.active or 0),
            "avg_content_size": float(row.avg_content_size or 0),
            "oldest": row.oldest,
            "newest": row.newest,
        }
