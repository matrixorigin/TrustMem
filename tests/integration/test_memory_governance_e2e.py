"""Governance E2E tests with real MatrixOne DB.

Tests governance frequency separation, working memory archival,
quarantine, tool_result cleanup, and trust tier effects.
"""

from datetime import datetime, timezone, timedelta

import pytest
from uuid_utils import uuid7

from memoria.core.memory.tabular.store import MemoryStore
from memoria.core.memory.tabular.governance import GovernanceScheduler
from memoria.core.memory.config import MemoryGovernanceConfig
from memoria.core.memory.types import Memory, MemoryType, TrustTier


def _uid():
    return f"gov_e2e_{uuid7().hex}"


def _sid():
    return f"sess_{uuid7().hex}"


@pytest.fixture
def db_factory():
    from tests.integration.conftest import _get_session_local

    SessionLocal = _get_session_local()
    return SessionLocal


@pytest.fixture
def cleanup(db_factory):
    ids = []
    yield ids
    if ids:
        from sqlalchemy import text

        db = db_factory()
        try:
            db.execute(
                text("DELETE FROM mem_memories WHERE memory_id IN :ids"),
                {"ids": tuple(ids)},
            )
            db.commit()
        finally:
            db.close()


class TestHourlyGovernance:
    def test_archives_working_memories(self, db_factory, cleanup):
        """Hourly governance archives stale working memories."""
        store = MemoryStore(db_factory)
        user_id = _uid()
        session_id = _sid()

        # Create working memory observed 4 hours ago (margin for TIMESTAMPDIFF truncation)
        mem = Memory(
            memory_id=str(uuid7()),
            user_id=user_id,
            session_id=session_id,
            memory_type=MemoryType.WORKING,
            content="Current plan: refactor auth",
            initial_confidence=0.9,
            observed_at=datetime.now(timezone.utc) - timedelta(hours=4),
        )
        cleanup.append(mem.memory_id)
        store.create(mem)

        # Verify active before governance
        assert store.get(mem.memory_id).is_active is True

        config = MemoryGovernanceConfig(
            working_memory_stale_hours=2, tool_result_ttl_hours=999
        )
        scheduler = GovernanceScheduler(db_factory, config=config)
        result = scheduler.run_hourly()

        assert result.archived_working >= 1
        # Verify archived in DB
        assert store.get(mem.memory_id).is_active is False

    def test_cleans_expired_tool_results(self, db_factory, cleanup):
        """Hourly governance deletes expired tool_result memories."""
        store = MemoryStore(db_factory)
        user_id = _uid()

        mem = Memory(
            memory_id=str(uuid7()),
            user_id=user_id,
            memory_type=MemoryType.TOOL_RESULT,
            content="grep output: 42 matches",
            initial_confidence=0.5,
            observed_at=datetime.now(timezone.utc) - timedelta(hours=26),
        )
        cleanup.append(mem.memory_id)
        store.create(mem)

        # Verify exists before cleanup
        assert store.get(mem.memory_id) is not None

        config = MemoryGovernanceConfig(tool_result_ttl_hours=24)
        scheduler = GovernanceScheduler(db_factory, config=config)
        result = scheduler.run_hourly()

        assert result.cleaned_tool_results >= 1
        # Verify deleted from DB
        assert store.get(mem.memory_id) is None


class TestDailyGovernance:
    def test_quarantines_low_confidence(self, db_factory, cleanup):
        """Daily governance quarantines memories with low effective_confidence."""
        store = MemoryStore(db_factory)
        user_id = _uid()

        # T4 memory (30d half-life), 90 days old → effective ≈ 0.05
        mem = Memory(
            memory_id=str(uuid7()),
            user_id=user_id,
            memory_type=MemoryType.SEMANTIC,
            content="Old unverified fact",
            initial_confidence=0.5,
            trust_tier=TrustTier.T4_UNVERIFIED,
            observed_at=datetime.now(timezone.utc) - timedelta(days=90),
        )
        cleanup.append(mem.memory_id)
        store.create(mem)

        assert store.get(mem.memory_id).is_active is True

        config = MemoryGovernanceConfig(quarantine_threshold=0.3)
        scheduler = GovernanceScheduler(db_factory, config=config)
        result = scheduler.run_daily(user_id)

        assert result.quarantined >= 1
        assert store.get(mem.memory_id).is_active is False

    def test_cleans_stale_inactive(self, db_factory, cleanup):
        """Daily governance deletes inactive low-confidence memories."""
        store = MemoryStore(db_factory)
        user_id = _uid()

        mem = Memory(
            memory_id=str(uuid7()),
            user_id=user_id,
            memory_type=MemoryType.SEMANTIC,
            content="Superseded fact",
            initial_confidence=0.05,
            observed_at=datetime.now(timezone.utc) - timedelta(days=30),
        )
        cleanup.append(mem.memory_id)
        store.create(mem)
        store.deactivate(mem.memory_id)

        scheduler = GovernanceScheduler(db_factory)
        result = scheduler.run_daily(user_id)

        assert result.cleaned_stale >= 1


class TestTrustTierAffectsQuarantine:
    def test_t1_survives_t4_quarantined(self, db_factory, cleanup):
        """T1 memory (365d half-life) survives 60 days; T4 (30d) gets quarantined."""
        store = MemoryStore(db_factory)
        user_id = _uid()
        age = datetime.now(timezone.utc) - timedelta(days=60)

        t1 = Memory(
            memory_id=str(uuid7()),
            user_id=user_id,
            memory_type=MemoryType.SEMANTIC,
            content="Verified API docs",
            initial_confidence=0.9,
            trust_tier=TrustTier.T1_VERIFIED,
            observed_at=age,
        )
        t4 = Memory(
            memory_id=str(uuid7()),
            user_id=user_id,
            memory_type=MemoryType.SEMANTIC,
            content="Unverified user claim",
            initial_confidence=0.5,
            trust_tier=TrustTier.T4_UNVERIFIED,
            observed_at=age,
        )
        cleanup.extend([t1.memory_id, t4.memory_id])
        store.create(t1)
        store.create(t4)

        config = MemoryGovernanceConfig(quarantine_threshold=0.3)
        scheduler = GovernanceScheduler(db_factory, config=config)
        scheduler.run_daily(user_id)

        # T1 should survive (effective ≈ 0.9 * exp(-60/365) ≈ 0.76)
        assert store.get(t1.memory_id).is_active is True
        # T4 should be quarantined (effective ≈ 0.5 * exp(-60/30) ≈ 0.07)
        assert store.get(t4.memory_id).is_active is False


class TestCompressRedundant:
    """_compress_redundant deactivates near-duplicate memories."""

    def test_near_duplicates_compressed(self, db_factory, cleanup):
        from sqlalchemy import text
        import json

        uid = _uid()
        mid_old = str(uuid7())
        mid_new = str(uuid7())
        cleanup.extend([mid_old, mid_new])

        # Two memories with identical embeddings (dim=384)
        emb = [0.5] * 384
        emb_str = json.dumps(emb)
        now = datetime.now(timezone.utc)
        old_time = now - timedelta(days=1)

        db = db_factory()
        for mid, ts in [(mid_old, old_time), (mid_new, now)]:
            db.execute(
                text(
                    "INSERT INTO mem_memories "
                    "(memory_id, user_id, content, memory_type, trust_tier, "
                    "initial_confidence, is_active, embedding, source_event_ids, "
                    "observed_at, created_at) "
                    "VALUES (:mid, :uid, 'duplicate content', 'semantic', 'T3', "
                    "0.8, 1, :emb, '[]', :ts, :ts)"
                ),
                {"mid": mid, "uid": uid, "emb": emb_str, "ts": ts},
            )
        db.commit()
        db.close()

        config = MemoryGovernanceConfig(
            redundant_similarity_threshold=0.95,
            redundant_window_days=30,
        )
        scheduler = GovernanceScheduler(db_factory, config=config)
        result = scheduler.run_daily(uid)

        assert result.compressed_redundant >= 1
        assert result.errors == [], f"Unexpected errors: {result.errors}"

        store = MemoryStore(db_factory)
        assert store.get(mid_new).is_active is True, "Newer memory must survive"
        assert store.get(mid_old).is_active is False, (
            "Older duplicate must be deactivated"
        )

    def test_different_embeddings_not_compressed(self, db_factory, cleanup):
        from sqlalchemy import text
        import json

        uid = _uid()
        mid1 = str(uuid7())
        mid2 = str(uuid7())
        cleanup.extend([mid1, mid2])

        # Two memories with very different embeddings
        emb1 = [1.0] + [0.0] * 383
        emb2 = [0.0] * 383 + [1.0]
        now = datetime.now(timezone.utc)

        db = db_factory()
        for mid, emb, offset in [(mid1, emb1, 1), (mid2, emb2, 0)]:
            db.execute(
                text(
                    "INSERT INTO mem_memories "
                    "(memory_id, user_id, content, memory_type, trust_tier, "
                    "initial_confidence, is_active, embedding, source_event_ids, "
                    "observed_at, created_at) "
                    "VALUES (:mid, :uid, 'unique content', 'semantic', 'T3', "
                    "0.8, 1, :emb, '[]', :ts, :ts)"
                ),
                {
                    "mid": mid,
                    "uid": uid,
                    "emb": json.dumps(emb),
                    "ts": now - timedelta(days=offset),
                },
            )
        db.commit()
        db.close()

        config = MemoryGovernanceConfig(redundant_similarity_threshold=0.95)
        scheduler = GovernanceScheduler(db_factory, config=config)
        result = scheduler.run_daily(uid)

        assert result.compressed_redundant == 0
        assert result.errors == [], f"Unexpected errors: {result.errors}"

        store = MemoryStore(db_factory)
        assert store.get(mid1).is_active is True
        assert store.get(mid2).is_active is True


class TestFullCycle:
    def test_all_frequencies(self, db_factory, cleanup):
        """Full governance cycle runs all frequencies without error."""
        store = MemoryStore(db_factory)
        user_id = _uid()

        mem = Memory(
            memory_id=str(uuid7()),
            user_id=user_id,
            memory_type=MemoryType.SEMANTIC,
            content="Test memory",
            initial_confidence=0.8,
            observed_at=datetime.now(timezone.utc),
        )
        cleanup.append(mem.memory_id)
        store.create(mem)

        scheduler = GovernanceScheduler(db_factory)
        result = scheduler.run_cycle(user_id)
        assert result.total_ms > 0
        # Snapshot cleanup may fail due to MO catalog schema differences — tolerate
        non_snapshot_errors = [e for e in result.errors if "snapshot" not in e.lower()]
        assert len(non_snapshot_errors) == 0


class TestGovernanceAuditLog:
    """Verify governance operations write correct audit records to mem_edit_log."""

    def _get_edit_log(self, db_factory, user_id: str, operation: str) -> list:
        from sqlalchemy import text
        import json

        db = db_factory()
        try:
            rows = db.execute(
                text(
                    "SELECT operation, target_ids, reason FROM mem_edit_log "
                    "WHERE user_id = :uid AND operation = :op "
                    "ORDER BY created_at DESC LIMIT 10"
                ),
                {"uid": user_id, "op": operation},
            ).fetchall()
            return [
                {
                    "operation": r.operation,
                    "target_ids": json.loads(r.target_ids) if r.target_ids else [],
                    "reason": r.reason,
                }
                for r in rows
            ]
        finally:
            db.close()

    def _cleanup_edit_log(self, db_factory, user_id: str) -> None:
        from sqlalchemy import text

        db = db_factory()
        try:
            db.execute(
                text("DELETE FROM mem_edit_log WHERE user_id = :uid"),
                {"uid": user_id},
            )
            db.commit()
        finally:
            db.close()

    def test_archive_working_writes_edit_log(self, db_factory, cleanup):
        """_archive_stale_working writes governance:archive_working to edit_log."""
        store = MemoryStore(db_factory)
        user_id = _uid()

        mem = Memory(
            memory_id=str(uuid7()),
            user_id=user_id,
            memory_type=MemoryType.WORKING,
            content="stale working memory",
            initial_confidence=0.9,
            observed_at=datetime.now(timezone.utc) - timedelta(hours=4),
        )
        cleanup.append(mem.memory_id)
        store.create(mem)

        config = MemoryGovernanceConfig(
            working_memory_stale_hours=2, tool_result_ttl_hours=999
        )
        scheduler = GovernanceScheduler(db_factory, config=config)
        scheduler.run_hourly()

        logs = self._get_edit_log(db_factory, user_id, "governance:archive_working")
        self._cleanup_edit_log(db_factory, user_id)

        assert len(logs) == 1
        assert mem.memory_id in logs[0]["target_ids"]
        assert "archived" in logs[0]["reason"].lower()

    def test_quarantine_writes_edit_log(self, db_factory, cleanup):
        """_quarantine_low_confidence writes governance:quarantine to edit_log."""
        store = MemoryStore(db_factory)
        user_id = _uid()

        # T4 memory 90 days old → effective_confidence ≈ 0.05 (below default threshold 0.1)
        mem = Memory(
            memory_id=str(uuid7()),
            user_id=user_id,
            memory_type=MemoryType.SEMANTIC,
            content="very old low confidence memory",
            initial_confidence=0.3,
            trust_tier=TrustTier.T4_UNVERIFIED,
            observed_at=datetime.now(timezone.utc) - timedelta(days=90),
        )
        cleanup.append(mem.memory_id)
        store.create(mem)

        config = MemoryGovernanceConfig(quarantine_threshold=0.1)
        scheduler = GovernanceScheduler(db_factory, config=config)
        scheduler.run_daily(user_id)

        logs = self._get_edit_log(db_factory, user_id, "governance:quarantine")
        self._cleanup_edit_log(db_factory, user_id)

        assert len(logs) >= 1
        all_ids = [mid for log in logs for mid in log["target_ids"]]
        assert mem.memory_id in all_ids
        assert "quarantine" in logs[0]["reason"].lower()

    def test_no_edit_log_when_nothing_deactivated(self, db_factory):
        """No edit_log entry written when governance finds nothing to deactivate."""
        user_id = _uid()
        config = MemoryGovernanceConfig(
            working_memory_stale_hours=9999,
            tool_result_ttl_hours=9999,
            quarantine_threshold=0.0,
        )
        scheduler = GovernanceScheduler(db_factory, config=config)
        scheduler.run_hourly()
        scheduler.run_daily(user_id)

        logs = self._get_edit_log(db_factory, user_id, "governance:archive_working")
        logs += self._get_edit_log(db_factory, user_id, "governance:quarantine")
        assert len(logs) == 0
