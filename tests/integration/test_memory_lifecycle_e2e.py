"""End-to-end memory lifecycle test — real DB, simulated time progression.

Validates the complete memory lifecycle with FIELD-LEVEL verification:
  1. Memory creation via observer (LLM-extracted)
  2. Contradiction detection & supersede
  3. Confidence decay & quarantine (via backdated observed_at)
  4. Reflection: cross-session clustering → LLM synthesis → scene persist
  5. Steady-state: tool_result TTL cleanup, working memory archival
  6. User isolation

Time simulation: instead of waiting real days, we backdate observed_at.
DB-side TIMESTAMPDIFF(DAY, observed_at, NOW()) naturally computes the gap.
"""

import json
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from sqlalchemy import text

from memoria.core.memory.models.memory import MemoryRecord
from memoria.core.memory.tabular.candidates import TabularCandidateProvider
from memoria.core.memory.tabular.governance import GovernanceScheduler
from memoria.core.memory.types import MemoryType, TrustTier


def _now():
    return datetime.now(timezone.utc)


def _ago(**kwargs):
    return _now() - timedelta(**kwargs)


def _uid():
    return f"u-{uuid.uuid4().hex[:8]}"


def _mid():
    return uuid.uuid4().hex


def _embed(val: float = 0.5):
    from tests.conftest import TEST_EMBEDDING_DIM

    return [val] * TEST_EMBEDDING_DIM


def _insert(
    db,
    user_id,
    content,
    *,
    memory_type="semantic",
    session_id="s1",
    embedding=None,
    observed_at=None,
    trust_tier="T3",
    initial_confidence=0.65,
    superseded_by=None,
    is_active=1,
):
    """Insert a memory record directly into DB and return it."""
    mid = _mid()
    obs = observed_at or _now()
    row = MemoryRecord(
        memory_id=mid,
        user_id=user_id,
        session_id=session_id,
        memory_type=memory_type,
        content=content,
        initial_confidence=initial_confidence,
        trust_tier=trust_tier,
        embedding=embedding,
        source_event_ids=[],
        superseded_by=superseded_by,
        is_active=is_active,
        observed_at=obs,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def _ts_eq(a, b):
    """Compare datetimes ignoring tzinfo (MatrixOne returns naive UTC)."""
    if a is None or b is None:
        return a is b
    a_naive = a.replace(tzinfo=None) if a.tzinfo else a
    b_naive = b.replace(tzinfo=None) if b.tzinfo else b
    # Allow 1 second tolerance for DB round-trip
    return abs((a_naive - b_naive).total_seconds()) < 1.0


def _get(db, memory_id):
    """Re-query a memory from DB (ground truth). Expires cache first."""
    db.expire_all()
    return db.query(MemoryRecord).filter(MemoryRecord.memory_id == memory_id).first()


def _active(db, user_id, memory_type=None):
    db.expire_all()
    q = db.query(MemoryRecord).filter(
        MemoryRecord.user_id == user_id, MemoryRecord.is_active == 1
    )
    if memory_type:
        q = q.filter(MemoryRecord.memory_type == memory_type)
    return q.all()


def _all(db, user_id, memory_type=None):
    db.expire_all()
    q = db.query(MemoryRecord).filter(MemoryRecord.user_id == user_id)
    if memory_type:
        q = q.filter(MemoryRecord.memory_type == memory_type)
    return q.all()


# ── Phase 1: Tool Result TTL ─────────────────────────────────────────


class TestToolResultTTL:
    def test_expired_cleaned_fresh_survives(self, db, db_factory):
        uid = _uid()
        old = _insert(
            db,
            uid,
            "old result",
            memory_type="tool_result",
            observed_at=_ago(hours=25),
            session_id="s1",
        )
        fresh = _insert(
            db,
            uid,
            "fresh result",
            memory_type="tool_result",
            observed_at=_ago(hours=1),
            session_id="s2",
        )
        old_id, fresh_id = old.memory_id, fresh.memory_id
        fresh_obs = fresh.observed_at
        fresh_conf = fresh.initial_confidence

        gov = GovernanceScheduler(db_factory)
        result = gov.run_hourly()

        assert result.cleaned_tool_results >= 1

        # Old one: permanently deleted (not just deactivated)
        assert _get(db, old_id) is None

        # Fresh one: all fields intact
        s = _get(db, fresh_id)
        assert s is not None
        assert s.memory_id == fresh_id
        assert s.user_id == uid
        assert s.session_id == "s2"
        assert s.memory_type == "tool_result"
        assert s.content == "fresh result"
        assert s.initial_confidence == fresh_conf
        assert s.trust_tier == "T3"
        assert s.embedding is None
        assert s.source_event_ids == []
        assert s.superseded_by is None
        assert s.is_active == 1
        assert _ts_eq(s.observed_at, fresh_obs)
        assert s.created_at is not None
        assert s.updated_at is not None


# ── Phase 2: Working Memory Archival ──────────────────────────────────


class TestWorkingMemoryArchival:
    def test_stale_archived_recent_intact(self, db, db_factory):
        uid = _uid()
        stale = _insert(
            db,
            uid,
            "old scratch",
            memory_type="working",
            observed_at=_ago(hours=3),
            session_id="s1",
        )
        recent = _insert(
            db,
            uid,
            "new scratch",
            memory_type="working",
            observed_at=_ago(minutes=30),
            session_id="s2",
        )

        gov = GovernanceScheduler(db_factory)
        result = gov.run_hourly()
        assert result.archived_working == 1

        # Stale: is_active flipped to 0, updated_at changed, rest unchanged
        s = _get(db, stale.memory_id)
        assert s.is_active == 0
        assert s.updated_at.replace(tzinfo=None) >= stale.updated_at.replace(
            tzinfo=None
        )  # updated_at bumped
        assert s.memory_id == stale.memory_id
        assert s.user_id == uid
        assert s.session_id == "s1"
        assert s.memory_type == "working"
        assert s.content == "old scratch"
        assert s.initial_confidence == stale.initial_confidence
        assert s.trust_tier == stale.trust_tier
        assert s.superseded_by is None
        assert _ts_eq(s.observed_at, stale.observed_at)  # observed_at NOT changed

        # Recent: completely untouched
        r = _get(db, recent.memory_id)
        assert r.is_active == 1
        assert r.content == "new scratch"
        assert r.session_id == "s2"
        assert _ts_eq(r.observed_at, recent.observed_at)
        assert _ts_eq(r.updated_at, recent.updated_at)  # not touched


# ── Phase 3: Confidence Decay & Quarantine ────────────────────────────


class TestConfidenceDecayQuarantine:
    """effective_confidence = initial_confidence × exp(-days / half_life)
    T4 hl=30d, T3 hl=60d, T1 hl=365d. quarantine_threshold=0.3
    """

    def test_t4_quarantined_fields(self, db, db_factory):
        """T4 (conf=0.4, hl=30d) @ 15d: 0.4×exp(-15/30)=0.243 < 0.3"""
        uid = _uid()
        obs = _ago(days=15)
        m = _insert(
            db,
            uid,
            "weak belief",
            trust_tier="T4",
            initial_confidence=0.4,
            observed_at=obs,
            session_id="s1",
        )

        gov = GovernanceScheduler(db_factory)
        result = gov.run_daily(uid)
        assert result.quarantined == 1

        s = _get(db, m.memory_id)
        assert s.is_active == 0  # quarantined
        assert s.updated_at.replace(tzinfo=None) >= m.updated_at.replace(
            tzinfo=None
        )  # updated_at bumped
        # All other fields unchanged
        assert s.memory_id == m.memory_id
        assert s.user_id == uid
        assert s.session_id == "s1"
        assert s.memory_type == "semantic"
        assert s.content == "weak belief"
        assert s.initial_confidence == 0.4  # NOT decayed in DB
        assert s.trust_tier == "T4"
        assert s.embedding is None
        assert s.source_event_ids == []
        assert s.superseded_by is None
        assert _ts_eq(s.observed_at, obs)  # NOT changed

    def test_t4_survives_fields(self, db, db_factory):
        """T4 (conf=0.4, hl=30d) @ 5d: 0.4×exp(-5/30)=0.339 > 0.3"""
        uid = _uid()
        m = _insert(
            db,
            uid,
            "recent belief",
            trust_tier="T4",
            initial_confidence=0.4,
            observed_at=_ago(days=5),
        )

        gov = GovernanceScheduler(db_factory)
        result = gov.run_daily(uid)
        assert result.quarantined == 0

        s = _get(db, m.memory_id)
        assert s.is_active == 1
        assert s.initial_confidence == 0.4
        assert s.trust_tier == "T4"
        assert s.content == "recent belief"

    def test_t3_survives_at_15d(self, db, db_factory):
        """T3 (conf=0.65, hl=60d) @ 15d: 0.65×exp(-15/60)=0.506 > 0.3"""
        uid = _uid()
        m = _insert(
            db,
            uid,
            "inferred fact",
            trust_tier="T3",
            initial_confidence=0.65,
            observed_at=_ago(days=15),
        )

        gov = GovernanceScheduler(db_factory)
        assert gov.run_daily(uid).quarantined == 0
        assert _get(db, m.memory_id).is_active == 1

    def test_t3_quarantined_at_50d(self, db, db_factory):
        """T3 (conf=0.65, hl=60d) @ 50d: 0.65×exp(-50/60)=0.283 < 0.3"""
        uid = _uid()
        m = _insert(
            db,
            uid,
            "old fact",
            trust_tier="T3",
            initial_confidence=0.65,
            observed_at=_ago(days=50),
        )

        gov = GovernanceScheduler(db_factory)
        assert gov.run_daily(uid).quarantined == 1
        assert _get(db, m.memory_id).is_active == 0

    def test_t1_survives_at_200d(self, db, db_factory):
        """T1 (conf=0.95, hl=365d) @ 200d: 0.95×exp(-200/365)=0.549 > 0.3"""
        uid = _uid()
        m = _insert(
            db,
            uid,
            "verified",
            trust_tier="T1",
            initial_confidence=0.95,
            observed_at=_ago(days=200),
        )

        gov = GovernanceScheduler(db_factory)
        assert gov.run_daily(uid).quarantined == 0
        assert _get(db, m.memory_id).is_active == 1

    def test_mixed_tiers_selective(self, db, db_factory):
        """Only the weak T4 quarantined; T3 and fresh T4 survive."""
        uid = _uid()
        t4_old = _insert(
            db,
            uid,
            "t4 old",
            trust_tier="T4",
            initial_confidence=0.4,
            observed_at=_ago(days=15),
        )
        t3 = _insert(
            db,
            uid,
            "t3",
            trust_tier="T3",
            initial_confidence=0.65,
            observed_at=_ago(days=15),
        )
        t4_fresh = _insert(
            db,
            uid,
            "t4 fresh",
            trust_tier="T4",
            initial_confidence=0.4,
            observed_at=_ago(days=2),
        )

        gov = GovernanceScheduler(db_factory)
        assert gov.run_daily(uid).quarantined == 1

        assert _get(db, t4_old.memory_id).is_active == 0
        assert _get(db, t3.memory_id).is_active == 1
        assert _get(db, t4_fresh.memory_id).is_active == 1


# ── Phase 4: Contradiction Supersede ──────────────────────────────────


class TestContradictionSupersede:
    def test_supersede_chain_candidate_fields(self, db, db_factory):
        uid = _uid()
        new = _insert(db, uid, "Python 3.12 is best", session_id="s2")
        old = _insert(
            db,
            uid,
            "Python 3.11 is best",
            session_id="s1",
            superseded_by=new.memory_id,
            is_active=0,
        )

        provider = TabularCandidateProvider(db_factory)
        candidates = provider.get_reflection_candidates(uid, since_hours=1)

        contras = [c for c in candidates if c.signal == "contradiction"]
        assert len(contras) == 1
        c = contras[0]
        assert len(c.memories) == 2
        assert c.importance_score > 0.3  # contradiction signal scores high

        by_content = {m.content: m for m in c.memories}
        old_m = by_content["Python 3.11 is best"]
        new_m = by_content["Python 3.12 is best"]

        # Old memory fields
        assert old_m.memory_id == old.memory_id
        assert old_m.user_id == uid
        assert old_m.session_id == "s1"
        assert old_m.memory_type == MemoryType.SEMANTIC
        assert old_m.is_active is False
        assert old_m.superseded_by == new.memory_id
        assert old_m.initial_confidence == 0.65
        assert old_m.trust_tier == TrustTier.T3_INFERRED
        assert old_m.embedding is None  # light query, no embedding

        # New memory fields
        assert new_m.memory_id == new.memory_id
        assert new_m.session_id == "s2"
        assert new_m.is_active is True
        assert new_m.superseded_by is None

        # Session IDs on candidate
        assert set(c.session_ids) == {"s1", "s2"}


# ── Phase 5: Reflection Full Cycle ───────────────────────────────────


class TestReflectionFullCycle:
    def test_scene_persisted_all_fields(self, db, db_factory):
        """Cross-session pattern → LLM synthesis → scene memory with all fields verified."""
        uid = _uid()
        source_ids = []
        for i in range(5):
            m = _insert(
                db,
                uid,
                f"Always run tests before commit v{i}",
                session_id=f"s{i}",
                embedding=_embed(0.5 + i * 0.1),
                memory_type="procedural",
            )
            source_ids.append(m.memory_id)

        mock_llm = MagicMock()
        mock_llm.chat.return_value = json.dumps(
            [
                {
                    "type": "procedural",
                    "content": "User consistently runs tests before committing",
                    "confidence": 0.6,
                    "evidence_summary": "Observed in 5 sessions",
                }
            ]
        )

        gov = GovernanceScheduler(db_factory, llm_client=mock_llm)
        result = gov.run_daily(uid)

        assert result.scenes_created >= 1

        # Find the scene in DB
        scene = (
            db.query(MemoryRecord)
            .filter(
                MemoryRecord.user_id == uid,
                MemoryRecord.content.like("%consistently runs tests%"),
            )
            .first()
        )
        assert scene is not None

        # Verify EVERY field
        assert len(scene.memory_id) > 0
        assert scene.user_id == uid
        assert scene.session_id is None  # scene has no session
        assert scene.memory_type == "procedural"
        assert "consistently runs tests" in scene.content
        assert scene.initial_confidence == 0.6
        assert scene.trust_tier == "T4"  # new scenes start at T4
        assert scene.embedding is None  # no embed_fn provided
        assert scene.superseded_by is None
        assert scene.is_active == 1
        assert scene.observed_at is not None
        assert scene.created_at is not None
        assert scene.updated_at is not None
        # source_event_ids should contain the source memory IDs
        assert isinstance(scene.source_event_ids, list)
        assert len(scene.source_event_ids) == 5
        assert set(scene.source_event_ids) == set(source_ids)

        # Original 5 memories still active and untouched
        for sid in source_ids:
            orig = _get(db, sid)
            assert orig.is_active == 1
            assert orig.memory_type == "procedural"

    def test_no_llm_skips_reflection(self, db, db_factory):
        uid = _uid()
        emb = _embed(0.9)
        for i in range(5):
            _insert(db, uid, f"pattern v{i}", session_id=f"s{i}", embedding=emb)

        gov = GovernanceScheduler(db_factory, llm_client=None)
        result = gov.run_daily(uid)

        assert result.scenes_created == 0
        # No extra memories created
        assert len(_all(db, uid)) == 5


# ── Phase 6: Full Multi-Day Lifecycle ─────────────────────────────────


class TestFullLifecycle:
    def test_multiday_simulation(self, db, db_factory):
        """Day 0 → hourly → Day 15: complete lifecycle with field verification."""
        uid = _uid()
        emb = _embed(0.7)

        # ── Day 0: Initial burst ──
        sem1 = _insert(
            db,
            uid,
            "User prefers dark mode",
            session_id="s1",
            embedding=emb,
            trust_tier="T4",
            initial_confidence=0.4,
        )
        sem2 = _insert(
            db,
            uid,
            "User prefers dark mode",
            session_id="s2",
            embedding=emb,
            trust_tier="T4",
            initial_confidence=0.4,
        )
        sem3 = _insert(
            db,
            uid,
            "Use pytest for testing",
            session_id="s3",
            embedding=_embed(0.1),
            trust_tier="T3",
            initial_confidence=0.65,
        )
        tr_old = _insert(
            db,
            uid,
            "grep output",
            memory_type="tool_result",
            observed_at=_ago(hours=25),
        )
        tr_new = _insert(
            db, uid, "ls output", memory_type="tool_result", observed_at=_ago(hours=2)
        )
        wk = _insert(
            db, uid, "scratch notes", memory_type="working", observed_at=_ago(hours=3)
        )
        tr_old_id = tr_old.memory_id
        tr_new_id = tr_new.memory_id
        wk_id = wk.memory_id
        wk_obs = wk.observed_at

        assert len(_active(db, uid)) == 6

        # ── Hourly: clean expired tool_result + archive stale working ──
        gov = GovernanceScheduler(db_factory)
        h = gov.run_hourly()

        assert h.cleaned_tool_results >= 1
        assert h.archived_working == 1

        # Expired tool_result: deleted
        assert _get(db, tr_old_id) is None
        # Fresh tool_result: intact
        assert _get(db, tr_new_id).is_active == 1
        assert _get(db, tr_new_id).content == "ls output"
        # Working: archived (is_active=0), content preserved
        wk_db = _get(db, wk_id)
        assert wk_db.is_active == 0
        assert wk_db.content == "scratch notes"
        assert _ts_eq(wk_db.observed_at, wk_obs)

        assert len(_active(db, uid)) == 4  # 3 semantic + 1 fresh tool_result

        # ── Simulate Day 15: backdate T4 memories ──
        db.execute(
            text("""
            UPDATE mem_memories
            SET observed_at = DATE_SUB(NOW(), INTERVAL 15 DAY)
            WHERE user_id = :uid AND trust_tier = 'T4'
        """),
            {"uid": uid},
        )
        db.commit()

        d = gov.run_daily(uid)

        # T4 (conf=0.4, hl=30d, 15d) → eff=0.243 < 0.3 → quarantined
        assert d.quarantined == 2
        s1 = _get(db, sem1.memory_id)
        s2 = _get(db, sem2.memory_id)
        assert s1.is_active == 0
        assert s2.is_active == 0
        # Content and confidence NOT mutated
        assert s1.content == "User prefers dark mode"
        assert s1.initial_confidence == 0.4
        assert s1.trust_tier == "T4"
        assert s2.content == "User prefers dark mode"

        # T3 pytest memory survives (eff=0.506)
        s3 = _get(db, sem3.memory_id)
        assert s3.is_active == 1
        assert s3.content == "Use pytest for testing"
        assert s3.trust_tier == "T3"
        assert s3.initial_confidence == 0.65

        # Final active count: T3 semantic + fresh tool_result
        assert len(_active(db, uid)) == 2


# ── Phase 7: User Isolation ───────────────────────────────────────────


class TestUserIsolation:
    def test_quarantine_does_not_cross_users(self, db, db_factory):
        uid_a, uid_b = _uid(), _uid()

        a = _insert(
            db,
            uid_a,
            "A old",
            trust_tier="T4",
            initial_confidence=0.4,
            observed_at=_ago(days=15),
        )
        b = _insert(
            db,
            uid_b,
            "B fresh",
            trust_tier="T4",
            initial_confidence=0.4,
            observed_at=_ago(days=2),
        )

        gov = GovernanceScheduler(db_factory)
        gov.run_daily(uid_a)

        # A quarantined
        a_db = _get(db, a.memory_id)
        assert a_db.is_active == 0
        # B completely untouched — verify ALL fields
        b_db = _get(db, b.memory_id)
        assert b_db.is_active == 1
        assert b_db.user_id == uid_b
        assert b_db.content == "B fresh"
        assert b_db.initial_confidence == 0.4
        assert b_db.trust_tier == "T4"
        assert _ts_eq(b_db.updated_at, b.updated_at)  # NOT touched

    def test_tool_result_cleanup_cross_user(self, db, db_factory):
        uid_a, uid_b = _uid(), _uid()
        a = _insert(
            db,
            uid_a,
            "A expired",
            memory_type="tool_result",
            observed_at=_ago(hours=25),
        )
        b = _insert(
            db, uid_b, "B fresh", memory_type="tool_result", observed_at=_ago(hours=1)
        )
        a_id, b_id = a.memory_id, b.memory_id

        GovernanceScheduler(db_factory).run_hourly()

        assert _get(db, a_id) is None
        b_db = _get(db, b_id)
        assert b_db is not None
        assert b_db.is_active == 1
        assert b_db.content == "B fresh"


# ── Phase 8: Stale Inactive Cleanup ──────────────────────────────────


class TestStaleInactiveCleanup:
    def test_inactive_low_conf_deleted_others_kept(self, db, db_factory):
        uid = _uid()
        garbage = _insert(db, uid, "garbage", is_active=0, initial_confidence=0.05)
        archived = _insert(db, uid, "archived", is_active=0, initial_confidence=0.5)
        alive = _insert(db, uid, "alive", is_active=1, initial_confidence=0.65)
        garbage_id, archived_id, alive_id = (
            garbage.memory_id,
            archived.memory_id,
            alive.memory_id,
        )

        gov = GovernanceScheduler(db_factory)
        result = gov.run_daily(uid)
        assert result.cleaned_stale == 1

        # Garbage: permanently deleted
        assert _get(db, garbage_id) is None
        # Archived: still exists, all fields intact
        a = _get(db, archived_id)
        assert a is not None
        assert a.is_active == 0
        assert a.initial_confidence == 0.5
        assert a.content == "archived"
        # Alive: untouched
        v = _get(db, alive_id)
        assert v.is_active == 1
        assert v.initial_confidence == 0.65
        assert v.content == "alive"


# ── Phase 9: run_daily_all ────────────────────────────────────────────


class TestRunDailyAll:
    def test_multi_user_batch(self, db, db_factory):
        uids = [_uid() for _ in range(3)]
        ids = {}
        for uid in uids:
            g = _insert(db, uid, "garbage", is_active=0, initial_confidence=0.05)
            w = _insert(
                db,
                uid,
                "weak T4",
                trust_tier="T4",
                initial_confidence=0.4,
                observed_at=_ago(days=15),
            )
            ids[uid] = {"garbage_id": g.memory_id, "weak_id": w.memory_id}

        gov = GovernanceScheduler(db_factory)
        result = gov.run_daily_all()

        assert result.cleaned_stale == 3
        assert result.quarantined == 3

        for uid in uids:
            assert _get(db, ids[uid]["garbage_id"]) is None
            w = _get(db, ids[uid]["weak_id"])
            assert w.is_active == 0
            assert w.content == "weak T4"
            assert w.initial_confidence == 0.4


class TestPurgeGraphNodeSync:
    """Regression tests: purge/correct must sync memory_graph_nodes.is_active.

    Bug: editor.purge() and store.supersede() only deactivated mem_memories,
    leaving memory_graph_nodes.is_active=1. Graph retriever would still return
    deleted/superseded memories.
    """

    def _insert_graph_node(self, db, user_id: str, memory_id: str, content: str):
        """Insert a graph node linked to a memory."""
        from memoria.core.memory.models.graph import GraphNode

        node_id = _mid()
        node = GraphNode(
            node_id=node_id,
            user_id=user_id,
            node_type="semantic",
            content=content,
            memory_id=memory_id,
            is_active=1,
        )
        db.add(node)
        db.commit()
        return node_id

    def _get_graph_node(self, db, node_id: str):
        from memoria.core.memory.models.graph import GraphNode

        db.expire_all()
        return db.query(GraphNode).filter(GraphNode.node_id == node_id).first()

    def test_purge_by_id_deactivates_graph_node(self, db, db_factory):
        """Purging a memory by ID must also deactivate its graph node."""
        from memoria.core.memory.factory import create_editor

        uid = _uid()
        mem = _insert(db, uid, "memory to purge")
        node_id = self._insert_graph_node(db, uid, mem.memory_id, mem.content)

        editor = create_editor(db_factory, user_id=uid, embed_client=None)
        result = editor.purge(uid, memory_ids=[mem.memory_id])

        assert result.deactivated == 1
        assert _get(db, mem.memory_id).is_active == 0
        # Graph node must also be deactivated
        gnode = self._get_graph_node(db, node_id)
        assert gnode.is_active == 0, (
            "Graph node still active after purge — stale retrieval bug"
        )

    def test_supersede_deactivates_graph_node(self, db, db_factory):
        """Correcting (superseding) a memory must also deactivate its graph node."""
        from memoria.core.memory.tabular.store import MemoryStore
        from memoria.core.memory.types import Memory, MemoryType, TrustTier
        from datetime import datetime, timezone

        uid = _uid()
        mem = _insert(db, uid, "old content")
        node_id = self._insert_graph_node(db, uid, mem.memory_id, mem.content)

        store = MemoryStore(db_factory)
        new_mem = Memory(
            memory_id=_mid(),
            user_id=uid,
            content="corrected content",
            memory_type=MemoryType.SEMANTIC,
            trust_tier=TrustTier.T2_CURATED,
            initial_confidence=0.9,
            observed_at=datetime.now(timezone.utc),
        )
        store.supersede(mem.memory_id, new_mem)

        assert _get(db, mem.memory_id).is_active == 0
        # Graph node for old memory must also be deactivated
        gnode = self._get_graph_node(db, node_id)
        assert gnode.is_active == 0, (
            "Graph node still active after supersede — stale retrieval bug"
        )
