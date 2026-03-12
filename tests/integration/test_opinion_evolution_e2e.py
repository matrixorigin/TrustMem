"""End-to-end tests for OpinionEvolver wired into TypedObserver.

Verifies that when a new memory is stored, nearby scene memories have their
confidence updated in the DB with full field-level verification.

Uses real DB, real l2_distance, real MemoryStore.
"""

import uuid
from datetime import datetime, timezone

import pytest

from memoria.core.memory.models.memory import MemoryRecord
from memoria.core.memory.tabular.store import MemoryStore
from memoria.core.memory.tabular.typed_observer import TypedObserver
from memoria.core.memory.types import Memory, MemoryType, TrustTier
from tests.conftest import TEST_EMBEDDING_DIM


def _now():
    return datetime.now(timezone.utc)


def _uid():
    return f"u-{uuid.uuid4().hex[:8]}"


def _mid():
    return uuid.uuid4().hex


def _embed_ortho():
    """Embedding orthogonal to _embed() — cosine similarity = 0.0."""
    return [1.0 if i % 2 == 0 else -1.0 for i in range(TEST_EMBEDDING_DIM)]


def _embed(val: float = 0.5):
    return [val] * TEST_EMBEDDING_DIM


def _embed_neutral():
    """Embedding with ~0.47 cosine similarity to _embed() — in neutral zone (0.3~0.8)."""
    half = TEST_EMBEDDING_DIM // 2
    return [
        0.9 if i < half else (1.0 if i % 2 == 0 else -1.0)
        for i in range(TEST_EMBEDDING_DIM)
    ]


def _insert_scene(db, user_id, content, *, embedding, confidence=0.5, trust_tier="T4"):
    """Insert a scene memory (session_id=None, like reflection output)."""
    mid = _mid()
    row = MemoryRecord(
        memory_id=mid,
        user_id=user_id,
        session_id=None,
        memory_type="procedural",
        content=content,
        initial_confidence=confidence,
        trust_tier=trust_tier,
        embedding=embedding,
        source_event_ids=[],
        superseded_by=None,
        is_active=1,
        observed_at=_now(),
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def _get(db, memory_id):
    db.expire_all()
    return db.query(MemoryRecord).filter(MemoryRecord.memory_id == memory_id).first()


def _make_observer(db_factory):
    store = MemoryStore(db_factory)
    return TypedObserver(
        store=store,
        llm_client=None,
        embed_fn=None,
        db_factory=db_factory,
    )


class TestOpinionSupportingEvidence:
    """New memory similar to scene → confidence +0.05."""

    def test_supporting_updates_confidence_only(self, db, db_factory):
        uid = _uid()
        emb = _embed(0.9)
        # Scene: reflection-produced, T4, conf=0.5
        scene = _insert_scene(
            db, uid, "User prefers dark mode", embedding=emb, confidence=0.5
        )

        # Store a very similar new memory (same embedding = cosine sim ~1.0)
        observer = _make_observer(db_factory)
        new_mem = Memory(
            memory_id=_mid(),
            user_id=uid,
            memory_type=MemoryType.PROCEDURAL,
            content="User likes dark mode",
            embedding=emb,
            session_id="s1",
            initial_confidence=0.65,
            trust_tier=TrustTier.T3_INFERRED,
        )
        observer.store.create(new_mem)
        observer._evolve_scene_opinions(new_mem)

        # Verify scene in DB: confidence bumped, everything else unchanged
        s = _get(db, scene.memory_id)
        assert s is not None
        assert s.initial_confidence == pytest.approx(0.55, abs=0.01)  # 0.5 + 0.05
        # All other fields unchanged
        assert s.memory_id == scene.memory_id
        assert s.user_id == uid
        assert s.session_id is None  # still a scene
        assert s.memory_type == "procedural"
        assert s.content == "User prefers dark mode"
        assert s.trust_tier == "T4"  # not promoted yet (0.55 < 0.8)
        assert s.is_active == 1
        assert s.embedding is not None
        assert s.source_event_ids == []
        assert s.superseded_by is None


class TestOpinionContradictingEvidence:
    """New memory orthogonal to scene → confidence -0.10."""

    def test_contradicting_decreases_confidence(self, db, db_factory):
        uid = _uid()
        scene_emb = _embed(0.9)
        # Orthogonal embedding → low cosine similarity
        contra_emb = _embed_ortho()

        scene = _insert_scene(
            db, uid, "User prefers tabs", embedding=scene_emb, confidence=0.5
        )

        observer = _make_observer(db_factory)
        new_mem = Memory(
            memory_id=_mid(),
            user_id=uid,
            memory_type=MemoryType.PROCEDURAL,
            content="User prefers spaces",
            embedding=contra_emb,
            session_id="s1",
            initial_confidence=0.65,
            trust_tier=TrustTier.T3_INFERRED,
        )
        observer.store.create(new_mem)
        observer._evolve_scene_opinions(new_mem)

        s = _get(db, scene.memory_id)
        assert s.initial_confidence == pytest.approx(0.38, abs=0.01)  # 0.5 - 0.12
        assert s.trust_tier == "T4"
        assert s.is_active == 1
        assert s.content == "User prefers tabs"


class TestOpinionNeutralEvidence:
    """Similarity between thresholds → no change at all."""

    def test_neutral_no_field_changes(self, db, db_factory):
        uid = _uid()
        scene_emb = _embed(0.9)
        neutral_emb = _embed_neutral()

        scene = _insert_scene(
            db, uid, "User prefers vim", embedding=scene_emb, confidence=0.5
        )

        observer = _make_observer(db_factory)
        new_mem = Memory(
            memory_id=_mid(),
            user_id=uid,
            memory_type=MemoryType.PROCEDURAL,
            content="User uses vim sometimes",
            embedding=neutral_emb,
            session_id="s1",
            initial_confidence=0.65,
            trust_tier=TrustTier.T3_INFERRED,
        )
        observer.store.create(new_mem)
        observer._evolve_scene_opinions(new_mem)

        s = _get(db, scene.memory_id)
        assert s.initial_confidence == 0.5  # unchanged
        assert s.trust_tier == "T4"  # unchanged
        assert s.is_active == 1  # unchanged
        assert s.content == "User prefers vim"


class TestOpinionPromotion:
    """Repeated supporting evidence pushes T4 → T3 at confidence ≥ 0.8."""

    def test_t4_promoted_to_t3(self, db, db_factory):
        uid = _uid()
        emb = _embed(0.9)
        # Start at 0.78 — one more supporting push crosses 0.8
        scene = _insert_scene(
            db, uid, "Always run linter", embedding=emb, confidence=0.78
        )

        observer = _make_observer(db_factory)
        new_mem = Memory(
            memory_id=_mid(),
            user_id=uid,
            memory_type=MemoryType.PROCEDURAL,
            content="Run linter before commit",
            embedding=emb,
            session_id="s1",
            initial_confidence=0.65,
            trust_tier=TrustTier.T3_INFERRED,
        )
        observer.store.create(new_mem)
        observer._evolve_scene_opinions(new_mem)

        s = _get(db, scene.memory_id)
        assert s.initial_confidence == pytest.approx(0.83, abs=0.01)  # 0.78 + 0.05
        assert s.trust_tier == "T3"  # PROMOTED
        assert s.is_active == 1
        assert s.content == "Always run linter"
        assert s.session_id is None
        assert s.superseded_by is None


class TestOpinionQuarantine:
    """Contradicting evidence drops confidence below 0.2 → quarantined."""

    def test_low_confidence_quarantined(self, db, db_factory):
        uid = _uid()
        scene_emb = _embed(0.9)
        contra_emb = _embed_ortho()

        # Start at 0.25 — one contradicting push drops to 0.15 < 0.2
        scene = _insert_scene(
            db, uid, "Outdated belief", embedding=scene_emb, confidence=0.25
        )

        observer = _make_observer(db_factory)
        new_mem = Memory(
            memory_id=_mid(),
            user_id=uid,
            memory_type=MemoryType.PROCEDURAL,
            content="Contradicting info",
            embedding=contra_emb,
            session_id="s1",
            initial_confidence=0.65,
            trust_tier=TrustTier.T3_INFERRED,
        )
        observer.store.create(new_mem)
        observer._evolve_scene_opinions(new_mem)

        s = _get(db, scene.memory_id)
        assert s.initial_confidence == pytest.approx(0.13, abs=0.01)  # 0.25 - 0.12
        assert s.is_active == 0  # QUARANTINED
        assert s.trust_tier == "T4"  # not promoted
        assert s.content == "Outdated belief"
        assert s.session_id is None


class TestOpinionNoEmbedding:
    """New memory without embedding → opinion evolution skipped entirely."""

    def test_no_embedding_no_change(self, db, db_factory):
        uid = _uid()
        scene = _insert_scene(
            db, uid, "Some scene", embedding=_embed(0.9), confidence=0.5
        )

        observer = _make_observer(db_factory)
        new_mem = Memory(
            memory_id=_mid(),
            user_id=uid,
            memory_type=MemoryType.PROCEDURAL,
            content="No embedding",
            embedding=None,
            session_id="s1",
            initial_confidence=0.65,
            trust_tier=TrustTier.T3_INFERRED,
        )
        observer.store.create(new_mem)
        observer._evolve_scene_opinions(new_mem)

        s = _get(db, scene.memory_id)
        assert s.initial_confidence == 0.5
        assert s.is_active == 1


class TestOpinionUserIsolation:
    """Scene from user B must not be affected by user A's new memory."""

    def test_other_user_scene_untouched(self, db, db_factory):
        uid_a, uid_b = _uid(), _uid()
        emb = _embed(0.9)

        scene_b = _insert_scene(db, uid_b, "B's scene", embedding=emb, confidence=0.5)

        observer = _make_observer(db_factory)
        new_mem = Memory(
            memory_id=_mid(),
            user_id=uid_a,
            memory_type=MemoryType.PROCEDURAL,
            content="A's memory",
            embedding=emb,
            session_id="s1",
            initial_confidence=0.65,
            trust_tier=TrustTier.T3_INFERRED,
        )
        observer.store.create(new_mem)
        observer._evolve_scene_opinions(new_mem)

        s = _get(db, scene_b.memory_id)
        assert s.initial_confidence == 0.5  # untouched
        assert s.is_active == 1
        assert s.trust_tier == "T4"
        assert s.user_id == uid_b


class TestOpinionMultipleScenes:
    """Multiple scenes: only similar ones affected, dissimilar ones untouched."""

    def test_selective_evolution(self, db, db_factory):
        uid = _uid()
        similar_emb = _embed(0.9)
        different_emb = _embed_ortho()

        scene_similar = _insert_scene(
            db, uid, "Similar scene", embedding=similar_emb, confidence=0.5
        )
        scene_different = _insert_scene(
            db, uid, "Different scene", embedding=different_emb, confidence=0.5
        )

        observer = _make_observer(db_factory)
        new_mem = Memory(
            memory_id=_mid(),
            user_id=uid,
            memory_type=MemoryType.PROCEDURAL,
            content="New evidence",
            embedding=similar_emb,
            session_id="s1",
            initial_confidence=0.65,
            trust_tier=TrustTier.T3_INFERRED,
        )
        observer.store.create(new_mem)
        observer._evolve_scene_opinions(new_mem)

        # Similar scene: supporting → +0.05
        s1 = _get(db, scene_similar.memory_id)
        assert s1.initial_confidence == pytest.approx(0.55, abs=0.01)

        # Different scene: contradicting → -0.12
        s2 = _get(db, scene_different.memory_id)
        assert s2.initial_confidence == pytest.approx(0.38, abs=0.01)
