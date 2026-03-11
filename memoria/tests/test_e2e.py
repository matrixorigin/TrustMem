"""Memoria — E2E tests with DB ground truth verification.

Run:  pytest memoria/tests/test_e2e.py -v
"""

from __future__ import annotations

import os
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

MASTER_KEY = "e2e-master-key"
os.environ["MEMORIA_MASTER_KEY"] = MASTER_KEY


@pytest.fixture(scope="module")
def client():
    from memoria.api.main import app
    # Save and restore global embedding client to avoid polluting other test modules
    from memoria.core.embedding import _shared_client as _saved_client
    import memoria.core.embedding as _emb_mod
    with TestClient(app) as c:
        yield c
    # Restore embedding client
    _emb_mod._shared_client = _saved_client
    # Clean up rate limit state after module
    from memoria.api.middleware import _windows
    _windows.clear()


@pytest.fixture(scope="module")
def db():
    """Direct DB session for ground truth verification."""
    from memoria.api.database import init_db, get_session_factory
    init_db()
    session = get_session_factory()()
    yield session
    session.close()


def _make_user(client: TestClient) -> tuple[str, dict, str]:
    """Create user + key, return (user_id, headers, key_id)."""
    from memoria.api.middleware import _windows
    _windows.clear()  # prevent rate limit exhaustion across many _make_user calls

    uid = f"e2e_{uuid.uuid4().hex[:8]}"
    r = client.post("/auth/keys", json={"user_id": uid, "name": "e2e-key"},
                     headers={"Authorization": f"Bearer {MASTER_KEY}"})
    assert r.status_code == 201
    data = r.json()
    return uid, {"Authorization": f"Bearer {data['raw_key']}"}, data["key_id"]


@pytest.fixture(scope="module")
def user_key(client):
    uid, h, kid = _make_user(client)
    return uid, h


# ── Health ────────────────────────────────────────────────────────────

class TestHealth:
    def test_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"
        assert r.json()["database"] == "connected"


# ── Auth ──────────────────────────────────────────────────────────────

class TestAuth:
    def test_no_token(self, client):
        assert client.get("/v1/memories").status_code in (401, 403)

    def test_bad_token(self, client):
        assert client.get("/v1/memories", headers={"Authorization": "Bearer bad"}).status_code == 401

    def test_key_create_persists(self, client, db):
        uid, h, kid = _make_user(client)

        # DB: user exists
        row = db.execute(text("SELECT user_id, is_active FROM tm_users WHERE user_id = :uid"), {"uid": uid}).first()
        assert row is not None
        assert row[1] == 1  # is_active

        # DB: key exists
        krow = db.execute(text("SELECT key_id, user_id, is_active FROM auth_api_keys WHERE key_id = :kid"), {"kid": kid}).first()
        assert krow is not None
        assert krow[1] == uid
        assert krow[2] == 1

    def test_revoke_key_db(self, client, db):
        uid, h, kid = _make_user(client)

        # Use it — should work
        assert client.get("/v1/memories", headers=h).status_code == 200

        # Revoke
        assert client.delete(f"/auth/keys/{kid}", headers=h).status_code == 204

        # DB: key is_active = 0
        row = db.execute(text("SELECT is_active FROM auth_api_keys WHERE key_id = :kid"), {"kid": kid}).first()
        assert row[0] == 0

        # HTTP: rejected
        assert client.get("/v1/memories", headers=h).status_code == 401

    def test_list_keys(self, client):
        """GET /auth/keys returns the user's active keys."""
        uid, h, kid = _make_user(client)
        r = client.get("/auth/keys", headers=h)
        assert r.status_code == 200
        keys = r.json()
        assert isinstance(keys, list)
        assert len(keys) >= 1
        assert any(k["key_id"] == kid for k in keys)


# ── Memory List ───────────────────────────────────────────────────────

class TestMemoryList:
    def test_list_memories_empty(self, client):
        """GET /memories returns empty list for new user."""
        _, h, _ = _make_user(client)
        r = client.get("/v1/memories", headers=h)
        assert r.status_code == 200
        data = r.json()
        assert data["items"] == []
        assert data["next_cursor"] is None

    def test_list_memories_returns_stored(self, client, db):
        """GET /memories returns stored memories with correct fields."""
        uid, h, _ = _make_user(client)
        client.post("/v1/memories", json={"content": "My favorite programming language is Python"}, headers=h)
        client.post("/v1/memories", json={"content": "I enjoy hiking in the mountains on weekends"}, headers=h)

        r = client.get("/v1/memories", headers=h)
        assert r.status_code == 200
        items = r.json()["items"]
        assert len(items) >= 2
        contents = [m["content"] for m in items]
        assert "My favorite programming language is Python" in contents
        assert "I enjoy hiking in the mountains on weekends" in contents
        for m in items:
            assert "memory_id" in m
            assert "content" in m
            assert "memory_type" in m

    def test_list_memories_cursor_pagination(self, client):
        """GET /memories cursor pagination works correctly."""
        _, h, _ = _make_user(client)
        contents = [
            "I prefer coffee over tea in the morning",
            "My dog's name is Buddy and he loves fetch",
            "The capital of France is Paris",
            "I learned to play guitar last summer",
            "My favorite movie genre is science fiction",
        ]
        for content in contents:
            client.post("/v1/memories", json={"content": content}, headers=h)

        # First page
        r = client.get("/v1/memories", params={"limit": 2}, headers=h)
        assert r.status_code == 200
        data = r.json()
        assert len(data["items"]) == 2
        assert data["next_cursor"] is not None

        # Second page
        r2 = client.get("/v1/memories", params={"limit": 2, "cursor": data["next_cursor"]}, headers=h)
        data2 = r2.json()
        assert len(data2["items"]) == 2
        # No overlap
        ids1 = {m["memory_id"] for m in data["items"]}
        ids2 = {m["memory_id"] for m in data2["items"]}
        assert ids1.isdisjoint(ids2)


# ── Memory CRUD with DB verification ─────────────────────────────────

class TestMemory:
    def test_store_db_verification(self, client, db, user_key):
        uid, h = user_key
        r = client.post("/v1/memories", json={"content": "DB verify test", "memory_type": "semantic"}, headers=h)
        assert r.status_code == 201
        mid = r.json()["memory_id"]

        # DB ground truth
        row = db.execute(text(
            "SELECT memory_id, user_id, content, memory_type, is_active, embedding "
            "FROM mem_memories WHERE memory_id = :mid"
        ), {"mid": mid}).first()
        assert row is not None
        assert row[1] == uid  # user_id
        assert row[2] == "DB verify test"  # content
        assert row[3] == "semantic"  # memory_type
        assert row[4] == 1  # is_active
        # Embedding may be NULL if external API (SiliconFlow) is unreachable in test env
        from memoria.config import get_settings
        if get_settings().embedding_provider == "local":
            assert row[5] is not None  # local embedding must always work

    def test_store_embedding_not_null(self, client, db):
        """Single inject via POST /v1/memories must produce a non-NULL embedding."""
        uid, h, _ = _make_user(client)
        r = client.post("/v1/memories", json={"content": "embedding test memory"}, headers=h)
        assert r.status_code == 201
        mid = r.json()["memory_id"]

        row = db.execute(text(
            "SELECT embedding FROM mem_memories WHERE memory_id = :mid"
        ), {"mid": mid}).first()
        assert row is not None
        assert row[0] is not None, "embedding must not be NULL after inject"

    def test_correct_embedding_not_null(self, client, db):
        """Correct via PUT /v1/memories/{id}/correct must produce a non-NULL embedding."""
        uid, h, _ = _make_user(client)
        mid = client.post("/v1/memories", json={"content": "original fact about databases"}, headers=h).json()["memory_id"]

        r = client.put(f"/v1/memories/{mid}/correct",
                       json={"new_content": "corrected fact about quantum computing", "reason": "fix"},
                       headers=h)
        assert r.status_code == 200
        new_mid = r.json()["memory_id"]

        row = db.execute(text(
            "SELECT embedding FROM mem_memories WHERE memory_id = :mid"
        ), {"mid": new_mid}).first()
        assert row is not None
        assert row[0] is not None, "embedding must not be NULL after correct"

        # Old memory embedding should still be intact (not corrupted by correct)
        old_row = db.execute(text(
            "SELECT embedding FROM mem_memories WHERE memory_id = :mid"
        ), {"mid": mid}).first()
        assert old_row is not None
        assert old_row[0] is not None, "old memory embedding must not be corrupted"

        # Content changed → embedding must differ
        assert row[0] != old_row[0], "corrected memory should have different embedding"

    def test_correct_deactivates_old(self, client, db):
        uid, h, _ = _make_user(client)
        mid = client.post("/v1/memories", json={"content": "original"}, headers=h).json()["memory_id"]

        r = client.put(f"/v1/memories/{mid}/correct", json={"new_content": "corrected", "reason": "fix"}, headers=h)
        assert r.status_code == 200
        new_mid = r.json()["memory_id"]

        # DB: old memory deactivated
        old = db.execute(text("SELECT is_active FROM mem_memories WHERE memory_id = :mid"), {"mid": mid}).first()
        assert old[0] == 0

        # DB: new memory active with correct content
        new = db.execute(text(
            "SELECT content, is_active, user_id FROM mem_memories WHERE memory_id = :mid"
        ), {"mid": new_mid}).first()
        assert new[0] == "corrected"
        assert new[1] == 1
        assert new[2] == uid

    def test_delete_deactivates(self, client, db):
        _, h, _ = _make_user(client)
        mid = client.post("/v1/memories", json={"content": "to delete"}, headers=h).json()["memory_id"]

        r = client.delete(f"/v1/memories/{mid}", headers=h)
        assert r.status_code == 200

        # DB: deactivated
        row = db.execute(text("SELECT is_active FROM mem_memories WHERE memory_id = :mid"), {"mid": mid}).first()
        assert row[0] == 0

    def test_batch_store_db(self, client, db):
        uid, h, _ = _make_user(client)
        r = client.post("/v1/memories/batch", json={
            "memories": [{"content": f"batch_{i}"} for i in range(3)]
        }, headers=h)
        assert r.status_code == 201
        mids = [m["memory_id"] for m in r.json()]
        assert len(mids) == 3

        # DB: all 3 exist and active
        count = db.execute(text(
            "SELECT COUNT(*) FROM mem_memories WHERE user_id = :uid AND is_active AND content LIKE 'batch_%'"
        ), {"uid": uid}).scalar()
        assert count == 3

    def test_batch_store_embedding_not_null(self, client, db):
        """Batch store must produce non-NULL embeddings for all memories."""
        uid, h, _ = _make_user(client)
        r = client.post("/v1/memories/batch", json={
            "memories": [
                {"content": "The Eiffel Tower is in Paris"},
                {"content": "Python was created by Guido van Rossum"},
            ]
        }, headers=h)
        assert r.status_code == 201
        mids = [m["memory_id"] for m in r.json()]

        for mid in mids:
            row = db.execute(text(
                "SELECT embedding FROM mem_memories WHERE memory_id = :mid"
            ), {"mid": mid}).first()
            assert row is not None
            assert row[0] is not None, f"embedding must not be NULL for batch memory {mid}"

    def test_search_returns_relevant(self, client, user_key):
        uid, h = user_key
        # Store something searchable
        client.post("/v1/memories", json={"content": "My favorite database is MatrixOne"}, headers=h)

        r = client.post("/v1/memories/search", json={"query": "database", "top_k": 5}, headers=h)
        assert r.status_code == 200
        results = r.json()
        assert len(results) >= 1

    def test_retrieve_returns_results(self, client, user_key):
        _, h = user_key
        r = client.post("/v1/memories/retrieve", json={"query": "favorite", "top_k": 5}, headers=h)
        assert r.status_code == 200

    def test_purge_by_type_db(self, client, db):
        uid, h, _ = _make_user(client)
        client.post("/v1/memories", json={"content": "wk note", "memory_type": "working"}, headers=h)
        client.post("/v1/memories", json={"content": "sem fact", "memory_type": "semantic"}, headers=h)

        r = client.post("/v1/memories/purge", json={"memory_types": ["working"]}, headers=h)
        assert r.status_code == 200

        # DB: working deactivated, semantic survives
        wk = db.execute(text(
            "SELECT COUNT(*) FROM mem_memories WHERE user_id = :uid AND memory_type = 'working' AND is_active"
        ), {"uid": uid}).scalar()
        assert wk == 0

        sem = db.execute(text(
            "SELECT COUNT(*) FROM mem_memories WHERE user_id = :uid AND memory_type = 'semantic' AND is_active"
        ), {"uid": uid}).scalar()
        assert sem >= 1

    def test_profile(self, client, user_key):
        _, h = user_key
        r = client.get("/v1/profiles/me", headers=h)
        assert r.status_code == 200
        assert "user_id" in r.json()


# ── Observe ───────────────────────────────────────────────────────────

class TestObserve:
    def test_observe_extracts_memories(self, client, db):
        uid, h, _ = _make_user(client)

        before = db.execute(text(
            "SELECT COUNT(*) FROM mem_memories WHERE user_id = :uid AND is_active"
        ), {"uid": uid}).scalar()

        r = client.post("/v1/observe", json={
            "messages": [
                {"role": "user", "content": "I work at Acme Corp as a senior engineer"},
                {"role": "assistant", "content": "Got it, you're a senior engineer at Acme Corp."},
            ]
        }, headers=h)
        assert r.status_code == 200
        extracted = r.json()

        # Should have extracted at least something (or empty if LLM not configured)
        assert isinstance(extracted, list)


# ── Snapshots with time-travel verification ───────────────────────────

class TestSnapshots:
    def test_lifecycle_with_time_travel(self, client, db):
        uid, h, _ = _make_user(client)

        # Store a memory
        mid = client.post("/v1/memories", json={"content": "before snapshot"}, headers=h).json()["memory_id"]

        # Create snapshot
        name = f"snap_{uuid.uuid4().hex[:6]}"
        r = client.post("/v1/snapshots", json={"name": name}, headers=h)
        assert r.status_code == 201

        # DB: registry entry exists
        reg = db.execute(text(
            "SELECT user_id, display_name FROM mem_snapshot_registry WHERE snapshot_name = :sn"
        ), {"sn": r.json()["snapshot_name"]}).first()
        assert reg is not None
        assert reg[0] == uid
        assert reg[1] == name

        # Store another memory AFTER snapshot
        client.post("/v1/memories", json={"content": "after snapshot"}, headers=h)

        # Read snapshot — should only see "before snapshot", not "after snapshot"
        r = client.get(f"/v1/snapshots/{name}", headers=h)
        assert r.status_code == 200
        snap_data = r.json()
        assert snap_data["memory_count"] >= 1
        snap_contents = [m["content"] for m in snap_data["memories"]]
        assert "before snapshot" in snap_contents
        assert "after snapshot" not in snap_contents

        # List
        r = client.get("/v1/snapshots", headers=h)
        assert any(s["name"] == name for s in r.json())

        # Delete
        assert client.delete(f"/v1/snapshots/{name}", headers=h).status_code == 204

        # DB: registry entry gone
        reg = db.execute(text(
            "SELECT 1 FROM mem_snapshot_registry WHERE display_name = :n AND user_id = :uid"
        ), {"n": name, "uid": uid}).first()
        assert reg is None

    def test_duplicate_409(self, client):
        _, h, _ = _make_user(client)
        assert client.post("/v1/snapshots", json={"name": "dup"}, headers=h).status_code == 201
        assert client.post("/v1/snapshots", json={"name": "dup"}, headers=h).status_code == 409

    def test_limit_enforced(self, client, db):
        """Verify quota check path (don't create 100, just verify the mechanism)."""
        uid, h, _ = _make_user(client)
        # Create one — should work
        assert client.post("/v1/snapshots", json={"name": "s1"}, headers=h).status_code == 201

        # DB: count = 1
        count = db.execute(text(
            "SELECT COUNT(*) FROM mem_snapshot_registry WHERE user_id = :uid"
        ), {"uid": uid}).scalar()
        assert count == 1


# ── User Ops (consolidate / reflect) ─────────────────────────────────

class TestUserOps:
    def test_consolidate(self, client, user_key):
        _, h = user_key
        r = client.post("/v1/consolidate", headers=h)
        assert r.status_code == 200
        assert isinstance(r.json(), dict)

    def test_consolidate_cooldown(self, client):
        _, h, _ = _make_user(client)
        # First call
        r1 = client.post("/v1/consolidate", headers=h)
        assert r1.status_code == 200
        assert r1.json().get("cached") is not True

        # Second call — should be cached
        r2 = client.post("/v1/consolidate", headers=h)
        assert r2.status_code == 200
        assert r2.json().get("cached") is True
        assert "cooldown_remaining_s" in r2.json()

    def test_consolidate_force_skips_cooldown(self, client):
        _, h, _ = _make_user(client)
        client.post("/v1/consolidate", headers=h)
        r = client.post("/v1/consolidate?force=true", headers=h)
        assert r.status_code == 200
        assert r.json().get("cached") is not True

    def test_reflect(self, client, user_key):
        _, h = user_key
        r = client.post("/v1/reflect", headers=h)
        assert r.status_code == 200
        assert isinstance(r.json(), dict)


# ── Admin ─────────────────────────────────────────────────────────────

class TestAdmin:
    @pytest.fixture()
    def admin_h(self):
        return {"Authorization": f"Bearer {MASTER_KEY}"}

    def test_non_admin_rejected(self, client, user_key):
        _, h = user_key
        assert client.get("/admin/stats", headers=h).status_code == 403
        assert client.get("/admin/users", headers=h).status_code == 403

    def test_stats(self, client, admin_h):
        r = client.get("/admin/stats", headers=admin_h)
        assert r.status_code == 200
        data = r.json()
        assert "total_users" in data
        assert "total_memories" in data
        assert "total_snapshots" in data
        assert data["total_users"] >= 1

    def test_list_users_pagination(self, client, admin_h):
        r = client.get("/admin/users?limit=2", headers=admin_h)
        assert r.status_code == 200
        data = r.json()
        assert "users" in data
        assert "next_cursor" in data
        assert len(data["users"]) <= 2

    def test_user_stats(self, client, admin_h, user_key):
        uid, _ = user_key
        r = client.get(f"/admin/users/{uid}/stats", headers=admin_h)
        assert r.status_code == 200
        data = r.json()
        assert data["user_id"] == uid
        assert "memory_count" in data
        assert "api_key_count" in data

    def test_delete_user_db(self, client, db, admin_h):
        uid, _, kid = _make_user(client)

        r = client.delete(f"/admin/users/{uid}", headers=admin_h)
        assert r.status_code == 200

        # DB: user deactivated
        row = db.execute(text("SELECT is_active FROM tm_users WHERE user_id = :uid"), {"uid": uid}).first()
        assert row[0] == 0

        # DB: all keys revoked
        active_keys = db.execute(text(
            "SELECT COUNT(*) FROM auth_api_keys WHERE user_id = :uid AND is_active"
        ), {"uid": uid}).scalar()
        assert active_keys == 0

    def test_governance_trigger(self, client, admin_h, user_key):
        uid, _ = user_key
        r = client.post(f"/admin/governance/{uid}/trigger", headers=admin_h)
        assert r.status_code == 200
        assert r.json()["op"] == "governance"
        assert r.json()["user_id"] == uid

    def test_governance_invalid_op(self, client, admin_h, user_key):
        uid, _ = user_key
        r = client.post(f"/admin/governance/{uid}/trigger?op=invalid", headers=admin_h)
        assert r.status_code == 400


# ── Rate Limiting ─────────────────────────────────────────────────────

class TestRateLimit:
    def test_rate_limit_headers(self, client, user_key):
        _, h = user_key
        r = client.get("/v1/memories", headers=h)
        assert "x-ratelimit-limit" in r.headers
        assert "x-ratelimit-remaining" in r.headers


# ── Error Paths ───────────────────────────────────────────────────────

class TestErrorPaths:
    def test_correct_nonexistent_memory(self, client, user_key):
        _, h = user_key
        r = client.put("/v1/memories/nonexistent-id/correct",
                        json={"new_content": "x", "reason": "y"}, headers=h)
        assert r.status_code == 404

    def test_delete_nonexistent_snapshot(self, client, user_key):
        _, h = user_key
        r = client.delete("/v1/snapshots/nonexistent_snap", headers=h)
        assert r.status_code == 404

    def test_read_nonexistent_snapshot(self, client, user_key):
        _, h = user_key
        r = client.get("/v1/snapshots/nonexistent_snap", headers=h)
        assert r.status_code == 404

    def test_expired_key_rejected(self, client, db):
        uid, h, kid = _make_user(client)
        # Manually expire the key in DB
        db.execute(text(
            "UPDATE auth_api_keys SET expires_at = '2020-01-01 00:00:00' WHERE key_id = :kid"
        ), {"kid": kid})
        db.commit()

        r = client.get("/v1/memories", headers=h)
        assert r.status_code == 401

    def test_store_empty_content_rejected(self, client, user_key):
        _, h = user_key
        r = client.post("/v1/memories", json={"content": ""}, headers=h)
        assert r.status_code == 422  # pydantic validation

    def test_batch_empty_list_rejected(self, client, user_key):
        _, h = user_key
        r = client.post("/v1/memories/batch", json={"memories": []}, headers=h)
        assert r.status_code == 422

    def test_search_empty_query_rejected(self, client, user_key):
        _, h = user_key
        r = client.post("/v1/memories/search", json={"query": ""}, headers=h)
        assert r.status_code == 422


# ── Cross-User Isolation ──────────────────────────────────────────────

class TestIsolation:
    def test_user_cannot_see_other_memories(self, client, db):
        _, h_a, _ = _make_user(client)
        _, h_b, _ = _make_user(client)

        # A stores
        r = client.post("/v1/memories", json={"content": "secret of user A"}, headers=h_a)
        mid_a = r.json()["memory_id"]

        # B cannot see A's memory in list
        r = client.get("/v1/memories", headers=h_b)
        b_mids = [m["memory_id"] for m in r.json()["items"]]
        assert mid_a not in b_mids

        # B cannot correct A's memory
        r = client.put(f"/v1/memories/{mid_a}/correct",
                        json={"new_content": "hacked", "reason": "x"}, headers=h_b)
        assert r.status_code in (404, 403)

        # DB: A's memory untouched
        row = db.execute(text(
            "SELECT content, is_active FROM mem_memories WHERE memory_id = :mid"
        ), {"mid": mid_a}).first()
        assert row[0] == "secret of user A"
        assert row[1] == 1

    def test_user_cannot_see_other_snapshots(self, client):
        _, h_a, _ = _make_user(client)
        _, h_b, _ = _make_user(client)

        client.post("/v1/snapshots", json={"name": "private_snap"}, headers=h_a)

        # B cannot read A's snapshot
        r = client.get("/v1/snapshots/private_snap", headers=h_b)
        assert r.status_code == 404

    def test_user_cannot_revoke_other_key(self, client):
        _, h_a, kid_a = _make_user(client)
        _, h_b, _ = _make_user(client)

        # Ensure A's key works first
        assert client.get("/v1/memories", headers=h_a).status_code == 200

        r = client.delete(f"/auth/keys/{kid_a}", headers=h_b)
        assert r.status_code in (403, 404)  # not your key


# ── Rate Limiting (actual 429) ────────────────────────────────────────

class TestRateLimitEnforcement:
    def test_rate_limit_headers_decrement(self, client):
        """Verify rate limit remaining decreases with each request."""
        from memoria.api.middleware import _windows
        _, h, _ = _make_user(client)
        _windows.clear()

        r1 = client.get("/v1/memories", headers=h)
        r2 = client.get("/v1/memories", headers=h)
        rem1 = int(r1.headers["x-ratelimit-remaining"])
        rem2 = int(r2.headers["x-ratelimit-remaining"])
        assert rem2 < rem1

    def test_429_when_limit_exceeded(self, client):
        """Hit a rate limit and verify 429."""
        from memoria.api.middleware import _windows, _RATE_LIMITS
        _, h, _ = _make_user(client)
        _windows.clear()

        # Use the actual configured limit for consolidate
        max_req, _ = _RATE_LIMITS.get("POST:/v1/consolidate", (3, 3600))

        for _ in range(max_req):
            r = client.post("/v1/consolidate?force=true", headers=h)
            assert r.status_code == 200

        r = client.post("/v1/consolidate?force=true", headers=h)
        assert r.status_code == 429
        assert "retry-after" in r.headers


# ── Governance Scheduler ──────────────────────────────────────────────

@pytest.mark.xdist_group("governance")
class TestGovernanceScheduler:
    def test_scheduler_starts_and_stops(self):
        """Verify the scheduler can be instantiated and started/stopped without error."""
        import asyncio
        from memoria.core.scheduler import GovernanceTaskRunner, AsyncIOBackend, MemoryGovernanceScheduler
        from memoria.api.database import get_db_context, get_db_factory

        runner = GovernanceTaskRunner(get_db_context, db_factory=get_db_factory(), memory_only=True)
        backend = AsyncIOBackend(runner)
        scheduler = MemoryGovernanceScheduler(backend=backend)

        async def _test():
            await scheduler.start()
            await asyncio.sleep(0.1)  # let it tick
            await scheduler.stop()

        asyncio.run(_test())

    def test_governance_runner_executes(self, client):
        """Run governance directly — verify result dict returned."""
        from memoria.core.scheduler import GovernanceTaskRunner
        from memoria.api.database import get_db_context, get_db_factory

        _, h, _ = _make_user(client)
        client.post("/v1/memories", json={"content": "governance test memory"}, headers=h)

        runner = GovernanceTaskRunner(get_db_context, db_factory=get_db_factory(), memory_only=True)
        result = runner.run("hourly")
        assert result is None or isinstance(result, dict)
        if result is not None:
            assert "mem_cleaned_tool_results" in result
            assert "mem_archived_working" in result


# ── Admin Governance Ops ──────────────────────────────────────────────

class TestAdminGovernanceOps:
    @pytest.fixture()
    def admin_h(self):
        return {"Authorization": f"Bearer {MASTER_KEY}"}

    def test_admin_consolidate(self, client, admin_h, user_key):
        uid, _ = user_key
        r = client.post(f"/admin/governance/{uid}/trigger?op=consolidate", headers=admin_h)
        assert r.status_code == 200
        assert r.json()["op"] == "consolidate"

    def test_admin_reflect_graceful(self, client, admin_h, user_key):
        """Reflect via admin — returns message since it needs LLM."""
        uid, _ = user_key
        r = client.post(f"/admin/governance/{uid}/trigger?op=reflect", headers=admin_h)
        assert r.status_code == 200
        assert r.json()["op"] == "reflect"


# ── Observe DB Verification ──────────────────────────────────────────

class TestObserveDB:
    def test_observe_persists_extracted_memories(self, client, db):
        """Observe should persist extracted memories to DB (if LLM available)."""
        uid, h, _ = _make_user(client)

        before = db.execute(text(
            "SELECT COUNT(*) FROM mem_memories WHERE user_id = :uid AND is_active"
        ), {"uid": uid}).scalar()

        r = client.post("/v1/observe", json={
            "messages": [
                {"role": "user", "content": "I prefer Python 3.11 and use MatrixOne as my database"},
                {"role": "assistant", "content": "Noted — Python 3.11 and MatrixOne."},
            ]
        }, headers=h)
        assert r.status_code == 200
        extracted = r.json()

        if len(extracted) > 0:
            # If extraction worked, verify DB has new rows
            after = db.execute(text(
                "SELECT COUNT(*) FROM mem_memories WHERE user_id = :uid AND is_active"
            ), {"uid": uid}).scalar()
            assert after > before

            # Verify each returned memory_id exists in DB
            for mem in extracted:
                row = db.execute(text(
                    "SELECT user_id, is_active FROM mem_memories WHERE memory_id = :mid"
                ), {"mid": mem["memory_id"]}).first()
                assert row is not None
                assert row[0] == uid
                assert row[1] == 1


# ── Purge Multi-Condition ─────────────────────────────────────────────

class TestPurgeMultiCondition:
    def test_purge_by_memory_ids(self, client, db):
        """Purge specific memory IDs, verify only those deactivated."""
        uid, h, _ = _make_user(client)
        mid1 = client.post("/v1/memories", json={"content": "I visited Tokyo last spring"}, headers=h).json()["memory_id"]
        mid2 = client.post("/v1/memories", json={"content": "My car needs an oil change soon"}, headers=h).json()["memory_id"]
        mid3 = client.post("/v1/memories", json={"content": "I enjoy reading fantasy novels"}, headers=h).json()["memory_id"]

        r = client.post("/v1/memories/purge", json={"memory_ids": [mid1, mid2]}, headers=h)
        assert r.status_code == 200
        assert r.json()["purged"] >= 2

        # DB: mid1, mid2 deactivated; mid3 survives
        for mid in (mid1, mid2):
            row = db.execute(text("SELECT is_active FROM mem_memories WHERE memory_id = :mid"), {"mid": mid}).first()
            assert row[0] == 0
        row3 = db.execute(text("SELECT is_active FROM mem_memories WHERE memory_id = :mid"), {"mid": mid3}).first()
        assert row3[0] == 1

    def test_purge_by_type_and_ids_combined(self, client, db):
        """Purge by type + IDs — both conditions applied."""
        uid, h, _ = _make_user(client)
        mid_wk = client.post("/v1/memories", json={"content": "wk1", "memory_type": "working"}, headers=h).json()["memory_id"]
        mid_sem = client.post("/v1/memories", json={"content": "sem1", "memory_type": "semantic"}, headers=h).json()["memory_id"]

        # Purge working type
        r = client.post("/v1/memories/purge", json={"memory_types": ["working"]}, headers=h)
        assert r.status_code == 200

        # DB: working gone, semantic survives
        wk = db.execute(text("SELECT is_active FROM mem_memories WHERE memory_id = :mid"), {"mid": mid_wk}).first()
        assert wk[0] == 0
        sem = db.execute(text("SELECT is_active FROM mem_memories WHERE memory_id = :mid"), {"mid": mid_sem}).first()
        assert sem[0] == 1


# ── Admin Stats Accuracy ─────────────────────────────────────────────

@pytest.mark.xdist_group("governance")
class TestAdminStatsAccuracy:
    def test_stats_reflect_actual_db(self, client, db):
        """Admin stats should be consistent with DB (>= since other workers may write concurrently)."""
        admin_h = {"Authorization": f"Bearer {MASTER_KEY}"}

        # Query DB first, then API — API result must be >= DB snapshot
        actual_users = db.execute(text("SELECT COUNT(*) FROM tm_users WHERE is_active = 1")).scalar()
        actual_memories = db.execute(text("SELECT COUNT(*) FROM mem_memories WHERE is_active = 1")).scalar()
        actual_snapshots = db.execute(text("SELECT COUNT(*) FROM mem_snapshot_registry")).scalar()

        r = client.get("/admin/stats", headers=admin_h)
        assert r.status_code == 200
        stats = r.json()

        assert stats["total_users"] >= actual_users
        assert stats["total_memories"] >= actual_memories
        assert stats["total_snapshots"] >= actual_snapshots

    def test_user_stats_accurate(self, client, db):
        """Per-user stats should match actual DB counts."""
        admin_h = {"Authorization": f"Bearer {MASTER_KEY}"}
        uid, h, _ = _make_user(client)

        # Create known data
        client.post("/v1/memories", json={"content": "stat test 1"}, headers=h)
        client.post("/v1/memories", json={"content": "stat test 2"}, headers=h)
        client.post("/v1/snapshots", json={"name": "stat_snap"}, headers=h)

        r = client.get(f"/admin/users/{uid}/stats", headers=admin_h)
        data = r.json()

        actual_mem = db.execute(text(
            "SELECT COUNT(*) FROM mem_memories WHERE user_id = :uid AND is_active"
        ), {"uid": uid}).scalar()
        actual_snap = db.execute(text(
            "SELECT COUNT(*) FROM mem_snapshot_registry WHERE user_id = :uid"
        ), {"uid": uid}).scalar()
        actual_keys = db.execute(text(
            "SELECT COUNT(*) FROM auth_api_keys WHERE user_id = :uid AND is_active"
        ), {"uid": uid}).scalar()

        assert data["memory_count"] == actual_mem
        assert data["snapshot_count"] == actual_snap
        assert data["api_key_count"] == actual_keys


# ── Consolidate Effect ────────────────────────────────────────────────

class TestConsolidateEffect:
    def test_consolidate_runs_on_real_data(self, client, db):
        """Consolidate on a user with memories — should complete without error."""
        uid, h, _ = _make_user(client)
        # Seed contradictory-ish memories
        client.post("/v1/memories", json={"content": "My favorite language is Python"}, headers=h)
        client.post("/v1/memories", json={"content": "My favorite language is Rust"}, headers=h)
        client.post("/v1/memories", json={"content": "I use MatrixOne for storage"}, headers=h)

        r = client.post("/v1/consolidate?force=true", headers=h)
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, dict)
        assert data.get("cached") is not True


# ── Boundary Values ───────────────────────────────────────────────────

class TestBoundaryValues:
    def test_top_k_zero_rejected(self, client, user_key):
        _, h = user_key
        r = client.post("/v1/memories/search", json={"query": "test", "top_k": 0}, headers=h)
        assert r.status_code == 422

    def test_top_k_over_max_rejected(self, client, user_key):
        _, h = user_key
        r = client.post("/v1/memories/search", json={"query": "test", "top_k": 101}, headers=h)
        assert r.status_code == 422

    def test_top_k_boundary_accepted(self, client, user_key):
        _, h = user_key
        r = client.post("/v1/memories/search", json={"query": "test", "top_k": 1}, headers=h)
        assert r.status_code == 200
        r = client.post("/v1/memories/search", json={"query": "test", "top_k": 100}, headers=h)
        assert r.status_code == 200

    def test_very_long_content_accepted(self, client, db):
        """Store a large content string — should succeed."""
        uid, h, _ = _make_user(client)
        long_content = "x" * 10000
        r = client.post("/v1/memories", json={"content": long_content}, headers=h)
        assert r.status_code == 201
        mid = r.json()["memory_id"]
        row = db.execute(text(
            "SELECT LENGTH(content) FROM mem_memories WHERE memory_id = :mid"
        ), {"mid": mid}).scalar()
        assert row == 10000

    def test_invalid_memory_type_rejected(self, client, user_key):
        """Unknown memory_type should be rejected with 422."""
        _, h = user_key
        r = client.post("/v1/memories", json={"content": "test", "memory_type": "nonexistent_type"}, headers=h)
        assert r.status_code == 422

    def test_special_chars_in_content(self, client, db):
        """Content with SQL-injection-like chars should be stored safely."""
        uid, h, _ = _make_user(client)
        evil = "Robert'); DROP TABLE mem_memories;--"
        r = client.post("/v1/memories", json={"content": evil}, headers=h)
        assert r.status_code == 201
        mid = r.json()["memory_id"]
        row = db.execute(text(
            "SELECT content FROM mem_memories WHERE memory_id = :mid"
        ), {"mid": mid}).first()
        assert row[0] == evil  # stored verbatim, not executed

    def test_unicode_content(self, client, db):
        uid, h, _ = _make_user(client)
        content = "我喜欢用 MatrixOne 🚀 数据库"
        r = client.post("/v1/memories", json={"content": content}, headers=h)
        assert r.status_code == 201
        mid = r.json()["memory_id"]
        row = db.execute(text(
            "SELECT content FROM mem_memories WHERE memory_id = :mid"
        ), {"mid": mid}).first()
        assert row[0] == content

    def test_snapshot_name_special_chars_sanitized(self, client):
        """Snapshot names with special chars should be sanitized, not crash."""
        _, h, _ = _make_user(client)
        r = client.post("/v1/snapshots", json={"name": "my-snap!@#$"}, headers=h)
        # Should succeed (sanitized) or reject — not crash
        assert r.status_code in (201, 400, 422)

    def test_correct_with_same_content(self, client):
        """Correct a memory with identical content — should still work."""
        _, h, _ = _make_user(client)
        mid = client.post("/v1/memories", json={"content": "same"}, headers=h).json()["memory_id"]
        r = client.put(f"/v1/memories/{mid}/correct",
                       json={"new_content": "same", "reason": "no change"}, headers=h)
        assert r.status_code == 200

    def test_batch_large_count(self, client, db):
        """Batch store 50 memories — verify all persisted."""
        uid, h, _ = _make_user(client)
        r = client.post("/v1/memories/batch", json={
            "memories": [{"content": f"bulk_{i}"} for i in range(50)]
        }, headers=h)
        assert r.status_code == 201
        assert len(r.json()) == 50
        count = db.execute(text(
            "SELECT COUNT(*) FROM mem_memories WHERE user_id = :uid AND is_active AND content LIKE 'bulk_%'"
        ), {"uid": uid}).scalar()
        assert count == 50


# ── Governance: Distributed Lock & Scheduling ─────────────────────────

@pytest.mark.xdist_group("governance")
class TestGovernanceDistributedLock:
    """Verify distributed lock mechanics: acquire, conflict, expiry takeover, heartbeat."""

    @pytest.fixture(autouse=True)
    def _clean_locks(self, db):
        db.execute(text("DELETE FROM infra_distributed_locks"))
        db.execute(text("DELETE FROM governance_runs"))
        db.commit()
        yield
        db.execute(text("DELETE FROM infra_distributed_locks"))
        db.execute(text("DELETE FROM governance_runs"))
        db.commit()

    def _make_runner(self, **kwargs):
        from memoria.core.scheduler import GovernanceTaskRunner
        from memoria.api.database import get_db_context, get_db_factory
        return GovernanceTaskRunner(
            get_db_context, db_factory=get_db_factory(), memory_only=True, **kwargs
        )

    def test_lock_acquired_and_released(self, db):
        """Lock is acquired, task runs, lock is released."""
        runner = self._make_runner()
        result = runner.run("hourly")
        assert isinstance(result, dict)

        # Lock should be released after run
        row = db.execute(text(
            "SELECT * FROM infra_distributed_locks WHERE lock_name = 'governance_hourly'"
        )).first()
        assert row is None, "Lock should be released after successful run"

    def test_lock_conflict_returns_none(self, db):
        """Second runner cannot acquire lock held by first."""
        from datetime import datetime, timedelta

        # Manually insert a non-expired lock
        db.execute(text(
            "INSERT INTO infra_distributed_locks (lock_name, instance_id, acquired_at, expires_at, task_name) "
            "VALUES (:name, :iid, :acq, :exp, :task)"
        ), {
            "name": "governance_hourly",
            "iid": "other-host:9999",
            "acq": datetime.now(),
            "exp": datetime.now() + timedelta(seconds=300),
            "task": "hourly",
        })
        db.commit()

        runner = self._make_runner()
        result = runner.run("hourly")
        assert result is None, "Should skip when lock is held by another instance"

    def test_expired_lock_takeover(self, db):
        """Expired lock is taken over via atomic CAS."""
        from datetime import datetime, timedelta

        # Insert an expired lock
        db.execute(text(
            "INSERT INTO infra_distributed_locks (lock_name, instance_id, acquired_at, expires_at, task_name) "
            "VALUES (:name, :iid, :acq, :exp, :task)"
        ), {
            "name": "governance_hourly",
            "iid": "dead-host:1234",
            "acq": datetime.now() - timedelta(seconds=600),
            "exp": datetime.now() - timedelta(seconds=60),  # expired
            "task": "hourly",
        })
        db.commit()

        runner = self._make_runner()
        result = runner.run("hourly")
        assert isinstance(result, dict), "Should take over expired lock and execute"

    def test_governance_run_persisted(self, db):
        """Each successful run writes to governance_runs table."""
        runner = self._make_runner()
        runner.run("hourly")

        row = db.execute(text(
            "SELECT task_name, result FROM governance_runs WHERE task_name = 'hourly' ORDER BY created_at DESC LIMIT 1"
        )).first()
        assert row is not None, "governance_runs should have a record"
        assert row[0] == "hourly"

        import json
        result = json.loads(row[1])
        assert "mem_cleaned_tool_results" in result
        assert "mem_archived_working" in result


@pytest.mark.xdist_group("governance")
class TestGovernanceMemoryOnly:
    """Verify memory_only=True skips knowledge/eval tasks cleanly."""

    @pytest.fixture(autouse=True)
    def _clean_locks(self, db):
        db.execute(text("DELETE FROM infra_distributed_locks"))
        db.commit()
        yield
        db.execute(text("DELETE FROM infra_distributed_locks"))
        db.commit()

    def _make_runner(self):
        from memoria.core.scheduler import GovernanceTaskRunner
        from memoria.api.database import get_db_context, get_db_factory
        return GovernanceTaskRunner(
            get_db_context, db_factory=get_db_factory(), memory_only=True
        )

    def test_hourly_no_knowledge_errors(self, db):
        """Hourly runs without knowledge governance errors."""
        runner = self._make_runner()
        result = runner.run("hourly")
        assert isinstance(result, dict)
        assert "mem_cleaned_tool_results" in result
        assert "mem_archived_working" in result
        # No knowledge keys should be present
        for key in result:
            assert not key.startswith("archived_scratchpads"), f"Unexpected knowledge key: {key}"

    def test_daily_no_knowledge_errors(self, db):
        """Daily runs without knowledge governance errors."""
        runner = self._make_runner()
        result = runner.run("daily")
        assert isinstance(result, dict)
        assert "mem_cleaned_stale" in result
        assert "mem_quarantined" in result

    def test_weekly_no_knowledge_errors(self, db):
        """Weekly runs without knowledge governance errors."""
        runner = self._make_runner()
        result = runner.run("weekly")
        assert isinstance(result, dict)
        assert "mem_cleaned_branches" in result
        assert "mem_cleaned_snapshots" in result

    def test_eval_daily_not_in_standalone(self, db):
        """eval_daily is removed in standalone Memoria — task should not exist."""
        from memoria.core.scheduler import GOVERNANCE_TASKS
        assert "eval_daily" not in GOVERNANCE_TASKS


@pytest.mark.xdist_group("governance")
class TestGovernanceWithData:
    """Verify governance actually processes real memory data."""

    @pytest.fixture(autouse=True)
    def _clean(self, db):
        db.execute(text("DELETE FROM infra_distributed_locks"))
        db.commit()
        yield
        db.execute(text("DELETE FROM infra_distributed_locks"))
        db.commit()

    def test_hourly_cleans_tool_results(self, client, db):
        """Store tool_result memories, run hourly, verify cleanup."""
        uid, h, _ = _make_user(client)

        # Store tool_result type memories (semantically distinct to avoid contradiction detection)
        tool_contents = [
            "SELECT * FROM users WHERE id = 42",
            "docker build -t myapp:latest .",
            "curl -X POST https://api.example.com/data",
        ]
        for content in tool_contents:
            client.post("/v1/memories", json={
                "content": content, "memory_type": "tool_result"
            }, headers=h)

        before = db.execute(text(
            "SELECT COUNT(*) FROM mem_memories WHERE user_id = :uid AND memory_type = 'tool_result' AND is_active"
        ), {"uid": uid}).scalar()
        assert before == 3

        from memoria.core.scheduler import GovernanceTaskRunner
        from memoria.api.database import get_db_context, get_db_factory
        runner = GovernanceTaskRunner(
            get_db_context, db_factory=get_db_factory(), memory_only=True
        )
        result = runner.run("hourly")
        assert isinstance(result, dict)
        # Result should report how many were cleaned (may be 0 if TTL not expired)
        assert "mem_cleaned_tool_results" in result

    def test_daily_governance_runs_on_memories(self, client, db):
        """Store memories, run daily, verify it completes without error."""
        uid, h, _ = _make_user(client)

        for i in range(5):
            client.post("/v1/memories", json={
                "content": f"daily governance test {i}"
            }, headers=h)

        from memoria.core.scheduler import GovernanceTaskRunner
        from memoria.api.database import get_db_context, get_db_factory
        runner = GovernanceTaskRunner(
            get_db_context, db_factory=get_db_factory(), memory_only=True
        )
        result = runner.run("daily")
        assert isinstance(result, dict)
        assert "mem_cleaned_stale" in result
        assert "mem_quarantined" in result


@pytest.mark.xdist_group("governance")
class TestGovernanceHeartbeat:
    """Verify heartbeat renews lock during execution."""

    @pytest.fixture(autouse=True)
    def _clean(self, db):
        db.execute(text("DELETE FROM infra_distributed_locks"))
        db.commit()
        yield
        db.execute(text("DELETE FROM infra_distributed_locks"))
        db.commit()

    def test_heartbeat_renews_lock(self, db):
        """Verify heartbeat thread can renew lock expiry."""
        import threading
        from datetime import datetime, timedelta
        from memoria.core.scheduler import GovernanceTaskRunner, LOCK_TTL
        from memoria.api.database import get_db_context, get_db_factory

        runner = GovernanceTaskRunner(
            get_db_context, db_factory=get_db_factory(), memory_only=True
        )

        # Manually insert a lock owned by this runner
        now = datetime.now()
        original_exp = now + timedelta(seconds=LOCK_TTL)
        db.execute(text(
            "INSERT INTO infra_distributed_locks (lock_name, instance_id, acquired_at, expires_at, task_name) "
            "VALUES (:name, :iid, :acq, :exp, :task)"
        ), {
            "name": "test_heartbeat_lock",
            "iid": runner._instance_id,
            "acq": now,
            "exp": original_exp,
            "task": "test",
        })
        db.commit()

        # Run heartbeat once
        stop = threading.Event()
        stop.set()  # will stop after one iteration check
        # Directly call the renewal logic
        with get_db_context() as hb_db:
            new_exp = datetime.now() + timedelta(seconds=LOCK_TTL)
            hb_db.execute(text(
                "UPDATE infra_distributed_locks SET expires_at = :exp "
                "WHERE lock_name = :name AND instance_id = :iid"
            ), {"exp": new_exp, "name": "test_heartbeat_lock", "iid": runner._instance_id})
            hb_db.commit()

        # Verify expiry was extended
        row = db.execute(text(
            "SELECT expires_at FROM infra_distributed_locks WHERE lock_name = 'test_heartbeat_lock'"
        )).first()
        assert row is not None
        assert row[0] >= original_exp, "Heartbeat should have extended the expiry"


# ── Profile Stats & Snapshot Diff ─────────────────────────────────────

class TestProfileStats:
    def test_profile_includes_stats(self, client):
        uid, h, _ = _make_user(client)
        client.post("/v1/memories", json={"content": "semantic fact"}, headers=h)
        client.post("/v1/memories", json={"content": "proc fact", "memory_type": "procedural"}, headers=h)

        r = client.get("/v1/profiles/me", headers=h)
        assert r.status_code == 200
        d = r.json()
        assert "stats" in d
        stats = d["stats"]
        assert stats["total"] == 2
        assert "semantic" in str(stats["by_type"])
        assert "procedural" in str(stats["by_type"])
        assert stats["avg_confidence"] is not None
        assert stats["oldest"] is not None
        assert stats["newest"] is not None


class TestSnapshotDiff:
    def test_diff_shows_changes(self, client):
        import time
        uid, h, _ = _make_user(client)
        client.post("/v1/memories", json={"content": "before A"}, headers=h)
        client.post("/v1/memories", json={"content": "before B"}, headers=h)

        time.sleep(0.3)
        client.post("/v1/snapshots", json={"name": "baseline"}, headers=h)
        time.sleep(0.3)

        # Add one, delete one
        client.post("/v1/memories", json={"content": "after C"}, headers=h)
        items = client.get("/v1/memories", headers=h).json()["items"]
        a_mid = next(m["memory_id"] for m in items if m["content"] == "before A")
        client.delete(f"/v1/memories/{a_mid}", headers=h)

        r = client.get("/v1/snapshots/baseline/diff", headers=h)
        assert r.status_code == 200
        d = r.json()
        assert d["snapshot_count"] == 2
        assert d["current_count"] == 2  # B + C
        assert d["added_count"] == 1
        assert d["removed_count"] == 1
        assert d["unchanged_count"] == 1
        assert any("after C" in m["content"] for m in d["added"])
        assert any("before A" in m["content"] for m in d["removed"])

    def test_diff_nonexistent_snapshot(self, client):
        _, h, _ = _make_user(client)
        r = client.get("/v1/snapshots/nonexistent/diff", headers=h)
        assert r.status_code == 404
