"""End-to-end MCP interface tests.

Tests the MCP tool layer (create_server / call_tool) with a fake in-memory
backend — no real DB or network required.  Covers:
  - Basic CRUD: store, retrieve, correct, purge, search, profile
  - Snapshot / branch lifecycle
  - Concurrent calls (asyncio.gather)
"""

from __future__ import annotations

import asyncio
import threading
import uuid
from typing import Any

import pytest

from mo_memory_mcp.server import MemoryBackend, create_server


# ── Fake backend ──────────────────────────────────────────────────────


class FakeBackend(MemoryBackend):
    """Thread-safe in-memory backend for testing."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._memories: dict[str, dict] = {}   # memory_id → record
        self._snapshots: dict[str, dict] = {}  # name → info
        self._branches: dict[str, dict] = {}   # name → info
        self._active_branch: str = "main"

    # ── CRUD ──────────────────────────────────────────────────────────

    def store(self, user_id: str, content: str, memory_type: str, session_id: str | None) -> dict:
        mid = str(uuid.uuid4())
        with self._lock:
            self._memories[mid] = {
                "memory_id": mid,
                "user_id": user_id,
                "content": content,
                "memory_type": memory_type,
                "session_id": session_id,
            }
        return {"memory_id": mid, "content": content}

    def retrieve(self, user_id: str, query: str, top_k: int, session_id: str | None = None) -> list[dict]:
        with self._lock:
            results = [
                {"memory_id": mid, "memory_type": r["memory_type"], "content": r["content"]}
                for mid, r in self._memories.items()
                if r["user_id"] == user_id and query.lower() in r["content"].lower()
            ]
        return results[:top_k]

    def correct(self, user_id: str, memory_id: str, new_content: str, reason: str) -> dict:
        with self._lock:
            if memory_id not in self._memories:
                raise KeyError(f"Memory {memory_id} not found")
            self._memories[memory_id]["content"] = new_content
        return {"memory_id": memory_id, "content": new_content}

    def purge(self, user_id: str, memory_id: str | None, topic: str | None, reason: str) -> dict:
        deleted = 0
        with self._lock:
            if memory_id:
                if memory_id in self._memories:
                    del self._memories[memory_id]
                    deleted = 1
            elif topic:
                to_del = [
                    mid for mid, r in self._memories.items()
                    if r["user_id"] == user_id and topic.lower() in r["content"].lower()
                ]
                for mid in to_del:
                    del self._memories[mid]
                deleted = len(to_del)
        return {"purged": deleted}

    def profile(self, user_id: str) -> dict:
        with self._lock:
            count = sum(1 for r in self._memories.values() if r["user_id"] == user_id)
        return {"user_id": user_id, "memory_count": count}

    def search(self, user_id: str, query: str, top_k: int) -> list[dict]:
        return self.retrieve(user_id, query, top_k)

    # ── Maintenance (stubs) ───────────────────────────────────────────

    def governance(self, user_id: str, force: bool = False) -> dict:
        return {"status": "ok", "needs_rebuild": False}

    def consolidate(self, user_id: str, force: bool = False) -> dict:
        return {"status": "ok"}

    def reflect(self, user_id: str, force: bool = False) -> dict:
        return {"status": "ok"}

    def rebuild_index(self, table: str) -> str:
        return f"Rebuilt index for {table}"

    def health_warnings(self, user_id: str) -> list[str]:
        return []

    # ── Snapshots ─────────────────────────────────────────────────────

    def snapshot_create(self, user_id: str, name: str, description: str) -> dict:
        with self._lock:
            self._snapshots[name] = {"name": name, "description": description, "timestamp": "2026-01-01 00:00"}
        return {"name": name}

    def snapshot_list(self, user_id: str) -> list[dict]:
        with self._lock:
            return list(self._snapshots.values())

    def snapshot_rollback(self, user_id: str, name: str) -> dict:
        with self._lock:
            if name not in self._snapshots:
                raise KeyError(f"Snapshot {name} not found")
        return {"rolled_back_to": name}

    # ── Branches ──────────────────────────────────────────────────────

    def branch_create(self, user_id: str, name: str, from_snapshot: str | None, from_timestamp: str | None) -> dict:
        with self._lock:
            self._branches[name] = {"name": name, "active": False}
        return {"name": name}

    def branch_list(self, user_id: str) -> list[dict]:
        with self._lock:
            return list(self._branches.values())

    def branch_checkout(self, user_id: str, name: str) -> dict:
        with self._lock:
            self._active_branch = name
        return {"active": name}

    def branch_delete(self, user_id: str, name: str) -> dict:
        with self._lock:
            self._branches.pop(name, None)
        return {"deleted": name}

    def branch_merge(self, user_id: str, source: str, strategy: str) -> dict:
        return {"merged": 0, "skipped": 0, "strategy": strategy}

    def branch_diff(self, user_id: str, source: str, limit: int) -> dict:
        return {"changes": [], "source": source}


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture()
def backend() -> FakeBackend:
    return FakeBackend()


@pytest.fixture()
def server(backend: FakeBackend):
    return create_server(backend, default_user="test_user")


async def call(server, tool: str, **kwargs) -> str:
    """Helper: call a tool and return the text content."""
    contents, _ = await server.call_tool(tool, kwargs)
    return contents[0].text


# ── Basic CRUD tests ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_store_and_retrieve(server):
    await call(server, "memory_store", content="I prefer Python over Java")
    result = await call(server, "memory_retrieve", query="Python")
    assert "Python" in result


@pytest.mark.asyncio
async def test_correct(server, backend):
    stored = backend.store("test_user", "old content", "semantic", None)
    mid = stored["memory_id"]
    result = await call(server, "memory_correct", memory_id=mid, new_content="new content", reason="update")
    assert "new content" in result


@pytest.mark.asyncio
async def test_purge_by_id(server, backend):
    stored = backend.store("test_user", "to delete", "semantic", None)
    mid = stored["memory_id"]
    result = await call(server, "memory_purge", memory_id=mid, reason="cleanup")
    assert mid not in backend._memories


@pytest.mark.asyncio
async def test_purge_by_topic(server, backend):
    backend.store("test_user", "python tip 1", "semantic", None)
    backend.store("test_user", "python tip 2", "semantic", None)
    backend.store("test_user", "unrelated memory", "semantic", None)
    await call(server, "memory_purge", topic="python", reason="cleanup")
    remaining = [r for r in backend._memories.values() if "python" in r["content"].lower()]
    assert len(remaining) == 0


@pytest.mark.asyncio
async def test_search(server, backend):
    backend.store("test_user", "matrixone database", "semantic", None)
    result = await call(server, "memory_search", query="matrixone")
    assert "matrixone" in result.lower()


@pytest.mark.asyncio
async def test_profile(server, backend):
    backend.store("test_user", "fact 1", "semantic", None)
    backend.store("test_user", "fact 2", "semantic", None)
    result = await call(server, "memory_profile")
    assert result  # non-empty response


# ── Snapshot / branch tests ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_snapshot_lifecycle(server, backend):
    await call(server, "memory_snapshot", name="snap1", description="test snapshot")
    assert "snap1" in backend._snapshots

    result = await call(server, "memory_snapshots")
    assert "snap1" in result

    result = await call(server, "memory_rollback", name="snap1")
    assert "snap1" in result


@pytest.mark.asyncio
async def test_branch_lifecycle(server, backend):
    await call(server, "memory_branch", name="exp1")
    assert "exp1" in backend._branches

    result = await call(server, "memory_branches")
    assert "exp1" in result

    await call(server, "memory_checkout", name="exp1")
    assert backend._active_branch == "exp1"

    await call(server, "memory_merge", source="exp1", strategy="append")
    await call(server, "memory_branch_delete", name="exp1")
    assert "exp1" not in backend._branches


# ── Concurrent tests ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_concurrent_store(server, backend):
    """50 concurrent store calls must all succeed without data loss."""
    n = 50
    tasks = [
        call(server, "memory_store", content=f"concurrent memory {i}")
        for i in range(n)
    ]
    results = await asyncio.gather(*tasks)
    assert len(results) == n
    assert all("Stored memory" in r for r in results)
    assert len(backend._memories) == n


@pytest.mark.asyncio
async def test_concurrent_store_and_retrieve(server, backend):
    """Interleaved stores and retrieves must not deadlock or corrupt state."""
    store_tasks = [
        call(server, "memory_store", content=f"fact about topic_{i % 5}")
        for i in range(20)
    ]
    retrieve_tasks = [
        call(server, "memory_retrieve", query=f"topic_{i % 5}")
        for i in range(20)
    ]
    results = await asyncio.gather(*store_tasks, *retrieve_tasks)
    assert len(results) == 40


@pytest.mark.asyncio
async def test_concurrent_mixed_operations(server, backend):
    """Store, retrieve, search, profile all running concurrently."""
    # Pre-populate
    for i in range(5):
        backend.store("test_user", f"pre-stored item {i}", "semantic", None)

    tasks = (
        [call(server, "memory_store", content=f"new item {i}") for i in range(10)]
        + [call(server, "memory_retrieve", query="item") for _ in range(10)]
        + [call(server, "memory_search", query="pre-stored") for _ in range(10)]
        + [call(server, "memory_profile") for _ in range(5)]
    )
    results = await asyncio.gather(*tasks)
    assert len(results) == 35
    assert all(r is not None for r in results)


@pytest.mark.asyncio
async def test_concurrent_snapshot_creation(server, backend):
    """Multiple snapshots created concurrently must all be persisted."""
    tasks = [
        call(server, "memory_snapshot", name=f"snap_{i}", description=f"snapshot {i}")
        for i in range(10)
    ]
    await asyncio.gather(*tasks)
    assert len(backend._snapshots) == 10


@pytest.mark.asyncio
async def test_concurrent_purge_no_double_delete(server, backend):
    """Concurrent purges on the same memory_id must not raise errors."""
    stored = backend.store("test_user", "to be purged", "semantic", None)
    mid = stored["memory_id"]

    # Fire 5 concurrent purges for the same ID — only one should delete, rest are no-ops
    tasks = [call(server, "memory_purge", memory_id=mid, reason="concurrent") for _ in range(5)]
    results = await asyncio.gather(*tasks)
    assert mid not in backend._memories
