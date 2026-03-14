"""MCP response format contract tests.

Validates that tool responses match the documented format constants in
memoria.mcp_local.messages.  Any format change must update the constant
first — these tests enforce that contract.

Uses the same FakeBackend from test_mcp_e2e.
"""

from __future__ import annotations

import json

import pytest

from memoria.mcp_local.messages import (
    MSG_CONSOLIDATION_DONE,
    MSG_CORRECT_NO_CONTENT,
    MSG_CORRECT_NO_TARGET,
    MSG_CORRECTED_BY_ID,
    MSG_GOVERNANCE_DONE,
    MSG_INDEX_NEEDS_REBUILD,
    MSG_PURGE_NO_TARGET,
    MSG_PURGED,
    MSG_REFLECTION_DONE,
    MSG_REFLECTION_NO_CANDIDATES,
    MSG_RETRIEVE_EMPTY,
    MSG_RETRIEVE_FOUND,
    MSG_RETRIEVE_ITEM,
    MSG_SEARCH_EMPTY,
    MSG_SEARCH_FOUND,
)
from memoria.mcp_local.server import create_server
from tests.unit.test_mcp_e2e import FakeBackend


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture()
def backend() -> FakeBackend:
    return FakeBackend()


@pytest.fixture()
def server(backend: FakeBackend):
    return create_server(backend, default_user="test_user")


async def call(server, tool: str, **kwargs) -> str:
    contents, _ = await server.call_tool(tool, kwargs)
    return contents[0].text


# ── Text format contract tests ────────────────────────────────────────


@pytest.mark.asyncio
async def test_store_text_uses_constant(server):
    result = await call(server, "memory_store", content="test fact")
    # Must start with the MSG_STORED prefix pattern
    assert result.startswith("Stored memory ")
    assert ": test fact" in result


@pytest.mark.asyncio
async def test_store_json_schema(server):
    result = await call(server, "memory_store", content="test fact", format="json")
    data = json.loads(result)
    assert data["status"] == "ok"
    assert "memory_id" in data
    assert data["content"] == "test fact"


@pytest.mark.asyncio
async def test_retrieve_text_found(server, backend):
    backend.store("test_user", "python preference", "profile", None)
    result = await call(server, "memory_retrieve", query="python")
    assert result.startswith(MSG_RETRIEVE_FOUND.format(count=1))
    assert (
        MSG_RETRIEVE_ITEM.format(type="profile", content="python preference") in result
    )


@pytest.mark.asyncio
async def test_retrieve_text_empty(server):
    result = await call(server, "memory_retrieve", query="nonexistent")
    assert result.startswith(MSG_RETRIEVE_EMPTY)


@pytest.mark.asyncio
async def test_retrieve_json_schema(server, backend):
    backend.store("test_user", "python preference", "profile", None)
    result = await call(server, "memory_retrieve", query="python", format="json")
    data = json.loads(result)
    assert data["status"] == "ok"
    assert data["count"] == 1
    mem = data["memories"][0]
    assert all(k in mem for k in ("memory_id", "type", "content"))
    assert mem["type"] == "profile"


@pytest.mark.asyncio
async def test_retrieve_json_empty(server):
    result = await call(server, "memory_retrieve", query="nonexistent", format="json")
    data = json.loads(result)
    assert data["status"] == "ok"
    assert data["count"] == 0
    assert data["memories"] == []


@pytest.mark.asyncio
async def test_search_text_found(server, backend):
    backend.store("test_user", "matrixone database", "semantic", None)
    result = await call(server, "memory_search", query="matrixone")
    assert result.startswith(MSG_SEARCH_FOUND.format(count=1))
    # search item includes memory_id in parens
    assert "- [semantic] (" in result
    assert ") matrixone database" in result


@pytest.mark.asyncio
async def test_search_text_empty(server):
    result = await call(server, "memory_search", query="nonexistent")
    assert result == MSG_SEARCH_EMPTY


@pytest.mark.asyncio
async def test_search_json_schema(server, backend):
    backend.store("test_user", "matrixone database", "semantic", None)
    result = await call(server, "memory_search", query="matrixone", format="json")
    data = json.loads(result)
    assert data["status"] == "ok"
    assert data["count"] == 1
    assert "memory_id" in data["memories"][0]


@pytest.mark.asyncio
async def test_correct_text_by_id(server, backend):
    stored = backend.store("test_user", "old content", "semantic", None)
    mid = stored["memory_id"]
    result = await call(
        server,
        "memory_correct",
        memory_id=mid,
        new_content="new content",
        reason="update",
    )
    expected_prefix = MSG_CORRECTED_BY_ID.format(memory_id=mid, content="new content")
    assert result.startswith(expected_prefix)


@pytest.mark.asyncio
async def test_correct_json_by_id(server, backend):
    stored = backend.store("test_user", "old content", "semantic", None)
    mid = stored["memory_id"]
    result = await call(
        server,
        "memory_correct",
        memory_id=mid,
        new_content="new content",
        reason="update",
        format="json",
    )
    data = json.loads(result)
    assert data["status"] == "ok"
    assert data["memory_id"] == mid
    assert data["content"] == "new content"


@pytest.mark.asyncio
async def test_correct_text_by_query(server, backend):
    backend.store("test_user", "uses black formatter", "profile", None)
    result = await call(
        server,
        "memory_correct",
        query="black",
        new_content="uses ruff",
        reason="switched",
    )
    assert "Found '" in result
    assert "→ corrected to" in result


@pytest.mark.asyncio
async def test_correct_json_by_query(server, backend):
    backend.store("test_user", "uses black formatter", "profile", None)
    result = await call(
        server,
        "memory_correct",
        query="black",
        new_content="uses ruff",
        reason="switched",
        format="json",
    )
    data = json.loads(result)
    assert data["status"] == "ok"
    assert data["content"] == "uses ruff"


@pytest.mark.asyncio
async def test_correct_no_match(server):
    result = await call(
        server, "memory_correct", query="nonexistent", new_content="x", reason="test"
    )
    assert "No memory found" in result


@pytest.mark.asyncio
async def test_correct_missing_content(server):
    result = await call(server, "memory_correct", memory_id="x")
    assert result == MSG_CORRECT_NO_CONTENT


@pytest.mark.asyncio
async def test_correct_missing_content_json(server):
    result = await call(server, "memory_correct", memory_id="x", format="json")
    data = json.loads(result)
    assert data["status"] == "error"
    assert data["error"] == MSG_CORRECT_NO_CONTENT


@pytest.mark.asyncio
async def test_correct_missing_target(server):
    result = await call(server, "memory_correct", new_content="x")
    assert result == MSG_CORRECT_NO_TARGET


@pytest.mark.asyncio
async def test_purge_text(server, backend):
    stored = backend.store("test_user", "to delete", "semantic", None)
    result = await call(
        server, "memory_purge", memory_id=stored["memory_id"], reason="cleanup"
    )
    assert result == MSG_PURGED.format(count=1)


@pytest.mark.asyncio
async def test_purge_json(server, backend):
    stored = backend.store("test_user", "to delete", "semantic", None)
    result = await call(
        server,
        "memory_purge",
        memory_id=stored["memory_id"],
        reason="cleanup",
        format="json",
    )
    data = json.loads(result)
    assert data["status"] == "ok"
    assert data["purged"] == 1


@pytest.mark.asyncio
async def test_purge_missing_args(server):
    result = await call(server, "memory_purge")
    assert result == MSG_PURGE_NO_TARGET


@pytest.mark.asyncio
async def test_purge_missing_args_json(server):
    result = await call(server, "memory_purge", format="json")
    data = json.loads(result)
    assert data["status"] == "error"
    assert data["error"] == MSG_PURGE_NO_TARGET


@pytest.mark.asyncio
async def test_governance_done(server):
    result = await call(server, "memory_governance")
    assert result.startswith(MSG_GOVERNANCE_DONE)
    assert "quarantined=1" in result


@pytest.mark.asyncio
async def test_governance_index_warning(server, backend):
    original = backend.governance
    backend.governance = lambda uid, force=False: {
        "quarantined": 0,
        "cleaned_stale": 0,
        "scenes_created": 0,
        "vector_index_health": {
            "mem_memories": {
                "needs_rebuild": True,
                "rebuilt": False,
                "total_rows": 500,
                "centroids": 4,
                "ratio": 125.0,
            }
        },
    }
    result = await call(server, "memory_governance")
    assert MSG_INDEX_NEEDS_REBUILD.format(table="mem_memories") in result
    backend.governance = original


@pytest.mark.asyncio
async def test_consolidation_done(server):
    result = await call(server, "memory_consolidate")
    assert result.startswith(MSG_CONSOLIDATION_DONE)
    assert "merged_nodes=1" in result


@pytest.mark.asyncio
async def test_reflection_done(server):
    result = await call(server, "memory_reflect", mode="internal")
    expected = MSG_REFLECTION_DONE.format(scenes_created=1, candidates_found=3)
    assert result == expected


@pytest.mark.asyncio
async def test_reflection_no_candidates(server):
    result = await call(server, "memory_reflect", mode="candidates")
    assert result == MSG_REFLECTION_NO_CANDIDATES


@pytest.mark.asyncio
async def test_retrieve_health_warnings(server, backend):
    backend.health_warnings = lambda uid: ["5 memories have low confidence"]
    backend.store("test_user", "some fact", "semantic", None)
    result = await call(server, "memory_retrieve", query="some")
    assert "⚠️ Memory health:" in result
    assert "low confidence" in result


@pytest.mark.asyncio
async def test_extract_entities_json_schema(server):
    result = await call(server, "memory_extract_entities", mode="internal")
    data = json.loads(result)
    assert data["status"] == "done"
    for key in ("total_memories", "entities_found", "edges_created"):
        assert isinstance(data[key], int)


@pytest.mark.asyncio
async def test_link_entities_json_schema(server, backend):
    stored = backend.store("test_user", "test content", "semantic", None)
    entities_json = json.dumps(
        [
            {
                "memory_id": stored["memory_id"],
                "entities": [{"name": "test", "type": "concept"}],
            }
        ]
    )
    result = await call(server, "memory_link_entities", entities=entities_json)
    data = json.loads(result)
    assert data["status"] == "done"
    for key in ("entities_created", "entities_reused", "edges_created"):
        assert isinstance(data[key], int)


# ── Explain Parameter Tests ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_retrieve_explain_bool_false(server, backend):
    """Test that explain=False works (no explain output)."""
    backend.store("test_user", "test content", "semantic", None)
    result = await call(server, "memory_retrieve", query="test", explain=False)
    # Should not contain explain output
    assert "Explain:" not in result
    assert "execution path" not in result.lower()


@pytest.mark.asyncio
async def test_retrieve_explain_bool_true(server, backend):
    """Test that explain=True works (basic explain output)."""
    backend.store("test_user", "test content", "semantic", None)
    result = await call(server, "memory_retrieve", query="test", explain=True)
    # Should contain explain output (format: [explain] total=X.Xms path=...)
    assert "[explain]" in result
    assert "total=" in result
    assert "ms" in result


@pytest.mark.asyncio
async def test_search_explain_bool_false(server, backend):
    """Test that explain=False works for search."""
    backend.store("test_user", "test content", "semantic", None)
    result = await call(server, "memory_search", query="test", explain=False)
    assert "[explain]" not in result


@pytest.mark.asyncio
async def test_search_explain_bool_true(server, backend):
    """Test that explain=True works for search."""
    backend.store("test_user", "test content", "semantic", None)
    result = await call(server, "memory_search", query="test", explain=True)
    assert "[explain]" in result
    assert "total=" in result


@pytest.mark.asyncio
async def test_retrieve_explain_string_values(server, backend):
    """Test that string values still work (backward compatibility)."""
    backend.store("test_user", "test content", "semantic", None)

    # Test "none"
    result = await call(server, "memory_retrieve", query="test", explain="none")
    assert "[explain]" not in result

    # Test "basic"
    result = await call(server, "memory_retrieve", query="test", explain="basic")
    assert "[explain]" in result
