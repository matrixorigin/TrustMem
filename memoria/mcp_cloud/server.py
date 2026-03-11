"""SaaS Memory MCP Server — proxies to SaaS API via HTTP."""

from __future__ import annotations

import argparse
import httpx
from mcp.server import FastMCP


def create_server(api_url: str, api_key: str) -> FastMCP:
    server = FastMCP("memoria")
    client = httpx.Client(base_url=api_url, headers={"Authorization": f"Bearer {api_key}"}, timeout=30)

    @server.tool()
    async def memory_store(content: str, memory_type: str = "semantic", session_id: str | None = None) -> str:
        """Store a memory."""
        r = client.post("/v1/memories", json={"content": content, "memory_type": memory_type, "session_id": session_id})
        r.raise_for_status()
        d = r.json()
        return f"Stored memory {d['memory_id']}: {d['content'][:80]}"

    @server.tool()
    async def memory_retrieve(query: str, top_k: int = 5) -> str:
        """Retrieve relevant memories for a query."""
        r = client.post("/v1/memories/retrieve", json={"query": query, "top_k": top_k})
        r.raise_for_status()
        items = r.json()
        if not items:
            return "No relevant memories found."
        lines = [f"- [{m['memory_type']}] {m['content']}" for m in items]
        return "\n".join(lines)

    @server.tool()
    async def memory_search(query: str, top_k: int = 10) -> str:
        """Semantic search over all memories."""
        r = client.post("/v1/memories/search", json={"query": query, "top_k": top_k})
        r.raise_for_status()
        items = r.json()
        if not items:
            return "No memories found."
        lines = [f"- [{m['memory_id']}] [{m['memory_type']}] {m['content']}" for m in items]
        return "\n".join(lines)

    @server.tool()
    async def memory_correct(memory_id: str, new_content: str, reason: str = "") -> str:
        """Correct an existing memory."""
        r = client.put(f"/v1/memories/{memory_id}/correct", json={"new_content": new_content, "reason": reason})
        r.raise_for_status()
        d = r.json()
        return f"Corrected memory {d['memory_id']}: {d['content'][:80]}"

    @server.tool()
    async def memory_purge(memory_id: str, reason: str = "") -> str:
        """Delete a memory."""
        r = client.delete(f"/v1/memories/{memory_id}", params={"reason": reason})
        r.raise_for_status()
        return f"Purged memory {memory_id}"

    @server.tool()
    async def memory_profile() -> str:
        """Get current user's memory profile."""
        # API resolves user_id from the API key — use a fixed placeholder
        r = client.get("/v1/profiles/me")
        r.raise_for_status()
        return str(r.json())

    @server.tool()
    async def memory_snapshot(name: str, description: str = "") -> str:
        """Create a read-only snapshot of current memories."""
        r = client.post("/v1/snapshots", json={"name": name, "description": description})
        r.raise_for_status()
        d = r.json()
        return f"Snapshot '{d['name']}' created (ts={d.get('timestamp', 'unknown')})"

    @server.tool()
    async def memory_snapshots() -> str:
        """List all snapshots."""
        r = client.get("/v1/snapshots")
        r.raise_for_status()
        items = r.json()
        if not items:
            return "No snapshots."
        lines = [f"- {s['name']} ({s.get('timestamp', '')})" for s in items]
        return "\n".join(lines)

    @server.tool()
    async def memory_consolidate(force: bool = False) -> str:
        """Detect contradicting memories, fix orphaned nodes. 30min cooldown."""
        r = client.post("/v1/consolidate", params={"force": force})
        r.raise_for_status()
        return str(r.json())

    @server.tool()
    async def memory_reflect(force: bool = False) -> str:
        """Analyze memory clusters and synthesize insights. 2h cooldown."""
        r = client.post("/v1/reflect", params={"force": force})
        r.raise_for_status()
        return str(r.json())

    @server.tool()
    async def memory_snapshot_diff(name: str) -> str:
        """Compare a snapshot with current memories. Shows added/removed since snapshot."""
        r = client.get(f"/v1/snapshots/{name}/diff")
        r.raise_for_status()
        d = r.json()
        lines = [f"Snapshot '{name}': {d['snapshot_count']} memories, Current: {d['current_count']} memories"]
        lines.append(f"Added: {d['added_count']}, Removed: {d['removed_count']}, Unchanged: {d['unchanged_count']}")
        for m in d.get("added", []):
            lines.append(f"  + [{m['memory_type']}] {m['content']}")
        for m in d.get("removed", []):
            lines.append(f"  - [{m['memory_type']}] {m['content']}")
        return "\n".join(lines)

    return server


def main():
    parser = argparse.ArgumentParser(description="SaaS Memory MCP Server")
    parser.add_argument("--api-url", required=True, help="SaaS API base URL")
    parser.add_argument("--api-key", required=True, help="API key for authentication")
    parser.add_argument("--transport", default="stdio", choices=["stdio", "sse"])
    args = parser.parse_args()

    server = create_server(args.api_url, args.api_key)
    server.run(transport=args.transport)


if __name__ == "__main__":
    main()
