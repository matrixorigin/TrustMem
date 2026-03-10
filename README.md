# TrustMem

Persistent memory for AI coding tools — your AI assistant remembers across conversations.

Works with **[Kiro](https://kiro.dev)**, **[Cursor](https://cursor.sh)**, and **[Claude Code](https://docs.anthropic.com/en/docs/claude-code)**. Stores memories in [MatrixOne](https://github.com/matrixorigin/matrixone) with vector search for semantic retrieval.

## How It Works

```
You: "Always use pytest, never unittest"
AI:  (stores this as a memory)

... days later, new conversation ...

You: "Write tests for this module"
AI:  (retrieves your preference → uses pytest automatically)
```

## Features

- **Cross-conversation memory** — preferences, facts, and decisions persist across sessions
- **Semantic search** — retrieves memories by meaning, not just keywords
- **Snapshots & rollback** — save memory state before risky changes, restore if things go wrong
- **Branches** — experiment with different memory states without affecting main (like git branches for memories)
- **Multi-tool support** — one `trustmem init` configures Kiro, Cursor, and Claude Code simultaneously
- **Self-maintaining** — built-in governance detects contradictions, quarantines low-confidence memories
- **Private by default** — local embedding model, no data leaves your machine

## Quick Start

### 1. Start MatrixOne (database)

```bash
git clone https://github.com/matrixorigin/TrustMem.git
cd TrustMem
docker compose up -d
# Wait ~30-60s for first-time initialization
```

Or use `docker run` directly:
```bash
docker run -d --name matrixone -p 6001:6001 -v ./data/matrixone:/mo-data --memory=2g matrixorigin/matrixone:latest
```

See [docker-compose.yml](docker-compose.yml) for configuration options. Don't want Docker? Use [MatrixOne Cloud](https://cloud.matrixorigin.cn) (free tier).

### 2. Install TrustMem

```bash
python3 -m venv .venv && source .venv/bin/activate

# Current package is on TestPyPI:
pip install --index-url https://pypi.org/simple/ --extra-index-url https://test.pypi.org/simple/ 'trust-mem-lite[local-embedding]'
```

> `[local-embedding]` installs a local embedding model (~80MB download on first use). If you skip it, memories won't be vectorized and retrieval quality will be significantly worse. See [Embedding providers](#embedding-providers) for alternatives.

### 3. Configure your AI tool

```bash
cd your-project
trustmem init
# Restart your AI tool — done!
```

### 4. Verify

```bash
trustmem status
```

Expected output:
```
TrustMem Status
  Database:    connected (mysql+pymysql://root:***@localhost:6001/trustmem)
  Tables:      8 tables OK
  Embedding:   local (all-MiniLM-L6-v2, dim=384)
  Kiro:        .kiro/settings/mcp.json ✓  |  .kiro/steering/memory.md ✓ (v0.2.2)
  Cursor:      not detected
  Claude Code: not detected
```

## Prerequisites

- Python 3.11+
- [MatrixOne](https://github.com/matrixorigin/matrixone) database (local Docker or [cloud](https://cloud.matrixorigin.cn))

## Setup by Tool

### Kiro

```bash
cd your-project
mkdir -p .kiro
trustmem init
# Restart Kiro
```

Files created:
- `.kiro/settings/mcp.json` — MCP server config
- `.kiro/steering/memory.md` — steering rules

### Cursor

```bash
cd your-project
mkdir -p .cursor
trustmem init
# Restart Cursor
```

Files created:
- `.cursor/mcp.json` — MCP server config
- `.cursor/rules/memory.mdc` — rules for Cursor

### Claude Code

```bash
cd your-project
trustmem init
# Restart Claude Code
```

Files created:
- `.claude/mcp.json` — MCP server config
- `CLAUDE.md` — rules appended (or created)

### All tools at once

If your project has `.kiro/`, `.cursor/`, and `CLAUDE.md`, `trustmem init` configures all of them in one go.

## Configuration Options

### Custom database URL

```bash
trustmem init --db-url 'mysql+pymysql://user:pass@host:6001/mydb'
```

### Embedding providers

TrustMem needs an embedding model to vectorize memories for semantic search.

| Provider | Quality | Cost | First-use latency | Ongoing latency |
|----------|---------|------|-------------------|-----------------|
| **Local** (default) | Good | Free, private | ~1 min model download (~80MB) + a few seconds to load model into memory on first query | Fast (in-process) |
| **OpenAI** | Better | API key required | None | Network round-trip |
| **Custom service** | Varies | Self-hosted | None | Network round-trip |

**Recommendation**: If you already have an embedding service running (OpenAI, Ollama, or custom endpoint), use it — avoids the local model download and cold-start latency. Otherwise, local embedding works well; the download and model load only happen once.

```bash
# Local (default) — free, private, ~80MB model download on first use
# First query takes a few seconds to load the model into memory; subsequent queries are fast.
trustmem init --embedding-provider local

# OpenAI — better quality, requires API key, no cold-start delay
trustmem init --embedding-provider openai --embedding-api-key sk-...

# Custom endpoint (Ollama, etc.) — use an existing embedding service
trustmem init --embedding-provider openai --embedding-base-url http://localhost:11434/v1
```

## Manual Tuning & Optimization

**Integration quality depends on your AI agent's capabilities and steering rules.** Out-of-the-box behavior may not be optimal — when/how memory tools are called is determined by:

1. **Agent reasoning ability** — how well it understands when to retrieve, store, or correct memories
2. **Steering rules** — the instructions in `.kiro/steering/memory.md`, `.cursor/rules/memory.mdc`, or `CLAUDE.md`

**If memory usage feels suboptimal:**
- Edit the steering rules to be more explicit about when to call `memory_retrieve`, `memory_store`, etc.
- Add examples of good/bad memory usage patterns
- Adjust retrieval triggers (e.g., "always retrieve at conversation start" vs. "retrieve only when user asks about past context")
- Tune memory type selection (semantic vs. profile vs. procedural)

**Example tuning**: If your agent forgets to retrieve memories at conversation start, add to steering rules:
```markdown
CRITICAL: At the start of EVERY conversation, call memory_retrieve with the user's first message.
```

## Adapting to Other Agents

TrustMem uses the **Model Context Protocol (MCP)** standard. Any agent system that supports MCP or external tool plugins can integrate TrustMem by:

1. **Pointing to the MCP server**: `python -m mo_memory_mcp`
2. **Providing steering rules**: Adapt the examples in `.kiro/steering/memory.md` to your agent's instruction format
3. **Configuring database**: Set `TRUSTMEM_DB_URL` environment variable or use `trustmem init`

**Generic integration template:**
```json
{
  "mcpServers": {
    "trustmem": {
      "command": "python",
      "args": ["-m", "mo_memory_mcp"],
      "env": {
        "TRUSTMEM_DB_URL": "mysql+pymysql://root:111@localhost:6001/trustmem"
      }
    }
  }
}
```

Then add steering instructions equivalent to:
- "Retrieve memories at conversation start"
- "Store new facts, preferences, and decisions"
- "Correct memories when user says they're wrong"
- "Forget memories when user requests"

See [SETUP_GUIDE.md](SETUP_GUIDE.md) for detailed integration examples.

## Commands

| Command | Description |
|---------|-------------|
| `trustmem init` | Configure everything (tables + MCP + rules) |
| `trustmem status` | Show configuration and rule versions |
| `trustmem update-rules` | Update steering rules to latest version |
| `trustmem migrate` | Create/update database tables only |
| `trustmem health` | Check memory service health |
| `trustmem governance` | Run memory cleanup and maintenance |

## Memory Types

| Type | What it stores | Example |
|------|---------------|---------|
| `semantic` | Project facts, technical decisions | "This project uses Go 1.22 with modules" |
| `profile` | User/agent preferences | "Always use pytest, never unittest" |
| `procedural` | How-to knowledge, workflows | "To deploy: run make build then kubectl apply" |
| `working` | Temporary context for current task | "Currently refactoring the auth module" |
| `tool_result` | Results from tool executions | Cached command outputs |

## API Reference

TrustMem exposes MCP tools that your AI tool calls automatically based on steering rules. You can also invoke them directly.

### Core CRUD

| Tool | Description |
|------|-------------|
| `memory_store` | Store a new memory |
| `memory_retrieve` | Retrieve relevant memories for a query (call at conversation start) |
| `memory_correct` | Update an existing memory with new content |
| `memory_purge` | Delete by ID or bulk-delete by topic keyword |
| `memory_search` | Semantic search across all memories |
| `memory_profile` | Get user's memory-derived profile summary |

### Snapshots

| Tool | Description |
|------|-------------|
| `memory_snapshot` | Create a named snapshot of current memory state |
| `memory_snapshots` | List all snapshots |
| `memory_rollback` | Restore memories to a previous snapshot |

### Branches

| Tool | Description |
|------|-------------|
| `memory_branch` | Create a new branch for isolated experimentation |
| `memory_branches` | List all branches |
| `memory_checkout` | Switch to a different branch |
| `memory_merge` | Merge a branch back into main |
| `memory_diff` | Preview what would change on merge |
| `memory_branch_delete` | Delete a branch |

### Maintenance

| Tool | Description |
|------|-------------|
| `memory_governance` | Quarantine low-confidence memories, clean stale data (1h cooldown) |
| `memory_consolidate` | Detect contradictions, fix orphaned graph nodes (30min cooldown) |
| `memory_reflect` | Synthesize high-level insights from memory clusters via LLM (2h cooldown) |
| `memory_rebuild_index` | Rebuild IVF vector index for a table |

## Usage Examples

### Store and Retrieve

```
You: "I prefer tabs over spaces, and always use black for formatting"
AI:  → calls memory_store("User prefers tabs over spaces, uses black for formatting", type="profile")

... next conversation ...

You: "Format this Python file"
AI:  → calls memory_retrieve("format python file")
     ← gets: [profile] User prefers tabs over spaces, uses black for formatting
     → formats with black, uses tabs
```

### Correct a Memory

```
You: "Actually, I switched to ruff instead of black"
AI:  → calls memory_correct(memory_id="abc123", new_content="User uses ruff for formatting", reason="switched from black")
```

### Forget Memories

```
You: "Forget everything about the old auth module"
AI:  → calls memory_purge(topic="old auth module", reason="no longer relevant")
     ← "Purged 3 memory(ies)."
```

### Snapshots: Save and Restore State

Snapshots capture the entire memory state at a point in time — useful before risky changes.

```
You: "Take a snapshot before we refactor the database layer"
AI:  → calls memory_snapshot(name="before_db_refactor", description="pre-refactor state")
     ← "Snapshot 'before_db_refactor' created."

... refactoring goes wrong, memories are messed up ...

You: "Roll back to before the refactor"
AI:  → calls memory_rollback(name="before_db_refactor")
     ← "Rolled back to snapshot 'before_db_refactor'."
```

List snapshots:
```
You: "Show my snapshots"
AI:  → calls memory_snapshots()
     ← before_db_refactor (2026-03-10 12:00:00)
        initial_setup (2026-03-08 09:30:00)
```

### Branches: Isolated Experimentation

Branches let you experiment with different memory states without affecting main. Think of it like git branches, but for memories.

**Create a branch and experiment:**
```
You: "Create a memory branch to evaluate switching from PostgreSQL to SQLite"
AI:  → calls memory_branch(name="eval_sqlite")
     ← "Branch 'eval_sqlite' created. Use memory_checkout to switch to it."
     → calls memory_checkout(name="eval_sqlite")
     ← "Switched to branch 'eval_sqlite'. 42 memories on this branch."

You: "We're now using SQLite instead of PostgreSQL"
AI:  → calls memory_store("Project uses SQLite for persistence", type="semantic")
     (stored on the eval_sqlite branch only — main is untouched)
```

**Preview changes before merging:**
```
You: "What would change if we merge this branch?"
AI:  → calls memory_diff(source="eval_sqlite")
     ← Branch 'eval_sqlite': 3 total changes
        Summary: 2 new, 1 conflict
          [new] Project uses SQLite for persistence
          [new] SQLite config: WAL mode enabled
          [conflict] Database engine is PostgreSQL 15 ↔ Project uses SQLite
```

**Merge or discard:**
```
You: "Merge it"
AI:  → calls memory_merge(source="eval_sqlite", strategy="replace")
     ← "Merged 3 memories from 'eval_sqlite' (skipped 0)."

# Or discard:
You: "Delete that branch, we're sticking with PostgreSQL"
AI:  → calls memory_branch_delete(name="eval_sqlite")
     ← "Branch 'eval_sqlite' deleted."
```

**Branch from a snapshot or timestamp:**
```
You: "Create a branch from the snapshot we took yesterday"
AI:  → calls memory_branch(name="alt_approach", from_snapshot="before_db_refactor")

You: "Branch from 5 minutes ago"
AI:  → calls memory_branch(name="quick_test", from_timestamp="2026-03-10 15:45:00")
```

**List and switch branches:**
```
You: "Show all branches"
AI:  → calls memory_branches()
     ←   * main
          eval_sqlite
          alt_approach

You: "Switch back to main"
AI:  → calls memory_checkout(name="main")
```

### Maintenance

```
You: "Clean up my memories"
AI:  → calls memory_governance()
     ← "Governance done: quarantined=2, cleaned_stale=5, scenes_created=0"

You: "Check for contradictions"
AI:  → calls memory_consolidate()
     ← "Consolidation done: merged_nodes=1, conflicts_detected=2, promoted=3, demoted=1"

You: "What do you know about me?"
AI:  → calls memory_profile()
     ← Profile for default:
        - Prefers pytest, uses ruff for formatting
        - Works on Go 1.22 project with MatrixOne backend
        - Deploys via kubectl, uses make for builds
```

## How Your AI Tool Uses It

Once configured, your AI tool will automatically:

1. **Retrieve** — At the start of every conversation, call `memory_retrieve` with your first message to load relevant memories
2. **Store** — After each response, store new facts, preferences, or decisions you mentioned
3. **Correct** — When you say a memory is wrong, update it via `memory_correct`
4. **Forget** — When you say "forget X", remove it via `memory_purge`

You can also ask:
- "What do you know about me?" → triggers `memory_profile`
- "Forget everything about X" → bulk delete by topic
- "Take a snapshot" → saves current state for later rollback
- "Create a branch for experimenting" → isolated memory sandbox

## Architecture

```
┌─────────────┐     MCP (stdio)     ┌──────────────┐     SQL      ┌────────────┐
│  Kiro /      │ ◄─────────────────► │  TrustMem    │ ◄──────────► │ MatrixOne  │
│  Cursor /    │   store / retrieve  │  MCP Server  │  vector +    │  Database   │
│  Claude Code │                     │              │  fulltext    │            │
└─────────────┘                      └──────────────┘              └────────────┘
```

## Troubleshooting

### "Cannot connect to database"

Make sure MatrixOne is running:
```bash
docker ps | grep matrixone
# If not running:
docker start matrixone
```

### "sentence-transformers not installed"

Install with local embedding support:
```bash
pip install --index-url https://pypi.org/simple/ --extra-index-url https://test.pypi.org/simple/ 'trust-mem-lite[local-embedding]'
```

### First query is slow

This is expected with local embedding. The model (~80MB) loads into memory on the first query, which takes a few seconds. Subsequent queries are fast. To avoid this, use an embedding service instead:
```bash
trustmem init --embedding-provider openai --embedding-api-key sk-...
# or a self-hosted endpoint:
trustmem init --embedding-provider openai --embedding-base-url http://localhost:11434/v1
```

### AI tool doesn't seem to use memory

1. Check config: `trustmem status`
2. Make sure you restarted the AI tool after `trustmem init`
3. Check that MCP server starts: `python -m mo_memory_mcp`

### Update steering rules after upgrading

```bash
pip install --upgrade trust-mem-lite
trustmem update-rules
```

## For AI Agents

If you're an AI agent helping a user set up TrustMem, **ask before acting**:

1. **Which AI tool?** Kiro / Cursor / Claude Code?
2. **MatrixOne database?** Three options:
   - Already have one → get the connection URL
   - Local Docker → `docker run -d --name matrixone -p 6001:6001 -v ./data/matrixone:/mo-data --memory=2g matrixorigin/matrixone:latest`
   - [MatrixOne Cloud](https://cloud.matrixorigin.cn) (free tier) → no Docker needed, register and get URL
3. **Embedding?** Already have an embedding service (OpenAI, Ollama, custom)? Use it. Otherwise local is fine.
4. **Virtual environment?** Create one if not already in one: `python3 -m venv .venv && source .venv/bin/activate`
5. **Install**: `pip install --index-url https://pypi.org/simple/ --extra-index-url https://test.pypi.org/simple/ 'trust-mem-lite[local-embedding]'`
6. **Configure**: `cd <user-project> && trustmem init` (add `--db-url` for cloud/remote DB, `--embedding-provider` if not local)
7. **Verify**: `trustmem status`
8. Tell user to **restart their AI tool**

For detailed troubleshooting, see [SETUP_GUIDE.md](SETUP_GUIDE.md).

## License

Apache-2.0
