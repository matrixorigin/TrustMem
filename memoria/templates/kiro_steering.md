---
inclusion: always
---

<!-- memoria-version: 0.1.0-->

# Memory Integration (Memoria Lite)

You have persistent memory via MCP tools. Memory survives across conversations.

## 🔴 MANDATORY: Every conversation start
Call `memory_retrieve` with the user's first message BEFORE responding.
- If results come back → use them as context.
- If "No relevant memories found" → this is normal for new users, proceed without.
- If ⚠️ health warnings appear → inform the user and offer to run `memory_governance`.

## 🔴 MANDATORY: Every conversation turn
After responding, decide if anything is worth remembering:
- User stated a preference, fact, or decision → `memory_store`
- User corrected you → `memory_store` the correction
- You learned something about the project/workflow → `memory_store`
- Do NOT store: greetings, trivial questions, things already in memory.

**Deduplication is automatic.** The system detects semantically similar memories and supersedes old ones. You do not need to check for duplicates before storing.

If `memory_store` or `memory_correct` response contains ⚠️, tell the user — it means the embedding service is down and retrieval will degrade to keyword-only search.

## Tool reference

### Write tools
| Tool | When to use | Key params |
|------|-------------|------------|
| `memory_store` | User shares a fact, preference, or decision | `content`, `memory_type` (default: semantic), `session_id` (optional) |
| `memory_correct` | User says a stored memory is wrong | `memory_id` or `query` (one required), `new_content`, `reason` |
| `memory_purge` | User asks to forget something | `memory_id` (single) or `topic` (bulk keyword match), `reason` |

### Read tools
| Tool | When to use | Key params |
|------|-------------|------------|
| `memory_retrieve` | Conversation start, or when context is needed | `query`, `top_k` (default 5), `session_id` (optional) |
| `memory_search` | User asks "what do you know about X" or you need to browse | `query`, `top_k` (default 10). Returns memory_id for each result |
| `memory_profile` | User asks "what do you know about me" | — |

### Memory types
| Type | Use for | Examples |
|------|---------|---------|
| `semantic` | Project facts, technical decisions (default) | "Uses MatrixOne as primary DB", "API follows REST conventions" |
| `profile` | User/agent identity and preferences | "Prefers concise answers", "Works on mo-dev-agent project" |
| `procedural` | How-to knowledge, workflows | "Deploy with: make dev-start", "Run tests with pytest -n auto" |
| `working` | Temporary context for current task | "Currently debugging embedding issue" |
| `tool_result` | Tool execution results worth caching | "Last CI run: 126 passed, 0 failed" |

### Snapshots (save/restore)
Use before risky changes. `memory_snapshot(name)` saves state, `memory_rollback(name)` restores it, `memory_snapshots()` lists all.

### Branches (isolated experiments)
Git-like workflow for memory. `memory_branch(name)` creates, `memory_checkout(name)` switches, `memory_diff(source)` previews changes, `memory_merge(source)` merges back, `memory_branch_delete(name)` cleans up. `memory_branches()` lists all.

### Entity graph (proactive — call when conditions are met)
| Tool | When to call | Key params |
|------|-------------|------------|
| `memory_extract_entities` | **Proactively** after storing ≥ 5 new memories in a session, OR when user discusses a new project/technology/person not yet in the graph | `mode` (default: auto) |
| `memory_link_entities` | After `extract_entities(mode='candidates')` returns memories — extract entities yourself, then call this | `entities` (JSON string) |

**Trigger heuristics — call `memory_extract_entities` when ANY of these are true:**
- You stored ≥ 5 memories this session and haven't extracted entities yet
- User mentions a project, technology, or person by name that you haven't seen in previous `memory_retrieve` results
- User asks about relationships between concepts ("how does X relate to Y")
- User starts working on a new codebase or topic area

**Do NOT extract entities when:**
- Conversation is short (< 3 turns) and no new named entities appeared
- User is only asking questions, not sharing new information
- You already ran extraction this session

### Maintenance (only when user explicitly asks)
| Tool | Trigger phrase | Cooldown |
|------|---------------|----------|
| `memory_governance` | "clean up memories", "check memory health" | 1 hour |
| `memory_consolidate` | "check for contradictions", "fix conflicts" | 30 min |
| `memory_reflect` | "find patterns", "summarize what you know" | 2 hours |
| `memory_rebuild_index` | Only when governance reports `needs_rebuild=True` | — |

`memory_reflect` and `memory_extract_entities` support `mode` parameter:
- `auto` (default): uses Memoria's internal LLM if configured, otherwise returns candidates for YOU to process
- `candidates`: always returns raw data for YOU to synthesize/extract, then store results via `memory_store` or `memory_link_entities`
- `internal`: always uses Memoria's internal LLM (fails if not configured)
