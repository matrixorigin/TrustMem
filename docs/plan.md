# Memoria Product Plan

Three-phase plan for evolving Memoria from a memory backend into a context-aware,
proactive, and visual knowledge system.

Prerequisites: all P0/P1 fixes from v0.2.8 are complete (entity dedup, unified linking,
entity_type, edge weight by source). See `roadmap.md` for technical backlog.

---

## Phase 1: Zero-Friction Capture + Contextual Retrieval (Weeks 1–4)

**Goal:** Automatically capture key information during conversations and retrieve
context-aware results — without interrupting the user.

### Deliverables

#### 1.1 Confidence-tiered auto-capture
- High confidence (≥ 0.85): silent ingest, no user prompt
- Medium confidence (0.5–0.85): ingest with `working` type, promote on reuse
- Low confidence (< 0.5): surface confirmation prompt to user/LLM
- Leverage existing `initial_confidence` + `trust_tier` fields in `mem_memories`
- **Files:** `typed_observer.py`, `typed_pipeline.py`, `config.py`

#### 1.2 Session-topic-anchored retrieval
- Compute session summary embedding at session close (exists: `session_summary.py`)
- Boost retrieval score for memories from recent sessions (recency decay)
- Add topic-anchor weight: memories linked to current session's entities score higher
- **Files:** `retriever.py`, `activation.py`, `scorer.py` (if tabular path)

#### 1.3 Dedup & merge baseline
- Current: cosine > 0.95 → supersede. Extend:
  - 0.85–0.95 range: suggest merge (surface to user or auto-merge with version note)
  - Track `superseded_by` chain for version history
  - Expose `/v1/memories/{id}/history` endpoint for version trail
- **Files:** `governance.py`, `consolidation.py`, memory router

### Acceptance Criteria
- Capture accuracy ≥ 80%, false-positive rate ≤ 5%
- Fuzzy recall ("that approach we discussed") improves ≥ 30% vs current baseline
- Measured via `make verify-talk` cases + new golden sessions

### Resources
2 backend + 1 product/UX

### Milestone M1 (Week 4)
Auto-capture + contextual retrieval live. Core experience: "useful without being noticed."

---

## Phase 2: Proactive Prompts + Natural Language Management (Weeks 5–8)

**Goal:** Surface relevant memories at the right moment without interrupting flow.
Let users manage memories through natural conversation.

### Deliverables

#### 2.1 Scene-based triggers
- When user starts a similar project/task, retrieve related lessons/decisions
- Trigger threshold: activation score > configurable gate (default 0.6)
- Suppress if user dismissed similar prompt within N turns (anti-annoyance)
- **Files:** `activation.py`, new `triggers.py` in `core/memory/`

#### 2.2 Memory management conversation flow
- Conflict detection: surface contradicting memories for resolution
  (existing: `consolidation.py` detects conflicts, needs user-facing flow)
- Version cleanup: "you have 3 versions of X, keep latest?" prompt
- Merge suggestions: "these 2 memories overlap, combine?"
- **Files:** `consolidation.py`, new MCP tools or chat commands

#### 2.3 Memory health prompts
- Low-confidence memory ratio warning (> 30% of active memories below 0.5)
- Stale memory alert (no access in 90 days, high importance)
- Surface via `/v1/health` response or proactive prompt
- **Files:** `health.py`, `governance.py`

### Acceptance Criteria
- Proactive prompt adoption rate ≥ 25%
- Management request completion rate ≥ 70%
- Measured via talk verification + user feedback

### Resources
2 backend + 1 frontend + 1 product/UX

### Milestone M2 (Week 8)
Proactive prompts and memory management conversation available.

---

## Phase 3: Visual Graph + Multimodal Preparation (Weeks 9–12)

**Goal:** Visualize knowledge structure. Prepare multimodal memory ingestion.

### Deliverables

#### 3.1 Graph API
- `GET /v1/graph?center={entity}&depth=2` — entity-centric subgraph
- `GET /v1/graph/timeline?session_id=...` — temporal event chain
- `GET /v1/graph/clusters` — scene-based topic clusters
- Return nodes + edges in a frontend-friendly format (e.g. vis.js / d3 compatible)
- **Files:** new `routers/graph.py`, `graph_store.py` queries

#### 3.2 Interactive graph MVP
- Frontend: entity-centric exploration (click entity → expand neighbors)
- Timeline view: session events as a horizontal chain
- Cluster view: scene nodes as groups with member memories
- **Stack:** lightweight SPA (React/Vue) or embedded in existing UI

#### 3.3 Multimodal schema
- Add `modality` VARCHAR(10) DEFAULT 'text' and `uri` TEXT to `memory_graph_nodes`
- Auto-migrate via `ensure_tables` (same pattern as `entity_type`)
- Prototype: store image/file URI, retrieve via text query → vector similarity
- **Files:** `graph.py` model, `schema.py` DDL, `graph_store.py`

### Acceptance Criteria
- Graph displays 3 layers of interactive relationships
- Multimodal prototype: text query retrieves image/file reference
- Measured via demo + integration test

### Resources
2 backend + 2 frontend + 1 design

### Milestone M3 (Week 12)
Graph MVP + multimodal entry point.

---

## Priority Reminders

**Must do first:**
- Confidence-tiered capture + contextual retrieval (Phase 1)
- Without these, proactive prompts become noise

**Can defer:**
- Graph frontend + multimodal (Phase 3)
- Without Phase 1–2, the visual layer has limited value

**Dependencies:**
- Phase 2 depends on Phase 1 (confidence tiers must exist before triggering prompts)
- Phase 3 is independent of Phase 2 (graph API can start in parallel with Phase 2)

---

## Existing Technical Backlog

See `roadmap.md` for implementation-level items:
- Async entity extraction (move off hot path)
- Hybrid entity extraction (regex + LLM threshold)
- Entity node garbage collection
- MatrixOne hybrid disambiguation
- Display name field for entities
