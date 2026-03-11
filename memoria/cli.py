"""Memoria Lite CLI — configure AI tools to use shared memory service.

Usage:
    memoria init       # Detect tools, write MCP config + steering rules
    memoria status     # Show connection status
    memoria health     # Health check
    memoria migrate    # Create memory tables in the database
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from memoria.core.db_consumer import DbFactory

_VERSION = "0.2.11"
_PRODUCT = "Memoria Lite"
_MCP_SERVER_KEY = "memoria-lite"

# ── MCP config templates ──────────────────────────────────────────────


def _mcp_config(mode: str = "stdio", db_url: str | None = None, **embed_opts: str) -> dict:
    import sys

    if mode == "remote":
        return {"url": "http://localhost:8100/mcp"}
    cfg: dict[str, Any] = {
        "command": sys.executable,
        "args": ["-m", "memoria.mcp_local"],
    }
    # Always emit all env vars so users can see what's configurable.
    # When user doesn't specify embedding options, write explicit defaults
    # so the MCP server, schema DDL, and embed client all agree on dimensions.
    provider = embed_opts.get("provider", "local")
    model = embed_opts.get("model", "all-MiniLM-L6-v2")
    dim = embed_opts.get("dim", "")
    if not dim:
        # Auto-infer from model name. Inline subset — keys use BOTH short and
        # fully-qualified names because users may configure either form.
        # Canonical source: core.embedding.client.KNOWN_DIMENSIONS (not imported
        # here to keep CLI startup fast and avoid heavy transitive imports).
        _MODEL_DIMS = {
            "all-MiniLM-L6-v2": "384", "all-MiniLM-L12-v2": "384",
            "sentence-transformers/all-MiniLM-L6-v2": "384",
            "sentence-transformers/all-MiniLM-L12-v2": "384",
            "BAAI/bge-m3": "1024", "BAAI/bge-base-en-v1.5": "768",
            "text-embedding-3-small": "1536", "text-embedding-ada-002": "1536",
        }
        dim = _MODEL_DIMS.get(model, "1024")
    env: dict[str, str] = {
        "MEMORIA_DB_URL": db_url or "",
        "EMBEDDING_PROVIDER": provider,
        "EMBEDDING_MODEL": model,
        "EMBEDDING_DIM": dim,
        "EMBEDDING_API_KEY": embed_opts.get("api_key", ""),
        "EMBEDDING_BASE_URL": embed_opts.get("base_url", ""),
    }
    cfg["env"] = env
    return cfg


# ── Steering rule content (loaded from templates/) ───────────────────
# Templates live alongside the MCP server package so they are the single
# source of truth for all AI-tool steering rules.  The CLI reads them at
# runtime and writes them into each tool's config directory.

def _get_templates_dir() -> Path:
    """Resolve templates dir, works both in normal Python and PyInstaller binary."""
    # PyInstaller sets sys._MEIPASS to the temp extraction dir
    base = Path(getattr(sys, '_MEIPASS', Path(__file__).parent.parent))
    return base / "memoria.mcp_local" / "templates"


_TEMPLATES_DIR = _get_templates_dir()

# Required sections that every steering template must contain.
# Prevents silently writing empty or broken rules.
_REQUIRED_KEYWORDS = ["Memory Integration", "memory_retrieve"]


def _load_template(name: str) -> str:
    """Load and validate a steering-rule template.

    Raises:
        FileNotFoundError: Template file does not exist.
        ValueError: Template is empty or missing required sections.
    """
    path = _TEMPLATES_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Template not found: {path}")

    content = path.read_text()
    if not content.strip():
        raise ValueError(f"Template is empty: {path}")

    for keyword in _REQUIRED_KEYWORDS:
        if keyword not in content:
            raise ValueError(
                f"Template {path.name} missing required section '{keyword}'"
            )
    return content


def _get_kiro_steering() -> str:
    return _load_template("kiro_steering.md")


def _get_cursor_rule() -> str:
    return _load_template("cursor_rule.md")


def _get_claude_rule() -> str:
    return _load_template("claude_rule.md")


def _safe_write_rule(rule_file: Path, new_content: str, *, force: bool, project_dir: Path) -> str:
    """Write a steering rule file, respecting user customizations.

    Returns a status string for display.
    """
    rel = rule_file.relative_to(project_dir)
    if not rule_file.exists():
        rule_file.parent.mkdir(parents=True, exist_ok=True)
        rule_file.write_text(new_content)
        return f"  ✅ {rel} (created)"

    existing = rule_file.read_text()
    installed_ver = _installed_rule_version(rule_file)
    import re
    new_ver_m = re.search(r"memoria-version:\s*([\d.]+)", new_content[:500])
    new_ver = new_ver_m.group(1) if new_ver_m else None

    # Same version, same content — skip
    if installed_ver == new_ver and existing.strip() == new_content.strip():
        return f"  ⏭️  {rel} (up to date)"

    # Same version but content differs — user customized
    if installed_ver == new_ver and existing.strip() != new_content.strip():
        if force:
            _backup(rule_file)
            rule_file.write_text(new_content)
            return f"  ⚠️  {rel} (overwritten — backup saved as {rel}.bak)"
        return f"  ⏭️  {rel} (user-customized, use --force to overwrite)"

    # Older version — auto-update, backup first
    if installed_ver != new_ver:
        _backup(rule_file)
        rule_file.write_text(new_content)
        return f"  ✅ {rel} (updated {installed_ver} → {new_ver}, backup saved as {rel}.bak)"

    rule_file.write_text(new_content)
    return f"  ✅ {rel}"


def _backup(path: Path) -> None:
    """Save a .bak copy of a file before overwriting."""
    bak = path.with_suffix(path.suffix + ".bak")
    bak.write_text(path.read_text())


# ── Detection & writing ───────────────────────────────────────────────

def _detect_tools(project_dir: Path) -> dict[str, bool]:
    return {
        "kiro": (project_dir / ".kiro").is_dir(),
        "cursor": (project_dir / ".cursor").is_dir() or (project_dir / ".cursorrc").exists(),
        "claude": (project_dir / "CLAUDE.md").exists() or (project_dir / ".claude").is_dir(),
    }


def _write_kiro(project_dir: Path, mode: str, db_url: str | None = None, force: bool = False, **embed_opts: str) -> list[str]:
    actions = []

    # MCP config
    mcp_dir = project_dir / ".kiro" / "settings"
    mcp_dir.mkdir(parents=True, exist_ok=True)
    mcp_file = mcp_dir / "mcp.json"

    if mcp_file.exists():
        config = json.loads(mcp_file.read_text())
    else:
        config = {"mcpServers": {}}

    config.setdefault("mcpServers", {})
    config["mcpServers"][_MCP_SERVER_KEY] = _mcp_config(mode, db_url, **embed_opts)
    mcp_file.write_text(json.dumps(config, indent=2) + "\n")
    actions.append(f"  ✅ {mcp_file.relative_to(project_dir)}")

    # Steering rule
    steering_dir = project_dir / ".kiro" / "steering"
    steering_dir.mkdir(parents=True, exist_ok=True)
    rule_file = steering_dir / "memory.md"
    actions.append(_safe_write_rule(rule_file, _get_kiro_steering(), force=force, project_dir=project_dir))

    return actions


def _write_cursor(project_dir: Path, mode: str, db_url: str | None = None, force: bool = False, **embed_opts: str) -> list[str]:
    actions = []

    # MCP config
    cursor_dir = project_dir / ".cursor"
    cursor_dir.mkdir(parents=True, exist_ok=True)
    mcp_file = cursor_dir / "mcp.json"

    if mcp_file.exists():
        config = json.loads(mcp_file.read_text())
    else:
        config = {"mcpServers": {}}

    config.setdefault("mcpServers", {})
    config["mcpServers"][_MCP_SERVER_KEY] = _mcp_config(mode, db_url, **embed_opts)
    mcp_file.write_text(json.dumps(config, indent=2) + "\n")
    actions.append(f"  ✅ {mcp_file.relative_to(project_dir)}")

    # Rule file
    rules_dir = cursor_dir / "rules"
    rules_dir.mkdir(parents=True, exist_ok=True)
    rule_file = rules_dir / "memory.mdc"
    actions.append(_safe_write_rule(rule_file, _get_cursor_rule(), force=force, project_dir=project_dir))

    return actions


def _write_claude(project_dir: Path, mode: str, db_url: str | None = None, force: bool = False, **embed_opts: str) -> list[str]:
    actions = []

    # MCP config for Claude Code
    claude_dir = project_dir / ".claude"
    claude_dir.mkdir(parents=True, exist_ok=True)
    mcp_file = claude_dir / "mcp.json"

    if mcp_file.exists():
        config = json.loads(mcp_file.read_text())
    else:
        config = {"mcpServers": {}}

    config.setdefault("mcpServers", {})
    config["mcpServers"][_MCP_SERVER_KEY] = _mcp_config(mode, db_url, **embed_opts)
    mcp_file.write_text(json.dumps(config, indent=2) + "\n")
    actions.append(f"  ✅ {mcp_file.relative_to(project_dir)}")

    # Append to CLAUDE.md
    claude_md = project_dir / "CLAUDE.md"
    if claude_md.exists():
        existing = claude_md.read_text()
        if _MCP_SERVER_KEY not in existing:
            claude_md.write_text(existing.rstrip() + "\n" + _get_claude_rule())
            actions.append(f"  ✅ {claude_md.relative_to(project_dir)} (appended)")
        else:
            actions.append(f"  ⏭️  {claude_md.relative_to(project_dir)} (already configured)")
    else:
        claude_md.write_text(_get_claude_rule().lstrip())
        actions.append(f"  ✅ {claude_md.relative_to(project_dir)} (created)")

    return actions


# ── Database helpers ──────────────────────────────────────────────────


def _resolve_engine(db_url: str | None) -> tuple[Any, str]:
    """Resolve a SQLAlchemy engine from --db-url, env var, or project default.

    Returns (engine, source_description). Always succeeds — falls back to DEFAULT_DB_URL.
    """
    from memoria.schema import DEFAULT_DB_URL

    url = db_url or os.environ.get("MEMORIA_DB_URL")
    if url:
        from sqlalchemy import create_engine
        return create_engine(url, pool_pre_ping=True), url

    # Fall back to project-local engine (dev mode).
    try:
        from api.database import engine as _engine
        return _engine, "project default (api.database)"
    except ImportError:
        pass

    # Last resort: local default.
    from sqlalchemy import create_engine
    return create_engine(DEFAULT_DB_URL, pool_pre_ping=True), f"{DEFAULT_DB_URL} (default)"


def _create_tables(engine: Any, *, dim: int | None = None, force: bool = False) -> list[str]:
    """Create memory tables using self-contained DDL (no core/ dependency).

    Returns list of table names created.
    """
    from memoria.schema import ensure_tables
    return ensure_tables(engine, dim=dim, force=force)


def _test_connection(engine: Any) -> bool:
    """Quick connectivity check. Returns True on success."""
    from sqlalchemy import text
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        print(f"❌ Cannot connect to database: {e}")
        print()
        print("  Check that MatrixOne is running and the URL is correct.")
        print("  Example: mysql+pymysql://root:111@localhost:6001/mydb")
        return False


# ── Commands ──────────────────────────────────────────────────────────

def cmd_init(args: argparse.Namespace) -> None:
    print(f"{_PRODUCT} v{_VERSION} — local single-user mode")
    print()
    project_dir = Path(args.dir).resolve()
    mode = args.mode
    db_url = args.db_url or os.environ.get("MEMORIA_DB_URL")
    embed_opts = {}
    for key in ("provider", "model", "dim", "api_key", "base_url"):
        val = getattr(args, f"embedding_{key}", None)
        if val:
            embed_opts[key] = str(val)

    tools = _detect_tools(project_dir)

    # --tool flag overrides auto-detection
    if args.tool:
        detected = args.tool
    else:
        detected = [name for name, found in tools.items() if found]
        if not detected:
            print("No AI tools detected. Use --tool to specify: --tool kiro, --tool cursor, --tool claude")
            return

    print(f"Detected tools: {', '.join(detected)}")
    print(f"Mode: {mode}")
    if db_url:
        print(f"DB URL: {db_url}")
    if embed_opts.get("provider"):
        print(f"Embedding: {embed_opts['provider']}")
    print()

    # ── Embedding check ───────────────────────────────────────────
    provider = embed_opts.get("provider", "local")
    if provider == "local":
        try:
            import sentence_transformers  # noqa: F401
            print("✅ sentence-transformers installed (local embedding ready)")
        except ImportError:
            print("⚠️  sentence-transformers not installed — memories won't be vectorized.")
            print("   Install it for better retrieval quality:")
            print("   pip install memoria-lite[local-embedding]")
            print("   Or use OpenAI embeddings:")
            print("   memoria init --embedding-provider openai --embedding-api-key sk-...")
        print()

    # ── Step 1: Write MCP configs + steering rules ────────────────
    effective_db_url = db_url
    if mode != "remote" and not db_url:
        # Resolve the default DB URL so it gets written into MCP config explicitly.
        # Tables are NOT created here — MCP server creates them on first start
        # using the EMBEDDING_DIM env var set in the config, so dim is always correct.
        engine, _src = _resolve_engine(None)
        effective_db_url = engine.url.render_as_string(hide_password=False)
        # Non-fatal connectivity check — warn early if DB is unreachable.
        if not _test_connection(engine):
            print("⚠️  Database not reachable — tables will be created when it becomes available.")
            print()
        engine.dispose()

    writers = {"kiro": _write_kiro, "cursor": _write_cursor, "claude": _write_claude}
    for tool_name in detected:
        print(f"Configuring {tool_name}:")
        actions = writers[tool_name](project_dir, mode, effective_db_url, force=args.force, **embed_opts)
        for a in actions:
            print(a)
        print()

    print("Done! Restart your AI tools to pick up the new MCP config.")
    print()
    print("Memory tables will be created automatically when the MCP server starts.")
    print("To change embedding provider before first use, edit the MCP config and")
    print("update EMBEDDING_PROVIDER / EMBEDDING_DIM, then restart your AI tool.")
    print("Or run manually: memoria migrate --db-url '...' --dim <dim>")
    if mode == "stdio" and not db_url:
        print("\nTip: pass --db-url to connect to a specific database:")
        print("  memoria init --db-url 'mysql+pymysql://user:pass@host:6001/db'")


def _rule_paths(project_dir: Path) -> dict[str, Path]:
    """Return {tool_name: rule_file_path} for each tool's steering rule."""
    return {
        "kiro": project_dir / ".kiro" / "steering" / "memory.md",
        "cursor": project_dir / ".cursor" / "rules" / "memory.mdc",
        "claude": project_dir / "CLAUDE.md",
    }


def _installed_rule_version(path: Path) -> str | None:
    """Extract memoria-version from an installed rule file."""
    if not path.exists():
        return None
    import re
    m = re.search(r"memoria-version:\s*([\d.]+)", path.read_text()[:500])
    return m.group(1) if m else None


def cmd_status(args: argparse.Namespace) -> None:
    project_dir = Path(args.dir).resolve()
    tools = _detect_tools(project_dir)
    rules = _rule_paths(project_dir)
    needs_update = False

    for name, found in tools.items():
        if found:
            if name == "kiro":
                cfg = project_dir / ".kiro" / "settings" / "mcp.json"
            elif name == "cursor":
                cfg = project_dir / ".cursor" / "mcp.json"
            else:
                cfg = project_dir / ".claude" / "mcp.json"

            if cfg.exists():
                data = json.loads(cfg.read_text())
                has_memory = _MCP_SERVER_KEY in data.get("mcpServers", {})
                status = "✅ configured" if has_memory else "❌ not configured"
            else:
                status = "❌ no MCP config"

            # Check rule version
            installed_ver = _installed_rule_version(rules[name])
            if installed_ver is None:
                status += " | rules: ❌ missing"
                needs_update = True
            elif installed_ver != _VERSION:
                status += f" | rules: ⚠️  outdated ({installed_ver} → {_VERSION})"
                needs_update = True
            else:
                status += f" | rules: ✅ v{installed_ver}"

            print(f"  {name}: {status}")
        else:
            print(f"  {name}: — not detected")

    if needs_update:
        print()
        print("  Run 'memoria update-rules' to update steering rules.")


def cmd_update_rules(args: argparse.Namespace) -> None:
    """Update steering rules for all detected AI tools."""
    project_dir = Path(args.dir).resolve()
    tools = _detect_tools(project_dir)
    detected = [name for name, found in tools.items() if found]
    if not detected:
        detected = ["kiro", "cursor", "claude"]

    writers = {
        "kiro": lambda d: _write_kiro(d, "stdio")[1:],  # skip mcp.json, only rule
        "cursor": lambda d: _write_cursor(d, "stdio")[1:],
        "claude": lambda d: _write_claude(d, "stdio")[1:],
    }

    # Directly write rule files (not MCP config)
    for name in detected:
        rules = _rule_paths(project_dir)
        rule_file = rules[name]
        if name == "kiro":
            rule_file.parent.mkdir(parents=True, exist_ok=True)
            rule_file.write_text(_get_kiro_steering())
        elif name == "cursor":
            rule_file.parent.mkdir(parents=True, exist_ok=True)
            rule_file.write_text(_get_cursor_rule())
        elif name == "claude":
            # For Claude, rewrite the whole CLAUDE.md section
            if rule_file.exists():
                existing = rule_file.read_text()
                if "memoria-version:" in existing:
                    # Replace existing section
                    import re
                    new_rule = _get_claude_rule()
                    existing = re.sub(
                        r"<!-- memoria-version:.*?(?=\n<!-- |$)",
                        new_rule.strip(),
                        existing,
                        flags=re.DOTALL,
                    )
                    rule_file.write_text(existing)
                else:
                    rule_file.write_text(existing.rstrip() + "\n" + _get_claude_rule())
            else:
                rule_file.write_text(_get_claude_rule().lstrip())
        print(f"  ✅ {name}: rules updated to v{_VERSION}")

    print(f"\nDone! {len(detected)} tools updated.")


def cmd_health(args: argparse.Namespace) -> None:
    import urllib.request
    url = args.api_url.rstrip("/") + "/health"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read())
            status = data.get("status", "unknown")
            print(f"Memory service: {status}")
            print(f"Database: {data.get('database', 'unknown')}")
    except Exception as e:
        print(f"❌ Cannot reach memory service at {url}: {e}")


def cmd_migrate(args: argparse.Namespace) -> None:
    """Create memory tables in the database."""
    engine, src = _resolve_engine(args.db_url)

    if not _test_connection(engine):
        sys.exit(1)

    dim = int(args.dim) if getattr(args, "dim", None) else None
    force = getattr(args, "force", False)
    try:
        tables = _create_tables(engine, dim=dim, force=force)
    except Exception as e:
        print(f"❌ Failed to create tables: {e}")
        sys.exit(1)

    for t in tables:
        print(f"  ✅ {t}")

    print(f"\n{len(tables)} memory tables ready.")


def _get_db_factory(args: argparse.Namespace) -> DbFactory:
    """Resolve a DbFactory from CLI args, env var, project default, or DEFAULT_DB_URL."""
    db_url = getattr(args, "db_url", None) or os.environ.get("MEMORIA_DB_URL")
    if db_url:
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        engine = create_engine(db_url, pool_pre_ping=True)
        return sessionmaker(bind=engine)
    try:
        from api.database import SessionLocal
        return SessionLocal
    except ImportError:
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from memoria.schema import DEFAULT_DB_URL
        engine = create_engine(DEFAULT_DB_URL, pool_pre_ping=True)
        return sessionmaker(bind=engine)


def cmd_governance(args: argparse.Namespace) -> None:
    """Run memory governance cycle."""
    from memoria.core.memory.tabular.governance import GovernanceScheduler
    db_factory = _get_db_factory(args)
    gs = GovernanceScheduler(db_factory)
    user_id = args.user_id or "all"
    print(f"Running governance for user={user_id}...")
    result = gs.run_cycle(user_id)
    print(f"  quarantined={result.quarantined}")
    print(f"  cleaned_stale={result.cleaned_stale}")
    print(f"  scenes_created={result.scenes_created}")
    for table, h in result.vector_index_health.items():
        if h.get("rebuilt"):
            print(f"  ✅ {table}: IVF index rebuilt")
        elif h.get("needs_rebuild"):
            print(f"  ⚠️  {table}: IVF index needs rebuild (ratio={h.get('ratio')})")
        elif "error" not in h:
            print(f"  ✅ {table}: IVF index healthy (ratio={h.get('ratio')})")
    if result.errors:
        for e in result.errors:
            print(f"  ❌ {e}")


def cmd_consolidate(args: argparse.Namespace) -> None:
    """Run graph consolidation."""
    from memoria.core.memory.graph.consolidation import GraphConsolidator
    db_factory = _get_db_factory(args)
    gc = GraphConsolidator(db_factory)
    print(f"Running consolidation for user={args.user_id}...")
    result = gc.consolidate(args.user_id)
    print(f"  merged_nodes={result.merged_nodes}")
    print(f"  conflicts_detected={result.conflicts_detected}")
    print(f"  orphaned_scenes={result.orphaned_scenes}")
    print(f"  promoted={result.promoted}, demoted={result.demoted}")
    if result.errors:
        for e in result.errors:
            print(f"  ❌ {e}")


def cmd_reflect(args: argparse.Namespace) -> None:
    """Run reflection (requires LLM)."""
    from memoria.core.memory.graph.candidates import GraphCandidateProvider
    from memoria.core.memory.graph.service import GraphMemoryService
    from memoria.core.memory.reflection.engine import ReflectionEngine
    db_factory = _get_db_factory(args)
    try:
        from memoria.core.llmclient import LLMClient
        llm = LLMClient(db_factory=db_factory)
    except ImportError as e:
        print(f"❌ LLM client not available (missing dependency): {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ LLM client initialization failed: {e}")
        sys.exit(1)
    provider = GraphCandidateProvider(db_factory)
    svc = GraphMemoryService(db_factory)
    engine = ReflectionEngine(provider, svc, llm)
    print(f"Running reflection for user={args.user_id}...")
    result = engine.reflect(args.user_id)
    print(f"  candidates_found={result.candidates_found}")
    print(f"  scenes_created={result.scenes_created}")
    print(f"  llm_calls={result.llm_calls}")
    if result.errors:
        for e in result.errors:
            print(f"  ❌ {e}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="memoria", description=f"{_PRODUCT} v{_VERSION} — configure AI tools for shared memory")
    parser.add_argument("--dir", default=".", help="Project directory")
    sub = parser.add_subparsers(dest="command")

    p_init = sub.add_parser("init", help="Configure MCP + steering rules + create tables")
    p_init.add_argument("--mode", choices=["stdio", "remote"], default="stdio", help="MCP transport mode")
    p_init.add_argument("--db-url", help="Database URL, e.g. mysql+pymysql://user:pass@host:6001/db")
    p_init.add_argument("--tool", choices=["kiro", "cursor", "claude"], action="append", help="Only configure specific tool(s). Can be repeated: --tool kiro --tool cursor")
    p_init.add_argument("--force", action="store_true", help="Overwrite steering rules even if user has customized them")
    p_init.add_argument("--embedding-provider", help="Embedding provider: local (default), openai, mock")
    p_init.add_argument("--embedding-model", help="Embedding model name (default: all-MiniLM-L6-v2)")
    p_init.add_argument("--embedding-dim", help="Embedding dimension (default: 384)")
    p_init.add_argument("--embedding-api-key", help="API key for OpenAI embedding provider")
    p_init.add_argument("--embedding-base-url", help="Custom API base URL (e.g. Ollama)")

    sub.add_parser("status", help="Show configuration status")
    sub.add_parser("update-rules", help="Update steering rules to latest version")

    p_migrate = sub.add_parser("migrate", help="Create memory tables in the database")
    p_migrate.add_argument("--db-url", help="Database URL (or set MEMORIA_DB_URL)")
    p_migrate.add_argument("--dim", help="Embedding vector dimension (default: 384)")
    p_migrate.add_argument("--force", action="store_true",
                           help="ALTER embedding column if dim mismatches (clears existing embeddings)")

    p_health = sub.add_parser("health", help="Check memory service health")
    p_health.add_argument("--api-url", default="http://localhost:8100", help="Memory service URL")

    p_gov = sub.add_parser("governance", help="Run memory governance: quarantine, cleanup, IVF index health")
    p_gov.add_argument("--user-id", help="User ID (default: all users)")
    p_gov.add_argument("--db-url", help="Database URL (or set MEMORIA_DB_URL)")

    p_con = sub.add_parser("consolidate", help="Run graph consolidation: conflict detection, orphan cleanup")
    p_con.add_argument("--user-id", required=True, help="User ID")
    p_con.add_argument("--db-url", help="Database URL (or set MEMORIA_DB_URL)")

    p_ref = sub.add_parser("reflect", help="Run reflection: synthesize insights from memory clusters (requires LLM)")
    p_ref.add_argument("--user-id", required=True, help="User ID")
    p_ref.add_argument("--db-url", help="Database URL (or set MEMORIA_DB_URL)")

    args = parser.parse_args()
    if args.command == "init":
        cmd_init(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "update-rules":
        cmd_update_rules(args)
    elif args.command == "migrate":
        cmd_migrate(args)
    elif args.command == "health":
        cmd_health(args)
    elif args.command == "governance":
        cmd_governance(args)
    elif args.command == "consolidate":
        cmd_consolidate(args)
    elif args.command == "reflect":
        cmd_reflect(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
