"""Sandbox cleanup — background task for orphaned/expired sandboxes.

Tier 1: Session-scoped cleanup (CodeExecutor.cleanup_session) — immediate.
Tier 2: Safety net (this module) — periodic background scan.
Tier 3: Data PR pending — kept alive until merge/discard, auto-discard after TTL.

Cleanup logic:
  1. session_id set + session CLOSED → sandbox should have been cleaned (Tier 1 missed)
  2. session_id set + session ACTIVE but no activity > TTL → zombie session
  3. session_id NULL + updated_at > TTL → manual sandbox expired
  4. sandbox_*/code_exec_* database with no metadata → orphan
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import text
from sqlalchemy.orm import Session

from memoria.core.logging_config import get_logger
from memoria.core.sandbox import Sandbox
from memoria.core.db_consumer import DbConsumer, DbFactory

logger = get_logger(__name__)

DEFAULT_TTL_HOURS = 24


class SandboxCleaner(DbConsumer):
    """Scans for and cleans up orphaned/expired sandboxes."""

    def __init__(self, db_factory: DbFactory, source_db: str = "dev_agent"):
        super().__init__(db_factory)
        self.source_db = source_db
        self.sandbox = Sandbox(db_factory=self._db_factory, source_db=source_db)

    def run(self, ttl_hours: int = DEFAULT_TTL_HOURS) -> dict:
        """Scan and clean expired sandboxes. Returns summary."""
        cleaned, failed = [], []

        # 1. Sandboxes bound to closed sessions (Tier 1 missed)
        for name in self._find_closed_session_sandboxes():
            self._try_delete(name, cleaned, failed)

        # 2. Sandboxes bound to zombie sessions (active but no activity > TTL)
        cutoff = datetime.now(timezone.utc) - timedelta(hours=ttl_hours)
        for name in self._find_zombie_session_sandboxes(cutoff):
            self._try_delete(name, cleaned, failed)

        # 3. Unbound sandboxes (no session_id) older than TTL
        for name in self._find_expired_unbound(cutoff):
            self._try_delete(name, cleaned, failed)

        # 4. Orphan databases (no metadata at all)
        for name in self._find_orphan_databases():
            self._try_force_delete(name, cleaned, failed)

        result = {"cleaned": len(cleaned), "failed": len(failed)}
        if cleaned or failed:
            logger.info(f"Sandbox cleanup: {result}")
        return result

    # ── Finders ──────────────────────────────────────────────

    def _find_closed_session_sandboxes(self) -> list[str]:
        """Sandboxes whose session is CLOSED — should have been cleaned by Tier 1."""
        with self._db() as db:
            try:
                r = db.execute(text(f"""
                    SELECT m.sandbox_name FROM {self.source_db}.infra_sandbox_metadata m
                    JOIN agent_sessions s ON m.session_id = s.session_id
                    WHERE m.status = 'active' AND s.status = 'closed'
                """))
                return [row._mapping["sandbox_name"] for row in r]
            except Exception:
                return []

    def _find_zombie_session_sandboxes(self, cutoff: datetime) -> list[str]:
        """Sandboxes whose session is ACTIVE but has no recent activity."""
        with self._db() as db:
            try:
                r = db.execute(text(f"""
                    SELECT m.sandbox_name FROM {self.source_db}.infra_sandbox_metadata m
                    JOIN agent_sessions s ON m.session_id = s.session_id
                    WHERE m.status = 'active' AND s.status = 'active'
                      AND s.updated_at < :cutoff
                """), {"cutoff": cutoff})
                return [row._mapping["sandbox_name"] for row in r]
            except Exception:
                return []

    def _find_expired_unbound(self, cutoff: datetime) -> list[str]:
        """Sandboxes with no session_id and updated_at older than cutoff."""
        with self._db() as db:
            try:
                r = db.execute(text(f"""
                    SELECT sandbox_name FROM {self.source_db}.infra_sandbox_metadata
                    WHERE status = 'active' AND session_id IS NULL
                      AND updated_at < :cutoff
                """), {"cutoff": cutoff})
                return [row._mapping["sandbox_name"] for row in r]
            except Exception:
                return []

    def _find_orphan_databases(self) -> list[str]:
        """sandbox_*/code_exec_* databases with no metadata entry."""
        with self._db() as db:
            try:
                r = db.execute(text("SHOW DATABASES"))
                all_dbs = [row[0] for row in r]
                sandbox_dbs = [
                    d for d in all_dbs
                    if d.startswith("sandbox_") or d.startswith("code_exec_")
                ]
                if not sandbox_dbs:
                    return []

                known = set()
                try:
                    r = db.execute(text(
                        f"SELECT sandbox_name FROM {self.source_db}.infra_sandbox_metadata"
                    ))
                    known = {row._mapping["sandbox_name"] for row in r}
                except Exception:
                    pass

                return [d for d in sandbox_dbs if d not in known]
            except Exception:
                return []

    # ── Helpers ──────────────────────────────────────────────

    def _try_delete(self, name: str, cleaned: list, failed: list) -> None:
        try:
            self.sandbox.delete(name)
            cleaned.append(name)
            logger.info(f"Sandbox cleanup: deleted {name}")
        except RuntimeError:
            failed.append(name)
            logger.warning(f"Sandbox cleanup: partial failure on {name}, metadata kept")
        except Exception as e:
            failed.append(name)
            logger.error(f"Sandbox cleanup: {name}: {e}")

    def _try_force_delete(self, name: str, cleaned: list, failed: list) -> None:
        try:
            self.sandbox.delete(name, force=True)
            cleaned.append(name)
            logger.info(f"Sandbox cleanup: force-deleted orphan {name}")
        except Exception as e:
            failed.append(name)
            logger.error(f"Sandbox cleanup: orphan {name}: {e}")
