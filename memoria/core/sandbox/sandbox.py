"""Sandbox for isolated experiments — backed by data branch."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from sqlalchemy import text
from sqlalchemy.orm import Session

from memoria.core.sandbox.branch import Branch
from memoria.core.validation import validate_identifier
from memoria.core.db_consumer import DbConsumer, DbFactory

if TYPE_CHECKING:
    from datetime import datetime


class Sandbox(DbConsumer):
    """Sandbox for isolated experiments with metadata management.

    Internal implementation uses `data branch` for zero-copy table branching.
    Snapshots are database-level (for sandbox, not account-level).
    PITR is created on sandbox database at creation time.
    """

    def __init__(
        self, db_factory: DbFactory, source_db: str = "dev_agent", account: str = "sys"
    ):
        super().__init__(db_factory)
        self.source_db = source_db
        self.account = account
        self.branch = Branch(self._db_factory, database=source_db)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def create(
        self,
        name: str,
        description: str = "",
        created_by: str = "system",
        tags: list[str] | None = None,
        tables: list[str] | None = None,
        session_id: str | None = None,
        pitr_range: int = 1,
        pitr_unit: str = "d",
    ) -> None:
        """Create sandbox database with optional table branching and PITR.

        Args:
            name: Sandbox database name.
            tables: Tables to branch from source (zero-copy). If None, empty DB.
            pitr_range: PITR retention length (default 1).
            pitr_unit: PITR retention unit: 'h','d','mo','y' (default 'w' = week? MO uses 'd').
        """
        with self._db() as db:
            import json

            validate_identifier(name)
            db.commit()

            # 1. Create empty database
            db.execute(text(f"DROP DATABASE IF EXISTS {name}"))
            db.commit()
            db.execute(text(f"CREATE DATABASE {name}"))
            db.commit()

            # 2. Branch tables (zero-copy)
            if tables:
                for t in tables:
                    self.branch.create(f"{name}.{t}", f"{self.source_db}.{t}")

            # 3. Create PITR for sandbox database
            pitr_name = f"{name}__pitr"
            try:
                db.execute(text(f"drop pitr if exists {pitr_name}"))
                db.commit()
                db.execute(text(
                    f"create pitr {pitr_name} for database {name} range {pitr_range} '{pitr_unit}'"
                ))
                db.commit()
            except Exception:
                pass  # PITR creation is best-effort

            # 4. Store metadata
            tags_json = json.dumps(tags) if tags else None
            db.execute(
                text(f"""
                    INSERT INTO {self.source_db}.infra_sandbox_metadata
                    (sandbox_name, user_id, data_source, description, created_by,
                     created_at, updated_at, tags, source_database, source_snapshot, status, session_id)
                    VALUES (:name, :created_by, :data_source, :description, :created_by,
                            CURRENT_TIMESTAMP(6), CURRENT_TIMESTAMP(6),
                            :tags, :source_db, :snapshot, 'active', :session_id)
                """),
                {
                    "name": name,
                    "created_by": created_by,
                    "data_source": json.dumps({"type": "matrixone", "database": name}),
                    "description": description,
                    "tags": tags_json,
                    "source_db": self.source_db,
                    "snapshot": None,
                    "session_id": session_id,
                },
            )
            db.commit()

    def delete(self, name: str, force: bool = False) -> None:
        """Delete sandbox: branch metadata + snapshots + PITR + database.

        If any resource cleanup fails and force=False, metadata is kept
        so the sandbox can be retried later. With force=True, metadata
        and database are always dropped (may leave orphan snapshots).
        """
        with self._db() as db:
            validate_identifier(name)
            db.commit()

            failures: list[str] = []

            # 1. Collect resources before any destructive ops
            tables = self.list_tables(name)
            snapshots = self._list_snapshots_raw(name)
            pitr_name = f"{name}__pitr"

            # 2. Delete branch metadata per table
            for t in tables:
                try:
                    self.branch.delete(f"{name}.{t}")
                except Exception as e:
                    failures.append(f"branch delete {name}.{t}: {e}")

            # 3. Drop all snapshots
            for sp in snapshots:
                try:
                    db.commit()
                    db.execute(text(f"drop snapshot {sp}"))
                    db.commit()
                except Exception as e:
                    failures.append(f"drop snapshot {sp}: {e}")

            # 4. Drop PITR
            try:
                db.commit()
                db.execute(text(f"drop pitr if exists {pitr_name}"))
                db.commit()
            except Exception as e:
                failures.append(f"drop pitr {pitr_name}: {e}")

            # 5. Verify snapshots actually gone
            remaining = self._list_snapshots_raw(name)
            if remaining:
                failures.append(f"snapshots still exist: {remaining}")

            # 6. Only drop DB + metadata if all clean (or force)
            if failures and not force:
                raise RuntimeError(
                    f"Sandbox {name} partially cleaned, metadata kept for retry. "
                    f"Failures: {failures}"
                )

            # Drop database
            try:
                db.commit()
                db.execute(text(f"DROP DATABASE IF EXISTS {name}"))
                db.commit()
            except Exception as e:
                failures.append(f"DROP DATABASE {name}: {e}")

            # Delete metadata
            try:
                db.execute(text(
                    f"DELETE FROM {self.source_db}.infra_sandbox_metadata WHERE sandbox_name = :name"
                ), {"name": name})
                db.commit()
            except Exception as e:
                if not force:
                    raise
                failures.append(f"DELETE metadata {name}: {e}")

    # ------------------------------------------------------------------
    # Table management
    # ------------------------------------------------------------------

    def add_table(self, sandbox: str, table: str) -> None:
        """Branch a table from source into sandbox (zero-copy)."""
        validate_identifier(sandbox)
        validate_identifier(table)
        self.branch.create(f"{sandbox}.{table}", f"{self.source_db}.{table}")
        self._touch_metadata(sandbox)

    def remove_table(self, sandbox: str, table: str) -> None:
        """Remove table from sandbox."""
        with self._db() as db:
            try:
                self.branch.delete(f"{sandbox}.{table}")
            except Exception:
                pass
            db.execute(text(f"DROP TABLE IF EXISTS {sandbox}.{table}"))
            db.commit()
            self._touch_metadata(sandbox)

    def list_tables(self, sandbox: str) -> list[str]:
        """List tables in sandbox."""
        with self._db() as db:
            try:
                # Validate sandbox name to prevent SQL injection
                if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', sandbox):
                    raise ValueError(f"Invalid sandbox name: {sandbox}")
                result = db.execute(text(f"SHOW TABLES FROM {sandbox}"))
                return [row._mapping[f"Tables_in_{sandbox}"] for row in result]
            except Exception:
                return []

    # ------------------------------------------------------------------
    # Snapshot & Restore (on sandbox database)
    # ------------------------------------------------------------------

    def snapshot(self, sandbox: str, name: str) -> None:
        """Create snapshot of sandbox database state."""
        with self._db() as db:
            result = db.execute(
                text(f"SELECT 1 FROM {self.source_db}.infra_sandbox_metadata WHERE sandbox_name = :s"),
                {"s": sandbox},
            )
            if not result.first():
                raise ValueError(f"Sandbox {sandbox} not found")

            snapshot_name = f"{sandbox}__{name}"
            validate_identifier(snapshot_name)
            db.commit()
            db.execute(text(f"create snapshot {snapshot_name} for database {sandbox}"))
            db.commit()
            self._touch_metadata(sandbox)

    def list_snapshots(self, sandbox: str) -> list[dict]:
        """List snapshots for a sandbox."""
        prefix = f"{sandbox}__"
        raw = self._list_snapshots_raw(sandbox)
        result = []
        for full_name in raw:
            short = full_name[len(prefix):]
            result.append({"name": short, "full_name": full_name})
        return result

    def restore(self, sandbox: str, snapshot_name: str) -> None:
        """Restore sandbox database from a named snapshot."""
        with self._db() as db:
            full_name = f"{sandbox}__{snapshot_name}"
            validate_identifier(full_name)
            db.commit()
            db.execute(text(
                f"restore account {self.account} database {sandbox} from snapshot {full_name}"
            ))
            db.commit()
            self._touch_metadata(sandbox)

    # ------------------------------------------------------------------
    # Diff & Merge (sandbox vs source)
    # ------------------------------------------------------------------

    def diff(self, sandbox: str, tables: list[str] | None = None) -> list[dict]:
        """Diff sandbox tables against source. Returns list of {table, rows}."""
        target_tables = tables or self.list_tables(sandbox)
        diffs = []
        for t in target_tables:
            try:
                rows = self.branch.diff(f"{sandbox}.{t}", f"{self.source_db}.{t}")
                if rows:
                    diffs.append({"table": t, "rows": rows})
            except Exception:
                continue
        return diffs

    def merge(
        self, sandbox: str, tables: list[str] | None = None, on_conflict: str = "skip"
    ) -> dict:
        """Merge sandbox changes back to source. Returns {merged, failed}."""
        target_tables = tables or self.list_tables(sandbox)
        merged, failed = [], []
        for t in target_tables:
            try:
                self.branch.merge(f"{sandbox}.{t}", f"{self.source_db}.{t}", on_conflict)
                merged.append(t)
            except Exception:
                failed.append(t)
        return {"merged": merged, "failed": failed}

    # ------------------------------------------------------------------
    # Query & Metadata
    # ------------------------------------------------------------------

    # use() removed — executing USE on a pooled connection pollutes the
    # connection's database context for all subsequent consumers.
    # Use fully-qualified table names (``sandbox_name.table``) instead.

    def info(self, sandbox: str) -> dict:
        """Get sandbox info with metadata."""
        with self._db() as db:
            result_meta = db.execute(
                text(f"SELECT * FROM {self.source_db}.infra_sandbox_metadata WHERE sandbox_name = :s"),
                {"s": sandbox},
            )
            metadata = result_meta.first()

            tables = self.list_tables(sandbox)
            table_info = []
            for t in tables:
                if t.startswith("_") or t == "infra_sandbox_metadata":
                    continue
                try:
                    cr = db.execute(text(f"SELECT COUNT(*) as count FROM {sandbox}.{t}"))
                    count = cr.scalar() or 0
                except Exception:
                    count = 0
                table_info.append({"table": t, "rows": count})

            result = {
                "sandbox_name": sandbox,
                "table_count": len(tables),
                "table_details": table_info,
            }
            if metadata:
                result.update(dict(metadata._mapping))
            return result

    def list_sandboxes(
        self,
        prefix: str = "sandbox_",
        pattern: str | None = None,
        status: str | None = None,
        created_by: str | None = None,
        created_after: datetime | None = None,
        updated_after: datetime | None = None,
        tags: list[str] | None = None,
    ) -> list[dict[str, str]]:
        """List sandboxes with filtering."""
        with self._db() as db:
            query = f"SELECT * FROM {self.source_db}.infra_sandbox_metadata WHERE 1=1"
            params: dict = {}

            if prefix:
                query += " AND sandbox_name LIKE :prefix"
                params["prefix"] = f"{prefix}%"
            if pattern:
                query += " AND sandbox_name LIKE :pattern"
                params["pattern"] = pattern
            if status:
                query += " AND status = :status"
                params["status"] = status
            if created_by:
                query += " AND created_by = :created_by"
                params["created_by"] = created_by
            if created_after:
                query += " AND created_at > :created_after"
                params["created_after"] = created_after.isoformat()
            if updated_after:
                query += " AND updated_at > :updated_after"
                params["updated_after"] = updated_after.isoformat()
            if tags:
                for i, tag in enumerate(tags):
                    query += f" AND tags LIKE :tag{i}"
                    params[f"tag{i}"] = f"%{tag}%"

            query += " ORDER BY created_at DESC"
            result = db.execute(text(query), params)
            return [dict(row._mapping) for row in result]

    def update(
        self,
        name: str,
        description: str | None = None,
        tags: list[str] | None = None,
        status: str | None = None,
    ) -> None:
        """Update sandbox metadata."""
        with self._db() as db:
            updates = []
            params: dict = {"name": name}

            if description is not None:
                updates.append("description = :description")
                params["description"] = description
            if tags is not None:
                import json
                updates.append("tags = :tags")
                params["tags"] = json.dumps(tags)
            if status is not None:
                updates.append("status = :status")
                params["status"] = status

            if updates:
                updates.append("updated_at = CURRENT_TIMESTAMP")
                q = f"UPDATE {self.source_db}.infra_sandbox_metadata SET " + ", ".join(updates) + " WHERE sandbox_name = :name"
                db.execute(text(q), params)
                db.commit()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _list_snapshots_raw(self, sandbox: str) -> list[str]:
        """Get snapshot names for a sandbox."""
        with self._db() as db:
            prefix = f"{sandbox}__"
            try:
                db.commit()
                result = db.execute(text(
                    f"show snapshots where snapshot_name like '{prefix}%'"
                ))
                return [row._mapping["snapshot_name"] for row in result]
            except Exception:
                return []

    def _touch_metadata(self, sandbox: str) -> None:
        """Update updated_at timestamp."""
        with self._db() as db:
            db.execute(text(
                f"UPDATE {self.source_db}.infra_sandbox_metadata SET updated_at = CURRENT_TIMESTAMP(6) "
                f"WHERE sandbox_name = :s"
            ), {"s": sandbox})
            db.commit()
