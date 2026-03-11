"""Git for Data operations for MatrixOne.

Provides snapshot, restore, and time-travel capabilities.
"""

from sqlalchemy import text
from sqlalchemy.orm import Session

from memoria.core.validation import validate_identifier, QueryRequest
from memoria.core.db_consumer import DbConsumer, DbFactory


class GitForData(DbConsumer):
    """Git for Data operations manager.

    Provides MatrixOne's snapshot and time-travel capabilities.
    Based on MatrixOne v3.0+ Git for Data features.
    """

    def __init__(self, db_factory: DbFactory) -> None:
        super().__init__(db_factory)

    def create_snapshot(self, snapshot_name: str, account: str = "sys") -> dict:
        """Create a snapshot of the current database state.

        Args:
            snapshot_name: Name for the snapshot
            account: Account name (default: sys)

        Returns:
            dict: Snapshot metadata with name and timestamp

        Example:
            >>> git = GitForData()
            >>> snapshot = git.create_snapshot("before_experiment")
            >>> print(snapshot["snapshot_name"])
        """
        with self._db() as db:
            db.commit()  # Commit before DDL
        
            # Validate inputs to prevent SQL injection
            safe_snapshot = validate_identifier(snapshot_name)
            safe_account = validate_identifier(account)
        
            query = f"CREATE SNAPSHOT {safe_snapshot} FOR ACCOUNT {safe_account}"
            db.execute(text(query))

            # Get snapshot info
            snapshots = self.list_snapshots()
            snapshot_info = next((s for s in snapshots if s["snapshot_name"] == snapshot_name), None)

            return snapshot_info or {"snapshot_name": snapshot_name, "timestamp": None}

    def list_snapshots(self) -> list[dict]:
        """List all available snapshots.

        Returns:
            list[dict]: List of snapshots with metadata
        """
        with self._db() as db:
            db.commit()  # Commit before DDL
            query = "SHOW SNAPSHOTS"
            result = db.execute(text(query))
            return [
                {
                    "snapshot_name": row._mapping["SNAPSHOT_NAME"],
                    "timestamp": row._mapping["TIMESTAMP"],
                    "snapshot_level": row._mapping["SNAPSHOT_LEVEL"],
                    "account_name": row._mapping["ACCOUNT_NAME"],
                    "database_name": row._mapping.get("DATABASE_NAME"),
                    "table_name": row._mapping.get("TABLE_NAME"),
                    "ts": row._mapping.get("TIMESTAMP"),  # Alias for compatibility
                }
                for row in result
            ]

    def query_at_snapshot(
        self, query: str, snapshot_name: str, params: dict | None = None
    ) -> list[dict]:
        """Execute a query at a specific snapshot (time-travel query).

        This is a READ-ONLY operation that doesn't affect the current state.
        Uses MatrixOne's {SNAPSHOT = 'name'} syntax.

        Args:
            query: SQL query (must be SELECT)
            snapshot_name: Snapshot to query
            params: Optional query parameters (dict for named params)

        Returns:
            list[dict]: Query results

        Example:
            >>> git = GitForData()
            >>> results = git.query_at_snapshot(
            ...     "SELECT * FROM agent_events WHERE session_id = :session_id",
            ...     "my_checkpoint",
            ...     {"session_id": "session_123"}
            ... )
        """
        with self._db() as db:
            import re

            # Validate snapshot name
            validate_identifier(snapshot_name)
        
            # Validate query for basic safety
            QueryRequest.validate_query(query)
        
            # Inject snapshot syntax into query
            # Replace FROM table with FROM table {SNAPSHOT = 'name'}
            snapshot_clause = f"{{SNAPSHOT = '{snapshot_name}'}}"

            def replace_match(match):
                full_match = match.group(0)
                # If snapshot clause already exists, don't modify
                if "{SNAPSHOT" in full_match.upper():
                    return full_match
                # Otherwise append snapshot clause
                return f"{full_match} {snapshot_clause}"

            # Regex to find FROM/JOIN clause and the table name
            # We also look ahead for existing snapshot clause to avoid double injection
            # Pattern: (FROM|JOIN) + whitespace + table_name + optional snapshot clause
            pattern = r'\b(FROM|JOIN)\s+([a-zA-Z0-9_.]+)(?:\s*\{SNAPSHOT\s*=[^}]+\})?'
        
            modified_query = re.sub(
                pattern, 
                replace_match, 
                query, 
                flags=re.IGNORECASE
            )
        
            result = db.execute(text(modified_query), params or {})
            return [dict(row._mapping) for row in result]

    def restore_from_snapshot(self, snapshot_name: str, account: str = "sys") -> None:
        """Restore database state from a snapshot.

        Args:
            snapshot_name: Name of the snapshot to restore
            account: Account name (default: sys)

        Warning:
            This operation will restore the entire account state.
            All changes after the snapshot will be lost.
            
        Note:
            This is a heavy operation that affects the entire account.
            For testing, consider using query_snapshot() for read-only access.
        """
        with self._db() as db:
            db.commit()  # Commit before DDL
        
            # Validate inputs
            safe_snapshot = validate_identifier(snapshot_name)
            safe_account = validate_identifier(account)
        
            query = f"RESTORE ACCOUNT {safe_account} FROM SNAPSHOT {safe_snapshot}"
            db.execute(text(query))

    def restore_table_from_snapshot(self, table_name: str, snapshot_name: str) -> None:
        """Restore a single table from snapshot using time-travel queries.
        
        This is a lighter alternative to restore_from_snapshot() that only
        affects one table instead of the entire account.
        
        Args:
            table_name: Name of the table to restore
            snapshot_name: Name of the snapshot to restore from
        """
        # Validate inputs to prevent SQL injection
        with self._db() as db:
            safe_table = validate_identifier(table_name)
            safe_snapshot = validate_identifier(snapshot_name)
        
            db.commit()  # Ensure clean transaction state
        
            # Step 1: Get snapshot timestamp
            snapshots = self.list_snapshots()
            snapshot_info = next((s for s in snapshots if s["snapshot_name"] == safe_snapshot), None)
            if not snapshot_info:
                raise ValueError(f"Snapshot {safe_snapshot} not found")
        
            snapshot_ts = snapshot_info["timestamp"]
        
            try:
                # Step 2: Clear current table data
                db.execute(text(f"DELETE FROM {safe_table}"))
            
                # Step 3: Insert data from snapshot using time-travel query
                # Note: This uses MatrixOne's {SNAPSHOT = 'name'} syntax
                insert_query = f"""
                INSERT INTO {safe_table} 
                SELECT * FROM {safe_table} {{SNAPSHOT = '{safe_snapshot}'}}
                """
                db.execute(text(insert_query))
                db.commit()
            except Exception as e:
                db.rollback()
                raise

    def drop_snapshot(self, snapshot_name: str) -> None:
        """Delete a snapshot.

        Args:
            snapshot_name: Name of the snapshot to delete
        """
        with self._db() as db:
            db.commit()  # Commit before DDL
            query = f"DROP SNAPSHOT {snapshot_name}"
            db.execute(text(query))

    def get_snapshot_info(self, snapshot_name: str) -> dict | None:
        """Get information about a specific snapshot.

        Args:
            snapshot_name: Name of the snapshot

        Returns:
            Optional[dict]: Snapshot metadata if found, None otherwise
        """
        snapshots = self.list_snapshots()
        return next((s for s in snapshots if s["snapshot_name"] == snapshot_name), None)

    def create_time_point_sandbox(self, snapshot_name: str, description: str | None = None) -> dict:
        """Create a time-point sandbox for experimentation.

        This creates a snapshot that can be used for isolated experiments
        without affecting the main database state.

        Args:
            snapshot_name: Name for the sandbox snapshot (alphanumeric and underscore only)
            description: Optional description of the sandbox purpose

        Returns:
            dict: Sandbox metadata
        """
        # Sanitize snapshot name (remove special characters)
        sanitized_name = "".join(c if c.isalnum() or c == "_" else "_" for c in snapshot_name)
        snapshot = self.create_snapshot(sanitized_name)
        return {
            "snapshot_name": sanitized_name,
            "timestamp": snapshot.get("timestamp"),
            "description": description,
            "type": "sandbox",
        }

    def cleanup_old_snapshots(self, keep_count: int = 10) -> list[str]:
        """Clean up old snapshots, keeping only the most recent ones.

        Args:
            keep_count: Number of recent snapshots to keep

        Returns:
            list[str]: Names of deleted snapshots
        """
        snapshots = self.list_snapshots()

        # Sort by timestamp (newest first)
        snapshots.sort(key=lambda s: s["timestamp"] if s["timestamp"] else "", reverse=True)

        deleted = []
        for snapshot in snapshots[keep_count:]:
            snapshot_name = snapshot["snapshot_name"]
            try:
                self.drop_snapshot(snapshot_name)
                deleted.append(snapshot_name)
            except Exception:
                # Skip if deletion fails (e.g., snapshot in use)
                pass

        return deleted
