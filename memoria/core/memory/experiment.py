"""MemoryExperiment — isolated memory experiments using Git-for-Data branching.

Lifecycle: create → mutate (via editor) → diff → evaluate → commit/discard.

Features:
- evaluate(): replay golden sessions against experiment branch (§7.3, §8.4)
- Optimistic locking on commit via base_snapshot timestamp (§7.4)
- TTL management with auto-expiry cleanup (§7.6)

See docs/design/memory/backend-management.md §7
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass, field
from datetime import datetime as dt
from datetime import timedelta, timezone
from typing import TYPE_CHECKING, Any

from sqlalchemy import func as sa_func
from sqlalchemy import text

from memoria.core.db_consumer import DbConsumer

if TYPE_CHECKING:
    from memoria.core.db_consumer import DbFactory
    from memoria.core.memory.service import MemoryService

logger = logging.getLogger(__name__)

# Tables to branch into experiment sandbox
_MEMORY_TABLES = [
    "mem_memories",
    "memory_graph_nodes",
    "memory_graph_edges",
]

# Max active experiments per user
DEFAULT_MAX_EXPERIMENTS = 3

# Default TTL in days
DEFAULT_TTL_DAYS = 7
MAX_TTL_DAYS = 30


@dataclass
class ExperimentInfo:
    """Metadata for a memory experiment."""

    experiment_id: str
    user_id: str
    name: str
    status: str
    branch_db: str
    base_snapshot: str | None = None
    description: str = ""
    strategy_key: str | None = None
    params_json: dict[str, Any] | None = None
    metrics_json: dict[str, Any] | None = None
    created_at: dt | None = None
    committed_at: dt | None = None
    expires_at: dt | None = None


@dataclass
class ExperimentDiff:
    """Structured diff between experiment branch and production."""

    table_diffs: list[dict[str, Any]] = field(default_factory=list)
    summary: str = ""


@dataclass
class EvalResult:
    """Result of evaluating an experiment against golden sessions."""

    sessions_tested: int = 0
    sessions_passed: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)
    replay_results: list[dict[str, Any]] = field(default_factory=list)


class ExperimentLimitError(Exception):
    """Raised when user exceeds max active experiments."""


class ExperimentConflictError(Exception):
    """Raised when production data changed since experiment branch point."""


def _model_to_info(row: Any) -> ExperimentInfo:
    """Convert a MemoryExperiment ORM instance to ExperimentInfo."""
    return ExperimentInfo(
        experiment_id=row.experiment_id,
        user_id=row.user_id,
        name=row.name,
        status=row.status,
        branch_db=row.branch_db,
        base_snapshot=row.base_snapshot,
        description=row.description or "",
        strategy_key=row.strategy_key,
        params_json=row.params_json,
        metrics_json=row.metrics_json,
        created_at=row.created_at,
        committed_at=row.committed_at,
        expires_at=row.expires_at,
    )


class MemoryExperimentManager(DbConsumer):
    """Manage isolated memory experiments using Git-for-Data branching.

    Each experiment creates a zero-copy branch of memory tables,
    allowing mutations without affecting production data.
    """

    def __init__(
        self,
        db_factory: DbFactory,
        source_db: str | None = None,
        replay_factory: Any | None = None,
    ) -> None:
        super().__init__(db_factory)
        if source_db is None:
            from config.settings import get_settings
            source_db = get_settings().matrixone_database
        self._source_db = source_db
        self._branch_engines: dict[str, Any] = {}  # experiment_id → engine
        # Optional: callable(db_factory) → object with .replay_session()
        # When None, _replay_sessions falls back to lazy import of ReplayService.
        self._replay_factory = replay_factory

    def __enter__(self) -> MemoryExperimentManager:
        return self

    def __exit__(self, *exc: object) -> None:
        self.dispose_engines()

    def create(
        self,
        user_id: str,
        name: str,
        *,
        description: str = "",
        strategy_key: str | None = None,
        params: dict[str, Any] | None = None,
        max_experiments: int = DEFAULT_MAX_EXPERIMENTS,
        ttl_days: int = DEFAULT_TTL_DAYS,
    ) -> ExperimentInfo:
        """Create an experiment: snapshot + branch memory tables.

        Args:
            user_id: Experiment owner.
            name: Human-readable experiment name.
            description: Optional description.
            strategy_key: Override strategy for this experiment.
            params: Strategy param overrides.
            max_experiments: Max active experiments per user.
            ttl_days: Days until auto-expiry (default 7, max 30).

        Returns:
            ExperimentInfo with branch_db and snapshot info.

        Raises:
            ExperimentLimitError: If user has too many active experiments.
        """
        from memoria.core.memory.models.memory_experiment import MemoryExperiment
        from memoria.core.memory.strategy.params import validate_strategy_params

        ttl_days = min(ttl_days, MAX_TTL_DAYS)

        active = self._count_active(user_id)
        if active >= max_experiments:
            raise ExperimentLimitError(
                f"User {user_id} has {active} active experiments "
                f"(max {max_experiments})"
            )

        # Validate params against strategy schema
        validated_params = validate_strategy_params(
            strategy_key or "vector:v1", params,
        )

        from memoria.core.utils.id_generator import generate_id

        exp_id = generate_id()
        branch_db = f"mem_exp_{exp_id}"
        snapshot_name = f"base_{exp_id}"

        snapshot_ok = self._create_snapshot(snapshot_name)
        self._create_branch(branch_db, snapshot_name if snapshot_ok else None)

        now = dt.now(timezone.utc)
        record = MemoryExperiment(
            experiment_id=exp_id,
            user_id=user_id,
            name=name,
            description=description,
            status="active",
            branch_db=branch_db,
            base_snapshot=snapshot_name if snapshot_ok else None,
            strategy_key=strategy_key,
            params_json=validated_params,
            expires_at=now + timedelta(days=ttl_days),
            created_by=user_id,
        )
        with self._db() as db:
            db.add(record)
            db.commit()

        info = self.get(exp_id)
        assert info is not None
        return info

    def get(self, experiment_id: str) -> ExperimentInfo | None:
        """Get experiment info by ID."""
        from memoria.core.memory.models.memory_experiment import MemoryExperiment

        with self._db() as db:
            row = db.query(MemoryExperiment).filter_by(
                experiment_id=experiment_id,
            ).first()
            if row is None:
                return None
            return _model_to_info(row)

    def list_active(self, user_id: str) -> list[ExperimentInfo]:
        """List active experiments for a user."""
        from memoria.core.memory.models.memory_experiment import MemoryExperiment

        with self._db() as db:
            rows = (
                db.query(MemoryExperiment)
                .filter_by(user_id=user_id, status="active")
                .order_by(MemoryExperiment.created_at.desc())
                .all()
            )
            return [_model_to_info(r) for r in rows]

    def get_service(
        self,
        experiment_id: str,
        *,
        llm_client: object | None = None,
        embed_fn: Any = None,
    ) -> MemoryService:
        """Get a MemoryService that reads/writes to the experiment branch.

        The returned service operates on the branch database tables,
        not production. All mutations are isolated.
        """
        info = self.get(experiment_id)
        if info is None:
            raise ValueError(f"Experiment {experiment_id} not found")
        if info.status != "active":
            raise ValueError(f"Experiment {experiment_id} is {info.status}, not active")

        branch_db_factory = self._make_branch_db_factory(info.branch_db, experiment_id)

        from memoria.core.memory.factory import create_memory_service

        return create_memory_service(
            branch_db_factory,
            strategy=info.strategy_key,
            params=info.params_json,
            llm_client=llm_client,
            embed_fn=embed_fn,
        )

    def diff(self, experiment_id: str) -> ExperimentDiff:
        """Diff experiment branch against production.

        Returns structured diff per memory table.
        """
        info = self.get(experiment_id)
        if info is None:
            raise ValueError(f"Experiment {experiment_id} not found")

        from memoria.core.sandbox.branch import Branch

        branch = Branch(self._db_factory, database=self._source_db)
        table_diffs: list[dict[str, Any]] = []

        for table in _MEMORY_TABLES:
            try:
                rows = branch.diff(
                    f"{info.branch_db}.{table}",
                    f"{self._source_db}.{table}",
                )
                if rows:
                    table_diffs.append({"table": table, "changes": rows})
            except Exception as e:
                logger.debug("Diff failed for %s: %s", table, e)

        return ExperimentDiff(table_diffs=table_diffs)

    # ── Evaluate ──────────────────────────────────────────────────────

    def evaluate(
        self,
        experiment_id: str,
        *,
        golden_session_ids: list[str] | None = None,
        golden_session_count: int = 50,
    ) -> EvalResult:
        """Replay golden sessions against experiment branch.

        Uses RegressionGate's golden session selection and replay
        infrastructure. Stores metrics on the experiment record.

        Args:
            experiment_id: Experiment to evaluate.
            golden_session_ids: Specific sessions to replay (optional).
            golden_session_count: Max golden sessions to auto-select.

        Returns:
            EvalResult with per-session results and aggregate metrics.
        """
        info = self.get(experiment_id)
        if info is None:
            raise ValueError(f"Experiment {experiment_id} not found")
        if info.status not in ("active", "evaluating"):
            raise ValueError(f"Experiment {experiment_id} is {info.status}")

        self._set_status(experiment_id, "evaluating")

        try:
            sessions = self._load_golden_sessions(
                golden_session_ids, golden_session_count,
            )
            if not sessions:
                result = EvalResult(metrics={"note": "no_golden_sessions"})
                self.update_metrics(experiment_id, result.metrics)
                self._set_status(experiment_id, "active")
                return result

            replay_results = self._replay_sessions(info, sessions)
            metrics = self._compute_eval_metrics(replay_results)
            result = EvalResult(
                sessions_tested=len(sessions),
                sessions_passed=sum(
                    1 for r in replay_results if r.get("successful", 0) > 0
                ),
                metrics=metrics,
                replay_results=replay_results,
            )

            self.update_metrics(experiment_id, {
                "sessions_tested": result.sessions_tested,
                "sessions_passed": result.sessions_passed,
                **metrics,
            })
            self._set_status(experiment_id, "active")
            return result

        except Exception:
            self._set_status(experiment_id, "active")
            raise

    # ── A/B Comparison ─────────────────────────────────────────────────

    def compare(
        self,
        experiment_id_a: str,
        experiment_id_b: str,
    ) -> dict[str, Any]:
        """Compare metrics of two experiments side-by-side.

        Both experiments must have been evaluated (metrics_json populated).

        Returns:
            Dict with metrics_a, metrics_b, and per-metric winner.
        """
        info_a = self.get(experiment_id_a)
        info_b = self.get(experiment_id_b)
        if info_a is None:
            raise ValueError(f"Experiment {experiment_id_a} not found")
        if info_b is None:
            raise ValueError(f"Experiment {experiment_id_b} not found")
        if not info_a.metrics_json:
            raise ValueError(f"Experiment {experiment_id_a} has no metrics (run evaluate first)")
        if not info_b.metrics_json:
            raise ValueError(f"Experiment {experiment_id_b} has no metrics (run evaluate first)")

        metrics_a = info_a.metrics_json
        metrics_b = info_b.metrics_json

        # Compare numeric metrics — higher is better for rates, lower for errors
        higher_better = {"pass_rate", "retrieval_precision_at_k", "retrieval_recall_at_k",
                         "response_quality_score", "profile_accuracy", "multi_hop_hit_rate"}
        lower_better = {"error_rate", "p50_retrieve_latency_ms", "p99_retrieve_latency_ms",
                        "avg_tokens_in_context"}

        comparison: dict[str, Any] = {}
        all_keys = set(metrics_a.keys()) | set(metrics_b.keys())
        for key in sorted(all_keys):
            val_a = metrics_a.get(key)
            val_b = metrics_b.get(key)
            if not isinstance(val_a, (int, float)) or not isinstance(val_b, (int, float)):
                comparison[key] = {"a": val_a, "b": val_b, "winner": None}
                continue
            if key in higher_better:
                winner = "a" if val_a > val_b else ("b" if val_b > val_a else "tie")
            elif key in lower_better:
                winner = "a" if val_a < val_b else ("b" if val_b < val_a else "tie")
            else:
                winner = None
            comparison[key] = {"a": val_a, "b": val_b, "winner": winner}

        return {
            "experiment_a": experiment_id_a,
            "experiment_b": experiment_id_b,
            "strategy_a": info_a.strategy_key,
            "strategy_b": info_b.strategy_key,
            "metrics_a": metrics_a,
            "metrics_b": metrics_b,
            "comparison": comparison,
        }

    # ── Commit with optimistic locking ────────────────────────────────

    def commit(self, experiment_id: str) -> None:
        """Merge experiment branch into production with optimistic locking.

        Compares base_snapshot timestamp against current production state.
        If production mem_memories changed since branch point, raises
        ExperimentConflictError — user must re-evaluate.

        Raises:
            ExperimentConflictError: If production changed since branch point.
        """
        from memoria.core.memory.models.memory_experiment import MemoryExperiment

        info = self.get(experiment_id)
        if info is None:
            raise ValueError(f"Experiment {experiment_id} not found")
        if info.status != "active":
            raise ValueError(f"Experiment {experiment_id} is {info.status}, not active")

        if info.base_snapshot:
            self._check_production_unchanged(info)

        from memoria.core.sandbox.branch import Branch

        branch = Branch(self._db_factory, database=self._source_db)
        for table in _MEMORY_TABLES:
            try:
                branch.merge(
                    f"{info.branch_db}.{table}",
                    f"{self._source_db}.{table}",
                    on_conflict="accept",
                )
            except Exception as e:
                if table == "mem_memories":
                    raise  # core table merge must not fail silently
                logger.debug("Merge skipped for %s: %s", table, e)

        with self._db() as db:
            db.query(MemoryExperiment).filter_by(
                experiment_id=experiment_id,
            ).update({"status": "committed", "committed_at": sa_func.now()})
            db.commit()

        self._dispose_engine(experiment_id)
        self._drop_branch_db(info.branch_db)

    def discard(self, experiment_id: str) -> None:
        """Discard experiment: drop branch DB, keep record for audit."""
        from memoria.core.memory.models.memory_experiment import MemoryExperiment

        info = self.get(experiment_id)
        if info is None:
            raise ValueError(f"Experiment {experiment_id} not found")
        if info.status not in ("active", "evaluating"):
            raise ValueError(f"Experiment {experiment_id} is {info.status}")

        with self._db() as db:
            db.query(MemoryExperiment).filter_by(
                experiment_id=experiment_id,
            ).update({"status": "discarded"})
            db.commit()

        self._dispose_engine(experiment_id)
        self._drop_branch_db(info.branch_db)

    # ── TTL management ────────────────────────────────────────────────

    def extend_ttl(self, experiment_id: str, days: int = DEFAULT_TTL_DAYS) -> None:
        """Extend experiment TTL. Total cannot exceed MAX_TTL_DAYS from creation.

        Args:
            experiment_id: Experiment to extend.
            days: Days to add (capped so total doesn't exceed MAX_TTL_DAYS).
        """
        info = self.get(experiment_id)
        if info is None:
            raise ValueError(f"Experiment {experiment_id} not found")
        if info.status != "active":
            raise ValueError(f"Experiment {experiment_id} is {info.status}")

        # Cap: expires_at cannot exceed created_at + MAX_TTL_DAYS
        # MatrixOne doesn't support LEAST(), use CASE WHEN
        with self._db() as db:
            db.execute(
                text(
                    "UPDATE mem_experiments SET expires_at = "
                    "CASE WHEN DATE_ADD(expires_at, INTERVAL :days DAY) "
                    "       > DATE_ADD(created_at, INTERVAL :max_days DAY) "
                    "     THEN DATE_ADD(created_at, INTERVAL :max_days DAY) "
                    "     ELSE DATE_ADD(expires_at, INTERVAL :days DAY) "
                    "END "
                    "WHERE experiment_id = :eid"
                ),
                {"eid": experiment_id, "days": days, "max_days": MAX_TTL_DAYS},
            )
            db.commit()

    def cleanup_expired(self, *, drop_snapshots: bool = False) -> int:
        """Expire and clean up experiments past their TTL.

        Sets status='expired', drops branch DBs.
        Optionally drops base snapshots (default: retained for audit per §7.6).
        Intended to be called by a daily governance job.

        Args:
            drop_snapshots: If True, also drop base_snapshot (saves storage).

        Returns:
            Number of experiments expired.
        """
        from memoria.core.memory.models.memory_experiment import MemoryExperiment

        with self._db() as db:
            rows = (
                db.query(MemoryExperiment)
                .filter(
                    MemoryExperiment.status == "active",
                    MemoryExperiment.expires_at.isnot(None),
                    MemoryExperiment.expires_at < sa_func.now(),
                )
                .all()
            )
            expired = [
                {
                    "experiment_id": r.experiment_id,
                    "branch_db": r.branch_db,
                    "base_snapshot": r.base_snapshot,
                }
                for r in rows
            ]

        count = 0
        for item in expired:
            exp_id = item["experiment_id"]
            try:
                self._set_status(exp_id, "expired")
                self._drop_branch_db(item["branch_db"])
                if drop_snapshots and item.get("base_snapshot"):
                    self._drop_snapshot(item["base_snapshot"])
                count += 1
            except Exception:
                logger.warning("Failed to expire experiment %s", exp_id, exc_info=True)

        return count

    # ── Metrics ───────────────────────────────────────────────────────

    def update_metrics(
        self, experiment_id: str, metrics: dict[str, Any],
    ) -> None:
        """Store evaluation metrics on the experiment record."""
        from memoria.core.memory.models.memory_experiment import MemoryExperiment

        with self._db() as db:
            db.query(MemoryExperiment).filter_by(
                experiment_id=experiment_id,
            ).update({"metrics_json": metrics})
            db.commit()

    # ── Internal helpers ──────────────────────────────────────────────

    def _set_status(self, experiment_id: str, status: str) -> None:
        from memoria.core.memory.models.memory_experiment import MemoryExperiment

        with self._db() as db:
            db.query(MemoryExperiment).filter_by(
                experiment_id=experiment_id,
            ).update({"status": status})
            db.commit()

    def _count_active(self, user_id: str) -> int:
        from memoria.core.memory.models.memory_experiment import MemoryExperiment

        with self._db() as db:
            return (
                db.query(sa_func.count())
                .select_from(MemoryExperiment)
                .filter_by(user_id=user_id, status="active")
                .scalar()
            ) or 0

    def _create_snapshot(self, name: str) -> bool:
        """Create account-level snapshot. Returns True on success."""
        try:
            from memoria.core.git_for_data import GitForData

            git = GitForData(self._db_factory)
            git.create_snapshot(name)
            return True
        except Exception:
            logger.warning("Failed to create experiment snapshot %s", name, exc_info=True)
            return False

    def _create_branch(self, branch_db: str, snapshot: str | None) -> None:
        """Create branch database with memory tables."""
        from memoria.core.sandbox.branch import Branch

        branch = Branch(self._db_factory, database=self._source_db)

        with self._db() as db:
            db.commit()
            db.execute(text(f"DROP DATABASE IF EXISTS `{branch_db}`"))
            db.commit()
            db.execute(text(f"CREATE DATABASE `{branch_db}`"))
            db.commit()

        for table in _MEMORY_TABLES:
            try:
                branch.create(
                    f"{branch_db}.{table}",
                    f"{self._source_db}.{table}",
                    snapshot=snapshot,
                )
            except Exception as e:
                if table == "mem_memories":
                    raise  # core table branch must not fail silently
                logger.debug("Branch table %s failed: %s", table, e)

    def _drop_branch_db(self, branch_db: str) -> None:
        """Drop branch database. Best-effort."""
        try:
            from memoria.core.sandbox.branch import Branch

            branch = Branch(self._db_factory, database=self._source_db)
            for table in _MEMORY_TABLES:
                with contextlib.suppress(Exception):
                    branch.delete(f"{branch_db}.{table}")

            with self._db() as db:
                db.commit()
                db.execute(text(f"DROP DATABASE IF EXISTS `{branch_db}`"))
                db.commit()
        except Exception:
            logger.warning("Failed to drop branch DB %s", branch_db, exc_info=True)

    def _drop_snapshot(self, name: str) -> None:
        """Drop a snapshot. Best-effort."""
        try:
            from memoria.core.git_for_data import GitForData

            git = GitForData(self._db_factory)
            git.drop_snapshot(name)
        except Exception:
            logger.debug("Failed to drop snapshot %s", name, exc_info=True)

    def _make_branch_db_factory(self, branch_db: str, experiment_id: str) -> DbFactory:
        """Create a db_factory that connects to the branch database.

        The engine is tracked and can be disposed via dispose_engines().
        """
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        from config.settings import get_settings

        settings = get_settings()
        url = (
            f"mysql+pymysql://{settings.matrixone_user}:{settings.matrixone_password}"
            f"@{settings.matrixone_host}:{settings.matrixone_port}/{branch_db}"
            "?charset=utf8mb4"
        )
        eng = create_engine(url, pool_pre_ping=True, pool_size=2)
        self._branch_engines[experiment_id] = eng
        factory = sessionmaker(bind=eng)
        return factory  # type: ignore[return-value]

    def dispose_engines(self) -> None:
        """Dispose all branch database engines to release connections."""
        for eng in self._branch_engines.values():
            with contextlib.suppress(Exception):
                eng.dispose()
        self._branch_engines.clear()

    def _dispose_engine(self, experiment_id: str) -> None:
        """Dispose a single branch engine if tracked."""
        eng = self._branch_engines.pop(experiment_id, None)
        if eng is not None:
            with contextlib.suppress(Exception):
                eng.dispose()

    def _check_production_unchanged(self, info: ExperimentInfo) -> None:
        """Optimistic lock: verify production hasn't changed since branch point.

        Compares snapshot timestamp against max(updated_at) in production
        mem_memories for this user. If production has newer writes,
        the experiment's assumptions may be stale.
        """
        from memoria.core.memory.models.memory import MemoryRecord
        from memoria.core.git_for_data import GitForData

        git = GitForData(self._db_factory)
        snap_info = git.get_snapshot_info(info.base_snapshot)  # type: ignore[arg-type]
        if snap_info is None:
            logger.warning(
                "Base snapshot %s not found, skipping optimistic lock",
                info.base_snapshot,
            )
            return

        snap_ts = snap_info.get("timestamp")
        if snap_ts is None:
            return

        with self._db() as db:
            cnt = (
                db.query(sa_func.count())
                .select_from(MemoryRecord)
                .filter(
                    MemoryRecord.user_id == info.user_id,
                    MemoryRecord.updated_at > snap_ts,
                )
                .scalar()
            ) or 0
            if cnt > 0:
                raise ExperimentConflictError(
                    f"Production has {cnt} memory changes since branch point "
                    f"({snap_ts}). Re-evaluate or discard the experiment."
                )

    def _load_golden_sessions(
        self,
        session_ids: list[str] | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Load golden sessions for evaluation.

        If session_ids provided, use those. Otherwise auto-select
        high-quality sessions via the same criteria as RegressionGate.

        Uses raw SQL to avoid depending on the Event ORM model (which
        lives in the orchestration layer).
        """
        if session_ids:
            placeholders = ",".join(f":s{i}" for i in range(len(session_ids)))
            params = {f"s{i}": sid for i, sid in enumerate(session_ids)}
            with self._db() as db:
                rows = db.execute(
                    text(
                        f"SELECT DISTINCT session_id, user_id"
                        f" FROM agent_events"
                        f" WHERE session_id IN ({placeholders})"
                    ),
                    params,
                ).fetchall()
                return [
                    {"session_id": r[0], "user_id": r[1], "avg_score": 0.0}
                    for r in rows
                ]

        cutoff = dt.now(timezone.utc) - timedelta(days=30)
        with self._db() as db:
            rows = db.execute(
                text(
                    "SELECT session_id, user_id, AVG(quality_score) AS avg_score"
                    " FROM agent_events"
                    " WHERE quality_score >= 4.0"
                    "   AND training_eligible = 1"
                    "   AND created_at > :cutoff"
                    " GROUP BY session_id, user_id"
                    " HAVING COUNT(*) >= 3"
                    " ORDER BY avg_score DESC"
                    " LIMIT :lim"
                ),
                {"cutoff": cutoff, "lim": limit},
            ).fetchall()
            return [
                {"session_id": r[0], "user_id": r[1],
                 "avg_score": float(r[2])}
                for r in rows
            ]

    def _replay_sessions(
        self,
        info: ExperimentInfo,
        sessions: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Replay sessions against experiment branch.

        Uses ReplayService to replay each session in the experiment's
        branch database context.

        Raises:
            ImportError: If ReplayService is not available.
        """
        results: list[dict[str, Any]] = []
        for session in sessions:
            try:
                with self._db() as db:
                    if self._replay_factory is not None:
                        replay_svc = self._replay_factory(lambda db=db: db)
                    else:
                        ReplayService = None  # not available in standalone Memoria
                        replay_svc = ReplayService(lambda db=db: db)
                    result = replay_svc.replay_session(
                        session_id=session["session_id"],
                        user_id=session["user_id"],
                        sandbox_name=info.branch_db,
                        mock_mode=True,
                    )
                    results.append({
                        "session_id": session["session_id"],
                        "original_score": session.get("avg_score", 0.0),
                        "replay_status": result.get("status", "unknown"),
                        "successful": result.get("result", {}).get("successful", 0),
                        "failed": result.get("result", {}).get("failed", 0),
                    })
            except Exception as e:
                results.append({
                    "session_id": session["session_id"],
                    "original_score": session.get("avg_score", 0.0),
                    "replay_status": "error",
                    "error": str(e),
                    "successful": 0,
                    "failed": 1,
                })

        return results

    @staticmethod
    def _compute_eval_metrics(
        replay_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Compute aggregate metrics from replay results."""
        total = len(replay_results)
        if total == 0:
            return {"sessions_tested": 0}

        passed = sum(1 for r in replay_results if r.get("successful", 0) > 0)
        failed = sum(1 for r in replay_results if r.get("replay_status") == "error")

        return {
            "sessions_tested": total,
            "pass_rate": passed / total if total else 0.0,
            "error_rate": failed / total if total else 0.0,
        }
