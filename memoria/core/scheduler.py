"""Memory governance scheduler — standalone version for Memoria.

Simplified from mo-dev-agent's core/context/scheduler.py.
Removes: knowledge governance, evaluation pipeline, input face learning.
Keeps: memory governance (hourly/daily/weekly) + distributed locking + asyncio backend.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import threading
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Callable

from sqlalchemy import text
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session

import logging

logger = logging.getLogger(__name__)

GOVERNANCE_TASKS: dict[str, dict[str, Any]] = {
    "hourly":  {"interval": 3600,   "lock_name": "governance_hourly"},
    "daily":   {"interval": 86400,  "lock_name": "governance_daily"},
    "weekly":  {"interval": 604800, "lock_name": "governance_weekly"},
}

LOCK_TTL = 300
HEARTBEAT_INTERVAL = 60


class SchedulerBackend(ABC):
    @abstractmethod
    async def start(self, tasks: dict[str, int]) -> None: ...
    @abstractmethod
    async def stop(self) -> None: ...


class GovernanceTaskRunner:
    def __init__(
        self,
        db_context_factory: Callable,
        db_factory: Callable | None = None,
        memory_only: bool = False,
    ):
        self._db_ctx = db_context_factory
        self._db_factory = db_factory
        self._memory_only = memory_only
        self._instance_id = f"{socket.gethostname()}:{os.getpid()}"

    def run(self, task_name: str) -> dict[str, int] | None:
        lock_name = GOVERNANCE_TASKS[task_name]["lock_name"]
        with self._db_ctx() as db:
            if not self._try_acquire(db, lock_name):
                return None

            stop_heartbeat = threading.Event()
            hb = threading.Thread(
                target=self._heartbeat_loop,
                args=(lock_name, self._instance_id, stop_heartbeat),
                daemon=True,
            )
            hb.start()
            try:
                result = self._dispatch(task_name, self._db_factory)
                self._persist_run(db, task_name, result)
                logger.info(f"Governance [{task_name}]: {result}")
                return result
            except Exception as e:
                logger.error(f"Governance [{task_name}] error: {e}")
                db.rollback()
                return None
            finally:
                stop_heartbeat.set()
                hb.join(timeout=5)
                self._release(db, lock_name)

    @staticmethod
    def _dispatch(task_name: str, db_factory: Callable) -> dict[str, int]:
        results: dict[str, int] = {}
        try:
            from memoria.core.memory import create_memory_service
            svc = create_memory_service(db_factory)
            if task_name == "hourly":
                r = svc.run_hourly()
                results["mem_cleaned_tool_results"] = r.cleaned_tool_results
                results["mem_archived_working"] = r.archived_working
            elif task_name == "daily":
                r = svc.run_daily_all()
                results["mem_cleaned_stale"] = r.cleaned_stale
                results["mem_quarantined"] = r.quarantined
            elif task_name == "weekly":
                r = svc.run_weekly()
                results["mem_cleaned_branches"] = r.cleaned_branches
                results["mem_cleaned_snapshots"] = r.cleaned_snapshots
        except Exception as e:
            logger.error("Memory governance [%s] failed: %s", task_name, e)
        return results

    def _try_acquire(self, db: Session, lock_name: str) -> bool:
        now = datetime.now()
        expires_at = now + timedelta(seconds=LOCK_TTL)
        try:
            db.execute(text(
                "INSERT INTO infra_distributed_locks (lock_name, instance_id, acquired_at, expires_at, task_name) "
                "VALUES (:name, :iid, :now, :exp, :task)"
            ), {"name": lock_name, "iid": self._instance_id, "now": now, "exp": expires_at, "task": lock_name.split("_", 1)[1]})
            db.commit()
            return True
        except (IntegrityError, OperationalError):
            db.rollback()
        result = db.execute(text(
            "UPDATE infra_distributed_locks SET instance_id = :iid, acquired_at = :now, expires_at = :exp "
            "WHERE lock_name = :name AND expires_at < :now"
        ), {"iid": self._instance_id, "now": now, "exp": expires_at, "name": lock_name})
        db.commit()
        return result.rowcount > 0

    @staticmethod
    def _release(db: Session, lock_name: str) -> None:
        try:
            db.execute(text("DELETE FROM infra_distributed_locks WHERE lock_name = :name"), {"name": lock_name})
            db.commit()
        except Exception as e:
            logger.error(f"Lock release failed: {e}")

    @staticmethod
    def _persist_run(db: Session, task_name: str, result: dict[str, int]) -> None:
        try:
            db.execute(text(
                "INSERT INTO governance_runs (task_name, result, created_at) VALUES (:task, :result, :ts)"
            ), {"task": task_name, "result": json.dumps(result), "ts": datetime.now()})
            db.commit()
        except Exception as e:
            logger.debug("governance_runs write skipped: %s", e)
            db.rollback()

    def _heartbeat_loop(self, lock_name: str, instance_id: str, stop: threading.Event):
        while not stop.wait(HEARTBEAT_INTERVAL):
            try:
                with self._db_ctx() as db:
                    db.execute(text(
                        "UPDATE infra_distributed_locks SET expires_at = :exp "
                        "WHERE lock_name = :name AND instance_id = :iid"
                    ), {"exp": datetime.now() + timedelta(seconds=LOCK_TTL), "name": lock_name, "iid": instance_id})
                    db.commit()
            except Exception:
                pass


class AsyncIOBackend(SchedulerBackend):
    def __init__(self, runner: GovernanceTaskRunner):
        self._runner = runner
        self._tasks: list[asyncio.Task] = []

    async def start(self, tasks: dict[str, int]) -> None:
        self._tasks = [asyncio.create_task(self._loop(n, i)) for n, i in tasks.items()]

    async def stop(self) -> None:
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def _loop(self, name: str, interval: int) -> None:
        while True:
            await asyncio.sleep(interval)
            try:
                self._runner.run(name)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Governance [{name}] failed: {e}")


class MemoryGovernanceScheduler:
    def __init__(self, backend: SchedulerBackend | None = None):
        self._enabled = os.environ.get("GOVERNANCE_ENABLED", "true").lower() == "true"
        self._backend = backend

    async def start(self) -> None:
        if not self._enabled or not self._backend:
            return
        tasks = {name: cfg["interval"] for name, cfg in GOVERNANCE_TASKS.items()}
        await self._backend.start(tasks)
        logger.info("Memory governance scheduler started")

    async def stop(self) -> None:
        if not self._enabled or not self._backend:
            return
        await self._backend.stop()
