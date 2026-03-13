"""Admin endpoints — user management, system stats. Cursor-based pagination."""

from fastapi import APIRouter, Depends
from sqlalchemy import func
from sqlalchemy.orm import Session

from memoria.api.database import get_db_session
from memoria.api.dependencies import require_admin
from memoria.api.models import ApiKey, SnapshotRegistry, User

router = APIRouter(tags=["admin"])


@router.get("/admin/stats")
def system_stats(
    _admin: str = Depends(require_admin),
    db: Session = Depends(get_db_session),
):
    """System-wide stats. Uses indexed COUNT for bounded tables, approximate for large ones."""
    from memoria.core.memory.models.memory import MemoryRecord as M

    total_users = (
        db.query(func.count(User.user_id)).filter(User.is_active == 1).scalar() or 0
    )
    total_memories = (
        db.query(func.count(M.memory_id)).filter(M.is_active > 0).scalar() or 0
    )
    total_snapshots = db.query(func.count(SnapshotRegistry.snapshot_name)).scalar() or 0
    return {
        "total_users": total_users,
        "total_memories": total_memories,
        "total_snapshots": total_snapshots,
    }


@router.get("/admin/users")
def list_users(
    cursor: str | None = None,
    limit: int = 100,
    _admin: str = Depends(require_admin),
    db: Session = Depends(get_db_session),
):
    """List users with cursor-based pagination. Pass last user_id as cursor for next page."""
    q = db.query(User.user_id, User.created_at).filter(User.is_active == 1)
    if cursor:
        q = q.filter(User.user_id > cursor)
    rows = q.order_by(User.user_id).limit(limit).all()
    next_cursor = rows[-1].user_id if len(rows) == limit else None
    return {
        "users": [
            {
                "user_id": r.user_id,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ],
        "next_cursor": next_cursor,
    }


@router.get("/admin/users/{user_id}/stats")
def user_stats(
    user_id: str,
    _admin: str = Depends(require_admin),
    db: Session = Depends(get_db_session),
):
    from memoria.core.memory.models.memory import MemoryRecord as M

    mem_count = (
        db.query(func.count(M.memory_id))
        .filter(M.user_id == user_id, M.is_active > 0)
        .scalar()
        or 0
    )
    snap_count = db.query(SnapshotRegistry).filter_by(user_id=user_id).count()
    key_count = db.query(ApiKey).filter_by(user_id=user_id, is_active=1).count()
    return {
        "user_id": user_id,
        "memory_count": mem_count,
        "snapshot_count": snap_count,
        "api_key_count": key_count,
    }


@router.get("/admin/users/{user_id}/keys")
def list_user_keys(
    user_id: str,
    _admin: str = Depends(require_admin),
    db: Session = Depends(get_db_session),
):
    """List all active API keys for a user (admin only)."""
    from memoria.api.routers.auth import _key_to_response

    rows = db.query(ApiKey).filter_by(user_id=user_id, is_active=1).all()
    return {"user_id": user_id, "keys": [_key_to_response(r) for r in rows]}


@router.delete("/admin/users/{user_id}/keys")
def revoke_all_user_keys(
    user_id: str,
    _admin: str = Depends(require_admin),
    db: Session = Depends(get_db_session),
):
    """Revoke all active API keys for a user (admin only)."""
    result = (
        db.query(ApiKey)
        .filter_by(user_id=user_id, is_active=1)
        .update({"is_active": 0})
    )
    db.commit()
    return {"user_id": user_id, "revoked": result}


@router.delete("/admin/users/{user_id}")
def delete_user(
    user_id: str,
    _admin: str = Depends(require_admin),
    db: Session = Depends(get_db_session),
):
    """Deactivate user and revoke all API keys."""
    db.query(User).filter_by(user_id=user_id).update({"is_active": 0})
    db.query(ApiKey).filter_by(user_id=user_id).update({"is_active": 0})
    db.commit()
    return {"status": "ok", "user_id": user_id}


@router.post("/admin/users/{user_id}/strategy")
def set_strategy(
    user_id: str,
    strategy: str = "vector:v1",
    _admin: str = Depends(require_admin),
):
    """Set retrieval strategy for a user. Used by benchmark to compare strategies."""
    from memoria.api.database import get_db_factory
    from memoria.core.memory.factory import switch_user_strategy

    result = switch_user_strategy(get_db_factory(), user_id, strategy)
    return {
        "user_id": user_id,
        "strategy": result.strategy_key,
        "previous": result.previous_key,
        "status": result.status,
    }


@router.post("/admin/governance/{user_id}/trigger")
def admin_trigger_governance(
    user_id: str,
    op: str = "governance",
    _admin: str = Depends(require_admin),
):
    """Admin triggers governance/consolidate/reflect/extract_entities for a user.

    Runs synchronously, skips all cooldowns.
    Used by benchmark executor for maturation phase.
    """
    valid_ops = ("governance", "consolidate", "reflect", "extract_entities")
    if op not in valid_ops:
        from fastapi import HTTPException

        raise HTTPException(
            status_code=400,
            detail=f"Invalid op. Must be one of: {', '.join(valid_ops)}",
        )

    from memoria.api.database import get_db_factory

    db_factory = get_db_factory()

    if op == "governance":
        from memoria.core.memory.tabular.governance import GovernanceScheduler

        r = GovernanceScheduler(db_factory).run_cycle(user_id)
        return {
            "op": op,
            "user_id": user_id,
            "result": {"quarantined": r.quarantined, "cleaned_stale": r.cleaned_stale},
        }

    if op == "extract_entities":
        from memoria.core.memory.strategy.activation_index import ActivationIndexManager

        mgr = ActivationIndexManager(db_factory)
        result = mgr.backfill(user_id)
        return {
            "op": op,
            "user_id": user_id,
            "result": {
                "processed": result.processed,
                "skipped": result.skipped,
                "errors": result.errors[:10],
            },
        }

    from memoria.core.memory.factory import create_memory_service

    svc = create_memory_service(db_factory, user_id=user_id)
    if op == "consolidate":
        result = svc.consolidate(user_id)
        return {"op": op, "user_id": user_id, "result": result}

    # reflect
    return {
        "op": op,
        "user_id": user_id,
        "result": "reflect requires LLM — use user endpoint",
    }
