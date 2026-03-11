"""Health check endpoint."""

from fastapi import APIRouter
from sqlalchemy import text

from memoria.api.database import get_session_factory

router = APIRouter(tags=["health"])


@router.get("/health")
def health():
    try:
        db = get_session_factory()()
        db.execute(text("SELECT 1"))
        db.close()
        return {"status": "ok", "database": "connected"}
    except Exception as e:
        return {"status": "error", "database": str(e)}
