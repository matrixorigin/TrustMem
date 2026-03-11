"""SaaS API key authentication."""

import hmac
from datetime import datetime, timezone

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import update
from sqlalchemy.orm import Session
from sqlalchemy.sql import func

from memoria.api.database import get_db_session
from memoria.api.models import ApiKey
from memoria.config import get_settings

security = HTTPBearer()


def get_current_user_id(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db_session),
) -> str:
    """Authenticate via API key, return user_id."""
    token = credentials.credentials
    settings = get_settings()

    # Master key → admin user (timing-safe comparison)
    if settings.master_key and hmac.compare_digest(token, settings.master_key):
        return "__admin__"

    key_hash = ApiKey.hash_key(token)
    row = db.query(ApiKey.key_id, ApiKey.user_id, ApiKey.expires_at).filter_by(key_hash=key_hash, is_active=1).first()
    if row is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")

    if row.expires_at:
        exp = row.expires_at.replace(tzinfo=timezone.utc) if row.expires_at.tzinfo is None else row.expires_at
        if exp < datetime.now(timezone.utc):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="API key expired")

    # Update last_used_at
    try:
        db.execute(
            update(ApiKey).where(ApiKey.key_id == row.key_id).values(last_used_at=func.now())
        )
        db.commit()
    except Exception:
        db.rollback()

    return row.user_id


def require_admin(user_id: str = Depends(get_current_user_id)) -> str:
    """Require admin (master key) access."""
    if user_id != "__admin__":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")
    return user_id
