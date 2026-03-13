"""SaaS API key authentication."""

import hmac
from datetime import datetime, timezone

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import update
from sqlalchemy.orm import Session
from sqlalchemy.sql import func

from memoria.api.database import get_db_session
from memoria.api.models import ApiKey
from memoria.config import get_settings

security = HTTPBearer()

ADMIN_USER_ID = "__admin__"


def _is_master_key(token: str) -> bool:
    """Check if token is the master key (timing-safe)."""
    settings = get_settings()
    return bool(settings.master_key and hmac.compare_digest(token, settings.master_key))


def get_current_user_id(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db_session),
) -> str:
    """Authenticate via API key, return user_id.

    Admin (master key) can impersonate a specific user via the
    ``X-Impersonate-User`` header.  Only master-key holders can use this;
    regular API keys ignore the header entirely.
    """
    token = credentials.credentials

    if _is_master_key(token):
        impersonate = request.headers.get("X-Impersonate-User")
        if impersonate:
            return impersonate
        return ADMIN_USER_ID

    key_hash = ApiKey.hash_key(token)
    row = (
        db.query(ApiKey.key_id, ApiKey.user_id, ApiKey.expires_at)
        .filter(
            ApiKey.key_hash == key_hash,
            ApiKey.is_active > 0,
        )
        .first()
    )
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key"
        )

    if row.expires_at:
        exp = (
            row.expires_at.replace(tzinfo=timezone.utc)
            if row.expires_at.tzinfo is None
            else row.expires_at
        )
        if exp < datetime.now(timezone.utc):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="API key expired"
            )

    try:
        db.execute(
            update(ApiKey)
            .where(ApiKey.key_id == row.key_id)
            .values(last_used_at=func.now())
        )
        db.commit()
    except Exception:
        db.rollback()

    return row.user_id


def require_admin(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    """Require admin (master key) access.

    Checks the raw token, NOT the impersonated user_id.
    This ensures admin endpoints remain accessible even when
    X-Impersonate-User is set.
    """
    if not _is_master_key(credentials.credentials):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required"
        )
    return ADMIN_USER_ID
