"""SaaS-specific ORM models.

NOTE: ApiKey mirrors api/models/auth.py but uses Memoria's own Base (separate DB/metadata).
This is intentional — Memoria runs as a standalone service with its own schema.
"""

import hashlib
import secrets

from sqlalchemy import Column, SmallInteger, String, Text
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.sql import func

from memoria.api._model_types import DateTime6


class Base(DeclarativeBase):
    pass


class User(Base):
    """Registered user — upserted when API key is created."""

    __tablename__ = "tm_users"

    user_id = Column(String(64), primary_key=True)
    display_name = Column(String(100), nullable=True)
    is_active = Column(SmallInteger, default=1, server_default="1", nullable=False)
    created_at = Column(DateTime6, default=func.now(), nullable=False)


class ApiKey(Base):
    __tablename__ = "auth_api_keys"

    key_id = Column(String(36), primary_key=True)
    user_id = Column(String(64), nullable=False, index=True)
    key_hash = Column(String(255), nullable=False, unique=True)
    key_prefix = Column(String(12), nullable=False, index=True)
    name = Column(String(100), nullable=False)
    is_active = Column(SmallInteger, default=1, server_default="1", nullable=False)
    created_at = Column(DateTime6, default=func.now(), nullable=False)
    expires_at = Column(DateTime6, nullable=True)
    last_used_at = Column(DateTime6, nullable=True)

    @staticmethod
    def generate_key() -> tuple[str, str, str]:
        raw = "sk-" + secrets.token_urlsafe(32)
        h = hashlib.sha256(raw.encode()).hexdigest()
        return raw, h, raw[:12]

    @staticmethod
    def hash_key(raw: str) -> str:
        return hashlib.sha256(raw.encode()).hexdigest()


class SnapshotRegistry(Base):
    """Tracks MatrixOne native snapshots per user for quota enforcement."""

    __tablename__ = "mem_snapshot_registry"

    snapshot_name = Column(String(200), primary_key=True)
    user_id = Column(String(64), nullable=False, index=True)
    display_name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime6, default=func.now(), nullable=False)
