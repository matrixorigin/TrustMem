"""Memory edit audit log model.

See docs/design/memory/backend-management.md §10
"""

from sqlalchemy import Column, Index, String, Text
from sqlalchemy.sql import func

from memoria.core.base import Base
from memoria.core.memory.models._sa_types import DateTime6
from memoria.core.memory.models._sa_types import NullableJSON as JSON


class MemoryEditLog(Base):
    """Audit log for memory injection, correction, and purge operations."""

    __tablename__ = "mem_edit_log"
    __table_args__ = (
        Index("idx_edit_user", "user_id"),
        Index("idx_edit_operation", "operation"),
    )

    edit_id = Column(String(36), primary_key=True)
    user_id = Column(String(64), nullable=False)
    operation = Column(String(20), nullable=False)  # inject | correct | purge
    target_ids = Column(JSON, nullable=True)
    reason = Column(Text, nullable=True)
    snapshot_before = Column(String(64), nullable=True)
    created_at = Column(DateTime6, server_default=func.now(), nullable=False)
    created_by = Column(String(64), nullable=False)
