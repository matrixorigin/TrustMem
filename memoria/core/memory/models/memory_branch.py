"""ORM model for mem_branches — long-lived memory branches."""

from sqlalchemy import Column, Index, String, func

from memoria.core.base import Base
from memoria.core.memory.models._sa_types import DateTime6


class MemoryBranch(Base):
    __tablename__ = "mem_branches"

    branch_id = Column(String(36), primary_key=True)
    user_id = Column(String(64), nullable=False)
    name = Column(String(100), nullable=False)
    branch_db = Column(String(64), nullable=False)
    base_snapshot = Column(String(128), nullable=True)
    status = Column(String(20), nullable=False, server_default="active")
    created_at = Column(DateTime6, nullable=False, server_default=func.now())
    updated_at = Column(DateTime6, nullable=True)

    __table_args__ = (
        Index("idx_branch_user", "user_id"),
        Index("idx_branch_user_status", "user_id", "status"),
        Index("idx_branch_user_name", "user_id", "name", unique=True),
    )
