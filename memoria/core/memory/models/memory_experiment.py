"""ORM model for mem_experiments — isolated memory experiments."""

from sqlalchemy import Column, Index, String, Text, func

from memoria.core.base import Base
from memoria.core.memory.models._sa_types import DateTime6, NullableJSON


class MemoryExperiment(Base):
    """Isolated memory experiment using Git-for-Data branching.

    See docs/design/memory/backend-management.md §7
    """

    __tablename__ = "mem_experiments"

    experiment_id = Column(String(36), primary_key=True)
    user_id = Column(String(64), nullable=False)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    status = Column(String(20), nullable=False, server_default="active")
    branch_db = Column(String(64), nullable=False)
    base_snapshot = Column(String(64), nullable=True)
    strategy_key = Column(String(32), nullable=True)
    params_json = Column(NullableJSON, nullable=True)
    metrics_json = Column(NullableJSON, nullable=True)
    created_at = Column(DateTime6, nullable=False, server_default=func.now())
    committed_at = Column(DateTime6, nullable=True)
    expires_at = Column(DateTime6, nullable=True)
    created_by = Column(String(64), nullable=False)

    __table_args__ = (
        Index("idx_exp_user", "user_id"),
        Index("idx_exp_status", "status"),
        Index("idx_exp_user_status", "user_id", "status"),
        Index("idx_exp_status_expires", "status", "expires_at"),
    )
