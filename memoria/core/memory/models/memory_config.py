"""Per-user memory strategy configuration.

See docs/design/memory/backend-management.md §4.1
"""

from sqlalchemy import Column, String
from sqlalchemy.sql import func

from memoria.core.base import Base
from memoria.core.memory.models._sa_types import DateTime6
from memoria.core.memory.models._sa_types import NullableJSON as JSON


class MemoryUserConfig(Base):
    """Per-user memory retrieval strategy binding."""

    __tablename__ = "mem_user_memory_config"

    user_id = Column(String(64), primary_key=True)
    strategy_key = Column(String(32), nullable=False, server_default="vector:v1")
    params_json = Column(JSON, nullable=True)
    migrated_from = Column(String(32), nullable=True)
    migration_snapshot = Column(String(64), nullable=True)
    index_status = Column(String(20), nullable=False, server_default="ready")
    created_at = Column(DateTime6, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime6, server_default=func.now(), onupdate=func.now())
