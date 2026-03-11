"""ORM model for mem_user_state — per-user persistent state (e.g. active branch)."""

from sqlalchemy import Column, String, func

from memoria.core.base import Base
from memoria.core.memory.models._sa_types import DateTime6


class MemoryUserState(Base):
    __tablename__ = "mem_user_state"

    user_id = Column(String(64), primary_key=True)
    active_branch = Column(String(100), nullable=False, server_default="main")
    updated_at = Column(DateTime6, nullable=False, server_default=func.now(), onupdate=func.now())
