"""Memory statistics table — separated from main table for frequent updates."""

from sqlalchemy import Column, Integer, String, func

from memoria.core.base import Base
from memoria.core.memory.models._sa_types import DateTime6


class MemoryStats(Base):
    """Statistics for memories — access_count and last_accessed_at.

    Separated from mem_memories to reduce write contention on the main table.
    Updated on every retrieve() call, but writes go to this small table only.
    """

    __tablename__ = "mem_memories_stats"

    memory_id = Column(String(64), primary_key=True)
    access_count = Column(Integer, server_default="0", nullable=False, default=0)
    last_accessed_at = Column(DateTime6, default=func.now())
