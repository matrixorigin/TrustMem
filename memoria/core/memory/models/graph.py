"""Graph SQLAlchemy models — nodes and edges for memory graph."""

from matrixone import VectorPrecision, VectorType
if not getattr(VectorType, "cache_ok", False):
    VectorType.cache_ok = True
from matrixone.sqlalchemy_ext import FulltextIndex, FulltextParserType
from sqlalchemy import Column, Float, Index, Integer, SmallInteger, String, Text
from sqlalchemy.sql import func

from memoria.core.base import Base
from memoria.core.memory.models._sa_types import EMBEDDING_DIM
from memoria.core.memory.models._sa_types import DateTime6


class GraphNode(Base):
    """Typed graph node.

    Three node types:
    - episodic: references agent_events (immutable)
    - semantic: references mem_memories (evolving)
    - scene: self-contained reflection insight (evolving)
    """

    __tablename__ = "memory_graph_nodes"
    __table_args__ = (
        FulltextIndex("ft_graph_content", ["content"], parser=FulltextParserType.NGRAM),
        Index("idx_graph_user_active", "user_id", "is_active", "node_type"),
        Index("idx_graph_event", "event_id"),
        Index("idx_graph_memory", "memory_id"),
        Index("idx_graph_conflicts", "user_id", "conflicts_with"),
    )

    node_id = Column(String(32), primary_key=True)
    user_id = Column(String(64), nullable=False)
    node_type = Column(String(10), nullable=False)

    content = Column(Text, nullable=False)
    embedding = Column(VectorType(EMBEDDING_DIM, VectorPrecision.F32))

    event_id = Column(String(32))
    memory_id = Column(String(64))
    session_id = Column(String(64))

    confidence = Column(Float, default=0.75)
    trust_tier = Column(String(4), default="T3")
    importance = Column(Float, nullable=False, default=0.0)

    source_nodes = Column(Text)  # "id1,id2,id3" (scene only)

    conflicts_with = Column(String(32))
    conflict_resolution = Column(String(10))

    access_count = Column(Integer, default=0)
    cross_session_count = Column(Integer, default=0)

    is_active = Column(SmallInteger, server_default="1", nullable=False)
    superseded_by = Column(String(32))
    created_at = Column(DateTime6, default=func.now(), nullable=False)


class GraphEdge(Base):
    """Directed edge between two graph nodes.

    Normalized edge table — enables DB-side multi-hop traversal
    without loading full graph into Python.

    Composite PK (source_id, target_id, edge_type) prevents duplicates.
    """

    __tablename__ = "memory_graph_edges"
    __table_args__ = (
        Index("idx_edge_target", "target_id"),
        Index("idx_edge_user", "user_id"),
    )

    source_id = Column(String(32), nullable=False, primary_key=True)
    target_id = Column(String(32), nullable=False, primary_key=True)
    edge_type = Column(String(15), nullable=False, primary_key=True)
    weight = Column(Float, nullable=False, default=1.0)
    user_id = Column(String(64), nullable=False)
