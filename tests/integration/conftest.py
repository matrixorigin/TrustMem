"""Integration test fixtures — uses Memoria's own database."""

import os
import pytest
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

# Default to memoria_test DB, override via env
os.environ.setdefault("MEMORIA_DB_HOST", "localhost")
os.environ.setdefault("MEMORIA_DB_PORT", "6001")
os.environ.setdefault("MEMORIA_DB_NAME", "memoria_test")

_engine = None
_SessionLocal = None


def _get_engine():
    global _engine
    if _engine is not None:
        return _engine
    from matrixone import Client as MoClient
    host = os.environ["MEMORIA_DB_HOST"]
    port = int(os.environ["MEMORIA_DB_PORT"])
    user = os.environ.get("MEMORIA_DB_USER", "root")
    password = os.environ.get("MEMORIA_DB_PASSWORD", "111")
    db_name = os.environ["MEMORIA_DB_NAME"]

    bootstrap = MoClient(host=host, port=port, user=user, password=password,
                         database="mo_catalog", sql_log_mode="off")
    with bootstrap._engine.connect() as c:
        c.execute(text(f"CREATE DATABASE IF NOT EXISTS `{db_name}`"))
        c.execute(text("COMMIT"))
    bootstrap._engine.dispose()

    client = MoClient(host=host, port=port, user=user, password=password,
                      database=db_name, sql_log_mode="off")
    _engine = client._engine
    return _engine


def _get_session_local():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=_get_engine())
    return _SessionLocal


def _init_tables():
    """Create all tables with correct embedding dim."""
    engine = _get_engine()
    from memoria.schema import ensure_tables
    dim = int(os.environ.get("MEMORIA_EMBEDDING_DIM", "384"))
    ensure_tables(engine, dim=dim, force=True)

    # Also create graph tables if not exist
    from memoria.core.base import Base
    Base.metadata.create_all(bind=engine, checkfirst=True)


# Run once per session
_tables_initialized = False


@pytest.fixture(scope="session", autouse=True)
def _init_db():
    global _tables_initialized
    if not _tables_initialized:
        _init_tables()
        _tables_initialized = True


@pytest.fixture(scope="session")
def db_factory():
    return _get_session_local()


@pytest.fixture
def db(db_factory):
    session = db_factory()
    yield session
    session.rollback()
    session.close()
