"""Shared test fixtures for Memoria."""

import os
import pytest
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

TEST_EMBEDDING_DIM = 384


@pytest.fixture(scope="session")
def db_engine():
    """Create test database engine. Uses MEMORIA_* env vars or defaults."""
    from matrixone import Client as MoClient

    host = os.environ.get("MEMORIA_DB_HOST", "localhost")
    port = int(os.environ.get("MEMORIA_DB_PORT", "6001"))
    user = os.environ.get("MEMORIA_DB_USER", "root")
    password = os.environ.get("MEMORIA_DB_PASSWORD", "111")
    db_name = os.environ.get("MEMORIA_DB_NAME", "memoria_test")

    # Bootstrap: create DB if not exists
    bootstrap = MoClient(host=host, port=port, user=user, password=password,
                         database="mo_catalog", sql_log_mode="off")
    with bootstrap._engine.connect() as c:
        c.execute(text(f"CREATE DATABASE IF NOT EXISTS `{db_name}`"))
        c.execute(text("COMMIT"))
    bootstrap._engine.dispose()

    client = MoClient(host=host, port=port, user=user, password=password,
                      database=db_name, sql_log_mode="off")
    engine = client._engine

    # Create tables
    from memoria.schema import ensure_tables
    ensure_tables(engine, dim=384)

    yield engine
    engine.dispose()


@pytest.fixture(scope="session")
def db_factory(db_engine):
    """Session factory for tests."""
    factory = sessionmaker(bind=db_engine)
    return factory


@pytest.fixture
def db(db_factory):
    """Per-test database session with rollback."""
    session = db_factory()
    yield session
    session.rollback()
    session.close()


@pytest.fixture(scope="session")
def embed_client():
    """Embedding client for tests."""
    from memoria.core.embedding import EmbeddingClient, set_embedding_client
    client = EmbeddingClient(provider="local", model="all-MiniLM-L6-v2", dim=384)
    set_embedding_client(client)
    return client
