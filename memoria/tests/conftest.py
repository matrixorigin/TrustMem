"""Memoria test configuration."""

import os
import pytest


def pytest_configure(config):
    """Fail fast if local embedding is configured in CI."""
    ci = os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS")
    provider = os.environ.get("MEMORIA_EMBEDDING_PROVIDER", "local")
    if ci and provider == "local":
        pytest.exit(
            "CI environment detected but MEMORIA_EMBEDDING_PROVIDER=local. "
            "Set MEMORIA_EMBEDDING_PROVIDER=openai and MEMORIA_EMBEDDING_API_KEY secret.",
            returncode=1,
        )


def pytest_collection_modifyitems(items):
    """Force governance tests to run in the same xdist group."""
    for item in items:
        if "Governance" in item.nodeid or "AdminStatsAccuracy" in item.nodeid:
            item.add_marker(pytest.mark.xdist_group("governance"))


@pytest.fixture(scope="session", autouse=True)
def isolated_test_db(request):
    """Each xdist worker gets its own isolated test database.

    worker_id is 'master' when not using xdist, 'gw0'/'gw1'/... with -n auto.
    Database is dropped after the session to avoid accumulation.
    """
    worker_id = getattr(request.config, "workerinput", {}).get("workerid", "master")
    db_name = f"memoria_test_{worker_id}"
    os.environ["MEMORIA_DB_NAME"] = db_name

    # Reset module-level singletons so they pick up the new DB name
    import memoria.api.database as _db_mod
    import memoria.config as _cfg_mod
    _db_mod._engine = None
    _db_mod._SessionLocal = None
    _cfg_mod._settings = None

    yield db_name

    # Teardown: drop the test database
    try:
        from sqlalchemy import text
        from matrixone import Client as MoClient
        from memoria.config import get_settings
        s = get_settings()
        bootstrap = MoClient(
            host=s.db_host, port=s.db_port,
            user=s.db_user, password=s.db_password,
            database="mo_catalog", sql_log_mode="off",
        )
        with bootstrap._engine.connect() as c:
            c.execute(text(f"DROP DATABASE IF EXISTS `{db_name}`"))
            c.execute(text("COMMIT"))
        bootstrap._engine.dispose()
    except Exception:
        pass  # best-effort cleanup
