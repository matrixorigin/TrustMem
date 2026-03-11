"""Unit tests for branch-aware CRUD in EmbeddedBackend."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def backend():
    """EmbeddedBackend instance with __init__ bypassed and all deps mocked."""
    from mo_memory_mcp.server import EmbeddedBackend

    with patch.object(EmbeddedBackend, "__init__", lambda self, **kw: None):
        b = EmbeddedBackend.__new__(EmbeddedBackend)

    b._engine = None
    b._db_factory = MagicMock()
    b._embed_client = None
    b._embed_client_initialized = False
    b._embed_client_standalone = False
    b._create_service = MagicMock()
    b._create_editor = MagicMock()
    # Instance vars (normally set in __init__)
    b._active_branches = {}
    b._branch_factory_cache = {}
    b._cooldown_cache = {}
    return b


def _db_ctx(backend, fetchone_return=None, fetchall_return=None):
    """Wire backend._db_factory to return a context-manager mock.

    fetchone_return=None means the mock's fetchone() returns None (no row found).
    Pass a MagicMock row object to simulate a found row.
    """
    db = MagicMock()
    # Always set fetchone explicitly — leaving it as a MagicMock would be truthy
    # and cause "if row:" checks to pass unexpectedly.
    db.execute.return_value.fetchone.return_value = fetchone_return
    if fetchall_return is not None:
        db.execute.return_value.fetchall.return_value = fetchall_return
    session = MagicMock()
    session.__enter__ = MagicMock(return_value=db)
    session.__exit__ = MagicMock(return_value=False)
    backend._db_factory.return_value = session
    return db


# ---------------------------------------------------------------------------
# _get_active_branch / _set_active_branch
# ---------------------------------------------------------------------------

class TestActiveBranch:
    """Test _get/_set_active_branch with DB persistence."""

    def test_default_is_main_when_no_db_row(self, backend):
        _db_ctx(backend, fetchone_return=None)
        assert backend._get_active_branch("alice") == "main"

    def test_restores_from_db(self, backend):
        row = MagicMock()
        row.active_branch = "experiment"
        _db_ctx(backend, fetchone_return=row)
        assert backend._get_active_branch("alice") == "experiment"

    def test_caches_after_first_load(self, backend):
        row = MagicMock()
        row.active_branch = "exp"
        db = _db_ctx(backend, fetchone_return=row)

        result1 = backend._get_active_branch("alice")
        result2 = backend._get_active_branch("alice")

        assert result1 == result2 == "exp"
        # DB queried only once; second call served from in-memory cache.
        assert db.execute.call_count == 1

    def test_set_persists_to_db(self, backend):
        db = _db_ctx(backend)
        backend._set_active_branch("alice", "feature")

        assert backend._get_active_branch("alice") == "feature"
        db.execute.assert_called_once()
        db.commit.assert_called_once()

    def test_set_survives_db_failure(self, backend):
        session = MagicMock()
        session.__enter__ = MagicMock(side_effect=Exception("db down"))
        session.__exit__ = MagicMock(return_value=False)
        backend._db_factory.return_value = session

        backend._set_active_branch("alice", "feature")

        # In-memory updated even when DB write fails.
        assert backend._active_branches["alice"] == "feature"

    def test_set_logs_warning_on_db_failure(self, backend):
        session = MagicMock()
        session.__enter__ = MagicMock(side_effect=Exception("db down"))
        session.__exit__ = MagicMock(return_value=False)
        backend._db_factory.return_value = session

        with patch("mo_memory_mcp.server.logger") as mock_log:
            backend._set_active_branch("alice", "feature")
        mock_log.warning.assert_called_once()

    def test_get_logs_warning_on_db_failure(self, backend):
        session = MagicMock()
        session.__enter__ = MagicMock(side_effect=Exception("db down"))
        session.__exit__ = MagicMock(return_value=False)
        backend._db_factory.return_value = session

        with patch("mo_memory_mcp.server.logger") as mock_log:
            result = backend._get_active_branch("bob")
        assert result == "main"
        mock_log.warning.assert_called_once()

    def test_set_invalidates_factory_cache_for_old_branch(self, backend):
        backend._active_branches["alice"] = "old"
        mock_engine = MagicMock()
        backend._branch_factory_cache[("alice", "old")] = (mock_engine, MagicMock())
        _db_ctx(backend)

        backend._set_active_branch("alice", "new")

        assert ("alice", "old") not in backend._branch_factory_cache
        mock_engine.dispose.assert_called_once()


# ---------------------------------------------------------------------------
# _branch_db_factory
# ---------------------------------------------------------------------------

class TestBranchDbFactory:
    """Test _branch_db_factory returns correct factory and caches it."""

    def test_main_returns_original_factory(self, backend):
        backend._active_branches["alice"] = "main"
        assert backend._branch_db_factory("alice") is backend._db_factory

    def test_missing_branch_resets_to_main(self, backend):
        backend._active_branches["alice"] = "gone"
        # Both the branch lookup and the subsequent _set_active_branch DB write
        # go through _db_factory — wire it to return None for fetchone.
        _db_ctx(backend, fetchone_return=None)

        result = backend._branch_db_factory("alice")

        assert result is backend._db_factory
        assert backend._active_branches["alice"] == "main"

    def test_branch_builds_factory_for_correct_db(self, backend):
        """Factory returned for a branch must point to the branch database."""
        backend._active_branches["alice"] = "exp"
        row = MagicMock()
        row.branch_db = "mem_br_abc123"
        _db_ctx(backend, fetchone_return=row)

        mock_url = MagicMock()
        mock_url.set.return_value = mock_url  # set() returns a new URL object
        backend._source_engine_url = MagicMock(return_value=mock_url)

        with patch("mo_memory_mcp.server.create_engine") as mock_engine, \
             patch("mo_memory_mcp.server.sessionmaker") as mock_sm:
            result = backend._branch_db_factory("alice")

        # URL.set called with the branch database name
        mock_url.set.assert_called_once_with(database="mem_br_abc123")
        mock_engine.assert_called_once_with(mock_url, pool_pre_ping=True)
        assert result is mock_sm.return_value

    def test_cache_hit_skips_db_lookup(self, backend):
        backend._active_branches["alice"] = "exp"
        sentinel_factory = MagicMock(name="cached_factory")
        backend._branch_factory_cache[("alice", "exp")] = (MagicMock(), sentinel_factory)

        result = backend._branch_db_factory("alice")

        assert result is sentinel_factory
        backend._db_factory.return_value.__enter__.assert_not_called()

    def test_cache_populated_on_miss(self, backend):
        backend._active_branches["alice"] = "exp"
        row = MagicMock()
        row.branch_db = "mem_br_abc"
        _db_ctx(backend, fetchone_return=row)

        mock_url = MagicMock()
        mock_url.set.return_value = mock_url
        backend._source_engine_url = MagicMock(return_value=mock_url)

        with patch("mo_memory_mcp.server.create_engine") as mock_ce, \
             patch("mo_memory_mcp.server.sessionmaker") as mock_sm:
            result = backend._branch_db_factory("alice")

        # Cache stores (engine, factory) tuple; result is the factory
        cached = backend._branch_factory_cache[("alice", "exp")]
        assert cached == (mock_ce.return_value, mock_sm.return_value)
        assert result is mock_sm.return_value

    def test_second_call_uses_cache(self, backend):
        backend._active_branches["alice"] = "exp"
        row = MagicMock()
        row.branch_db = "mem_br_abc"
        db = _db_ctx(backend, fetchone_return=row)

        mock_url = MagicMock()
        mock_url.set.return_value = mock_url
        backend._source_engine_url = MagicMock(return_value=mock_url)

        with patch("mo_memory_mcp.server.create_engine"), \
             patch("mo_memory_mcp.server.sessionmaker"):
            backend._branch_db_factory("alice")
            backend._branch_db_factory("alice")

        # DB lookup only on first call
        assert db.execute.call_count == 1


# ---------------------------------------------------------------------------
# _source_engine_url
# ---------------------------------------------------------------------------

class TestSourceEngineUrl:
    """Test _source_engine_url for both init modes."""

    def test_standalone_mode_uses_self_engine(self, backend):
        mock_engine = MagicMock()
        backend._engine = mock_engine

        url = backend._source_engine_url()

        assert url is mock_engine.url

    def test_dev_mode_reads_session_local(self, backend):
        backend._engine = None
        mock_url = MagicMock()

        # SessionLocal is imported locally inside _source_engine_url from api.database
        with patch("api.database.SessionLocal") as mock_sl:
            mock_sl.kw = {"bind": MagicMock(url=mock_url)}
            url = backend._source_engine_url()

        assert url is mock_url


# ---------------------------------------------------------------------------
# store — branch field in return value
# ---------------------------------------------------------------------------

class TestBranchAwareStore:
    """store() must include the active branch name in its return value."""

    def test_store_returns_main_branch(self, backend):
        backend._active_branches["alice"] = "main"
        mock_mem = MagicMock()
        mock_mem.memory_id = "m1"
        mock_mem.content = "test"
        backend._create_editor.return_value.inject.return_value = mock_mem

        result = backend.store("alice", "test", "semantic", None)

        assert result["branch"] == "main"

    def test_store_returns_active_branch_name(self, backend):
        """When user is on a non-main branch, store() must report that branch."""
        backend._active_branches["alice"] = "experiment"
        # Inject a cached (engine, factory) so _branch_db_factory doesn't hit DB
        backend._branch_factory_cache[("alice", "experiment")] = (MagicMock(), backend._db_factory)
        mock_mem = MagicMock()
        mock_mem.memory_id = "m2"
        mock_mem.content = "hello"
        backend._create_editor.return_value.inject.return_value = mock_mem

        result = backend.store("alice", "hello", "semantic", None)

        assert result["branch"] == "experiment"
        assert result["memory_id"] == "m2"


# ---------------------------------------------------------------------------
# Lazy embedding client
# ---------------------------------------------------------------------------

class TestLazyEmbedClient:
    """_get_embed_client() must initialize once and cache the result."""

    def test_not_initialized_at_startup(self, backend):
        """embed_client_initialized starts False — model not loaded at startup."""
        assert backend._embed_client_initialized is False
        assert backend._embed_client is None

    def test_first_call_initializes_in_standalone_mode(self, backend):
        """First _get_embed_client() call triggers _make_embed_client() in standalone mode."""
        backend._embed_client_standalone = True
        mock_client = MagicMock()
        with patch.object(type(backend), "_make_embed_client", return_value=mock_client):
            result = backend._get_embed_client()
        assert result is mock_client
        assert backend._embed_client is mock_client
        assert backend._embed_client_initialized is True

    def test_second_call_skips_make(self, backend):
        """Subsequent calls return cached client without calling _make_embed_client again."""
        backend._embed_client_standalone = True
        mock_client = MagicMock()
        with patch.object(type(backend), "_make_embed_client", return_value=mock_client) as mock_make:
            backend._get_embed_client()
            backend._get_embed_client()
        mock_make.assert_called_once()

    def test_dev_mode_never_calls_make(self, backend):
        """In dev mode (standalone=False), _make_embed_client is never called."""
        backend._embed_client_standalone = False
        with patch.object(type(backend), "_make_embed_client") as mock_make:
            result = backend._get_embed_client()
        mock_make.assert_not_called()
        assert result is None

    def test_store_uses_lazy_embed_client(self, backend):
        """store() calls _get_embed_client(), not _embed_client directly."""
        backend._active_branches["u"] = "main"
        mock_mem = MagicMock()
        mock_mem.memory_id = "m1"
        mock_mem.content = "hi"
        backend._create_editor.return_value.inject.return_value = mock_mem
        mock_client = MagicMock()

        with patch.object(backend, "_get_embed_client", return_value=mock_client) as mock_get:
            backend.store("u", "hi", "semantic", None)

        mock_get.assert_called_once()
        backend._create_editor.assert_called_once_with(
            backend._db_factory, user_id="u", embed_client=mock_client
        )

    def test_retrieve_passes_query_embedding(self, backend):
        """retrieve() embeds the query and passes it to svc.retrieve()."""
        backend._active_branches["u"] = "main"
        mock_client = MagicMock()
        mock_client.embed.return_value = [0.1, 0.2, 0.3]
        mock_svc = MagicMock()
        mock_svc.retrieve.return_value = ([], None)
        backend._create_service.return_value = mock_svc

        with patch.object(backend, "_get_embed_client", return_value=mock_client):
            backend.retrieve("u", "test query", 5)

        mock_client.embed.assert_called_once_with("test query")
        call_kwargs = mock_svc.retrieve.call_args[1]
        assert call_kwargs["query_embedding"] == [0.1, 0.2, 0.3]

    def test_retrieve_falls_back_to_keyword_when_no_embed(self, backend):
        """retrieve() passes query_embedding=None when embed client unavailable."""
        backend._active_branches["u"] = "main"
        mock_svc = MagicMock()
        mock_svc.retrieve.return_value = ([], None)
        backend._create_service.return_value = mock_svc

        with patch.object(backend, "_get_embed_client", return_value=None):
            backend.retrieve("u", "test query", 5)

        call_kwargs = mock_svc.retrieve.call_args[1]
        assert call_kwargs["query_embedding"] is None
