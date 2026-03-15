"""Tests for memory router _get_service function."""

import pytest
from unittest.mock import MagicMock, patch


class TestGetService:
    """Tests for _get_service function."""

    @pytest.fixture(autouse=True)
    def reset_cache(self):
        """Reset module-level cache before each test."""
        # Import and reset the cache
        from memoria.api.routers import memory as memory_router

        memory_router._llm_client_cache = None
        memory_router._embed_client_cache = None
        yield
        # Clean up after test
        memory_router._llm_client_cache = None
        memory_router._embed_client_cache = None

    @pytest.fixture
    def mock_db_factory(self):
        """Create a mock database factory."""
        return MagicMock()

    def test_get_service_caches_llm_client(self, mock_db_factory):
        """Test that _get_service caches LLM client across calls."""
        from memoria.api.routers.memory import _get_service

        with (
            patch("memoria.core.llm.get_llm_client") as mock_get_llm,
            patch("memoria.core.embedding.get_embedding_client") as mock_get_embed,
            patch("memoria.core.memory.factory.create_memory_service") as mock_create,
        ):
            mock_llm = MagicMock()
            mock_get_llm.return_value = mock_llm
            mock_embed = MagicMock()
            mock_embed.embed = MagicMock(return_value=[0.1, 0.2])
            mock_get_embed.return_value = mock_embed
            mock_service = MagicMock()
            mock_create.return_value = mock_service

            # First call
            service1 = _get_service(mock_db_factory, "user1")
            assert mock_get_llm.call_count == 1

            # Second call - should use cached client
            service2 = _get_service(mock_db_factory, "user2")
            assert mock_get_llm.call_count == 1  # Not called again
            assert service1 is service2 is mock_service

    def test_get_service_caches_embedding_client(self, mock_db_factory):
        """Test that _get_service caches embedding client across calls."""
        from memoria.api.routers.memory import _get_service

        with (
            patch("memoria.core.llm.get_llm_client") as mock_get_llm,
            patch("memoria.core.embedding.get_embedding_client") as mock_get_embed,
            patch("memoria.core.memory.factory.create_memory_service") as mock_create,
        ):
            mock_get_llm.return_value = None
            mock_embed = MagicMock()
            mock_embed.embed = MagicMock(return_value=[0.1, 0.2])
            mock_get_embed.return_value = mock_embed
            mock_create.return_value = MagicMock()

            # First call
            _get_service(mock_db_factory, "user1")
            assert mock_get_embed.call_count == 1

            # Second call - should use cached client
            _get_service(mock_db_factory, "user2")
            assert mock_get_embed.call_count == 1  # Not called again

    def test_get_service_without_embedding(self, mock_db_factory):
        """Test _get_service handles missing embedding client gracefully."""
        from memoria.api.routers.memory import _get_service

        with (
            patch("memoria.core.llm.get_llm_client") as mock_get_llm,
            patch("memoria.core.embedding.get_embedding_client") as mock_get_embed,
            patch("memoria.core.memory.factory.create_memory_service") as mock_create,
        ):
            mock_get_llm.return_value = None
            mock_get_embed.side_effect = Exception("Embedding not configured")
            mock_service = MagicMock()
            mock_create.return_value = mock_service

            service = _get_service(mock_db_factory, "user1")

            assert service is mock_service
            # Verify embed_fn is None when embedding client fails
            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["embed_fn"] is None

    def test_get_service_passes_correct_params(self, mock_db_factory):
        """Test _get_service passes correct parameters to create_memory_service."""
        from memoria.api.routers.memory import _get_service

        with (
            patch("memoria.core.llm.get_llm_client") as mock_get_llm,
            patch("memoria.core.embedding.get_embedding_client") as mock_get_embed,
            patch("memoria.core.memory.factory.create_memory_service") as mock_create,
        ):
            mock_llm = MagicMock()
            mock_get_llm.return_value = mock_llm
            mock_embed = MagicMock()
            mock_embed.embed = MagicMock(return_value=[0.1, 0.2])
            mock_get_embed.return_value = mock_embed
            mock_create.return_value = MagicMock()

            _get_service(mock_db_factory, "test_user")

            # Verify create_memory_service was called with correct params
            call_args = mock_create.call_args
            assert call_args.kwargs["user_id"] == "test_user"
            assert call_args.kwargs["llm_client"] is mock_llm
            assert call_args.args[0] is mock_db_factory
            # embed_fn should be the embed method
            assert call_args.kwargs["embed_fn"] is mock_embed.embed

    def test_get_service_embed_fn_works(self, mock_db_factory):
        """Test that the embed_fn passed to service works correctly."""
        from memoria.api.routers.memory import _get_service

        with (
            patch("memoria.core.llm.get_llm_client") as mock_get_llm,
            patch("memoria.core.embedding.get_embedding_client") as mock_get_embed,
            patch("memoria.core.memory.factory.create_memory_service") as mock_create,
        ):
            mock_get_llm.return_value = None
            mock_embed = MagicMock()
            expected_embedding = [0.1, 0.2, 0.3]
            mock_embed.embed.return_value = expected_embedding
            mock_get_embed.return_value = mock_embed
            mock_create.return_value = MagicMock()

            _get_service(mock_db_factory, "user1")

            # Get the embed_fn that was passed
            embed_fn = mock_create.call_args.kwargs["embed_fn"]
            # Verify it works
            result = embed_fn("test text")
            assert result == expected_embedding
            mock_embed.embed.assert_called_once_with("test text")

    def test_get_service_with_none_llm(self, mock_db_factory):
        """Test _get_service works when LLM client is None."""
        from memoria.api.routers.memory import _get_service

        with (
            patch("memoria.core.llm.get_llm_client") as mock_get_llm,
            patch("memoria.core.embedding.get_embedding_client") as mock_get_embed,
            patch("memoria.core.memory.factory.create_memory_service") as mock_create,
        ):
            mock_get_llm.return_value = None
            mock_embed = MagicMock()
            mock_get_embed.return_value = mock_embed
            mock_service = MagicMock()
            mock_create.return_value = mock_service

            service = _get_service(mock_db_factory, "user1")

            assert service is mock_service
            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["llm_client"] is None
            assert call_kwargs["embed_fn"] is not None

    def test_get_service_with_both_clients_none(self, mock_db_factory):
        """Test _get_service works when both LLM and embedding are None."""
        from memoria.api.routers.memory import _get_service

        with (
            patch("memoria.core.llm.get_llm_client") as mock_get_llm,
            patch("memoria.core.embedding.get_embedding_client") as mock_get_embed,
            patch("memoria.core.memory.factory.create_memory_service") as mock_create,
        ):
            mock_get_llm.return_value = None
            mock_get_embed.side_effect = Exception("No embedding")
            mock_service = MagicMock()
            mock_create.return_value = mock_service

            service = _get_service(mock_db_factory, "user1")

            assert service is mock_service
            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["llm_client"] is None
            assert call_kwargs["embed_fn"] is None


class TestClearServiceCache:
    """Tests for _clear_service_cache function."""

    @pytest.fixture(autouse=True)
    def setup_cache(self):
        """Set up cache with mock values before each test."""
        from memoria.api.routers import memory as memory_router

        # Set up cache with mock values
        memory_router._llm_client_cache = MagicMock()
        memory_router._embed_client_cache = MagicMock()
        yield
        # Clean up
        memory_router._llm_client_cache = None
        memory_router._embed_client_cache = None

    def test_clear_service_cache_resets_llm_client(self):
        """Test _clear_service_cache resets LLM client cache."""
        from memoria.api.routers.memory import _clear_service_cache
        from memoria.api.routers import memory as memory_router

        # Verify cache is set
        assert memory_router._llm_client_cache is not None

        _clear_service_cache()

        # Verify cache is cleared
        assert memory_router._llm_client_cache is None

    def test_clear_service_cache_resets_embed_client(self):
        """Test _clear_service_cache resets embedding client cache."""
        from memoria.api.routers.memory import _clear_service_cache
        from memoria.api.routers import memory as memory_router

        # Verify cache is set
        assert memory_router._embed_client_cache is not None

        _clear_service_cache()

        # Verify cache is cleared
        assert memory_router._embed_client_cache is None

    def test_clear_service_cache_allows_recreation(self):
        """Test that after clearing, clients are recreated on next call."""
        from memoria.api.routers.memory import _get_service, _clear_service_cache
        from memoria.api.routers import memory as memory_router

        # Start with clean cache (fixture sets mock values, we need None)
        memory_router._llm_client_cache = None
        memory_router._embed_client_cache = None

        with (
            patch("memoria.core.llm.get_llm_client") as mock_get_llm,
            patch("memoria.core.embedding.get_embedding_client") as mock_get_embed,
            patch("memoria.core.memory.factory.create_memory_service") as mock_create,
        ):
            mock_llm1 = MagicMock()
            mock_llm2 = MagicMock()
            mock_get_llm.side_effect = [mock_llm1, mock_llm2]
            mock_embed = MagicMock()
            mock_embed.embed = MagicMock(return_value=[0.1, 0.2])
            mock_get_embed.return_value = mock_embed
            mock_create.return_value = MagicMock()

            # First call creates first client
            _get_service(MagicMock(), "user1")
            assert memory_router._llm_client_cache is mock_llm1
            assert mock_get_llm.call_count == 1

            # Clear cache
            _clear_service_cache()
            assert memory_router._llm_client_cache is None

            # Second call creates new client
            _get_service(MagicMock(), "user2")
            assert memory_router._llm_client_cache is mock_llm2
            assert mock_get_llm.call_count == 2
