"""Unit tests for ProfileManager — Task 5."""

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from memoria.core.memory.tabular.profile import ProfileManager, _DEFAULT_PROFILE
from memoria.core.memory.types import Memory, MemoryType


@pytest.fixture
def mock_store():
    store = MagicMock()
    store.list_active.return_value = []
    return store


@pytest.fixture
def manager(mock_store):
    return ProfileManager(store=mock_store, max_tokens=200)


def _profile_mem(content, initial_confidence=0.8):
    return Memory(
        memory_id=f"m{hash(content) % 1000}",
        user_id="u1",
        memory_type=MemoryType.PROFILE,
        content=content,
        initial_confidence=initial_confidence,
        observed_at=datetime(2026, 2, 26),
    )


class TestGetProfile:
    def test_returns_default_when_empty(self, manager, mock_store):
        mock_store.list_active.return_value = []
        assert manager.get_profile("u1") == _DEFAULT_PROFILE

    def test_synthesizes_from_memories(self, manager, mock_store):
        mock_store.list_active.return_value = [
            _profile_mem("prefers Go"),
            _profile_mem("uses vim"),
        ]
        profile = manager.get_profile("u1")
        assert "prefers Go" in profile
        assert "uses vim" in profile
        assert profile.startswith("User Profile:")

    def test_caches_result(self, manager, mock_store):
        mock_store.list_active.return_value = [_profile_mem("test")]
        manager.get_profile("u1")
        manager.get_profile("u1")
        # Should only call store once
        assert mock_store.list_active.call_count == 1

    def test_respects_token_limit(self, manager, mock_store):
        # Create many long memories
        mock_store.list_active.return_value = [
            _profile_mem("x" * 500) for _ in range(10)
        ]
        profile = manager.get_profile("u1")
        # Should be truncated
        assert len(profile) < 5000


class TestInvalidate:
    def test_invalidate_clears_cache(self, manager, mock_store):
        mock_store.list_active.return_value = [_profile_mem("v1")]
        manager.get_profile("u1")

        mock_store.list_active.return_value = [_profile_mem("v2")]
        manager.invalidate("u1")
        profile = manager.get_profile("u1")

        assert "v2" in profile
        assert mock_store.list_active.call_count == 2


class TestUpdateFromMemories:
    def test_invalidates_on_profile_memory(self, manager, mock_store):
        mock_store.list_active.return_value = [_profile_mem("old")]
        manager.get_profile("u1")

        new_mems = [_profile_mem("new")]
        result = manager.update_from_memories("u1", new_mems)

        assert result is True
        assert "u1" not in manager._cache

    def test_no_invalidate_on_non_profile(self, manager, mock_store):
        mock_store.list_active.return_value = [_profile_mem("old")]
        manager.get_profile("u1")

        episodic = Memory(
            memory_id="e1", user_id="u1", memory_type=MemoryType.SEMANTIC,
            content="event", initial_confidence=0.7,
        )
        result = manager.update_from_memories("u1", [episodic])

        assert result is False
        assert "u1" in manager._cache
