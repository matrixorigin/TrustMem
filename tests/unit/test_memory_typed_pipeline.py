"""Unit tests for typed memory pipeline."""

import json
from unittest.mock import MagicMock, patch

import pytest

from memoria.core.memory.tabular.typed_pipeline import run_typed_memory_pipeline, TypedPipelineResult
from memoria.core.memory.config import MemoryGovernanceConfig
from memoria.core.memory.types import Memory, MemoryType


@pytest.fixture
def mock_db():
    return MagicMock()


class TestTypedPipeline:
    def test_extracts_memories(self, mock_db):
        mock_llm = MagicMock()
        mock_llm.chat_with_tools.return_value = {"content": json.dumps([
            {"type": "profile", "content": "likes Go", "confidence": 0.9},
        ])}

        with patch("memoria.core.memory.tabular.typed_pipeline.MemoryStore") as MockStore:
            mock_store = MagicMock()
            mock_store.create.side_effect = lambda m: m
            mock_store.list_active.return_value = []
            MockStore.return_value = mock_store

            result = run_typed_memory_pipeline(
                db_factory=lambda: mock_db,
                user_id="u1",
                messages=[{"role": "user", "content": "I like Go"}],
                llm_client=mock_llm,
            )

        assert result.memories_extracted == 1

    def test_detects_profile_change(self, mock_db):
        mock_llm = MagicMock()
        mock_llm.chat_with_tools.return_value = {"content": json.dumps([
            {"type": "profile", "content": "prefers vim", "confidence": 0.9},
        ])}

        with patch("memoria.core.memory.tabular.typed_pipeline.MemoryStore") as MockStore:
            mock_store = MagicMock()
            mock_store.create.side_effect = lambda m: m
            mock_store.list_active.return_value = []
            MockStore.return_value = mock_store

            result = run_typed_memory_pipeline(
                db_factory=lambda: mock_db,
                user_id="u1",
                messages=[{"role": "user", "content": "I use vim"}],
                llm_client=mock_llm,
            )

        assert result.profile_changed is True

    def test_no_profile_change_for_semantic(self, mock_db):
        mock_llm = MagicMock()
        mock_llm.chat_with_tools.return_value = {"content": json.dumps([
            {"type": "semantic", "content": "discussed testing", "confidence": 0.7},
        ])}

        with patch("memoria.core.memory.tabular.typed_pipeline.MemoryStore") as MockStore:
            mock_store = MagicMock()
            mock_store.create.side_effect = lambda m: m
            mock_store.list_active.return_value = []
            MockStore.return_value = mock_store

            result = run_typed_memory_pipeline(
                db_factory=lambda: mock_db,
                user_id="u1",
                messages=[{"role": "user", "content": "test"}],
                llm_client=mock_llm,
            )

        assert result.profile_changed is False

    def test_uses_custom_config(self, mock_db):
        config = MemoryGovernanceConfig(contradiction_similarity_threshold=0.9)
        mock_llm = MagicMock()
        mock_llm.chat_with_tools.return_value = {"content": "[]"}

        with patch("memoria.core.memory.tabular.typed_pipeline.MemoryStore"):
            result = run_typed_memory_pipeline(
                db_factory=lambda: mock_db,
                user_id="u1",
                messages=[{"role": "user", "content": "test"}],
                llm_client=mock_llm,
                config=config,
            )

        assert result.memories_extracted == 0

    def test_handles_observer_error(self, mock_db):
        with patch("memoria.core.memory.tabular.typed_pipeline.TypedObserver") as MockObs:
            MockObs.side_effect = Exception("Observer failed")

            result = run_typed_memory_pipeline(
                db_factory=lambda: mock_db,
                user_id="u1",
                messages=[{"role": "user", "content": "test"}],
            )

        assert len(result.errors) > 0
        assert "observer" in result.errors[0]

    def test_no_reflector_in_result(self, mock_db):
        """Pipeline no longer has clusters_promoted — reflector removed."""
        result = TypedPipelineResult()
        assert not hasattr(result, "clusters_promoted")
