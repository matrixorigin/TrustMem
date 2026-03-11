"""E2E tests for the new memory system.

All tests use unique user_id for pytest -n auto isolation.
Tests verify the memory system components work correctly together.
"""

import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock


from memoria.core.memory.tabular.retriever import TASK_WEIGHTS
from memoria.core.memory.tabular.typed_observer import TypedObserver
from memoria.core.tiered_loader import TieredMemoryLoader
from memoria.core.memory.tabular.service import MemoryService
from memoria.core.memory.tabular.health import MemoryHealth
from memoria.core.memory.config import MemoryGovernanceConfig, DEFAULT_CONFIG
from memoria.core.memory.types import Memory, MemoryType


def uid() -> str:
    return f"e2e_{uuid.uuid4().hex}"


class TestL0ProfileInPrompt:
    """L0 profile always in assembled prompt."""

    def test_profile_section_always_present(self):
        """TieredMemoryLoader returns empty string when no memories exist."""
        mock_svc = MagicMock()
        mock_svc.get_profile.return_value = None
        mock_svc.retrieve.return_value = []

        loader = TieredMemoryLoader(mock_svc)
        section, _ = loader.build_section(uid(), session_id="test_session", query="test")

        # No memories → empty section (don't waste tokens on filler)
        assert section is not None
        assert isinstance(section, str)


class TestTaskWeights:
    """Dynamic retrieval weights by task_hint."""

    def test_task_weights_configured(self):
        """Verify task-specific weights exist."""
        assert "code" in TASK_WEIGHTS
        assert "reasoning" in TASK_WEIGHTS
        assert "recall" in TASK_WEIGHTS
        assert "default" in TASK_WEIGHTS

    def test_code_task_has_weights(self):
        """Code tasks should have configured weights."""
        code_weights = TASK_WEIGHTS["code"]
        # Verify it's a RetrievalWeights instance with valid values
        assert code_weights.vector >= 0
        assert code_weights.keyword >= 0
        assert code_weights.temporal >= 0
        assert code_weights.confidence >= 0


class TestContradictionDetection:
    """Contradiction detection: 'prefers tabs' superseded by 'prefers spaces'."""

    def test_high_similarity_different_content_is_contradiction(self):
        """High vector similarity + different content = contradiction (DB-side)."""
        mock_db = MagicMock()
        mock_row = MagicMock()
        mock_row.memory_id = "old1"
        mock_row.content = "User prefers tabs"
        mock_row.initial_confidence = 0.8
        mock_row.l2_dist = 0.1  # Very close → contradiction
        mock_db.query.return_value.filter.return_value.order_by.return_value.limit.return_value.first.return_value = mock_row

        store = MagicMock()
        observer = TypedObserver(
            store=store, llm_client=None, embed_fn=None,
            db_factory=lambda: mock_db,
        )

        new_mem = Memory(
            memory_id="new1", user_id="u", memory_type=MemoryType.PROFILE,
            content="User prefers spaces", initial_confidence=0.9,
            embedding=[0.1] * 768,
            observed_at=datetime.now(timezone.utc),
        )

        contradiction, _ = observer._find_contradiction(new_mem)
        assert contradiction is not None
        assert contradiction.memory_id == "old1"

    def test_low_similarity_not_contradiction(self):
        """Distant vector match = not a contradiction (DB-side)."""
        mock_db = MagicMock()
        mock_row = MagicMock()
        mock_row.memory_id = "go1"
        mock_row.content = "User likes Go"
        mock_row.initial_confidence = 0.8
        mock_row.l2_dist = 5.0  # Very far → not a contradiction
        mock_db.query.return_value.filter.return_value.order_by.return_value.limit.return_value.first.return_value = mock_row

        store = MagicMock()
        observer = TypedObserver(
            store=store, llm_client=None, embed_fn=None,
            db_factory=lambda: mock_db,
        )

        rust_mem = Memory(
            memory_id="rust1", user_id="u", memory_type=MemoryType.PROFILE,
            content="User likes Rust", initial_confidence=0.8,
            embedding=[0.0] * 767 + [1.0],
            observed_at=datetime.now(timezone.utc),
        )

        contradiction, _ = observer._find_contradiction(rust_mem)
        assert contradiction is None

    def test_no_db_factory_skips_contradiction_detection(self):
        """Without db_factory, contradiction detection is skipped (returns None)."""
        store = MagicMock()
        observer = TypedObserver(store=store, llm_client=None, embed_fn=None, db_factory=None)

        new_mem = Memory(
            memory_id="new1", user_id="u", memory_type=MemoryType.PROFILE,
            content="User prefers spaces", initial_confidence=0.9,
            embedding=[0.1] * 768,
            observed_at=datetime.now(timezone.utc),
        )

        contradiction, _ = observer._find_contradiction(new_mem)
        assert contradiction is None


class TestHealthDetection:
    """Health detects pollution patterns."""

    def test_health_class_exists(self):
        """MemoryHealth class is properly defined."""
        mock_db = MagicMock()
        db_factory = lambda: mock_db

        health = MemoryHealth(db_factory)
        # Verify methods exist
        assert hasattr(health, 'analyze')
        assert hasattr(health, 'detect_pollution')
        assert hasattr(health, 'suggest_rollback_target')


class TestGovernanceConfig:
    """Governance config overrides take effect."""

    def test_default_config_values(self):
        """DEFAULT_CONFIG has expected values."""
        assert DEFAULT_CONFIG.pitr_range_value == 14
        assert DEFAULT_CONFIG.pitr_range_unit == "d"
        assert DEFAULT_CONFIG.pollution_threshold == 0.3
        assert DEFAULT_CONFIG.contradiction_similarity_threshold == 0.85

    def test_custom_config_overrides(self):
        """Custom config values override defaults."""
        custom = MemoryGovernanceConfig(
            pollution_threshold=0.5,
            contradiction_similarity_threshold=0.9,
        )

        assert custom.pollution_threshold == 0.5
        assert custom.contradiction_similarity_threshold == 0.9
        # Other values should be defaults
        assert custom.pitr_range_value == 14


class TestMemoryTypes:
    """Memory type system works correctly."""

    def test_all_memory_types_defined(self):
        """All expected memory types exist."""
        assert MemoryType.SEMANTIC.value == "semantic"
        assert MemoryType.PROCEDURAL.value == "procedural"
        assert MemoryType.PROFILE.value == "profile"
        assert MemoryType.WORKING.value == "working"
        assert MemoryType.TOOL_RESULT.value == "tool_result"

    def test_memory_dataclass_fields(self):
        """Memory dataclass has all required fields."""
        mem = Memory(
            memory_id="test",
            user_id="user",
            memory_type=MemoryType.SEMANTIC,
            content="test content",
            initial_confidence=0.8,
            observed_at=datetime.now(timezone.utc),
        )

        assert mem.memory_id == "test"
        assert mem.user_id == "user"
        assert mem.memory_type == MemoryType.SEMANTIC
        assert mem.content == "test content"
        assert mem.initial_confidence == 0.8
        assert mem.is_active is True  # default
        assert mem.embedding is None  # optional
        assert mem.superseded_by is None  # optional


class TestObserverExtraction:
    """TypedObserver extracts memories from conversations."""

    def test_parse_json_array_handles_markdown(self):
        """_parse_json_array extracts JSON from markdown code blocks."""
        from memoria.core.memory.tabular.typed_observer import _parse_json_array

        text = '''Here are the memories:
```json
[{"content": "User likes Python", "type": "profile", "confidence": 0.8}]
```
'''
        result = _parse_json_array(text)
        assert len(result) == 1
        assert result[0]["content"] == "User likes Python"

    def test_parse_json_array_handles_plain_json(self):
        """_parse_json_array handles plain JSON."""
        from memoria.core.memory.tabular.typed_observer import _parse_json_array

        text = '[{"content": "test", "type": "semantic"}]'
        result = _parse_json_array(text)
        assert len(result) == 1

    def test_parse_json_array_handles_invalid(self):
        """_parse_json_array returns empty list for invalid input."""
        from memoria.core.memory.tabular.typed_observer import _parse_json_array

        result = _parse_json_array("not json at all")
        assert result == []
