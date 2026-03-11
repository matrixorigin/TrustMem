"""ProfileManager — L0 profile synthesis and caching."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from memoria.core.memory.tabular.store import MemoryStore
from memoria.core.memory.types import Memory, MemoryType

logger = logging.getLogger(__name__)

_DEFAULT_PROFILE = ""


class ProfileManager:
    """Synthesize and cache user profile from profile-type memories."""

    def __init__(self, store: MemoryStore, max_tokens: int = 200):
        self.store = store
        self.max_tokens = max_tokens
        self._cache: dict[str, str] = {}

    def get_profile(self, user_id: str) -> str:
        """Get condensed profile string (~200 tokens).

        Returns cached version if available.
        """
        if user_id in self._cache:
            return self._cache[user_id]

        profile = self._synthesize(user_id)
        self._cache[user_id] = profile
        return profile

    def invalidate(self, user_id: str) -> None:
        """Invalidate cache when new profile memories are written."""
        self._cache.pop(user_id, None)

    def update_from_memories(self, user_id: str, new_memories: list[Memory]) -> bool:
        """Check if any new memories are profile type and invalidate cache.

        Returns True if cache was invalidated.
        """
        has_profile = any(m.memory_type == MemoryType.PROFILE for m in new_memories)
        if has_profile:
            self.invalidate(user_id)
        return has_profile

    def _synthesize(self, user_id: str) -> str:
        """Synthesize profile from all active profile memories."""
        memories = self.store.list_active(user_id, MemoryType.PROFILE)
        if not memories:
            return _DEFAULT_PROFILE

        # Sort by confidence (highest first), then by recency (newest first).
        # observed_at may be None — treat as epoch 0 so they sort last.
        _epoch = datetime(1970, 1, 1)
        memories.sort(key=lambda m: (-m.initial_confidence, -(m.observed_at or _epoch).timestamp()))

        lines = []
        total_chars = 0
        char_limit = self.max_tokens * 4  # rough estimate

        for m in memories:
            if total_chars + len(m.content) > char_limit:
                break
            lines.append(f"- {m.content}")
            total_chars += len(m.content) + 3

        if not lines:
            return _DEFAULT_PROFILE

        return "User Profile:\n" + "\n".join(lines)
