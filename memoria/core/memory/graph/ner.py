"""Named Entity Recognition abstraction layer.

Currently only the regex backend is used at runtime. The ABC is kept so
alternative backends (spaCy, LLM) can be plugged in later without changing
call sites.

Usage:
    from memoria.core.memory.graph.ner import get_ner_backend
    entities = get_ner_backend().extract("Alice 负责 auth-service 模块")
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from memoria.core.memory.graph.entity_extractor import (
    ExtractedEntity,
    extract_entities_lightweight,
)


class NERBackend(ABC):
    """Abstract NER backend."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def extract(self, text: str) -> list[ExtractedEntity]: ...


class RegexNERBackend(NERBackend):
    """Zero-dependency regex/heuristic backend (default)."""

    @property
    def name(self) -> str:
        return "regex"

    def extract(self, text: str) -> list[ExtractedEntity]:
        return extract_entities_lightweight(text)


def get_ner_backend() -> NERBackend:
    """Return the default NER backend."""
    return RegexNERBackend()
