"""Tests for NER abstraction layer (ner.py)."""

from memoria.core.memory.graph.entity_extractor import ExtractedEntity
from memoria.core.memory.graph.ner import NERBackend, RegexNERBackend, get_ner_backend


class TestRegexNERBackend:
    def test_name(self):
        assert RegexNERBackend().name == "regex"

    def test_delegates_to_lightweight(self):
        ner = RegexNERBackend()
        result = ner.extract("后端微服务用 Go 编写，使用 gin 框架")
        names = {e.name for e in result}
        assert "go" in names
        assert "gin" in names

    def test_returns_extracted_entity_list(self):
        ner = RegexNERBackend()
        result = ner.extract("Deploy with Docker")
        assert all(isinstance(e, ExtractedEntity) for e in result)


class TestGetNERBackend:
    def test_returns_regex_backend(self):
        assert isinstance(get_ner_backend(), RegexNERBackend)

    def test_is_abstract_base(self):
        assert issubclass(RegexNERBackend, NERBackend)
