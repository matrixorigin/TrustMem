"""Embedding providers — pluggable backends for EmbeddingClient."""

import hashlib
from abc import ABC, abstractmethod


class BaseEmbeddingProvider(ABC):
    """Abstract base for embedding providers."""

    @abstractmethod
    def embed(self, text: str) -> list[float]: ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts. Default: sequential fallback."""
        return [self.embed(t) for t in texts]

    @abstractmethod
    def dimension(self) -> int: ...

    @abstractmethod
    def model_name(self) -> str: ...


class MockProvider(BaseEmbeddingProvider):
    """Deterministic hash-based embeddings. No semantic similarity."""

    def __init__(self, dim: int):
        self._dim = dim

    def embed(self, text: str) -> list[float]:
        h = hashlib.sha256(text.encode()).digest()
        vec = [(h[i] * 256 + h[i + 1]) / 65535.0 * 2 - 1 for i in range(0, len(h), 2)]
        while len(vec) < self._dim:
            vec.extend(vec[: self._dim - len(vec)])
        return vec[: self._dim]

    def dimension(self) -> int:
        return self._dim

    def model_name(self) -> str:
        return "mock"


# Module-level cache for sentence-transformers model (loaded once per process).
_local_model_cache: dict[str, object] = {}


class LocalProvider(BaseEmbeddingProvider):
    """Local sentence-transformers model."""

    def __init__(self, model: str, dim: int):
        self._model_name = model
        self._dim = dim
        self._model = self._load(model)
        actual = self._model.get_sentence_embedding_dimension()
        if actual != dim:
            raise ValueError(
                f"Model {model} produces {actual}-dim vectors but config says {dim}"
            )

    @staticmethod
    def _load(model: str):
        if model not in _local_model_cache:
            from sentence_transformers import SentenceTransformer
            _local_model_cache[model] = SentenceTransformer(model)
        return _local_model_cache[model]

    def embed(self, text: str) -> list[float]:
        return self._model.encode(text).tolist()

    def dimension(self) -> int:
        return self._dim

    def model_name(self) -> str:
        return self._model_name


class OpenAIProvider(BaseEmbeddingProvider):
    """OpenAI-compatible embedding API."""

    def __init__(self, api_key: str, model: str, dim: int, base_url: str | None = None):
        if not api_key:
            raise ValueError("OpenAI embedding provider requires api_key")
        import openai as _openai
        kwargs: dict = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = _openai.OpenAI(**kwargs)
        self._model_name = model
        self._dim = dim
        # Only OpenAI text-embedding-3-* supports the `dimensions` param.
        self._supports_dimensions = not base_url and "text-embedding-3" in model

    def _create_kwargs(self, input: str | list[str]) -> dict:
        kwargs: dict = {"model": self._model_name, "input": input}
        if self._supports_dimensions:
            kwargs["dimensions"] = self._dim
        return kwargs

    def embed(self, text: str) -> list[float]:
        resp = self._client.embeddings.create(**self._create_kwargs(text))
        return resp.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = self._client.embeddings.create(**self._create_kwargs(texts))
        # API returns embeddings in same order as input
        return [d.embedding for d in resp.data]

    def dimension(self) -> int:
        return self._dim

    def model_name(self) -> str:
        return self._model_name
