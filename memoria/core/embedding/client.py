"""EmbeddingClient — unified embedding interface.

Provider is determined by config. No runtime fallback — misconfigured = fail fast.
"""

from memoria.core.embedding.providers import (
    BaseEmbeddingProvider,
    LocalProvider,
    MockProvider,
    OpenAIProvider,
)

# Known fixed-dimension models. Used to catch misconfiguration early.
# Models with variable dimensions (e.g. text-embedding-3-* via `dimensions` param) are NOT listed.
KNOWN_DIMENSIONS: dict[str, int] = {
    # BGE family (BAAI)
    "BAAI/bge-m3": 1024,
    "BAAI/bge-large-en-v1.5": 1024,
    "BAAI/bge-large-zh-v1.5": 1024,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-base-zh-v1.5": 768,
    "BAAI/bge-small-en-v1.5": 512,
    "BAAI/bge-small-zh-v1.5": 512,
    # Sentence-transformers
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-MiniLM-L12-v2": 384,
    "sentence-transformers/all-mpnet-base-v2": 768,
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": 384,
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": 768,
    # OpenAI fixed-dim models (ada-002 has no `dimensions` param)
    "text-embedding-ada-002": 1536,
    # Cohere
    "embed-english-v3.0": 1024,
    "embed-multilingual-v3.0": 1024,
    "embed-english-light-v3.0": 384,
    "embed-multilingual-light-v3.0": 384,
    # Jina
    "jina-embeddings-v2-base-en": 768,
    "jina-embeddings-v3": 1024,
    # Nomic
    "nomic-embed-text-v1": 768,
    "nomic-embed-text-v1.5": 768,
}


class EmbeddingClient:
    """Unified embedding client. Hides local vs API difference."""

    def __init__(self, provider: str, model: str, dim: int, **kwargs):
        expected = KNOWN_DIMENSIONS.get(model)
        if expected is not None and dim != expected:
            raise ValueError(
                f"Model {model!r} has fixed dimension {expected}, but config says {dim}"
            )
        self._provider: BaseEmbeddingProvider
        if provider == "mock":
            self._provider = MockProvider(dim)
        elif provider == "local":
            self._provider = LocalProvider(model, dim)
        elif provider == "openai":
            self._provider = OpenAIProvider(
                api_key=kwargs.get("api_key", ""),
                model=model,
                dim=dim,
                base_url=kwargs.get("base_url"),
            )
        else:
            raise ValueError(f"Unknown embedding provider: {provider!r}")

    def embed(self, text: str) -> list[float]:
        return self._provider.embed(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in one call (native batch if provider supports it)."""
        return self._provider.embed_batch(texts)

    @property
    def dimension(self) -> int:
        return self._provider.dimension()

    @property
    def model_name(self) -> str:
        return self._provider.model_name()
