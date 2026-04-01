"""
Embedding model wrapper.

Supports two providers:
- "local"  — sentence-transformers (paraphrase-multilingual-mpnet-base-v2)
             No API key required. Downloads ~420 MB on first use.
- "openai" — OpenAI text-embedding-3-small (requires OPENAI_API_KEY in .env)

Configure via settings.EMBEDDING_PROVIDER.

Usage:
    from rag.embeddings import get_embedder
    embedder = get_embedder()
    vectors = embedder.embed(["text one", "text two"])  # list[list[float]]
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from config.settings import settings

logger = logging.getLogger(__name__)


class BaseEmbedder(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns list of float vectors."""

    @abstractmethod
    def dimension(self) -> int:
        """Return the vector dimension produced by this model."""


class LocalEmbedder(BaseEmbedder):
    """sentence-transformers multilingual embedder — no API key needed."""

    def __init__(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers not installed. Run: pip install sentence-transformers"
            ) from exc

        logger.info("Loading local embedding model: %s", settings.EMBEDDING_MODEL_LOCAL)
        self._model = SentenceTransformer(settings.EMBEDDING_MODEL_LOCAL)
        self._dim = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, batch_size=settings.EMBEDDING_BATCH_SIZE,
                                     show_progress_bar=False, normalize_embeddings=True)
        return [v.tolist() for v in vectors]

    def dimension(self) -> int:
        return self._dim


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI text-embedding-3-small — requires OPENAI_API_KEY."""

    DIM = 1536

    def __init__(self) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("openai not installed. Run: pip install openai") from exc

        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set in .env")

        self._client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self._model = settings.EMBEDDING_MODEL_OPENAI
        logger.info("Using OpenAI embedding model: %s", self._model)

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(input=texts, model=self._model)
        return [item.embedding for item in response.data]

    def dimension(self) -> int:
        return self.DIM


_embedder_instance: BaseEmbedder | None = None


def get_embedder() -> BaseEmbedder:
    """Return a cached embedder instance (lazy-loaded on first call)."""
    global _embedder_instance
    if _embedder_instance is None:
        provider = settings.EMBEDDING_PROVIDER.lower()
        if provider == "local":
            _embedder_instance = LocalEmbedder()
        elif provider == "openai":
            _embedder_instance = OpenAIEmbedder()
        else:
            raise ValueError(f"Unknown EMBEDDING_PROVIDER: {provider!r}. Use 'local' or 'openai'.")
    return _embedder_instance
