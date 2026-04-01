"""
Vector similarity retriever for the Qdrant collection.

Usage:
    from rag.retriever import Retriever

    retriever = Retriever()
    results = retriever.search("дозировка магния при преэклампсии", top_k=5)
    for r in results:
        print(r.score, r.metadata["source_url"])
        print(r.text[:300])
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from config.settings import settings
from rag.embeddings import get_embedder

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict = field(default_factory=dict)

    @property
    def source_url(self) -> str:
        return self.metadata.get("source_url", "")

    @property
    def protocol_name(self) -> str:
        # Extract from context header: "[Protocol: NAME, YEAR]"
        import re
        m = re.search(r"\[Protocol:\s*(.+?),\s*\d+\]", self.text)
        return m.group(1).strip() if m else self.metadata.get("document_id", "")

    @property
    def section(self) -> str:
        return self.metadata.get("section_name", "")


class Retriever:
    def __init__(
        self,
        collection: str = settings.QDRANT_COLLECTION,
        qdrant_url: str = settings.QDRANT_URL,
        qdrant_api_key: str = settings.QDRANT_API_KEY,
    ) -> None:
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            raise ImportError("qdrant-client not installed. Run: pip install qdrant-client") from exc

        self._client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key or None)
        self._collection = collection
        self._embedder = get_embedder()

    def search(
        self,
        query: str,
        top_k: int = settings.TOP_K,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        """
        Search for chunks relevant to `query`.

        Args:
            query: Natural language question in Russian or English.
            top_k: Number of results to return.
            filters: Optional Qdrant filter dict, e.g.
                     {"category": "Кардиология"} or {"icd_codes": {"any": ["I21"]}}

        Returns:
            List of SearchResult ordered by descending similarity score.
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

        query_vector = self._embedder.embed([query])[0]

        qdrant_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                if isinstance(value, list):
                    conditions.append(FieldCondition(key=key, match=MatchAny(any=value)))
                else:
                    conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
            qdrant_filter = Filter(must=conditions)

        # qdrant-client >= 1.7: query_points replaces the removed search()
        response = self._client.query_points(
            collection_name=self._collection,
            query=query_vector,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )

        results = []
        for hit in response.points:
            payload = dict(hit.payload or {})
            text = payload.pop("text", "")
            results.append(SearchResult(text=text, score=hit.score, metadata=payload))

        return results

    def search_by_icd(self, icd_code: str, top_k: int = 10) -> list[SearchResult]:
        """Convenience method: retrieve all chunks for a specific ICD-10 code."""
        return self.search("", top_k=top_k, filters={"icd_codes": [icd_code]})
