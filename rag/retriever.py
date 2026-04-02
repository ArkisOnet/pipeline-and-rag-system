"""
Vector similarity retriever for Qdrant collections.

Key behaviours
--------------
* ICD-10 detection  — if the query contains a code like "Z35.8", we first do an
  exact metadata filter (scroll) to surface guaranteed matches, then fall back
  to semantic search for anything above the similarity threshold.
* Score threshold   — results below MIN_SIMILARITY_SCORE are dropped so
  irrelevant documents never reach the LLM context.
* Source tagging    — SearchResult.source returns "service" or "protocol" so
  callers can render attribution correctly.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from config.settings import settings
from rag.embeddings import get_embedder

logger = logging.getLogger(__name__)

# ICD-10 pattern: one capital letter + 2 digits + optional .digit(s)
# Examples: A00, Z35.8, I21.0, O80
_ICD_RE = re.compile(r"\b([A-Z]\d{2}(?:\.\d{1,2})?)\b")


def extract_icd_codes(text: str) -> list[str]:
    """Return all ICD-10 codes found in *text* (uppercased)."""
    return _ICD_RE.findall(text.upper())


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict = field(default_factory=dict)

    # ── convenience properties ────────────────────────────────────────────────

    @property
    def source(self) -> str:
        """'service' for service-registry rows, 'protocol' for clinical protocols."""
        return self.metadata.get("source", "protocol")

    @property
    def source_url(self) -> str:
        return self.metadata.get("source_url", "")

    @property
    def protocol_name(self) -> str:
        m = re.search(r"\[Protocol:\s*(.+?),\s*\d+\]", self.text)
        return m.group(1).strip() if m else self.metadata.get("document_id", "")

    @property
    def section(self) -> str:
        return self.metadata.get("section_name", "")


class Retriever:
    """
    Searches one Qdrant collection with optional ICD-10 exact-match priority.

    Parameters
    ----------
    collection : str
        Qdrant collection name.
    icd_field : str
        Payload field that stores ICD codes.
        • "icd_codes"  — protocols collection (array of strings)
        • "icd_code"   — services collection  (single string)
    """

    def __init__(
        self,
        collection: str = settings.QDRANT_COLLECTION,
        qdrant_url: str = settings.QDRANT_URL,
        qdrant_api_key: str = settings.QDRANT_API_KEY,
        icd_field: str = "icd_codes",
    ) -> None:
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            raise ImportError("qdrant-client not installed. Run: pip install qdrant-client") from exc

        self._client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key or None)
        self._collection = collection
        self._icd_field = icd_field
        self._embedder = get_embedder()

    # ── public API ────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = settings.TOP_K,
        filters: dict | None = None,
        icd_codes: list[str] | None = None,
        min_score: float = settings.MIN_SIMILARITY_SCORE,
    ) -> list[SearchResult]:
        """
        Search for chunks relevant to *query*.

        If *icd_codes* is provided (non-empty), exact metadata matches are
        fetched first and prepended to the semantic results (deduped).
        Semantic results are filtered to score >= *min_score*.

        Parameters
        ----------
        query      : Natural language question (Russian or English).
        top_k      : Maximum results to return.
        filters    : Extra Qdrant payload filters (e.g. {"category": "…"}).
        icd_codes  : ICD-10 codes already extracted from the query.
        min_score  : Minimum cosine similarity for semantic results.
        """
        exact: list[SearchResult] = []
        if icd_codes:
            exact = self._exact_icd_search(icd_codes, top_k)

        semantic = self._semantic_search(query, top_k, filters, min_score)

        # Merge: exact matches first, then semantic.
        # Dedup by service_code (services) or text prefix (protocols).
        seen: set[str] = set()
        merged: list[SearchResult] = []
        for r in exact + semantic:
            key = r.metadata.get("service_code") or r.text[:80]
            if key not in seen:
                seen.add(key)
                merged.append(r)

        return merged[:top_k]

    def search_by_icd(self, icd_code: str, top_k: int = 10) -> list[SearchResult]:
        """Fetch all chunks for a specific ICD-10 code via exact filter."""
        return self._exact_icd_search([icd_code], top_k)

    # ── internals ─────────────────────────────────────────────────────────────

    def _exact_icd_search(self, icd_codes: list[str], top_k: int) -> list[SearchResult]:
        """
        Scroll the collection using a metadata filter on the ICD field.
        Returns results with score=1.0 (exact match, no vector needed).
        """
        from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

        # Build filter: MatchAny works for both array fields and single-value fields
        condition = FieldCondition(
            key=self._icd_field,
            match=MatchAny(any=icd_codes),
        )
        qdrant_filter = Filter(must=[condition])

        try:
            points, _ = self._client.scroll(
                collection_name=self._collection,
                scroll_filter=qdrant_filter,
                limit=top_k,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:
            logger.warning("ICD exact search failed (%s): %s", self._collection, exc)
            return []

        results = []
        for point in points:
            payload = dict(point.payload or {})
            text = payload.pop("text", "")
            results.append(SearchResult(text=text, score=1.0, metadata=payload))

        logger.info("ICD exact match: %d results for %s in '%s'",
                    len(results), icd_codes, self._collection)
        return results

    def _semantic_search(
        self,
        query: str,
        top_k: int,
        filters: dict | None,
        min_score: float,
    ) -> list[SearchResult]:
        """Vector similarity search, dropping results below *min_score*."""
        from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

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

        response = self._client.query_points(
            collection_name=self._collection,
            query=query_vector,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )

        results = []
        for hit in response.points:
            if hit.score < min_score:
                logger.debug("Dropped low-score result (%.3f < %.3f)", hit.score, min_score)
                continue
            payload = dict(hit.payload or {})
            text = payload.pop("text", "")
            results.append(SearchResult(text=text, score=hit.score, metadata=payload))

        return results
