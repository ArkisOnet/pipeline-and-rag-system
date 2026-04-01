"""
Unit tests for rag/retriever.py — mocked, no live Qdrant required.

qdrant-client is an optional dependency; tests that need it are skipped
automatically when it is not installed.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

qdrant_client = pytest.importorskip("qdrant_client", reason="qdrant-client not installed")

from rag.retriever import Retriever, SearchResult  # noqa: E402


class TestSearchResult:
    def test_protocol_name_extracted_from_context_header(self):
        r = SearchResult(
            text="[Protocol: HELLP-синдром, 2023] [Section: Диагностика]\n\nContent.",
            score=0.95,
            metadata={"section_name": "Диагностика", "source_url": "https://example.com"},
        )
        assert r.protocol_name == "HELLP-синдром"

    def test_section_from_metadata(self):
        r = SearchResult(
            text="Some text",
            score=0.8,
            metadata={"section_name": "Лечение > Стационар"},
        )
        assert r.section == "Лечение > Стационар"

    def test_source_url_from_metadata(self):
        url = "https://diseases.medelement.com/disease/hellp/17522"
        r = SearchResult(text="t", score=0.9, metadata={"source_url": url})
        assert r.source_url == url


class TestRetriever:
    @patch("rag.retriever.get_embedder")
    @patch("rag.retriever.QdrantClient", create=True)
    def test_search_calls_qdrant(self, mock_qdrant_cls, mock_get_embedder):
        # Mock embedder
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [[0.1] * 768]
        mock_get_embedder.return_value = mock_embedder

        # Mock Qdrant hit
        mock_hit = MagicMock()
        mock_hit.score = 0.95
        mock_hit.payload = {
            "text": "[Protocol: Test, 2023] [Section: Диагностика]\n\nContent.",
            "source_url": "https://example.com",
            "section_name": "Диагностика",
        }

        mock_client = MagicMock()
        mock_client.search.return_value = [mock_hit]
        mock_qdrant_cls.return_value = mock_client

        with patch.dict("sys.modules", {"qdrant_client": MagicMock(QdrantClient=mock_qdrant_cls)}):
            retriever = Retriever.__new__(Retriever)
            retriever._client = mock_client
            retriever._collection = "medprotocols"
            retriever._embedder = mock_embedder

            results = retriever.search("test query", top_k=3)

        assert len(results) == 1
        assert results[0].score == 0.95
        assert results[0].section == "Диагностика"
