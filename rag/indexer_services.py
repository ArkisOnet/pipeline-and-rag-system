"""
Indexes services.xlsx into a Qdrant vector collection (one point per row).

Run:
    python -m rag.indexer_services [--input data/raw/services.xlsx]
                                   [--collection medservices]
                                   [--recreate]
                                   [--batch-size 64]

Prereqs:
    pip install qdrant-client sentence-transformers pandas openpyxl
    docker run -p 6333:6333 qdrant/qdrant
"""

from __future__ import annotations

import argparse
import logging
import sys

from tqdm import tqdm

from config.settings import settings
from pipeline.parser.services_parser import load_service_records
from rag.embeddings import get_embedder

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def _get_client():
    try:
        from qdrant_client import QdrantClient
    except ImportError as exc:
        raise ImportError("qdrant-client not installed. Run: pip install qdrant-client") from exc
    return QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY or None)


def _ensure_collection(client, collection: str, dim: int, recreate: bool) -> None:
    from qdrant_client.models import Distance, VectorParams

    existing = {c.name for c in client.get_collections().collections}

    if recreate and collection in existing:
        client.delete_collection(collection)
        logger.info("Deleted existing collection: %s", collection)
        existing.discard(collection)

    if collection not in existing:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        logger.info("Created collection: %s (dim=%d)", collection, dim)
    else:
        logger.info("Using existing collection: %s", collection)


def index(
    input_path: str = settings.SERVICES_XLSX_PATH,
    collection: str = settings.SERVICES_COLLECTION,
    batch_size: int = settings.EMBEDDING_BATCH_SIZE,
    recreate: bool = False,
) -> None:
    from qdrant_client.models import PointStruct

    records = load_service_records(input_path)
    if not records:
        logger.error("No records loaded from %s", input_path)
        sys.exit(1)

    embedder = get_embedder()
    client = _get_client()
    _ensure_collection(client, collection, embedder.dimension(), recreate)

    texts = [r["text"] for r in records]
    points: list[PointStruct] = []

    logger.info("Generating embeddings for %d service records (batch_size=%d)...", len(texts), batch_size)
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch_texts = texts[i : i + batch_size]
        batch_records = records[i : i + batch_size]
        vectors = embedder.embed(batch_texts)

        for j, (vec, rec) in enumerate(zip(vectors, batch_records)):
            payload = dict(rec)  # already flat — text + all metadata keys
            points.append(PointStruct(
                id=i + j,
                vector=vec,
                payload=payload,
            ))

    logger.info("Upserting %d points to collection '%s'...", len(points), collection)
    for i in tqdm(range(0, len(points), batch_size), desc="Upserting"):
        client.upsert(
            collection_name=collection,
            points=points[i : i + batch_size],
        )

    logger.info("Done. Collection '%s' now has %d vectors.", collection, len(points))


def main() -> None:
    parser = argparse.ArgumentParser(description="Index services.xlsx into Qdrant")
    parser.add_argument("--input", default=settings.SERVICES_XLSX_PATH,
                        help=f"Path to services.xlsx (default: {settings.SERVICES_XLSX_PATH})")
    parser.add_argument("--collection", default=settings.SERVICES_COLLECTION)
    parser.add_argument("--batch-size", type=int, default=settings.EMBEDDING_BATCH_SIZE)
    parser.add_argument("--recreate", action="store_true",
                        help="Drop and recreate the collection before indexing")
    args = parser.parse_args()

    index(
        input_path=args.input,
        collection=args.collection,
        batch_size=args.batch_size,
        recreate=args.recreate,
    )


if __name__ == "__main__":
    main()
