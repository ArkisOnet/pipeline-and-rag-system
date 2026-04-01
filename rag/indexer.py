"""
Indexes protocols.jsonl into a Qdrant vector collection.

Run:
    python -m rag.indexer [--input data/output/protocols.jsonl]
                          [--collection medprotocols]
                          [--recreate]
                          [--batch-size 64]

Prereqs:
    pip install qdrant-client sentence-transformers
    docker run -p 6333:6333 qdrant/qdrant   # or point QDRANT_URL at a remote instance
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from tqdm import tqdm

from config.settings import settings
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
    input_path: str = settings.OUTPUT_JSONL_PATH,
    collection: str = settings.QDRANT_COLLECTION,
    batch_size: int = settings.EMBEDDING_BATCH_SIZE,
    recreate: bool = False,
) -> None:
    from qdrant_client.models import PointStruct

    path = Path(input_path)
    if not path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    logger.info("Loaded %d chunks from %s", len(records), input_path)

    embedder = get_embedder()
    client = _get_client()
    _ensure_collection(client, collection, embedder.dimension(), recreate)

    texts = [r["text"] for r in records]
    points: list[PointStruct] = []

    logger.info("Generating embeddings (batch_size=%d)...", batch_size)
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch_texts = texts[i : i + batch_size]
        batch_records = records[i : i + batch_size]
        vectors = embedder.embed(batch_texts)

        for j, (vec, rec) in enumerate(zip(vectors, batch_records)):
            point_id = i + j
            points.append(PointStruct(
                id=point_id,
                vector=vec,
                payload={
                    "text": rec["text"],
                    **rec["metadata"],
                },
            ))

    logger.info("Upserting %d points to Qdrant...", len(points))
    # Upsert in batches to avoid large payloads
    for i in tqdm(range(0, len(points), batch_size), desc="Upserting"):
        client.upsert(
            collection_name=collection,
            points=points[i : i + batch_size],
        )

    logger.info("Indexing complete. Collection '%s' now has %d vectors.", collection, len(points))


def main() -> None:
    parser = argparse.ArgumentParser(description="Index protocols.jsonl into Qdrant")
    parser.add_argument("--input", default=settings.OUTPUT_JSONL_PATH,
                        help=f"Path to protocols.jsonl (default: {settings.OUTPUT_JSONL_PATH})")
    parser.add_argument("--collection", default=settings.QDRANT_COLLECTION)
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
