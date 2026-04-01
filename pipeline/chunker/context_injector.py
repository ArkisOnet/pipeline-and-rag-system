"""
Prepends a context header to each chunk and packages it with metadata.

Output record format:
{
    "text": "[Protocol: NAME, YEAR] [Section: SECTION]\n\nchunk text...",
    "metadata": { source_url, document_id, icd_codes, country, category, ... }
}
"""

from __future__ import annotations

from pipeline.chunker.splitter import Chunk
from pipeline.parser.metadata_extractor import ProtocolMetadata


def inject(chunks: list[Chunk], metadata: ProtocolMetadata) -> list[dict]:
    records = []
    for idx, chunk in enumerate(chunks):
        context_header = (
            f"[Protocol: {metadata.name}, {metadata.version_year}]"
            f" [Section: {chunk.section_name}]"
        )
        full_text = f"{context_header}\n\n{chunk.text}"

        records.append({
            "text": full_text,
            "metadata": {
                "source_url": metadata.source_url,
                "document_id": metadata.document_id,
                "icd_codes": metadata.icd_codes,
                "country": metadata.country,
                "category": metadata.category,
                "version_year": metadata.version_year,
                "date_published": metadata.date_published,
                "date_modified": metadata.date_modified,
                "section_name": chunk.section_name,
                "chunk_index": idx,
            },
        })

    return records
