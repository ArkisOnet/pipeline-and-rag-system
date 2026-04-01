"""
Markdown-aware document splitter.

Uses MarkdownHeaderTextSplitter as the primary splitter, then applies a
table-safety guard to ensure no GFM table is ever split across two chunks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from config.settings import settings

logger = logging.getLogger(__name__)

MIN_CHUNK_BODY = 30

_HEADERS_TO_SPLIT = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]

_HEADER_SPLITTER = MarkdownHeaderTextSplitter(
    headers_to_split_on=_HEADERS_TO_SPLIT,
    strip_headers=False,
)

_CHAR_SPLITTER = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " "],
    chunk_size=settings.MAX_CHUNK_TOKENS * 4,
    chunk_overlap=0,
)


@dataclass
class Chunk:
    text: str
    section_name: str


def _chunk_ends_in_table(text: str) -> bool:
    lines = text.rstrip().splitlines()
    for line in reversed(lines):
        stripped = line.strip()
        if stripped:
            return stripped.startswith("|")
    return False


def _contains_table(text: str) -> bool:
    return any(line.strip().startswith("|") for line in text.splitlines())


def _next_chunk_continues_table(text: str) -> bool:
    first = next((l for l in text.splitlines() if l.strip()), "")
    return first.strip().startswith("|")


def _merge_split_tables(texts: list[str]) -> list[str]:
    merged: list[str] = []
    i = 0
    while i < len(texts):
        current = texts[i]
        while (
            _chunk_ends_in_table(current)
            and i + 1 < len(texts)
            and _next_chunk_continues_table(texts[i + 1])
        ):
            i += 1
            current = current + "\n\n" + texts[i]
        merged.append(current)
        i += 1
    return merged


def _build_section_name(metadata: dict) -> str:
    parts = []
    for key in ("h1", "h2", "h3"):
        val = metadata.get(key)
        if val:
            parts.append(val.strip())
    return " > ".join(parts) if parts else "General"


def split(markdown: str, protocol_name: str = "") -> list[Chunk]:
    if not markdown.strip():
        return []

    raw_docs = _HEADER_SPLITTER.split_text(markdown)

    texts: list[str] = []
    section_names: list[str] = []

    for doc in raw_docs:
        meta = doc.metadata if hasattr(doc, "metadata") else {}
        if not meta.get("h1") and protocol_name:
            meta = {"h1": protocol_name, **meta}
        texts.append(doc.page_content)
        section_names.append(_build_section_name(meta))

    merged_texts = _merge_split_tables(texts)

    while len(section_names) < len(merged_texts):
        section_names.append(section_names[-1] if section_names else "General")

    final_chunks: list[Chunk] = []
    for text, section in zip(merged_texts, section_names):
        estimated_tokens = len(text) // 4
        if estimated_tokens > settings.MAX_CHUNK_TOKENS and not _contains_table(text):
            sub_texts = _CHAR_SPLITTER.split_text(text)
            for sub in sub_texts:
                final_chunks.append(Chunk(text=sub.strip(), section_name=section))
        else:
            final_chunks.append(Chunk(text=text.strip(), section_name=section))

    return [c for c in final_chunks if len(c.text.strip()) >= MIN_CHUNK_BODY]
