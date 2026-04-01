"""Tests for pipeline/chunker/splitter.py"""

import pytest
from pipeline.chunker.splitter import (
    Chunk,
    _chunk_ends_in_table,
    _contains_table,
    _merge_split_tables,
    split,
)


class TestChunkEndsInTable:
    def test_detects_table_ending(self):
        assert _chunk_ends_in_table("Some header\n\n| Col1 | Col2 |\n|------|------|") is True

    def test_normal_text_not_table(self):
        assert _chunk_ends_in_table("Some text paragraph.\n\nAnother line.") is False

    def test_empty_string(self):
        assert _chunk_ends_in_table("") is False

    def test_whitespace_only(self):
        assert _chunk_ends_in_table("   \n  \n  ") is False


class TestMergeSplitTables:
    def test_merges_split_table(self):
        chunk1 = "## Header\n\n| A | B |\n|---|---|"
        chunk2 = "| val1 | val2 |\n| val3 | val4 |"
        chunk3 = "## Next section\n\nSome text."
        result = _merge_split_tables([chunk1, chunk2, chunk3])
        assert len(result) == 2
        assert "val1" in result[0]
        assert "Next section" in result[1]

    def test_no_merge_needed(self):
        chunks = ["Paragraph one.", "Paragraph two.", "Paragraph three."]
        assert _merge_split_tables(chunks) == chunks

    def test_single_chunk(self):
        assert _merge_split_tables(["Only one chunk."]) == ["Only one chunk."]

    def test_empty_list(self):
        assert _merge_split_tables([]) == []


class TestSplit:
    def test_basic_split_by_headers(self):
        md = (
            "# Protocol\n\n"
            "## Diagnosis\n\nDiagnostic criteria are based on laboratory findings.\n\n"
            "## Treatment\n\nTreatment protocol includes magnesium sulphate infusion.\n"
        )
        chunks = split(md, protocol_name="Test Protocol")
        assert len(chunks) >= 2
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_table_not_split(self):
        md = "# Protocol\n\n## Lab Values\n\n| Test | Value | Range |\n|------|-------|-------|\n| Hb | 80 | >120 |\n| PLT | 50 | >150 |\n| ALT | 150 | <40 |\n"
        chunks = split(md, protocol_name="Lab Test")
        all_text = "\n".join(c.text for c in chunks)
        assert "Hb" in all_text
        assert "PLT" in all_text
        assert "ALT" in all_text

    def test_section_name_populated(self):
        md = "# Protocol\n\n## Section A\n\nContent here.\n\n### Sub-section A1\n\nSub content.\n"
        chunks = split(md, protocol_name="My Protocol")
        assert any("Section A" in c.section_name for c in chunks)

    def test_empty_markdown_returns_empty(self):
        assert split("") == []
        assert split("   \n  ") == []

    def test_nb_preserved_in_chunk(self):
        md = "# Protocol\n\n## Treatment\n\n> **NB!** Critical warning.\n\nStandard treatment.\n"
        chunks = split(md)
        assert any("NB!" in c.text for c in chunks)

    def test_no_empty_chunks(self):
        md = "# Title\n\n## Empty\n\n\n\n## Has Content\n\nSome text here."
        chunks = split(md, protocol_name="Test")
        assert all(c.text.strip() for c in chunks)
