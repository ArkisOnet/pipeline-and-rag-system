"""Tests for pipeline/parser/metadata_extractor.py"""

from pathlib import Path

import pytest
from pipeline.parser.metadata_extractor import extract, ProtocolMetadata
from pipeline.scraper.listing_scraper import ProtocolStub


FIXTURE_HTML = Path("tests/fixtures/sample_protocol.html").read_text(encoding="utf-8")


def _make_stub(**kwargs) -> ProtocolStub:
    defaults = dict(
        url="https://diseases.medelement.com/disease/hellp-синдром-кп-рк-2022/17522",
        name="HELLP-синдром",
        icd_codes=[],
        category="Акушерство и гинекология",
        country="Kazakhstan",
        version_year=2022,
        content_type=4,
        document_id="17522",
    )
    defaults.update(kwargs)
    return ProtocolStub(**defaults)


class TestExtract:
    def test_returns_metadata_instance(self):
        assert isinstance(extract(FIXTURE_HTML, _make_stub()), ProtocolMetadata)

    def test_name_from_json_ld(self):
        meta = extract(FIXTURE_HTML, _make_stub(name="Fallback"))
        assert meta.name == "HELLP-синдром"

    def test_date_published_from_json_ld(self):
        meta = extract(FIXTURE_HTML, _make_stub())
        assert "2022" in meta.date_published

    def test_icd_codes_extracted_from_html(self):
        meta = extract(FIXTURE_HTML, _make_stub(icd_codes=[]))
        assert "O14.2" in meta.icd_codes

    def test_stub_icd_codes_used_when_present(self):
        meta = extract(FIXTURE_HTML, _make_stub(icd_codes=["O14.2", "O15.0"]))
        assert "O14.2" in meta.icd_codes
        assert "O15.0" in meta.icd_codes

    def test_country_from_stub(self):
        meta = extract(FIXTURE_HTML, _make_stub(country="Kazakhstan"))
        assert meta.country == "Kazakhstan"

    def test_document_id_preserved(self):
        meta = extract(FIXTURE_HTML, _make_stub(document_id="17522"))
        assert meta.document_id == "17522"

    def test_source_url_preserved(self):
        url = "https://diseases.medelement.com/disease/hellp/17522"
        meta = extract(FIXTURE_HTML, _make_stub(url=url))
        assert meta.source_url == url

    def test_category_from_stub(self):
        meta = extract(FIXTURE_HTML, _make_stub(category="Акушерство и гинекология"))
        assert meta.category == "Акушерство и гинекология"
