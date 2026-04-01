"""
Extracts structured metadata from a protocol page.

Priority order:
1. JSON-LD <script type="application/ld+json"> block
2. HTML fallback: full-page text scan for ICD-10 codes
3. ProtocolStub values from the listing page
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from bs4 import BeautifulSoup

from pipeline.scraper.listing_scraper import ProtocolStub

logger = logging.getLogger(__name__)

_ICD_RE = re.compile(r"\b([A-Z]\d{2}(?:\.\d+)?(?:[–\-][A-Z]\d{2}(?:\.\d+)?)?)\b")
_YEAR_RE = re.compile(r"\b(20\d{2})\b")


@dataclass
class ProtocolMetadata:
    source_url: str
    document_id: str
    name: str
    icd_codes: list[str] = field(default_factory=list)
    country: str = "Kazakhstan"
    category: str = ""
    version_year: int = 0
    date_published: str = ""
    date_modified: str = ""


def _parse_json_ld(soup: BeautifulSoup) -> dict:
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
            if isinstance(data, dict):
                return data
            if isinstance(data, list) and data:
                return data[0]
        except (json.JSONDecodeError, AttributeError):
            continue
    return {}


def _extract_icd_from_html(soup: BeautifulSoup) -> list[str]:
    full_text = soup.get_text(" ", strip=True)
    full_text = full_text.translate(str.maketrans("О", "O"))
    return list(dict.fromkeys(_ICD_RE.findall(full_text)))


def extract(raw_html: str, stub: ProtocolStub) -> ProtocolMetadata:
    soup = BeautifulSoup(raw_html, "lxml")
    ld = _parse_json_ld(soup)

    name = ld.get("name") or ld.get("headline") or stub.name

    icd_codes = stub.icd_codes.copy()
    if not icd_codes:
        about = ld.get("about") or []
        if isinstance(about, list):
            for item in about:
                if isinstance(item, dict):
                    code = item.get("sameAs") or item.get("identifier") or ""
                    icd_codes.extend(_ICD_RE.findall(str(code)))
        if not icd_codes:
            icd_codes = _extract_icd_from_html(soup)

    date_published = ld.get("datePublished") or ld.get("dateCreated") or ""
    date_modified = ld.get("dateModified") or ""

    version_year = stub.version_year
    if not version_year:
        for date_str in (date_published, date_modified):
            match = _YEAR_RE.search(date_str)
            if match:
                version_year = int(match.group(1))
                break

    category = stub.category
    if not category:
        section = ld.get("articleSection") or ld.get("keywords") or ""
        if isinstance(section, list):
            section = section[0] if section else ""
        category = str(section)

    return ProtocolMetadata(
        source_url=stub.url,
        document_id=stub.document_id,
        name=str(name),
        icd_codes=list(dict.fromkeys(icd_codes)),
        country=stub.country,
        category=category,
        version_year=version_year,
        date_published=date_published,
        date_modified=date_modified,
    )
