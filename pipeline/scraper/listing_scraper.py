"""
Crawls disease listing pages on diseases.medelement.com and yields ProtocolStub
objects for every protocol found.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlencode

from bs4 import BeautifulSoup

from config.settings import settings
from pipeline.scraper.browser import fetch_html
from pipeline.utils.retry import async_retry

logger = logging.getLogger(__name__)

_ICD_RE = re.compile(r"\b([A-Z]\d{2}(?:\.\d+)?(?:[–\-][A-Z]\d{2}(?:\.\d+)?)?)\b")
_YEAR_RE = re.compile(r"\b(20\d{2})\b")

_COUNTRY_LOGO_MAP: dict[str, str] = {
    "logo__pdl_kz": "Kazakhstan",
    "logo__pdl_ru": "Russia",
    "logo__pdl_by": "Belarus",
}

_DOC_ID_RE = re.compile(r"/disease/[^/]+/(\d+)")


@dataclass
class ProtocolStub:
    url: str
    name: str
    icd_codes: list[str] = field(default_factory=list)
    category: str = ""
    country: str = "Kazakhstan"
    version_year: int = 0
    content_type: int = 4
    document_id: str = ""


def _build_listing_url(content_type: int, specialty_id: str, skip: int = 0) -> str:
    params = urlencode({
        "searched_data": "diseases",
        "diseases_filter_type": "section_medicine",
        "diseases_content_type": content_type,
        "section_medicine": specialty_id,
        "category_mkb": 0,
        "parent_category_mkb": 0,
        "skip": skip,
    })
    return f"{settings.BASE_URL}/?{params}"


def _extract_icd_codes(text: str) -> list[str]:
    text = text.translate(str.maketrans("О", "O"))
    return list(dict.fromkeys(_ICD_RE.findall(text)))


def _extract_year(text: str) -> int:
    match = _YEAR_RE.search(text)
    return int(match.group(1)) if match else 0


def _detect_country(soup_fragment: BeautifulSoup) -> str:
    for img in soup_fragment.find_all("img"):
        src: str = (img.get("src") or "").lower()
        for key, country in _COUNTRY_LOGO_MAP.items():
            if key in src:
                return country
    return "Kazakhstan"


def _parse_stubs(html: str, category: str, content_type: int) -> list[ProtocolStub]:
    soup = BeautifulSoup(html, "lxml")
    stubs: list[ProtocolStub] = []

    for anchor in soup.find_all("a", href=re.compile(r"^/disease/")):
        href: str = anchor.get("href", "")
        doc_id_match = _DOC_ID_RE.search(href)
        if not doc_id_match:
            continue

        document_id = doc_id_match.group(1)
        name = anchor.get_text(strip=True)
        if not name:
            continue

        full_url = urljoin(settings.BASE_URL, href)
        container = anchor.find_parent(["div", "li", "article", "section"]) or anchor.parent
        raw_text = container.get_text(" ", strip=True) if container else ""

        icd_codes = _extract_icd_codes(raw_text)
        year = _extract_year(raw_text)
        country = _detect_country(BeautifulSoup(str(container), "lxml"))

        if content_type == 4 and country not in ("Kazakhstan", ""):
            continue

        stubs.append(ProtocolStub(
            url=full_url,
            name=name,
            icd_codes=icd_codes,
            category=category,
            country=country or "Kazakhstan",
            version_year=year,
            content_type=content_type,
            document_id=document_id,
        ))

    return stubs


def _has_next_page(html: str) -> bool:
    soup = BeautifulSoup(html, "lxml")
    for a in soup.find_all("a"):
        text = a.get_text(strip=True)
        if "вперед" in text.lower() or "→" in text or "»" in text:
            return True
    return False


@async_retry(max_attempts=5, base_delay=2.0, max_delay=60.0)
async def _fetch_listing(url: str) -> str:
    return await fetch_html(url)


async def crawl_specialty(
    specialty_name: str,
    specialty_id: str,
    content_type: int = 4,
    state_manager=None,
) -> list[ProtocolStub]:
    all_stubs: list[ProtocolStub] = []
    skip = 0

    while True:
        url = _build_listing_url(content_type, specialty_id, skip)

        if state_manager and state_manager.is_listing_done(url):
            logger.debug("Listing already crawled, skipping: %s", url)
            break

        logger.info("Fetching listing: %s", url)
        html = await _fetch_listing(url)

        stubs = _parse_stubs(html, category=specialty_name, content_type=content_type)
        logger.info("  Found %d protocols on page (skip=%d)", len(stubs), skip)

        all_stubs.extend(stubs)

        if state_manager:
            state_manager.mark_listing_done(url)

        if not stubs or not _has_next_page(html):
            break

        skip += settings.PAGE_SIZE

    return all_stubs
