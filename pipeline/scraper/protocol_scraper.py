"""
Fetches the full HTML of an individual protocol page.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from bs4 import BeautifulSoup

from pipeline.scraper.browser import fetch_html
from pipeline.scraper.listing_scraper import ProtocolStub
from pipeline.utils.retry import async_retry

logger = logging.getLogger(__name__)


@dataclass
class ProtocolPage:
    stub: ProtocolStub
    raw_html: str
    fetched_at: str


def _validate_protocol_html(html: str, url: str) -> None:
    soup = BeautifulSoup(html, "lxml")
    if not soup.find("h1"):
        raise ValueError(f"No <h1> found — page may be a 404 or login wall: {url}")


@async_retry(max_attempts=5, base_delay=2.0, max_delay=60.0)
async def fetch_protocol(stub: ProtocolStub) -> ProtocolPage:
    logger.info("Fetching protocol [%s]: %s", stub.document_id, stub.url)
    html = await fetch_html(stub.url)
    _validate_protocol_html(html, stub.url)
    return ProtocolPage(
        stub=stub,
        raw_html=html,
        fetched_at=datetime.now(timezone.utc).isoformat(),
    )
