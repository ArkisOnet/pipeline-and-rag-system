"""
Optional OCR for scanned image tables from files.medelement.com/uploads/.

Install easyocr to enable: pip install easyocr
If easyocr is not installed all calls return None and the pipeline
falls back to [IMAGE TABLE: label] placeholders.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

try:
    import easyocr
    _reader = easyocr.Reader(["ru", "en"], verbose=False)
    _EASYOCR_AVAILABLE = True
    logger.info("easyocr loaded — image OCR enabled")
except ImportError:
    _reader = None
    _EASYOCR_AVAILABLE = False
    logger.debug("easyocr not installed — image OCR disabled (pip install easyocr)")

try:
    import httpx
    _HTTPX_AVAILABLE = True
except ImportError:
    _HTTPX_AVAILABLE = False

_cache: dict[str, str | None] = {}


def ocr_image_url(url: str) -> str | None:
    """Download image at `url` and return OCR'd text, or None if unavailable."""
    if not _EASYOCR_AVAILABLE or not _HTTPX_AVAILABLE:
        return None

    if url in _cache:
        return _cache[url]

    try:
        response = httpx.get(url, timeout=15, follow_redirects=True)
        response.raise_for_status()
        image_bytes = response.content
    except Exception as exc:
        logger.warning("Failed to download image %s: %s", url, exc)
        _cache[url] = None
        return None

    try:
        results = _reader.readtext(image_bytes, detail=0, paragraph=True)
        text = _clean_ocr_text(results)
        _cache[url] = text or None
        return _cache[url]
    except Exception as exc:
        logger.warning("OCR failed for %s: %s", url, exc)
        _cache[url] = None
        return None


def _clean_ocr_text(results: list[str]) -> str:
    lines = []
    for line in results:
        line = line.strip()
        if len(line) <= 1:
            continue
        if re.match(r"^[^\w]*$", line):
            continue
        line = line.translate(str.maketrans("О", "O"))
        lines.append(line)
    return "\n".join(lines)
