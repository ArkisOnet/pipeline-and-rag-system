"""
Isolates the main article content from a raw protocol page HTML.

Steps:
1. Find the main content root (div/article containing the protocol <h1>)
2. Remove navigation chrome: nav, header, footer, sidebars, scripts, etc.
3. Strip the "Внимание!" / "Прикреплённые файлы" legal boilerplate sections
4. Convert <table> elements to GFM Markdown before html2text sees them
5. Replace image-based tables with labelled [IMAGE TABLE: ...] placeholders
   (or OCR'd text if easyocr is installed)
6. Remove site logo <img> tags
7. Return the cleaned HTML string
"""

from __future__ import annotations

import logging
import re
from urllib.parse import urlparse

from bs4 import BeautifulSoup, Comment, NavigableString, Tag

logger = logging.getLogger(__name__)

_UPLOAD_DOMAIN = "files.medelement.com"

_STRIP_SELECTORS = [
    "nav", "header", "footer", "script", "style", "noscript", "iframe",
    "[class*='menu']", "[class*='banner']", "[class*='sidebar']",
    "[class*='breadcrumb']", "[class*='cookie']", "[class*='social']",
    "[class*='share']", "[class*='ad']", "[class*='promo']",
]

_ATTENTION_RE = re.compile(
    r"^\s*(внимание|attention|предупреждение|прикреплённые\s+файлы|attached\s+files)\s*[!:]?\s*$",
    re.I,
)


def _remove_comments(soup: BeautifulSoup) -> None:
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        comment.extract()


def _strip_elements(soup: BeautifulSoup) -> None:
    for selector in _STRIP_SELECTORS:
        for el in soup.select(selector):
            el.decompose()


def _strip_attention_section(root: Tag) -> None:
    for heading in root.find_all(re.compile(r"^h[1-6]$")):
        if _ATTENTION_RE.match(heading.get_text(strip=True)):
            for sibling in list(heading.find_next_siblings()):
                sibling.decompose()
            heading.decompose()
            return


def _strip_javascript_links(root: Tag) -> None:
    for a in root.find_all("a", href=re.compile(r"^javascript:", re.I)):
        a.decompose()


def _convert_html_tables(root: Tag) -> None:
    """Convert <table> elements to GFM Markdown before html2text processing."""

    def _cell_text(td: Tag) -> str:
        text = td.get_text(" ", strip=True)
        text = " ".join(text.split())
        return text.replace("|", "\\|")

    for table in root.find_all("table"):
        rows: list[list[str]] = []
        for tr in table.find_all("tr"):
            cells = [_cell_text(td) for td in tr.find_all(["th", "td"])]
            if cells:
                rows.append(cells)

        if not rows:
            table.decompose()
            continue

        max_cols = max(len(r) for r in rows)
        rows = [r + [""] * (max_cols - len(r)) for r in rows]

        header = "| " + " | ".join(rows[0]) + " |"
        separator = "| " + " | ".join(["---"] * max_cols) + " |"
        body_lines = ["| " + " | ".join(r) + " |" for r in rows[1:]]

        md_table = "\n" + "\n".join([header, separator] + body_lines) + "\n"
        table.replace_with(NavigableString(md_table))


def _nearest_label(img: Tag) -> str:
    alt = (img.get("alt") or "").strip()
    if alt:
        return alt
    for candidate in img.find_all_previous(limit=10):
        if not isinstance(candidate, Tag):
            continue
        if candidate.name in ("strong", "b"):
            text = candidate.get_text(" ", strip=True)
            if text:
                return text
        if re.match(r"^h[1-6]$", candidate.name or ""):
            text = candidate.get_text(" ", strip=True)
            if text:
                return text
    return ""


def _replace_upload_images(root: Tag, source_url: str) -> None:
    from pipeline.parser import image_ocr

    for img in root.find_all("img"):
        src: str = img.get("src") or ""
        parsed = urlparse(src)
        if _UPLOAD_DOMAIN in parsed.netloc or "/uploads/" in src:
            label = _nearest_label(img)
            filename = parsed.path.rsplit("/", 1)[-1] if parsed.path else "unknown"
            if not label:
                label = filename

            ocr_text = image_ocr.ocr_image_url(src)
            if ocr_text:
                placeholder = f"\n\n[TABLE: {label}]\n{ocr_text}\n\n"
                logger.debug("OCR table: %s (from %s)", label, source_url)
            else:
                placeholder = f"\n\n[IMAGE TABLE: {label}]\n\n"
                logger.debug("Image table (no OCR): %s (from %s)", label, source_url)

            img.replace_with(NavigableString(placeholder))
        else:
            img.decompose()


def _find_content_root(soup: BeautifulSoup) -> Tag | None:
    h1 = soup.find("h1")
    if not h1:
        return None
    candidate = h1.parent
    for _ in range(5):
        if candidate is None or candidate.name == "body":
            break
        tag_name = getattr(candidate, "name", "")
        if tag_name in ("article", "main"):
            return candidate
        if tag_name == "div" and len(candidate.get_text(strip=True)) > 500:
            return candidate
        candidate = candidate.parent
    return soup.find("body") or soup  # type: ignore[return-value]


def clean(raw_html: str, source_url: str = "") -> str:
    soup = BeautifulSoup(raw_html, "lxml")
    _remove_comments(soup)
    _strip_elements(soup)
    _strip_javascript_links(soup)

    root = _find_content_root(soup)
    if root is None:
        logger.warning("Could not find content root for %s", source_url)
        root = soup

    _strip_attention_section(root)
    _convert_html_tables(root)
    _replace_upload_images(root, source_url)

    return str(root)
