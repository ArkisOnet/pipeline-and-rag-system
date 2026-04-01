"""
Converts cleaned protocol HTML to GitHub-flavored Markdown.

Post-conversion fixups:
- NB! paragraphs → Markdown blockquotes
- Evidence level notation → standardised [EL:X] tags
"""

from __future__ import annotations

import re

import html2text


def _build_converter() -> html2text.HTML2Text:
    h = html2text.HTML2Text()
    h.body_width = 0
    h.protect_links = True
    h.wrap_links = False
    h.ignore_images = False
    h.mark_code = True
    h.unicode_snob = True
    h.ignore_links = False
    # Tables are pre-converted to Markdown by html_cleaner._convert_html_tables()
    h.bypass_tables = True
    return h


_CONVERTER = _build_converter()

_NB_RE = re.compile(r"^(\*\*NB!\*\*.*)", re.MULTILINE)

_EL_LETTER = r"([ABCDАВСДabcdавсд])"
_EL_PATTERNS = [
    re.compile(r"\(УД\s*[-–]\s*" + _EL_LETTER + r"\)", re.IGNORECASE),
    re.compile(r"\(уровень\s+доказательности\s*[-–]?\s*" + _EL_LETTER + r"\)", re.IGNORECASE),
    re.compile(r"\(УД:\s*" + _EL_LETTER + r"\)", re.IGNORECASE),
]

_CYR_TO_LAT = str.maketrans("АВСДавсд", "ABCDabcd")


def _apply_nb_blockquotes(md: str) -> str:
    return _NB_RE.sub(r"> \1", md)


def _normalise_evidence_levels(md: str) -> str:
    for pattern in _EL_PATTERNS:
        md = pattern.sub(
            lambda m: f"[EL:{m.group(1).translate(_CYR_TO_LAT).upper()}]", md
        )
    return md


def convert(clean_html: str) -> str:
    md = _CONVERTER.handle(clean_html)
    md = _apply_nb_blockquotes(md)
    md = _normalise_evidence_levels(md)
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()
