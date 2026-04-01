"""Tests for pipeline/parser/markdown_converter.py"""

import pytest
from pipeline.parser.markdown_converter import convert, _apply_nb_blockquotes, _normalise_evidence_levels


class TestNbBlockquotes:
    def test_nb_wrapped_in_blockquote(self):
        md = "Some text.\n\n**NB!** Важное предупреждение о лечении.\n\nЕщё текст."
        result = _apply_nb_blockquotes(md)
        assert "> **NB!**" in result

    def test_non_nb_lines_unchanged(self):
        md = "Regular paragraph.\n\nAnother one."
        result = _apply_nb_blockquotes(md)
        assert result == md


class TestEvidenceLevels:
    @pytest.mark.parametrize("raw, expected_tag", [
        ("Назначить препарат (УД - А).", "[EL:A]"),
        ("Применять осторожно (УД-B).", "[EL:B]"),
        ("Рекомендуется (уровень доказательности - C).", "[EL:C]"),
        ("Эффективность доказана (уровень доказательности D).", "[EL:D]"),
        ("(УД: A) в конце.", "[EL:A]"),
    ])
    def test_evidence_level_normalised(self, raw, expected_tag):
        result = _normalise_evidence_levels(raw)
        assert expected_tag in result

    def test_original_notation_removed(self):
        raw = "Препарат А (УД - А) назначается."
        result = _normalise_evidence_levels(raw)
        assert "(УД" not in result


class TestHtmlTableConversion:
    def test_basic_table_to_markdown(self):
        from pipeline.parser.html_cleaner import clean
        html = """<div><h1>Protocol</h1>
        <table>
          <tr><th>Препарат</th><th>Доза</th><th>Путь</th></tr>
          <tr><td>Магния сульфат</td><td>4 г</td><td>в/в</td></tr>
          <tr><td>Лабеталол</td><td>20 мг</td><td>в/в</td></tr>
        </table></div>"""
        result = clean(html, source_url="https://example.com")
        assert "| Препарат |" in result
        assert "| Доза |" in result
        assert "| Магния сульфат |" in result
        assert "| --- |" in result

    def test_table_separator_row(self):
        from pipeline.parser.html_cleaner import clean
        html = """<div><h1>T</h1><table>
          <tr><th>A</th><th>B</th></tr>
          <tr><td>1</td><td>2</td></tr>
        </table></div>"""
        result = clean(html, source_url="https://example.com")
        lines = [l for l in result.splitlines() if l.strip()]
        sep_lines = [l for l in lines if "---" in l]
        assert len(sep_lines) == 1
        assert "|" in sep_lines[0]

    def test_pipe_in_cell_escaped(self):
        from pipeline.parser.html_cleaner import clean
        html = """<div><h1>T</h1><table>
          <tr><th>Range</th></tr>
          <tr><td>A | B</td></tr>
        </table></div>"""
        result = clean(html, source_url="https://example.com")
        assert r"A \| B" in result

    def test_empty_table_removed(self):
        from pipeline.parser.html_cleaner import clean
        html = "<div><h1>T</h1><table></table><p>Content here.</p></div>"
        result = clean(html, source_url="https://example.com")
        assert "<table>" not in result
        assert "Content here." in result

    def test_html2text_sees_markdown_table(self):
        from pipeline.parser.html_cleaner import clean
        from pipeline.parser.markdown_converter import convert
        html = """<div><h1>Protocol</h1>
        <table>
          <tr><th>МКБ</th><th>Название</th></tr>
          <tr><td>O14.2</td><td>HELLP-синдром</td></tr>
        </table></div>"""
        cleaned = clean(html, source_url="https://example.com")
        md = convert(cleaned)
        assert "| МКБ |" in md
        assert "| O14.2 |" in md
        assert "HELLP-синдром" in md


class TestJavascriptLinks:
    def test_javascript_links_stripped(self):
        from pipeline.parser.html_cleaner import clean
        html = '<div><h1>Title</h1><a href="javascript:window.print()">Версия для печати</a><p>Content.</p></div>'
        result = clean(html, source_url="https://example.com")
        assert "javascript:" not in result
        assert "Версия для печати" not in result

    def test_normal_links_preserved(self):
        from pipeline.parser.html_cleaner import clean
        html = '<div><h1>Title</h1><a href="https://example.com/ref">Reference</a><p>Content.</p></div>'
        result = clean(html, source_url="https://example.com")
        assert "Reference" in result


class TestConvert:
    def test_table_preserved(self):
        html = """<table>
          <tr><th>Препарат</th><th>Доза</th></tr>
          <tr><td>Магния сульфат</td><td>4 г в/в</td></tr>
        </table>"""
        result = convert(html)
        assert "Магния сульфат" in result
        assert "4 г" in result

    def test_cyrillic_preserved(self):
        html = "<p>Акушерство и гинекология — клинический протокол.</p>"
        result = convert(html)
        assert "Акушерство" in result

    def test_boilerplate_stripped_by_cleaner(self):
        from pipeline.parser.html_cleaner import clean
        from pathlib import Path
        html = Path("tests/fixtures/sample_protocol.html").read_text(encoding="utf-8")
        cleaned = clean(html, source_url="https://example.com/test")
        md = convert(cleaned)
        assert "не предназначен для самолечения" not in md
