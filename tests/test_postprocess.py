"""Tests for ocr_kb.postprocess — HTML cleaning, Markdown conversion, layout parsing, image extraction."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import pytest
from PIL import Image


# ---------------------------------------------------------------------------
# html_parser
# ---------------------------------------------------------------------------

class TestCleanHtml:
    def test_removes_script_tags(self):
        from ocr_kb.postprocess.html_parser import clean_html
        raw = "<p>text</p><script>alert('xss')</script>"
        result = clean_html(raw)
        assert "<script>" not in result
        assert "text" in result

    def test_removes_style_tags(self):
        from ocr_kb.postprocess.html_parser import clean_html
        raw = "<p>text</p><style>body{color:red}</style>"
        result = clean_html(raw)
        assert "<style>" not in result
        assert "text" in result

    def test_removes_meta_and_link(self):
        from ocr_kb.postprocess.html_parser import clean_html
        raw = "<head><meta charset='utf-8'><link rel='stylesheet' href='x.css'></head><p>ok</p>"
        result = clean_html(raw)
        assert "<meta" not in result
        assert "<link" not in result

    def test_preserves_semantic_tags(self):
        from ocr_kb.postprocess.html_parser import clean_html
        raw = "<h1>Title</h1><p>Para</p><ul><li>item</li></ul>"
        result = clean_html(raw)
        assert "<h1>" in result
        assert "<p>" in result
        assert "<li>" in result

    def test_collapses_whitespace_in_text(self):
        from ocr_kb.postprocess.html_parser import clean_html
        raw = "<p>hello    world</p>"
        result = clean_html(raw)
        assert "hello    world" not in result
        assert "hello world" in result

    def test_does_not_collapse_whitespace_in_pre(self):
        from ocr_kb.postprocess.html_parser import clean_html
        raw = "<pre>  indented  code  </pre>"
        result = clean_html(raw)
        assert "  indented  code  " in result

    def test_empty_string_returns_string(self):
        from ocr_kb.postprocess.html_parser import clean_html
        assert isinstance(clean_html(""), str)


class TestStripToFragment:
    def test_strips_html_body_wrapper(self):
        from ocr_kb.postprocess.html_parser import strip_to_fragment
        full = "<html><head></head><body><p>content</p></body></html>"
        result = strip_to_fragment(full)
        assert "<html>" not in result
        assert "<body>" not in result
        assert "content" in result

    def test_returns_fragment_unchanged(self):
        from ocr_kb.postprocess.html_parser import strip_to_fragment
        fragment = "<h1>Title</h1><p>Para</p>"
        result = strip_to_fragment(fragment)
        assert "Title" in result
        assert "Para" in result

    def test_empty_body_returns_empty(self):
        from ocr_kb.postprocess.html_parser import strip_to_fragment
        result = strip_to_fragment("<html><body></body></html>")
        assert result.strip() == ""


class TestIsBlank:
    def test_empty_string_is_blank(self):
        from ocr_kb.postprocess.html_parser import is_blank
        assert is_blank("") is True

    def test_whitespace_only_is_blank(self):
        from ocr_kb.postprocess.html_parser import is_blank
        assert is_blank("   \n\t  ") is True

    def test_tags_only_no_text_is_blank(self):
        from ocr_kb.postprocess.html_parser import is_blank
        assert is_blank("<p></p><div></div>") is True

    def test_short_text_under_threshold_is_blank(self):
        from ocr_kb.postprocess.html_parser import is_blank
        assert is_blank("hi") is True  # under 10 chars

    def test_content_page_is_not_blank(self):
        from ocr_kb.postprocess.html_parser import is_blank
        assert is_blank("<p>This is a real paragraph with content.</p>") is False

    def test_html_with_text_not_blank(self):
        from ocr_kb.postprocess.html_parser import is_blank
        assert is_blank("<h1>Introduction</h1>") is False


class TestExtractText:
    def test_extracts_visible_text(self):
        from ocr_kb.postprocess.html_parser import extract_text
        html = "<h1>Title</h1><p>Body text here.</p>"
        result = extract_text(html)
        assert "Title" in result
        assert "Body text here." in result

    def test_strips_tags(self):
        from ocr_kb.postprocess.html_parser import extract_text
        html = "<p>hello <strong>world</strong></p>"
        result = extract_text(html)
        assert "<strong>" not in result
        assert "hello world" in result

    def test_normalises_whitespace(self):
        from ocr_kb.postprocess.html_parser import extract_text
        result = extract_text("<p>  too   many   spaces  </p>")
        assert "  " not in result


# ---------------------------------------------------------------------------
# markdown_converter
# ---------------------------------------------------------------------------

class TestHtmlToMarkdown:
    def test_h1_becomes_atx_heading(self):
        from ocr_kb.postprocess.markdown_converter import html_to_markdown
        result = html_to_markdown("<h1>Title</h1>")
        assert result.startswith("# Title")

    def test_h2_heading(self):
        from ocr_kb.postprocess.markdown_converter import html_to_markdown
        result = html_to_markdown("<h2>Sub</h2>")
        assert "## Sub" in result

    def test_paragraph_preserved(self):
        from ocr_kb.postprocess.markdown_converter import html_to_markdown
        result = html_to_markdown("<p>Hello world</p>")
        assert "Hello world" in result

    def test_ul_becomes_bullets(self):
        from ocr_kb.postprocess.markdown_converter import html_to_markdown
        result = html_to_markdown("<ul><li>a</li><li>b</li></ul>")
        assert "- a" in result
        assert "- b" in result

    def test_script_stripped(self):
        from ocr_kb.postprocess.markdown_converter import html_to_markdown
        result = html_to_markdown("<p>text</p><script>bad()</script>")
        assert "bad()" not in result


class TestCleanMarkdown:
    def test_collapses_triple_newlines(self):
        from ocr_kb.postprocess.markdown_converter import clean_markdown
        result = clean_markdown("a\n\n\n\nb")
        assert "\n\n\n" not in result

    def test_strips_leading_trailing(self):
        from ocr_kb.postprocess.markdown_converter import clean_markdown
        assert clean_markdown("\n\nhello\n\n") == "hello"

    def test_preserves_double_newline(self):
        from ocr_kb.postprocess.markdown_converter import clean_markdown
        result = clean_markdown("para one\n\npara two")
        assert "para one\n\npara two" == result


class TestPlainToMarkdown:
    def test_strips_and_collapses(self):
        from ocr_kb.postprocess.markdown_converter import plain_to_markdown
        result = plain_to_markdown("\n\n\nsome text\n\n\n")
        assert result == "some text"

    def test_returns_string(self):
        from ocr_kb.postprocess.markdown_converter import plain_to_markdown
        assert isinstance(plain_to_markdown(""), str)


class TestConvertMathBlocks:
    def test_inline_latex_parens(self):
        from ocr_kb.postprocess.markdown_converter import convert_math_blocks
        result = convert_math_blocks(r"formula \(x^2\) here")
        assert "$x^2$" in result
        assert r"\(" not in result

    def test_display_latex_brackets(self):
        from ocr_kb.postprocess.markdown_converter import convert_math_blocks
        result = convert_math_blocks(r"see \[E=mc^2\] below")
        assert "$$E=mc^2$$" in result
        assert r"\[" not in result

    def test_no_math_unchanged(self):
        from ocr_kb.postprocess.markdown_converter import convert_math_blocks
        text = "plain text no math"
        assert convert_math_blocks(text) == text

    def test_multiple_inline_converted(self):
        from ocr_kb.postprocess.markdown_converter import convert_math_blocks
        result = convert_math_blocks(r"\(a\) and \(b\)")
        assert result.count("$") == 4  # two pairs


# ---------------------------------------------------------------------------
# layout_parser
# ---------------------------------------------------------------------------

class TestParseLayout:
    def test_empty_html_returns_empty(self):
        from ocr_kb.postprocess.layout_parser import parse_layout
        assert parse_layout("") == []

    def test_h1_classified_as_heading_level_1(self):
        from ocr_kb.postprocess.layout_parser import parse_layout
        blocks = parse_layout("<h1>Title</h1>")
        assert len(blocks) == 1
        assert blocks[0].type == "heading"
        assert blocks[0].level == 1
        assert blocks[0].content == "Title"

    def test_h2_heading_level_2(self):
        from ocr_kb.postprocess.layout_parser import parse_layout
        blocks = parse_layout("<h2>Sub</h2>")
        assert blocks[0].level == 2

    def test_h3_heading_level_3(self):
        from ocr_kb.postprocess.layout_parser import parse_layout
        blocks = parse_layout("<h3>Sub-sub</h3>")
        assert blocks[0].level == 3

    def test_paragraph_classified(self):
        from ocr_kb.postprocess.layout_parser import parse_layout
        blocks = parse_layout("<p>Some paragraph text.</p>")
        assert blocks[0].type == "paragraph"
        assert blocks[0].level is None

    def test_empty_paragraph_skipped(self):
        from ocr_kb.postprocess.layout_parser import parse_layout
        blocks = parse_layout("<p></p><p>real content</p>")
        assert len(blocks) == 1

    def test_ul_classified_as_list(self):
        from ocr_kb.postprocess.layout_parser import parse_layout
        blocks = parse_layout("<ul><li>a</li><li>b</li></ul>")
        assert blocks[0].type == "list"
        assert "- a" in blocks[0].content
        assert "- b" in blocks[0].content

    def test_ol_classified_as_list(self):
        from ocr_kb.postprocess.layout_parser import parse_layout
        blocks = parse_layout("<ol><li>first</li><li>second</li></ol>")
        assert blocks[0].type == "list"

    def test_table_classified(self):
        from ocr_kb.postprocess.layout_parser import parse_layout
        html = "<table><tr><td>A</td><td>B</td></tr></table>"
        blocks = parse_layout(html)
        assert blocks[0].type == "table"
        assert "table" in blocks[0].content

    def test_figure_classified(self):
        from ocr_kb.postprocess.layout_parser import parse_layout
        blocks = parse_layout("<figure><img src='x.jpg'/></figure>")
        assert blocks[0].type == "figure"

    def test_img_classified_as_figure(self):
        from ocr_kb.postprocess.layout_parser import parse_layout
        blocks = parse_layout("<img src='x.jpg'/>")
        assert blocks[0].type == "figure"

    def test_pre_classified_as_code(self):
        from ocr_kb.postprocess.layout_parser import parse_layout
        blocks = parse_layout("<pre>def foo(): pass</pre>")
        assert blocks[0].type == "code"

    def test_order_preserved(self):
        from ocr_kb.postprocess.layout_parser import parse_layout
        html = "<h1>A</h1><p>B</p><ul><li>C</li></ul>"
        blocks = parse_layout(html)
        assert [b.type for b in blocks] == ["heading", "paragraph", "list"]

    def test_body_wrapper_transparent(self):
        from ocr_kb.postprocess.layout_parser import parse_layout
        html = "<body><h1>Title</h1><p>Para</p></body>"
        blocks = parse_layout(html)
        assert len(blocks) == 2

    def test_index_assigned(self):
        from ocr_kb.postprocess.layout_parser import parse_layout
        blocks = parse_layout("<h1>A</h1><p>B</p>")
        assert all(isinstance(b.index, int) for b in blocks)


# ---------------------------------------------------------------------------
# image_extractor
# ---------------------------------------------------------------------------

def _png_data_uri() -> str:
    img = Image.new("RGB", (10, 10), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


class TestSavePageImage:
    def test_creates_file(self, tmp_path):
        from ocr_kb.postprocess.image_extractor import save_page_image
        img = Image.new("RGB", (100, 100))
        out = save_page_image(img, Path("doc.pdf"), 1, tmp_path / "out")
        assert out.exists()

    def test_filename_convention(self, tmp_path):
        from ocr_kb.postprocess.image_extractor import save_page_image
        img = Image.new("RGB", (100, 100))
        out = save_page_image(img, Path("my_doc.pdf"), 3, tmp_path / "out")
        assert out.name == "my_doc_p003.jpg"

    def test_creates_output_dir(self, tmp_path):
        from ocr_kb.postprocess.image_extractor import save_page_image
        img = Image.new("RGB", (100, 100))
        deep = tmp_path / "a" / "b" / "c"
        save_page_image(img, Path("f.pdf"), 1, deep)
        assert deep.exists()

    def test_output_is_valid_jpeg(self, tmp_path):
        from ocr_kb.postprocess.image_extractor import save_page_image
        img = Image.new("RGB", (100, 100))
        out = save_page_image(img, Path("f.pdf"), 1, tmp_path)
        loaded = Image.open(out)
        assert loaded.format == "JPEG"


class TestExtractImagesFromHtml:
    def test_extracts_data_uri_image(self, tmp_path):
        from ocr_kb.postprocess.image_extractor import extract_images_from_html
        uri = _png_data_uri()
        html = f'<p>text</p><img src="{uri}"/>'
        saved = extract_images_from_html(html, tmp_path / "out")
        assert len(saved) == 1
        assert saved[0].exists()

    def test_skips_non_data_uri(self, tmp_path):
        from ocr_kb.postprocess.image_extractor import extract_images_from_html
        html = '<img src="https://example.com/img.jpg"/>'
        saved = extract_images_from_html(html, tmp_path / "out")
        assert saved == []

    def test_multiple_images_extracted(self, tmp_path):
        from ocr_kb.postprocess.image_extractor import extract_images_from_html
        uri = _png_data_uri()
        html = f'<img src="{uri}"/><img src="{uri}"/>'
        saved = extract_images_from_html(html, tmp_path / "out")
        assert len(saved) == 2

    def test_prefix_in_filename(self, tmp_path):
        from ocr_kb.postprocess.image_extractor import extract_images_from_html
        uri = _png_data_uri()
        html = f'<img src="{uri}"/>'
        saved = extract_images_from_html(html, tmp_path / "out", prefix="page1")
        assert saved[0].name.startswith("page1_")

    def test_no_images_returns_empty(self, tmp_path):
        from ocr_kb.postprocess.image_extractor import extract_images_from_html
        saved = extract_images_from_html("<p>no images here</p>", tmp_path / "out")
        assert saved == []

    def test_bad_data_uri_skipped(self, tmp_path):
        from ocr_kb.postprocess.image_extractor import extract_images_from_html
        html = '<img src="data:image/png;base64,NOT_VALID_BASE64!!!"/>'
        saved = extract_images_from_html(html, tmp_path / "out")
        assert saved == []


# ---------------------------------------------------------------------------
# postprocess __init__ re-exports
# ---------------------------------------------------------------------------

def test_postprocess_init_exports():
    import ocr_kb.postprocess as pp
    for name in ["clean_html", "html_to_markdown", "parse_layout", "LayoutBlock",
                 "save_page_image", "is_blank", "strip_to_fragment"]:
        assert hasattr(pp, name)
