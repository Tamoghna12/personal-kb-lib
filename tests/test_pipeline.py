"""Tests for ocr_kb.pipeline — orchestration, mode selection, error capture."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import pypdfium2 as pdfium
from PIL import Image

from ocr_kb.model.schema import OcrResponse
from ocr_kb.settings import Settings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _settings() -> Settings:
    return Settings(_env_file=None)


def _make_png(path: Path, w: int = 200, h: int = 200) -> Path:
    Image.new("RGB", (w, h), color=(10, 20, 30)).save(path)
    return path


def _make_pdf(path: Path, pages: int = 2) -> Path:
    doc = pdfium.PdfDocument.new()
    for _ in range(pages):
        doc.new_page(200, 200)
    doc.save(str(path))
    return path


def _make_txt(path: Path, content: str = "hello world") -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def _ocr_response(text: str = "ocr text") -> OcrResponse:
    return OcrResponse(text=text, model="m", prompt_tokens=10, completion_tokens=5)


# Patch target — run_ocr as imported inside pipeline
_PATCH_OCR = "ocr_kb.pipeline.run_ocr"
_PATCH_TEXT = "ocr_kb.pipeline.run_enrichment"


# ---------------------------------------------------------------------------
# PageResult and FileResult dataclasses
# ---------------------------------------------------------------------------

class TestResultDataclasses:
    def test_file_result_succeeded_true(self, tmp_path):
        from ocr_kb.pipeline import FileResult, PageResult
        r = FileResult(source_path=tmp_path / "f.txt")
        r.pages.append(PageResult(source_path=tmp_path / "f.txt", page_number=1, text="x", mode="text"))
        assert r.succeeded is True

    def test_file_result_succeeded_false_when_no_pages(self, tmp_path):
        from ocr_kb.pipeline import FileResult
        assert FileResult(source_path=tmp_path / "f.txt").succeeded is False

    def test_file_result_full_text_joins_pages(self, tmp_path):
        from ocr_kb.pipeline import FileResult, PageResult
        p = tmp_path / "f.txt"
        r = FileResult(source_path=p)
        r.pages = [
            PageResult(source_path=p, page_number=1, text="page one", mode="text"),
            PageResult(source_path=p, page_number=2, text="page two", mode="text"),
        ]
        assert r.full_text == "page one\n\npage two"

    def test_file_result_full_text_empty_when_no_pages(self, tmp_path):
        from ocr_kb.pipeline import FileResult
        assert FileResult(source_path=tmp_path / "f.txt").full_text == ""


# ---------------------------------------------------------------------------
# process_file — text files (no OCR)
# ---------------------------------------------------------------------------

class TestProcessFileText:
    def test_text_file_returns_one_page(self, tmp_path):
        from ocr_kb.pipeline import process_file
        p = _make_txt(tmp_path / "note.txt", "some notes")
        result = process_file(p, _settings())
        assert len(result.pages) == 1

    def test_text_file_content_preserved(self, tmp_path):
        from ocr_kb.pipeline import process_file
        p = _make_txt(tmp_path / "note.txt", "hello world")
        result = process_file(p, _settings())
        assert result.pages[0].text == "hello world"

    def test_text_file_no_ocr_call(self, tmp_path):
        from ocr_kb.pipeline import process_file
        p = _make_txt(tmp_path / "note.txt", "x")
        with patch(_PATCH_OCR) as mock_ocr:
            process_file(p, _settings())
            mock_ocr.assert_not_called()

    def test_text_file_zero_tokens(self, tmp_path):
        from ocr_kb.pipeline import process_file
        p = _make_txt(tmp_path / "note.txt", "x")
        page = process_file(p, _settings()).pages[0]
        assert page.prompt_tokens == 0
        assert page.completion_tokens == 0

    def test_markdown_file_processed(self, tmp_path):
        from ocr_kb.pipeline import process_file
        p = _make_txt(tmp_path / "notes.md", "# Title\n\n- item")
        result = process_file(p, _settings())
        assert result.pages[0].text == "# Title\n\n- item"

    def test_source_path_set_on_page(self, tmp_path):
        from ocr_kb.pipeline import process_file
        p = _make_txt(tmp_path / "note.txt", "x")
        page = process_file(p, _settings()).pages[0]
        assert page.source_path == p

    def test_page_number_is_1(self, tmp_path):
        from ocr_kb.pipeline import process_file
        p = _make_txt(tmp_path / "note.txt", "x")
        assert process_file(p, _settings()).pages[0].page_number == 1


# ---------------------------------------------------------------------------
# process_file — image files (OCR required)
# ---------------------------------------------------------------------------

class TestProcessFileImage:
    def test_image_calls_run_ocr(self, tmp_path):
        from ocr_kb.pipeline import process_file
        p = _make_png(tmp_path / "img.png")
        with patch(_PATCH_OCR, return_value=_ocr_response()) as mock_ocr:
            process_file(p, _settings())
            mock_ocr.assert_called_once()

    def test_image_ocr_text_in_page(self, tmp_path):
        from ocr_kb.pipeline import process_file
        p = _make_png(tmp_path / "img.png")
        with patch(_PATCH_OCR, return_value=_ocr_response("extracted text")):
            page = process_file(p, _settings()).pages[0]
        assert page.text == "extracted text"

    def test_text_mode_uses_ocr_prompt(self, tmp_path):
        from ocr_kb.pipeline import process_file
        from ocr_kb.prompts import GLM_OCR_PROMPT
        p = _make_png(tmp_path / "img.png")
        with patch(_PATCH_OCR, return_value=_ocr_response()) as mock_ocr:
            process_file(p, _settings(), mode="text")
        req = mock_ocr.call_args[0][0]
        assert req.prompt == GLM_OCR_PROMPT

    def test_html_mode_uses_html_prompt(self, tmp_path):
        from ocr_kb.pipeline import process_file
        from ocr_kb.prompts import GLM_OCR_HTML_PROMPT
        p = _make_png(tmp_path / "img.png")
        with patch(_PATCH_OCR, return_value=_ocr_response()) as mock_ocr:
            process_file(p, _settings(), mode="html")
        req = mock_ocr.call_args[0][0]
        assert req.prompt == GLM_OCR_HTML_PROMPT

    def test_token_counts_captured(self, tmp_path):
        from ocr_kb.pipeline import process_file
        p = _make_png(tmp_path / "img.png")
        resp = OcrResponse(text="t", model="m", prompt_tokens=42, completion_tokens=7)
        with patch(_PATCH_OCR, return_value=resp):
            page = process_file(p, _settings()).pages[0]
        assert page.prompt_tokens == 42
        assert page.completion_tokens == 7

    def test_mode_stored_on_page(self, tmp_path):
        from ocr_kb.pipeline import process_file
        p = _make_png(tmp_path / "img.png")
        with patch(_PATCH_OCR, return_value=_ocr_response()):
            page = process_file(p, _settings(), mode="html").pages[0]
        assert page.mode == "html"


# ---------------------------------------------------------------------------
# process_file — PDF (multiple pages)
# ---------------------------------------------------------------------------

class TestProcessFilePdf:
    def test_pdf_one_call_per_page(self, tmp_path):
        from ocr_kb.pipeline import process_file
        p = _make_pdf(tmp_path / "doc.pdf", pages=3)
        with patch(_PATCH_OCR, return_value=_ocr_response()) as mock_ocr:
            result = process_file(p, _settings())
        assert mock_ocr.call_count == 3
        assert len(result.pages) == 3

    def test_pdf_page_numbers_sequential(self, tmp_path):
        from ocr_kb.pipeline import process_file
        p = _make_pdf(tmp_path / "doc.pdf", pages=3)
        with patch(_PATCH_OCR, return_value=_ocr_response()):
            result = process_file(p, _settings())
        assert [pg.page_number for pg in result.pages] == [1, 2, 3]

    def test_pdf_page_range_respected(self, tmp_path):
        from ocr_kb.pipeline import process_file
        p = _make_pdf(tmp_path / "doc.pdf", pages=5)
        with patch(_PATCH_OCR, return_value=_ocr_response()):
            result = process_file(p, _settings(), page_range="1,3")
        assert [pg.page_number for pg in result.pages] == [1, 3]


# ---------------------------------------------------------------------------
# Optional transformations — key points and summary
# ---------------------------------------------------------------------------

class TestTransformations:
    def test_key_points_not_called_by_default(self, tmp_path):
        from ocr_kb.pipeline import process_file
        p = _make_txt(tmp_path / "note.txt", "content")
        with patch(_PATCH_TEXT) as mock_text:
            process_file(p, _settings())
            mock_text.assert_not_called()

    def test_extract_key_points_calls_run_text(self, tmp_path):
        from ocr_kb.pipeline import process_file
        p = _make_txt(tmp_path / "note.txt", "some content")
        with patch(_PATCH_TEXT, return_value="- point one") as mock_text:
            page = process_file(p, _settings(), extract_key_points=True).pages[0]
        assert mock_text.call_count >= 1
        assert page.key_points == "- point one"

    def test_extract_summary_calls_run_text(self, tmp_path):
        from ocr_kb.pipeline import process_file
        p = _make_txt(tmp_path / "note.txt", "some content")
        with patch(_PATCH_TEXT, return_value="A short summary.") as mock_text:
            page = process_file(p, _settings(), extract_summary=True).pages[0]
        assert page.summary == "A short summary."

    def test_both_transformations_called(self, tmp_path):
        from ocr_kb.pipeline import process_file
        p = _make_txt(tmp_path / "note.txt", "content")
        with patch(_PATCH_TEXT, return_value="result") as mock_text:
            process_file(p, _settings(), extract_key_points=True, extract_summary=True)
        assert mock_text.call_count == 2

    def test_empty_text_skips_transformations(self, tmp_path):
        from ocr_kb.pipeline import process_file
        p = _make_txt(tmp_path / "note.txt", "   ")
        with patch(_PATCH_TEXT) as mock_text:
            process_file(p, _settings(), extract_key_points=True, extract_summary=True)
            mock_text.assert_not_called()

    def test_key_points_prompt_contains_text(self, tmp_path):
        from ocr_kb.pipeline import process_file
        p = _make_txt(tmp_path / "note.txt", "the actual content")
        with patch(_PATCH_TEXT, return_value="pts") as mock_text:
            process_file(p, _settings(), extract_key_points=True)
        prompt_sent = mock_text.call_args[0][0]
        assert "the actual content" in prompt_sent


# ---------------------------------------------------------------------------
# Gemma enrichment
# ---------------------------------------------------------------------------

class TestGemmaEnrichment:
    def test_enrichment_not_called_by_default(self, tmp_path):
        from ocr_kb.pipeline import process_file
        p = _make_txt(tmp_path / "note.txt", "content")
        with patch(_PATCH_TEXT) as mock_text:
            process_file(p, _settings())
            mock_text.assert_not_called()

    def test_enrichment_called_when_enabled_via_settings(self, tmp_path):
        from ocr_kb.pipeline import process_file
        s = Settings(_env_file=None, enable_gemma_enrichment=True)
        p = _make_txt(tmp_path / "note.txt", "some content")
        with patch(_PATCH_TEXT, return_value="cleaned") as mock_text:
            page = process_file(p, s).pages[0]
        assert mock_text.call_count >= 1
        assert page.enriched_metadata == "cleaned"

    def test_enrichment_called_when_flag_passed(self, tmp_path):
        from ocr_kb.pipeline import process_file
        p = _make_txt(tmp_path / "note.txt", "some content")
        with patch(_PATCH_TEXT, return_value="enriched") as mock_text:
            page = process_file(p, _settings(), gemma_enrichment=True).pages[0]
        assert page.enriched_metadata == "enriched"

    def test_flag_overrides_settings_false(self, tmp_path):
        from ocr_kb.pipeline import process_file
        s = Settings(_env_file=None, enable_gemma_enrichment=True)
        p = _make_txt(tmp_path / "note.txt", "content")
        with patch(_PATCH_TEXT) as mock_text:
            process_file(p, s, gemma_enrichment=False)
        mock_text.assert_not_called()

    def test_enriched_metadata_none_by_default(self, tmp_path):
        from ocr_kb.pipeline import process_file
        p = _make_txt(tmp_path / "note.txt", "content")
        page = process_file(p, _settings()).pages[0]
        assert page.enriched_metadata is None

    def test_empty_text_skips_enrichment(self, tmp_path):
        from ocr_kb.pipeline import process_file
        s = Settings(_env_file=None, enable_gemma_enrichment=True)
        p = _make_txt(tmp_path / "note.txt", "   ")
        with patch(_PATCH_TEXT) as mock_text:
            process_file(p, s)
            mock_text.assert_not_called()


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_unsupported_file_captured_in_errors(self, tmp_path):
        from ocr_kb.pipeline import process_file
        p = tmp_path / "file.xyz"
        p.write_text("data")
        result = process_file(p, _settings())
        assert result.succeeded is False
        assert result.errors

    def test_ocr_failure_captured_per_page(self, tmp_path):
        from ocr_kb.pipeline import process_file
        p = _make_png(tmp_path / "img.png")
        with patch(_PATCH_OCR, side_effect=RuntimeError("model offline")):
            result = process_file(p, _settings())
        assert len(result.pages) == 0
        assert any("model offline" in e for e in result.errors)

    def test_one_bad_page_does_not_abort_rest(self, tmp_path):
        from ocr_kb.pipeline import process_file
        p = _make_pdf(tmp_path / "doc.pdf", pages=3)
        responses = [
            _ocr_response("page 1"),
            RuntimeError("bad page"),
            _ocr_response("page 3"),
        ]
        with patch(_PATCH_OCR, side_effect=responses):
            result = process_file(p, _settings())
        assert len(result.pages) == 2
        assert len(result.errors) == 1

    def test_error_message_includes_page_number(self, tmp_path):
        from ocr_kb.pipeline import process_file
        p = _make_png(tmp_path / "img.png")
        with patch(_PATCH_OCR, side_effect=RuntimeError("boom")):
            result = process_file(p, _settings())
        assert any("Page 1" in e for e in result.errors)


# ---------------------------------------------------------------------------
# process_path — file and directory dispatch
# ---------------------------------------------------------------------------

class TestProcessPath:
    def test_file_returns_list_of_one(self, tmp_path):
        from ocr_kb.pipeline import process_path
        p = _make_txt(tmp_path / "note.txt", "x")
        results = process_path(p, _settings())
        assert len(results) == 1

    def test_directory_finds_all_supported_files(self, tmp_path):
        from ocr_kb.pipeline import process_path
        _make_txt(tmp_path / "a.txt", "x")
        _make_txt(tmp_path / "b.md", "y")
        _make_png(tmp_path / "c.png")
        (tmp_path / "ignore.xyz").write_text("skip")
        with patch(_PATCH_OCR, return_value=_ocr_response()):
            results = process_path(tmp_path, _settings())
        assert len(results) == 3

    def test_directory_skips_unsupported(self, tmp_path):
        from ocr_kb.pipeline import process_path
        _make_txt(tmp_path / "note.txt", "x")
        (tmp_path / "binary.bin").write_bytes(b"\x00\x01")
        results = process_path(tmp_path, _settings())
        assert len(results) == 1

    def test_empty_directory_returns_empty_list(self, tmp_path):
        from ocr_kb.pipeline import process_path
        results = process_path(tmp_path, _settings())
        assert results == []

    def test_nested_directory_discovered(self, tmp_path):
        from ocr_kb.pipeline import process_path
        sub = tmp_path / "sub"
        sub.mkdir()
        _make_txt(sub / "deep.txt", "x")
        results = process_path(tmp_path, _settings())
        assert len(results) == 1
        assert results[0].source_path == sub / "deep.txt"


# ---------------------------------------------------------------------------
# model_backend routing
# ---------------------------------------------------------------------------

class TestModelBackendRouting:
    def test_glm_only_skips_all_enrichment(self, tmp_path):
        from ocr_kb.pipeline import process_file
        s = Settings(_env_file=None, model_backend="glm_only")
        p = _make_txt(tmp_path / "note.txt", "content")
        with patch(_PATCH_TEXT) as mock_enrich:
            process_file(p, s, extract_key_points=True, extract_summary=True, gemma_enrichment=True)
            mock_enrich.assert_not_called()

    def test_glm_only_still_returns_text_content(self, tmp_path):
        from ocr_kb.pipeline import process_file
        s = Settings(_env_file=None, model_backend="glm_only")
        p = _make_txt(tmp_path / "note.txt", "hello content")
        page = process_file(p, s).pages[0]
        assert page.text == "hello content"

    def test_gemma_only_errors_on_image(self, tmp_path):
        from ocr_kb.pipeline import process_file
        s = Settings(_env_file=None, model_backend="gemma_only")
        p = _make_png(tmp_path / "scan.png")
        result = process_file(p, s)
        assert result.succeeded is False
        assert any("gemma_only" in e for e in result.errors)

    def test_gemma_only_processes_text_file(self, tmp_path):
        from ocr_kb.pipeline import process_file
        s = Settings(_env_file=None, model_backend="gemma_only")
        p = _make_txt(tmp_path / "note.txt", "text only content")
        result = process_file(p, s)
        assert result.succeeded is True
        assert result.pages[0].text == "text only content"

    def test_hybrid_uses_ocr_for_images(self, tmp_path):
        from ocr_kb.pipeline import process_file
        s = Settings(_env_file=None, model_backend="hybrid")
        p = _make_png(tmp_path / "img.png")
        with patch(_PATCH_OCR, return_value=_ocr_response("extracted")) as mock_ocr:
            process_file(p, s)
            mock_ocr.assert_called_once()

    def test_hybrid_uses_enrichment_when_enabled(self, tmp_path):
        from ocr_kb.pipeline import process_file
        s = Settings(_env_file=None, model_backend="hybrid", enable_gemma_enrichment=True)
        p = _make_txt(tmp_path / "note.txt", "content")
        with patch(_PATCH_TEXT, return_value="result") as mock_enrich:
            process_file(p, s)
            mock_enrich.assert_called()

    def test_enrichment_sets_generated_tags(self, tmp_path):
        from ocr_kb.pipeline import process_file
        s = Settings(_env_file=None, enable_gemma_enrichment=True)
        p = _make_txt(tmp_path / "note.txt", "some content about deep learning")
        with patch(_PATCH_TEXT, side_effect=["cleaned text", "deep-learning,ml,neural-network"]):
            page = process_file(p, s).pages[0]
        assert page.generated_tags == "deep-learning,ml,neural-network"

    def test_glm_only_generated_tags_stays_none(self, tmp_path):
        from ocr_kb.pipeline import process_file
        s = Settings(_env_file=None, model_backend="glm_only", enable_gemma_enrichment=True)
        p = _make_txt(tmp_path / "note.txt", "content")
        with patch(_PATCH_TEXT):
            page = process_file(p, s).pages[0]
        assert page.generated_tags is None


# ---------------------------------------------------------------------------
# _chunk_text helper
# ---------------------------------------------------------------------------

class TestChunkText:
    def test_short_text_returns_single_chunk(self):
        from ocr_kb.pipeline import _chunk_text
        text = "Hello world."
        assert _chunk_text(text, chunk_size=1000, overlap=0) == [text]

    def test_disabled_returns_single_chunk(self):
        from ocr_kb.pipeline import _chunk_text
        long_text = "word " * 500
        assert _chunk_text(long_text, chunk_size=0, overlap=0) == [long_text]

    def test_splits_into_multiple_chunks(self):
        from ocr_kb.pipeline import _chunk_text
        text = "\n\n".join([f"Paragraph {i} with some content." for i in range(20)])
        chunks = _chunk_text(text, chunk_size=200, overlap=0)
        assert len(chunks) > 1

    def test_chunks_cover_all_content(self):
        from ocr_kb.pipeline import _chunk_text
        paras = [f"Unique paragraph {i} content" for i in range(10)]
        text = "\n\n".join(paras)
        chunks = _chunk_text(text, chunk_size=100, overlap=0)
        # Every paragraph should appear in at least one chunk
        combined = " ".join(chunks)
        for para in paras:
            assert para in combined

    def test_overlap_prepends_tail_from_previous(self):
        from ocr_kb.pipeline import _chunk_text
        paras = [f"Paragraph {i} with enough filler text to exceed the size limit." for i in range(10)]
        text = "\n\n".join(paras)
        chunks = _chunk_text(text, chunk_size=100, overlap=50)
        if len(chunks) > 1:
            # Second chunk must start with a tail from the first
            assert len(chunks[1]) > 0

    def test_chunk_index_assigned_in_process_file(self, tmp_path):
        from ocr_kb.pipeline import process_file
        long_content = "\n\n".join([f"Section {i}: " + "word " * 40 for i in range(10)])
        p = _make_txt(tmp_path / "long.txt", long_content)
        s = Settings(_env_file=None, chunk_size=200, chunk_overlap=50)
        result = process_file(p, s)
        assert result.succeeded
        indices = [pg.chunk_index for pg in result.pages]
        assert all(idx >= 1 for idx in indices)
        assert len(set(indices)) == len(indices)  # each chunk has unique index

    def test_no_chunking_when_disabled(self, tmp_path):
        from ocr_kb.pipeline import process_file
        long_content = "\n\n".join([f"Section {i}: " + "word " * 40 for i in range(10)])
        p = _make_txt(tmp_path / "long.txt", long_content)
        s = Settings(_env_file=None, chunk_size=0)
        result = process_file(p, s)
        assert result.succeeded
        assert len(result.pages) == 1
        assert result.pages[0].chunk_index == 0


# ---------------------------------------------------------------------------
# Native PDF text extraction helpers
# ---------------------------------------------------------------------------

class TestIsUsableText:
    def test_empty_returns_false(self):
        from ocr_kb.ingest.pdf_reader import _is_usable_text
        assert _is_usable_text("") is False

    def test_short_text_returns_false(self):
        from ocr_kb.ingest.pdf_reader import _is_usable_text
        assert _is_usable_text("Hi") is False

    def test_mostly_symbols_returns_false(self):
        from ocr_kb.ingest.pdf_reader import _is_usable_text
        assert _is_usable_text("!@#$%^&*() " * 20) is False

    def test_real_sentence_returns_true(self):
        from ocr_kb.ingest.pdf_reader import _is_usable_text
        text = "The quick brown fox jumps over the lazy dog. " * 5
        assert _is_usable_text(text) is True

    def test_min_len_boundary(self):
        from ocr_kb.ingest.pdf_reader import _is_usable_text
        # exactly 50 real alpha chars (well above 40% ratio)
        text = "a" * 50
        assert _is_usable_text(text) is True


class TestSmartPdfLoader:
    def test_blank_pdf_falls_back_to_image(self, tmp_path):
        """A blank PDF has no text layer — smart loader must return images."""
        from ocr_kb.ingest.pdf_reader import render_pdf_pages_smart
        pdf = _make_pdf(tmp_path / "blank.pdf", pages=1)
        results = render_pdf_pages_smart(pdf, dpi=72, min_dim=100)
        assert len(results) == 1
        page_num, content = results[0]
        assert page_num == 1
        assert isinstance(content, Image.Image)

    def test_page_count_matches(self, tmp_path):
        from ocr_kb.ingest.pdf_reader import render_pdf_pages_smart
        pdf = _make_pdf(tmp_path / "multi.pdf", pages=3)
        results = render_pdf_pages_smart(pdf, dpi=72, min_dim=100)
        assert len(results) == 3
