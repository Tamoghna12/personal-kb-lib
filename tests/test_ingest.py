"""Tests for ocr_kb.ingest — image loading, PDF rendering, loader dispatch, batch building."""

from pathlib import Path

import pytest
import pypdfium2 as pdfium
from PIL import Image

from ocr_kb.settings import Settings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _settings(**kwargs) -> Settings:
    defaults = dict(
        image_dpi=72,
        min_image_dim=100,
        min_pdf_image_dim=100,
    )
    defaults.update(kwargs)
    return Settings(_env_file=None, **defaults)


def _make_png(path: Path, width: int = 200, height: int = 200) -> Path:
    img = Image.new("RGB", (width, height), color=(128, 64, 32))
    img.save(path)
    return path


def _make_pdf(path: Path, pages: int = 2) -> Path:
    doc = pdfium.PdfDocument.new()
    for _ in range(pages):
        doc.new_page(200, 200)
    doc.save(str(path))
    return path


# ---------------------------------------------------------------------------
# image_reader
# ---------------------------------------------------------------------------

class TestLoadImage:
    def test_returns_rgb_image(self, tmp_path):
        p = _make_png(tmp_path / "img.png")
        from ocr_kb.ingest.image_reader import load_image
        img = load_image(p)
        assert img.mode == "RGB"

    def test_size_preserved(self, tmp_path):
        p = _make_png(tmp_path / "img.png", 320, 240)
        from ocr_kb.ingest.image_reader import load_image
        assert load_image(p).size == (320, 240)

    def test_missing_file_raises(self, tmp_path):
        from ocr_kb.ingest.image_reader import load_image
        with pytest.raises(Exception):
            load_image(tmp_path / "nonexistent.png")


class TestUpscaleIfNeeded:
    def test_upscales_small_image(self):
        from ocr_kb.ingest.image_reader import upscale_if_needed
        img = Image.new("RGB", (50, 80))
        result = upscale_if_needed(img, min_dim=100)
        assert min(result.size) >= 100

    def test_preserves_aspect_ratio(self):
        from ocr_kb.ingest.image_reader import upscale_if_needed
        img = Image.new("RGB", (50, 100))  # 1:2 ratio
        result = upscale_if_needed(img, min_dim=200)
        w, h = result.size
        assert abs(h / w - 2.0) < 0.05

    def test_no_upscale_when_large_enough(self):
        from ocr_kb.ingest.image_reader import upscale_if_needed
        img = Image.new("RGB", (800, 600))
        result = upscale_if_needed(img, min_dim=100)
        assert result.size == (800, 600)

    def test_exact_min_dim_is_not_upscaled(self):
        from ocr_kb.ingest.image_reader import upscale_if_needed
        img = Image.new("RGB", (100, 200))
        result = upscale_if_needed(img, min_dim=100)
        assert result.size == (100, 200)


# ---------------------------------------------------------------------------
# pdf_reader — parse_page_range
# ---------------------------------------------------------------------------

class TestParsePageRange:
    def test_single_page(self):
        from ocr_kb.ingest.pdf_reader import parse_page_range
        assert parse_page_range("1", 5) == [0]

    def test_comma_list(self):
        from ocr_kb.ingest.pdf_reader import parse_page_range
        assert parse_page_range("1,3", 5) == [0, 2]

    def test_range(self):
        from ocr_kb.ingest.pdf_reader import parse_page_range
        assert parse_page_range("2-4", 5) == [1, 2, 3]

    def test_mixed(self):
        from ocr_kb.ingest.pdf_reader import parse_page_range
        assert parse_page_range("1,3-5,7", 10) == [0, 2, 3, 4, 6]

    def test_out_of_range_dropped(self):
        from ocr_kb.ingest.pdf_reader import parse_page_range
        assert parse_page_range("1,99", 3) == [0]

    def test_deduplicates(self):
        from ocr_kb.ingest.pdf_reader import parse_page_range
        assert parse_page_range("1,1,1", 5) == [0]

    def test_result_is_sorted(self):
        from ocr_kb.ingest.pdf_reader import parse_page_range
        result = parse_page_range("5,1,3", 10)
        assert result == sorted(result)

    def test_all_out_of_range_returns_empty(self):
        from ocr_kb.ingest.pdf_reader import parse_page_range
        assert parse_page_range("10,20", 3) == []


# ---------------------------------------------------------------------------
# pdf_reader — render_pdf_pages
# ---------------------------------------------------------------------------

class TestRenderPdfPages:
    def test_renders_all_pages_by_default(self, tmp_path):
        pdf = _make_pdf(tmp_path / "doc.pdf", pages=3)
        from ocr_kb.ingest.pdf_reader import render_pdf_pages
        results = render_pdf_pages(pdf, dpi=72, min_dim=50)
        assert len(results) == 3

    def test_page_numbers_are_1_based(self, tmp_path):
        pdf = _make_pdf(tmp_path / "doc.pdf", pages=3)
        from ocr_kb.ingest.pdf_reader import render_pdf_pages
        nums = [n for n, _ in render_pdf_pages(pdf, dpi=72, min_dim=50)]
        assert nums == [1, 2, 3]

    def test_page_range_filters(self, tmp_path):
        pdf = _make_pdf(tmp_path / "doc.pdf", pages=5)
        from ocr_kb.ingest.pdf_reader import render_pdf_pages
        results = render_pdf_pages(pdf, dpi=72, min_dim=50, page_range="1,3,5")
        assert [n for n, _ in results] == [1, 3, 5]

    def test_returns_pil_images(self, tmp_path):
        pdf = _make_pdf(tmp_path / "doc.pdf", pages=1)
        from ocr_kb.ingest.pdf_reader import render_pdf_pages
        _, img = render_pdf_pages(pdf, dpi=72, min_dim=50)[0]
        assert isinstance(img, Image.Image)

    def test_upscales_below_min_dim(self, tmp_path):
        pdf = _make_pdf(tmp_path / "doc.pdf", pages=1)
        from ocr_kb.ingest.pdf_reader import render_pdf_pages
        _, img = render_pdf_pages(pdf, dpi=72, min_dim=800)[0]
        assert min(img.size) >= 800


# ---------------------------------------------------------------------------
# loader
# ---------------------------------------------------------------------------

class TestIsDetection:
    def test_is_pdf(self):
        from ocr_kb.ingest.loader import is_pdf
        assert is_pdf(Path("file.pdf")) is True
        assert is_pdf(Path("file.PDF")) is True
        assert is_pdf(Path("file.png")) is False

    def test_is_image(self):
        from ocr_kb.ingest.loader import is_image
        for ext in (".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"):
            assert is_image(Path(f"file{ext}")) is True
        assert is_image(Path("file.pdf")) is False

    def test_supported_suffixes_contains_pdf_and_images(self):
        from ocr_kb.ingest.loader import SUPPORTED_SUFFIXES
        assert ".pdf" in SUPPORTED_SUFFIXES
        assert ".png" in SUPPORTED_SUFFIXES
        assert ".jpg" in SUPPORTED_SUFFIXES


class TestLoadFile:
    def test_loads_image_returns_single_page(self, tmp_path):
        p = _make_png(tmp_path / "img.png", 200, 200)
        from ocr_kb.ingest.loader import load_file
        pages = load_file(p, _settings())
        assert len(pages) == 1
        assert pages[0][0] == 1

    def test_loads_image_returns_pil(self, tmp_path):
        p = _make_png(tmp_path / "img.png")
        from ocr_kb.ingest.loader import load_file
        _, img = load_file(p, _settings())[0]
        assert isinstance(img, Image.Image)

    def test_loads_pdf_returns_all_pages(self, tmp_path):
        p = _make_pdf(tmp_path / "doc.pdf", pages=3)
        from ocr_kb.ingest.loader import load_file
        pages = load_file(p, _settings())
        assert len(pages) == 3

    def test_loads_pdf_with_page_range(self, tmp_path):
        p = _make_pdf(tmp_path / "doc.pdf", pages=5)
        from ocr_kb.ingest.loader import load_file
        pages = load_file(p, _settings(), page_range="2-4")
        assert [n for n, _ in pages] == [2, 3, 4]

    def test_unsupported_extension_raises(self, tmp_path):
        p = tmp_path / "file.xyz"
        p.write_text("garbage")
        from ocr_kb.ingest.loader import load_file
        with pytest.raises(ValueError, match="Unsupported"):
            load_file(p, _settings())

    def test_image_upscaled_to_min_dim(self, tmp_path):
        p = _make_png(tmp_path / "small.png", 40, 40)
        from ocr_kb.ingest.loader import load_file
        _, img = load_file(p, _settings(min_image_dim=200))[0]
        assert min(img.size) >= 200


# ---------------------------------------------------------------------------
# batch_builder
# ---------------------------------------------------------------------------

class TestBuildBatch:
    def test_batch_length_matches_pages(self, tmp_path):
        p = _make_pdf(tmp_path / "doc.pdf", pages=2)
        from ocr_kb.ingest.batch_builder import build_batch
        batch = build_batch(p, _settings())
        assert len(batch) == 2

    def test_batch_item_fields(self, tmp_path):
        p = _make_png(tmp_path / "img.png")
        from ocr_kb.ingest.batch_builder import build_batch, BatchItem
        item = build_batch(p, _settings())[0]
        assert isinstance(item, BatchItem)
        assert item.source_path == p
        assert item.page_number == 1
        assert isinstance(item.image, Image.Image)

    def test_batch_page_numbers_sequential(self, tmp_path):
        p = _make_pdf(tmp_path / "doc.pdf", pages=3)
        from ocr_kb.ingest.batch_builder import build_batch
        nums = [item.page_number for item in build_batch(p, _settings())]
        assert nums == [1, 2, 3]

    def test_batch_page_range(self, tmp_path):
        p = _make_pdf(tmp_path / "doc.pdf", pages=5)
        from ocr_kb.ingest.batch_builder import build_batch
        batch = build_batch(p, _settings(), page_range="1,3")
        assert [item.page_number for item in batch] == [1, 3]

    def test_source_path_preserved_for_all_items(self, tmp_path):
        p = _make_pdf(tmp_path / "doc.pdf", pages=2)
        from ocr_kb.ingest.batch_builder import build_batch
        batch = build_batch(p, _settings())
        assert all(item.source_path == p for item in batch)

    def test_prefer_pdf_text_blank_pdf_uses_image_path(self, tmp_path):
        """Blank PDFs have no text layer — smart loader falls back to image OCR."""
        p = _make_pdf(tmp_path / "blank.pdf", pages=2)
        from ocr_kb.ingest.batch_builder import build_batch
        s = _settings(prefer_pdf_text=True)
        batch = build_batch(p, s)
        assert len(batch) == 2
        # Blank pages have no usable text, so each item must be an image
        for item in batch:
            assert item.needs_ocr
            assert isinstance(item.image, Image.Image)

    def test_prefer_pdf_text_false_uses_image_path(self, tmp_path):
        """When prefer_pdf_text=False, always use rendered images (old behaviour)."""
        p = _make_pdf(tmp_path / "doc.pdf", pages=2)
        from ocr_kb.ingest.batch_builder import build_batch
        s = _settings(prefer_pdf_text=False)
        batch = build_batch(p, s)
        for item in batch:
            assert item.needs_ocr

    def test_text_file_unaffected_by_prefer_pdf_text(self, tmp_path):
        p = tmp_path / "note.txt"
        p.write_text("Hello world", encoding="utf-8")
        from ocr_kb.ingest.batch_builder import build_batch
        s = _settings(prefer_pdf_text=True)
        batch = build_batch(p, s)
        assert len(batch) == 1
        assert batch[0].text_content == "Hello world"
        assert not batch[0].needs_ocr
