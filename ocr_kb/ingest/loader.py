from pathlib import Path

from PIL import Image

from ocr_kb.ingest.image_reader import load_image, upscale_if_needed
from ocr_kb.ingest.pdf_reader import render_pdf_pages
from ocr_kb.settings import Settings

IMAGE_SUFFIXES: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}
)
PDF_SUFFIX = ".pdf"
TEXT_SUFFIXES: frozenset[str] = frozenset({".txt", ".md"})
SUPPORTED_SUFFIXES: frozenset[str] = IMAGE_SUFFIXES | frozenset({PDF_SUFFIX}) | TEXT_SUFFIXES


def is_pdf(path: Path) -> bool:
    return path.suffix.lower() == PDF_SUFFIX


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_SUFFIXES


def is_text(path: Path) -> bool:
    return path.suffix.lower() in TEXT_SUFFIXES


def load_text_file(path: Path, encoding: str = "utf-8") -> str:
    return path.read_text(encoding=encoding)


def load_file(
    path: Path,
    settings: Settings,
    page_range: str | None = None,
) -> list[tuple[int, Image.Image]]:
    """Return (1-based page number, PIL image) pairs for visual files (PDF, image).

    For plain-text files use load_text_file; build_batch handles dispatch automatically.
    """
    if is_pdf(path):
        return render_pdf_pages(path, settings.image_dpi, settings.min_pdf_image_dim, page_range)
    if is_image(path):
        img = upscale_if_needed(load_image(path), settings.min_image_dim)
        return [(1, img)]
    raise ValueError(f"Unsupported file type: {path.suffix!r}")
