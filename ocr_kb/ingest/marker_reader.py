"""PDF-to-markdown extraction using marker-pdf (marker_pdf).

Produces high-quality structured markdown via Surya layout detection + OCR.
Models are loaded lazily and cached for the process lifetime — first call is
slow; subsequent calls are fast.

Falls back gracefully when marker_pdf is not installed.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

_log = logging.getLogger(__name__)

# Model cache keyed by device string so the cache stays valid across calls
# even if the device setting is changed between processes.
_artifact_cache: dict[str, dict] = {}


def _get_artifact_dict(device: str = "cpu") -> dict:
    if device not in _artifact_cache:
        from marker.models import create_model_dict
        _log.info("Loading marker-pdf models on device=%s (one-time load)", device)
        _artifact_cache[device] = create_model_dict(device=device)
    return _artifact_cache[device]


def _clean(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def pdf_to_markdown(
    path: Path,
    page_range: list[int] | None = None,
    device: str = "cpu",
) -> list[tuple[int, str]] | None:
    """Return [(1-based page number, markdown), ...] for a PDF via marker-pdf.

    *page_range* is a list of 0-based page indices (marker's convention).
    Returns None when marker_pdf is not installed or conversion fails.
    """
    try:
        from marker.converters.pdf import PdfConverter
        from marker.output import text_from_rendered
    except ImportError:
        _log.debug("marker_pdf not installed; skipping marker extraction")
        return None

    try:
        config: dict = {"output_format": "markdown", "extract_images": False}
        if page_range is not None:
            config["page_range"] = page_range

        converter = PdfConverter(
            config=config,
            artifact_dict=_get_artifact_dict(device),
            processor_list=None,
            renderer="marker.renderers.markdown.MarkdownRenderer",
        )
        rendered = converter(str(path))
        full_md, _, _ = text_from_rendered(rendered)
    except Exception as exc:
        _log.warning("marker_pdf failed on %s: %s", path.name, exc)
        return None

    # marker returns one big markdown string; split on page-break markers.
    # marker inserts "\n\n---\n\n" between pages, but we fall back to treating
    # the whole document as page 1 if no dividers are found.
    page_divider = re.compile(r"\n{0,2}-{3,}\n{0,2}")
    page_texts = page_divider.split(full_md)

    if not page_texts:
        return None

    # Build 1-based page numbers.  If a page_range was supplied we map back to
    # the original document page numbers; otherwise we number from 1.
    if page_range is not None:
        # page_range is 0-based; convert to 1-based and align with split result
        base_pages = [p + 1 for p in page_range]
    else:
        base_pages = list(range(1, len(page_texts) + 1))

    results: list[tuple[int, str]] = []
    for i, text in enumerate(page_texts):
        text = _clean(text)
        if not text:
            continue
        page_num = base_pages[i] if i < len(base_pages) else base_pages[-1] + i
        results.append((page_num, text))

    return results or None


def is_text_page(text: str, min_len: int = 50, min_alpha: float = 0.4) -> bool:
    """True when extracted text looks like real content, not a scan artifact."""
    t = text.strip()
    if len(t) < min_len:
        return False
    return sum(c.isalpha() for c in t) / len(t) >= min_alpha
