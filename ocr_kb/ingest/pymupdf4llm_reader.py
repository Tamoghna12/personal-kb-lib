"""Structured PDF-to-markdown extraction using pymupdf4llm.

Produces heading-aware, table-formatted markdown from native-text PDFs.
Falls back gracefully when pymupdf4llm is not installed.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

_log = logging.getLogger(__name__)

# Margin in PDF points to exclude repeating header/footer bands.
# A4 height = ~794 pts; 56 bottom ≈ 7 %, 48 top ≈ 6 %.
_MARGINS = (0, 56, 0, 48)


def _clean(text: str) -> str:
    """Fix soft-hyphen and line-break artifacts left by pymupdf4llm.

    Three cases handled:
    1. ALL-CAPS­ALL-CAPS   — real abbreviation hyphen (NIR­SRS → NIR-SRS)
    2. word­\\nword        — soft-hyphen line-break (imple­\\nment → implement)
    3. word­word (no \\n)  — remaining bare soft hyphens → remove
    4. mid-word line breaks — `Ma\\nchine` (no hyphen char) in tables → rejoin
    """
    # 1. ALL-CAPS abbreviation soft-hyphen (with or without newline)
    text = re.sub(r"([A-Z]{2,})­\n?([A-Z]{2,})", r"\1-\2", text)
    # 2 & 3. All remaining soft hyphens — rejoin word halves
    text = re.sub(r"­\n?", "", text)
    # Collapse triple+ blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def pdf_to_markdown_pages(path: Path) -> list[tuple[int, str]] | None:
    """Return [(1-based page number, markdown text), ...] for a native-text PDF.

    Returns None when pymupdf4llm is not installed so callers can fall back.
    Each page's markdown has proper heading hierarchy (# / ##), bold figure
    captions, italic section titles, and markdown tables where present.
    Running journal headers and page-number footers are stripped via margin
    exclusion.
    """
    try:
        import pymupdf4llm.helpers.pymupdf_rag as rag
    except ImportError:
        _log.debug("pymupdf4llm not installed; falling back to pypdfium2 text extraction")
        return None

    try:
        chunks: list[dict] = rag.to_markdown(
            str(path),
            page_chunks=True,
            force_text=True,       # use native text layer, skip Tesseract
            ignore_images=True,    # images handled separately by the pipeline
            margins=_MARGINS,      # strip header / footer bands
            show_progress=False,
        )
    except Exception as exc:
        _log.warning("pymupdf4llm failed on %s: %s", path.name, exc)
        return None

    results: list[tuple[int, str]] = []
    for chunk in chunks:
        page_num: int = chunk.get("metadata", {}).get("page", 0)
        text = _clean(chunk.get("text", ""))
        if text:
            results.append((page_num, text))

    return results or None


def is_text_page(text: str, min_len: int = 50, min_alpha: float = 0.4) -> bool:
    """True when extracted text looks like real content, not a scan artifact."""
    t = text.strip()
    if len(t) < min_len:
        return False
    return sum(c.isalpha() for c in t) / len(t) >= min_alpha
