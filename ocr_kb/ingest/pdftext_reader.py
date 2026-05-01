"""Zero-ML native text extraction via pdftext.

pdftext reads the PDF text layer directly (same source as marker uses
internally) without loading any ML models.  For digital PDFs this is
instant.  Scanned PDFs return empty/short strings which the pipeline
detects and routes to OCR.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

_log = logging.getLogger(__name__)


def _clean(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def pdf_to_text_pdftext(
    path: Path,
    page_range: list[int] | None = None,
) -> list[tuple[int, str]] | None:
    """Return [(1-based page number, text), ...] using pdftext (no ML).

    *page_range* is a list of 0-based page indices (pdftext convention).
    Returns None when pdftext is not installed or extraction yields nothing.
    """
    try:
        from pdftext.extraction import paginated_plain_text_output
    except ImportError:
        _log.debug("pdftext not installed; skipping")
        return None

    try:
        pages: list[str] = paginated_plain_text_output(
            str(path),
            sort=True,
            hyphens=False,
            page_range=page_range,
        )
    except Exception as exc:
        _log.warning("pdftext failed on %s: %s", path.name, exc)
        return None

    # page_range is 0-based; build matching 1-based page numbers
    if page_range is not None:
        base = [p + 1 for p in page_range]
    else:
        base = list(range(1, len(pages) + 1))

    results: list[tuple[int, str]] = []
    for i, text in enumerate(pages):
        text = _clean(text)
        if text:
            page_num = base[i] if i < len(base) else base[-1] + i
            results.append((page_num, text))

    return results or None


def is_text_page(text: str, min_len: int = 50, min_alpha: float = 0.4) -> bool:
    t = text.strip()
    if len(t) < min_len:
        return False
    return sum(c.isalpha() for c in t) / len(t) >= min_alpha
