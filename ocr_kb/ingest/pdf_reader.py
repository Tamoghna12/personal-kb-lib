from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image

from ocr_kb.ingest.image_reader import upscale_if_needed

# FPDF_PAGEOBJ_IMAGE = 3  (from pypdfium2.raw — confirmed via introspection)
# Earlier code incorrectly used 4 (which is FPDF_PAGEOBJ_SHADING).
_FPDF_PAGEOBJ_IMAGE = 3


def extract_page_images(
    path: Path,
    min_pixels: int = 10_000,
) -> dict[int, list[Image.Image]]:
    """Extract embedded image objects from each page of a PDF.

    Returns a dict mapping 1-based page number → list of PIL RGB images.
    Objects with area < *min_pixels* (e.g. icons, decorative rules) are skipped.
    Errors on individual objects are silently ignored so a corrupt embedded
    image never aborts the rest of the page.
    """
    doc = pdfium.PdfDocument(str(path))
    result: dict[int, list[Image.Image]] = {}

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        images: list[Image.Image] = []

        try:
            for obj in page.get_objects():
                obj_type = getattr(obj, "type", None)
                if obj_type is None:
                    continue
                type_val = obj_type.value if hasattr(obj_type, "value") else int(obj_type)
                if type_val != _FPDF_PAGEOBJ_IMAGE:
                    continue
                try:
                    bitmap = obj.get_bitmap()
                    pil_img = bitmap.to_pil().convert("RGB")
                    w, h = pil_img.size
                    if w * h >= min_pixels:
                        images.append(pil_img)
                except Exception:
                    continue
        except Exception:
            pass  # page.get_objects() may fail on malformed PDFs

        if images:
            result[page_idx + 1] = images

    return result


def parse_page_range(spec: str, total_pages: int) -> list[int]:
    """Convert a human page-range string into sorted 0-based page indices.

    "1,3-5,7" with total_pages=10 → [0, 2, 3, 4, 6]
    Out-of-range values are silently dropped.
    """
    indices: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            indices.update(range(int(lo) - 1, int(hi)))
        else:
            indices.add(int(part) - 1)
    return sorted(i for i in indices if 0 <= i < total_pages)


def _render_page(page: pdfium.PdfPage, dpi: int) -> Image.Image:
    scale = dpi / 72.0
    bitmap = page.render(scale=scale)
    return bitmap.to_pil()


def _clean_pdf_text(text: str) -> str:
    """Fix character-level artifacts produced by pypdfium2.

    1. U+FFFE / U+00AD  — soft-hyphen misencoding that splits words across
       lines.  ALL-CAPS sequences keep a real '-' (e.g. NIR-SRS); otherwise
       the artifact is removed to rejoin the halves.
    2. U+0002 (STX)     — column-break control character that also splits words
       across two-column layouts (e.g. 'trans\x02formations').
    3. Normalise CRLF → LF and collapse runs of blank lines to two newlines.
    """
    import re
    # Soft-hyphen: all-caps on both sides → preserve as real hyphen
    text = re.sub(r"([A-Z]+)￾([A-Z]+)", r"\1-\2", text)
    # Remaining soft-hyphens and column-break chars → join word halves
    text = re.sub(r"[­￾\x02]", "", text)
    # Normalise line endings and blank-line runs
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_figure_captions(footer_text: str) -> str:
    """Pull figure captions from the footer band, stripping bare page numbers."""
    import re
    lines = [ln.strip() for ln in footer_text.splitlines() if ln.strip()]
    caption_lines: list[str] = []
    for ln in lines:
        if re.fullmatch(r"\d{1,4}", ln):
            continue  # bare page number — discard
        if re.match(r"(Fig\.|Figure|Table)\s*\d+", ln, re.IGNORECASE):
            caption_lines.append(ln)
        elif caption_lines:
            # continuation line after a caption start
            caption_lines.append(ln)
    return "\n".join(caption_lines)


def _extract_page_text(page: pdfium.PdfPage) -> str:
    """Return structured body text for a native-text PDF page.

    Uses pypdfium2's bounded-rectangle extraction to strip the repeating
    journal running-header (top 6 %) and page-number footer (bottom 7 %).
    Figure captions found in the footer band are preserved and appended after
    the body.  All remaining soft-hyphen and column-break artefacts are then
    cleaned by _clean_pdf_text.
    """
    try:
        tp = page.get_textpage()
        w, h = page.get_size()

        bot_cut = h * 0.07   # footer zone upper bound
        top_cut = h * 0.94   # header zone lower bound

        body_raw  = tp.get_text_bounded(0, bot_cut, w, top_cut) or ""
        footer_raw = tp.get_text_bounded(0, 0,       w, bot_cut) or ""

        body = _clean_pdf_text(body_raw)
        captions = _extract_figure_captions(footer_raw)
        if captions:
            body = body + "\n\n" + captions
        return body
    except Exception:
        try:
            return _clean_pdf_text(page.get_textpage().get_text_range() or "")
        except Exception:
            return ""


def _is_usable_text(text: str, min_len: int = 50, min_alpha_ratio: float = 0.4) -> bool:
    """Return True when text looks like real document content rather than a scan artifact."""
    text = text.strip()
    if len(text) < min_len:
        return False
    alpha = sum(c.isalpha() for c in text)
    return (alpha / len(text)) >= min_alpha_ratio


def render_pdf_pages_smart(
    path: Path,
    dpi: int,
    min_dim: int,
    page_range: str | None = None,
) -> list[tuple[int, Image.Image | str]]:
    """Like render_pdf_pages but yields ``str`` for pages with a usable text layer.

    For digital PDFs this avoids a round-trip through the vision model entirely.
    Pages that look scanned (empty or mostly non-alpha text) still get rendered
    as images so the OCR pipeline can process them normally.

    Returns list of ``(1-based page number, content)`` where content is either
    a ``str`` (native text, no OCR needed) or a ``PIL.Image.Image`` (needs OCR).
    """
    doc = pdfium.PdfDocument(str(path))
    total = len(doc)
    indices = parse_page_range(page_range, total) if page_range else list(range(total))

    results: list[tuple[int, Image.Image | str]] = []
    for idx in indices:
        page = doc[idx]
        text = _extract_page_text(page)
        if _is_usable_text(text):
            results.append((idx + 1, text))
        else:
            img = _render_page(page, dpi)
            img = upscale_if_needed(img, min_dim)
            results.append((idx + 1, img))
    return results


def render_pdf_pages(
    path: Path,
    dpi: int,
    min_dim: int,
    page_range: str | None = None,
) -> list[tuple[int, Image.Image]]:
    """Render PDF pages to PIL images.

    Returns list of (1-based page number, image).
    If page_range is None, all pages are rendered.
    """
    doc = pdfium.PdfDocument(str(path))
    total = len(doc)
    indices = parse_page_range(page_range, total) if page_range else list(range(total))

    results: list[tuple[int, Image.Image]] = []
    for idx in indices:
        img = _render_page(doc[idx], dpi)
        img = upscale_if_needed(img, min_dim)
        results.append((idx + 1, img))
    return results
