from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from ocr_kb.ingest.loader import is_pdf, is_text, load_file, load_text_file
from ocr_kb.settings import Settings


@dataclass
class BatchItem:
    source_path: Path
    page_number: int
    image: Image.Image | None = field(default=None)
    text_content: str | None = field(default=None)
    # Embedded images extracted from this page (figures, diagrams, charts).
    # Populated for PDF pages when settings.extract_embedded_images is True.
    embedded_images: list[Image.Image] = field(default_factory=list)

    @property
    def needs_ocr(self) -> bool:
        """True when the pipeline must call the vision model to extract text."""
        return self.image is not None and self.text_content is None


def build_batch(
    path: Path,
    settings: Settings,
    page_range: str | None = None,
) -> list[BatchItem]:
    """Return one BatchItem per page/section for the given file.

    Text and Markdown files produce a single item with text_content pre-filled.
    Images and PDFs produce one item per page with image set (OCR required).
    For PDF files, embedded images (figures, diagrams) are extracted per page
    when settings.extract_embedded_images is True.
    """
    if is_text(path):
        return [BatchItem(
            source_path=path,
            page_number=1,
            text_content=load_text_file(path),
        )]

    if is_pdf(path) and settings.prefer_pdf_text:
        # Extraction priority (first success wins):
        #   pdftext (instant, no ML) → marker-pdf → pymupdf4llm → pypdfium2
        # Scanned pages from any text path are re-routed to OCR below.
        from ocr_kb.ingest.pdf_reader import parse_page_range
        import fitz as _fitz
        _doc = _fitz.open(str(path))
        _total = len(_doc)
        _doc.close()
        _zero_range = parse_page_range(page_range, _total) if page_range else None

        md_pages: list[tuple[int, str]] | None = None

        if md_pages is None and settings.use_pdftext:
            from ocr_kb.ingest.pdftext_reader import pdf_to_text_pdftext
            md_pages = pdf_to_text_pdftext(path, page_range=_zero_range)

        if md_pages is None and settings.use_marker_pdf:
            from ocr_kb.ingest.marker_reader import pdf_to_markdown
            md_pages = pdf_to_markdown(path, page_range=_zero_range, device=settings.marker_device)

        if md_pages is None and settings.use_pymupdf4llm:
            from ocr_kb.ingest.pymupdf4llm_reader import pdf_to_markdown_pages
            md_pages = pdf_to_markdown_pages(path)

        if md_pages is not None:
            from ocr_kb.ingest.pdftext_reader import is_text_page

            # pymupdf4llm doesn't apply the page range internally — filter here.
            if page_range and _zero_range is not None:
                allowed = set(i + 1 for i in _zero_range)
                md_pages = [(pn, t) for pn, t in md_pages if pn in allowed]

            # Lazily open the PDF once for rendering scanned pages to images.
            _pdf_doc = None

            def _render_scanned(pn: int) -> Image.Image:
                nonlocal _pdf_doc
                import pypdfium2 as _pdfium
                from ocr_kb.ingest.image_reader import upscale_if_needed as _up
                if _pdf_doc is None:
                    _pdf_doc = _pdfium.PdfDocument(str(path))
                scale = settings.image_dpi / 72.0
                bm = _pdf_doc[pn - 1].render(scale=scale)
                return _up(bm.to_pil(), settings.min_pdf_image_dim)

            items = []
            for page_num, text in md_pages:
                if not is_text_page(text):
                    # Scanned page — try Surya OCR or fall back to vision model
                    if settings.use_surya_ocr:
                        from ocr_kb.ingest.surya_reader import pdf_to_text_surya
                        surya_pages = pdf_to_text_surya(
                            path,
                            page_range=[page_num - 1],
                            device=settings.marker_device,
                            dpi=settings.image_dpi,
                        )
                        if surya_pages:
                            items.append(BatchItem(
                                source_path=path,
                                page_number=page_num,
                                text_content=surya_pages[0][1],
                            ))
                            continue
                    # Fall back to vision model — render page to image
                    items.append(BatchItem(
                        source_path=path,
                        page_number=page_num,
                        image=_render_scanned(page_num),
                    ))
                else:
                    items.append(BatchItem(
                        source_path=path,
                        page_number=page_num,
                        text_content=text,
                    ))
        else:
            from ocr_kb.ingest.pdf_reader import render_pdf_pages_smart
            pages = render_pdf_pages_smart(
                path, settings.image_dpi, settings.min_pdf_image_dim, page_range
            )
            items = [
                BatchItem(
                    source_path=path,
                    page_number=page_num,
                    image=content if not isinstance(content, str) else None,
                    text_content=content if isinstance(content, str) else None,
                )
                for page_num, content in pages
            ]
    else:
        items = [
            BatchItem(source_path=path, page_number=page_num, image=img)
            for page_num, img in load_file(path, settings, page_range)
        ]

    if settings.extract_embedded_images and path.suffix.lower() == ".pdf":
        from ocr_kb.ingest.pdf_reader import extract_page_images
        embedded = extract_page_images(path, min_pixels=settings.embedded_image_min_pixels)
        for item in items:
            item.embedded_images = embedded.get(item.page_number, [])

    return items
