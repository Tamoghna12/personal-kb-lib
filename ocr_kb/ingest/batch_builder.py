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
        # Try structured markdown extraction first (pymupdf4llm).
        # Falls back to pypdfium2 raw text when not installed or on failure.
        md_pages: list[tuple[int, str]] | None = None
        if settings.use_pymupdf4llm:
            from ocr_kb.ingest.pymupdf4llm_reader import (
                is_text_page,
                pdf_to_markdown_pages,
            )
            md_pages = pdf_to_markdown_pages(path)

        if md_pages is not None:
            from ocr_kb.ingest.pdf_reader import render_pdf_pages_smart
            from ocr_kb.ingest.pymupdf4llm_reader import is_text_page

            # Filter page range if specified
            if page_range:
                from ocr_kb.ingest.pdf_reader import parse_page_range
                import fitz
                doc = fitz.open(str(path))
                allowed = set(i + 1 for i in parse_page_range(page_range, len(doc)))
                doc.close()
                md_pages = [(pn, t) for pn, t in md_pages if pn in allowed]

            items = []
            scanned_pages = {pn for pn, t in md_pages if not is_text_page(t)}
            for page_num, text in md_pages:
                if page_num in scanned_pages:
                    # Page looks like a scan — render for OCR instead
                    items.append(BatchItem(source_path=path, page_number=page_num, image=None))
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
