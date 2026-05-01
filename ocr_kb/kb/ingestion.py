"""Shared ingestion helpers used by both the CLI and the MCP server."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from ocr_kb.ingest.metadata_extractor import DocMetadata
from ocr_kb.kb.exporter import _entry_to_markdown
from ocr_kb.kb.schema import KBEntry
from ocr_kb.kb.store import save
from ocr_kb.pipeline import FileResult, PageResult
from ocr_kb.postprocess import clean_html, html_to_markdown, is_blank, parse_layout
from ocr_kb.postprocess.markdown_converter import plain_to_markdown
from ocr_kb.settings import Settings


def page_result_to_entry(
    result: PageResult,
    tags: str,
    category: str,
    metadata: DocMetadata | None = None,
) -> KBEntry:
    """Convert a pipeline PageResult to a KBEntry ready for upsert."""
    if result.mode == "html":
        html = clean_html(result.text)
        md = html_to_markdown(html)
        blocks = parse_layout(html)
    else:
        html = ""
        md = plain_to_markdown(result.text)
        blocks = []

    layout_json = json.dumps([
        {"type": b.type, "content": b.content, "level": b.level, "index": b.index}
        for b in blocks
    ])
    effective_tags = tags or result.generated_tags or ""
    return KBEntry(
        source_path=str(result.source_path),
        page_number=result.page_number,
        chunk_index=result.chunk_index,
        raw_text=result.text,
        markdown=md,
        html=html,
        layout_blocks=layout_json,
        tags=effective_tags,
        category=category,
        key_points=result.key_points or "",
        summary=result.summary or "",
        enriched_metadata=result.enriched_metadata or "",
        doc_title=metadata.title if metadata else "",
        authors=metadata.authors if metadata else "",
        year=metadata.year if metadata else None,
        doi=metadata.doi if metadata else "",
        abstract=metadata.abstract if metadata else "",
        journal=metadata.journal if metadata else "",
    )


def save_file_result(
    file_result: FileResult,
    conn: sqlite3.Connection,
    settings: Settings,
    tags: str,
    category: str,
) -> tuple[int, list[str]]:
    """Persist all non-blank pages from a FileResult.

    Returns ``(saved_count, errors)``.  Each saved page also writes a Markdown
    mirror to ``settings.markdown_dir`` for external tooling (Cursor, etc.).
    """
    from ocr_kb.model.embedder import embed_text

    saved = 0
    errors = list(file_result.errors)
    meta = file_result.metadata
    for page in file_result.pages:
        if is_blank(page.text):
            continue
        try:
            entry = page_result_to_entry(page, tags, category, metadata=meta)
            entry.embedding = embed_text(
                page.text,
                settings,
                title=meta.title if meta else "",
                abstract=meta.abstract if meta else "",
            )
            save(conn, entry)
            settings.markdown_dir.mkdir(parents=True, exist_ok=True)
            stem = Path(entry.source_path).stem
            md_path = settings.markdown_dir / f"{stem}_p{entry.page_number:03d}.md"
            md_path.write_text(_entry_to_markdown(entry), encoding="utf-8")
            saved += 1
        except Exception as exc:
            errors.append(f"Save failed for page {page.page_number}: {exc}")
    return saved, errors
