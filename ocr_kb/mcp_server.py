"""MCP server exposing the full ocr-kb pipeline as tools.

Start:
    ocr-kb-mcp                      # stdio transport (default, for Claude Code / OpenCode)
    ocr-kb-mcp --transport sse      # SSE transport (for web-based agents)

Configure in your agent:
    Claude Code:  add to ~/.claude/claude_desktop_config.json
    Cursor:       add to .cursor/mcp.json
    OpenCode:     add to opencode.json  (tool.exec field)

All model calls go through LM Studio at the URLs configured in local.env.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "ocr-kb",
    instructions=(
        "Personal research knowledge base built from OCR'd PDFs, images, and text files. "
        "All document text is stored locally in SQLite — no cloud storage. "
        "LM Studio provides the vision model (OCR) and text model (RAG answers). "
        "\n\n"
        "Core workflow:\n"
        "1. kb_ingest — add a file or folder (PDF, image, text, Markdown).\n"
        "2. kb_ask — ask a question; get a cited answer from the stored content.\n"
        "3. kb_search — keyword or phrase search with optional filters.\n"
        "4. kb_show / kb_recent — browse entries by ID or recency.\n"
        "5. kb_tag / kb_retag — organise entries without re-ingesting.\n"
        "6. kb_compile_wiki — build a linked concept graph from the entire KB.\n"
        "\n"
        "Use kb_ask first for research questions. "
        "Use kb_search when you need to locate a specific term or source. "
        "Use kb_ingest whenever the user drops a new document."
    ),
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_conn_and_settings(db: str):
    from ocr_kb.kb.store import init_db
    from ocr_kb.settings import get_settings
    s = get_settings()
    path = Path(db) if db else s.kb_db_path
    return init_db(path), s


def _fmt_entry(entry) -> str:
    lines = [
        f"ID {entry.id}  ·  {Path(entry.source_path).name}  p.{entry.page_number}",
        f"Category: {entry.category or '—'}  |  Tags: {entry.tags or '—'}",
        f"Ingested: {entry.created_at.strftime('%Y-%m-%d %H:%M')}",
    ]
    if entry.summary:
        lines += ["", f"Summary:\n{entry.summary[:600]}"]
    if entry.key_points:
        lines += ["", f"Key Points:\n{entry.key_points[:400]}"]
    lines += ["", "─── Content ───", entry.markdown or entry.raw_text]
    return "\n".join(lines)


def _log(msg: str) -> None:
    print(f"[ocr-kb] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# kb_ingest
# ---------------------------------------------------------------------------

@mcp.tool()
def kb_ingest(
    path: str,
    tags: str = "",
    category: str = "",
    mode: str = "text",
    key_points: bool = False,
    summary: bool = False,
    vision_model: str = "",
    text_model: str = "",
    db: str = "",
) -> str:
    """Ingest a file or directory into the knowledge base.

    Supports: PDF (OCR via vision LLM), PNG/JPG/TIFF (OCR), plain text, Markdown.
    Re-ingesting the same file updates pages in-place — no duplicates are created.
    Existing tags and category are preserved when not explicitly specified.
    Embedded images (figures, diagrams, charts) in PDFs are extracted and described
    automatically using the vision model.

    Args:
        path: Absolute or relative path to a file or directory.
        tags: Comma-separated tags to apply, e.g. "ml,paper,2024".
        category: Category label, e.g. "biology" or "lecture-notes".
        mode: "text" for plain OCR output; "html" to preserve document structure.
        key_points: Run an extra LLM pass after OCR to extract key points.
        summary: Run an extra LLM pass after OCR to generate a one-paragraph summary.
        vision_model: Override the vision/OCR model name for this call (e.g. "llava:13b").
        text_model: Override the text/enrichment model name for this call (e.g. "gemma3:12b").
        db: Override the default KB database path (uses settings default when empty).
    """
    from ocr_kb.pipeline import process_path
    from ocr_kb.kb.ingestion import save_file_result

    target = Path(path)
    if not target.exists():
        return f"ERROR: path not found: {path}"

    conn, s = _get_conn_and_settings(db)
    total_saved = 0
    all_errors: list[str] = []
    lines: list[str] = [f"Ingesting {target} → {s.kb_db_path}", ""]

    def _on_page(current: int, total: int) -> None:
        _log(f"{target.name}: page {current}/{total}")

    results = process_path(
        target, s,
        mode=mode,  # type: ignore[arg-type]
        extract_key_points=key_points,
        extract_summary=summary,
        vision_model=vision_model or None,
        text_model=text_model or None,
        on_page=_on_page,
    )

    for fr in results:
        saved, errors = save_file_result(fr, conn, s, tags, category)
        total_saved += saved
        all_errors.extend(errors)
        icon = "✓" if saved else "⚠"
        lines.append(f"  {icon}  {fr.source_path.name}: {saved} page(s) saved")

    if all_errors:
        lines.append("\nErrors:")
        lines.extend(f"  • {e}" for e in all_errors)

    lines.append(f"\nDone. {total_saved} entr{'y' if total_saved == 1 else 'ies'} saved.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# kb_search
# ---------------------------------------------------------------------------

@mcp.tool()
def kb_search(
    query: str,
    limit: int = 10,
    source: str = "",
    category: str = "",
    tag: str = "",
    after: str = "",
    db: str = "",
) -> str:
    """Full-text search the knowledge base with optional column-level filters.

    Returns matching entries with ID, source file, page number, tags, category,
    and a text excerpt. Punctuation and question marks in the query are handled
    automatically — pass natural language freely.

    Args:
        query: Search terms (natural language OK).
        limit: Maximum number of results to return.
        source: Restrict to entries whose source path contains this substring.
        category: Restrict to entries with exactly this category (case-insensitive).
        tag: Restrict to entries whose tags contain this substring.
        after: Only entries ingested on or after this ISO date, e.g. "2024-01-01".
        db: Override the default KB database path.
    """
    from ocr_kb.kb.indexer import filtered_search, hybrid_search

    conn, s = _get_conn_and_settings(db)
    has_filters = any([source, category, tag, after])

    if has_filters:
        results = filtered_search(
            conn, query,
            source=source or None, category=category or None,
            tag=tag or None, after=after or None,
            limit=limit,
        )
    else:
        results = hybrid_search(conn, query, s, limit=limit)

    if not results:
        return f"No results found for '{query}'."

    lines: list[str] = [f"Found {len(results)} result(s) for '{query}':", ""]
    for entry in results:
        excerpt = (entry.markdown or entry.raw_text)[:300].replace("\n", " ")
        lines.append(
            f"[id={entry.id}]  {Path(entry.source_path).name}  p.{entry.page_number}"
            f"  [{entry.category or '—'}]  tags: {entry.tags or '—'}"
        )
        lines.append(f"  {excerpt}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# kb_ask
# ---------------------------------------------------------------------------

@mcp.tool()
def kb_ask(
    question: str,
    context_size: int = 5,
    chunk_chars: int = 0,
    text_model: str = "",
    db: str = "",
) -> str:
    """Ask a natural-language question and get a cited answer from the knowledge base.

    Retrieves the most relevant stored documents via full-text search, builds a
    context window, and calls the configured LLM for a grounded answer.
    The response includes inline citations and a numbered source list.

    Args:
        question: The question to answer.
        context_size: Number of documents to retrieve as context (default 5).
        chunk_chars: Characters per document snippet (0 = use the rag_chunk_chars setting).
        text_model: Override the text model name for this call (e.g. "gemma3:12b").
        db: Override the default KB database path.
    """
    from ocr_kb.kb.indexer import hybrid_search
    from ocr_kb.model import run_enrichment
    from ocr_kb.prompts import format_rag_prompt

    conn, s = _get_conn_and_settings(db)
    chars = chunk_chars if chunk_chars > 0 else s.rag_chunk_chars

    hits = hybrid_search(conn, question, s, limit=context_size)
    if not hits:
        return "No relevant documents found in the knowledge base."

    context_parts: list[str] = []
    for i, entry in enumerate(hits, start=1):
        source_label = f"{Path(entry.source_path).name} p.{entry.page_number}"
        text = (entry.markdown or entry.raw_text)[:chars]
        context_parts.append(f"[{i}] {source_label}:\n{text}")

    context = "\n\n---\n\n".join(context_parts)
    prompt = format_rag_prompt(question, context)

    _log(f"asking with {len(hits)} context doc(s)…")
    answer = run_enrichment(prompt, s, text_model=text_model or None)

    source_map = "\n".join(
        f"  [{i}] {Path(e.source_path).name}  p.{e.page_number}"
        f"  id={e.id}  [{e.tags or 'no tags'}]"
        for i, e in enumerate(hits, start=1)
    )
    return f"{answer}\n\nSources:\n{source_map}"


# ---------------------------------------------------------------------------
# kb_show
# ---------------------------------------------------------------------------

@mcp.tool()
def kb_show(entry_id: int, db: str = "") -> str:
    """Retrieve the full content of a single knowledge base entry by its numeric ID.

    Displays source path, page number, category, tags, ingestion date, summary,
    key points (if available), and the complete extracted text.

    Args:
        entry_id: The numeric ID shown in kb_search or kb_recent output.
        db: Override the default KB database path.
    """
    from ocr_kb.kb.store import get

    conn, _ = _get_conn_and_settings(db)
    entry = get(conn, entry_id)
    if entry is None:
        return f"No entry found with id={entry_id}."
    return _fmt_entry(entry)


# ---------------------------------------------------------------------------
# kb_recent
# ---------------------------------------------------------------------------

@mcp.tool()
def kb_recent(limit: int = 20, db: str = "") -> str:
    """List the most recently ingested entries, newest first.

    Useful for reviewing what was just added or for browsing the KB.

    Args:
        limit: Maximum number of entries to return (default 20).
        db: Override the default KB database path.
    """
    from ocr_kb.kb.store import get_recent

    conn, _ = _get_conn_and_settings(db)
    entries = get_recent(conn, limit=limit)
    if not entries:
        return "No entries in the knowledge base yet. Use kb_ingest to add documents."

    lines: list[str] = [f"Most recent {len(entries)} entries:", ""]
    for e in entries:
        lines.append(
            f"[id={e.id}]  {e.created_at.strftime('%Y-%m-%d %H:%M')}"
            f"  {Path(e.source_path).name}  p.{e.page_number}"
            f"  [{e.category or '—'}]  {e.tags or '—'}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# kb_tag
# ---------------------------------------------------------------------------

@mcp.tool()
def kb_tag(
    entry_id: int,
    add: str = "",
    remove: str = "",
    category: str = "",
    db: str = "",
) -> str:
    """Add or remove tags and optionally change the category of an entry.

    Modifies metadata in-place — no re-ingestion needed.
    Existing tags not mentioned in 'remove' are preserved.

    Args:
        entry_id: The entry ID to modify.
        add: Comma-separated tags to add, e.g. "reviewed,important".
        remove: Comma-separated tags to remove, e.g. "draft".
        category: If non-empty, replace the entry's category with this value.
        db: Override the default KB database path.
    """
    from ocr_kb.kb.store import get, update_category, update_tags

    conn, _ = _get_conn_and_settings(db)
    entry = get(conn, entry_id)
    if entry is None:
        return f"No entry found with id={entry_id}."

    current = {t.strip() for t in entry.tags.split(",") if t.strip()}
    if add:
        current |= {t.strip() for t in add.split(",") if t.strip()}
    if remove:
        current -= {t.strip() for t in remove.split(",") if t.strip()}

    new_tags = ",".join(sorted(current))
    update_tags(conn, entry_id, new_tags)
    result_lines = [f"id={entry_id}  tags updated: {new_tags or '(none)'}"]

    if category:
        update_category(conn, entry_id, category)
        result_lines.append(f"  category set: {category}")
    return "\n".join(result_lines)


# ---------------------------------------------------------------------------
# kb_retag
# ---------------------------------------------------------------------------

@mcp.tool()
def kb_retag(
    query: str,
    add: str = "",
    remove: str = "",
    limit: int = 100,
    db: str = "",
) -> str:
    """Bulk-edit tags on all entries matching a search query.

    Useful for reorganising the KB after ingesting a batch of documents.

    Args:
        query: FTS search query to select entries to modify.
        add: Comma-separated tags to add to every matched entry.
        remove: Comma-separated tags to remove from every matched entry.
        limit: Maximum number of entries to modify (default 100).
        db: Override the default KB database path.
    """
    from ocr_kb.kb.indexer import fts_search
    from ocr_kb.kb.store import update_tags

    conn, _ = _get_conn_and_settings(db)
    entries = fts_search(conn, query, limit=limit)
    if not entries:
        return "No entries matched the query."

    add_set = {t.strip() for t in add.split(",") if t.strip()}
    remove_set = {t.strip() for t in remove.split(",") if t.strip()}
    for entry in entries:
        current = {t.strip() for t in entry.tags.split(",") if t.strip()}
        current = (current | add_set) - remove_set
        update_tags(conn, entry.id, ",".join(sorted(current)))  # type: ignore[arg-type]

    return f"Updated tags on {len(entries)} entr{'y' if len(entries) == 1 else 'ies'}."


# ---------------------------------------------------------------------------
# kb_delete
# ---------------------------------------------------------------------------

@mcp.tool()
def kb_delete(
    entry_id: int = 0,
    source: str = "",
    db: str = "",
) -> str:
    """Delete one entry by numeric ID, or all entries from a source file.

    Specify exactly one of entry_id (non-zero) or source (non-empty string).

    Args:
        entry_id: Numeric ID of a single entry to delete (0 = not used).
        source: Source path string — deletes all pages ingested from this file.
        db: Override the default KB database path.
    """
    from ocr_kb.kb.store import delete, delete_by_source, get

    if not entry_id and not source:
        return "ERROR: specify a non-zero entry_id or a non-empty source path."

    conn, _ = _get_conn_and_settings(db)

    if entry_id:
        entry = get(conn, entry_id)
        if entry is None:
            return f"No entry found with id={entry_id}."
        delete(conn, entry_id)
        return f"Deleted entry {entry_id} ({Path(entry.source_path).name} p.{entry.page_number})."

    count = delete_by_source(conn, source)
    if count == 0:
        return f"No entries found for source: {source}"
    return f"Deleted {count} entr{'y' if count == 1 else 'ies'} from '{source}'."


# ---------------------------------------------------------------------------
# kb_export
# ---------------------------------------------------------------------------

@mcp.tool()
def kb_export(
    output_path: str,
    format: str = "json",
    filter_tag: str = "",
    filter_category: str = "",
    db: str = "",
) -> str:
    """Export knowledge base entries to Markdown, JSON, or Obsidian vault format.

    Args:
        output_path: Destination file path (JSON) or directory (markdown/obsidian).
        format: Export format — "json", "markdown", or "obsidian".
        filter_tag: Only export entries whose tags contain this value.
        filter_category: Only export entries with exactly this category.
        db: Override the default KB database path.
    """
    from ocr_kb.kb.exporter import export_json, export_markdown, export_obsidian

    conn, _ = _get_conn_and_settings(db)
    out = Path(output_path)
    fmt = format.lower()
    ft = filter_tag or None
    fc = filter_category or None

    if fmt == "json":
        export_json(conn, out, filter_tag=ft, filter_category=fc)
        return f"Exported JSON to {out}"
    if fmt == "markdown":
        paths = export_markdown(conn, out, filter_tag=ft, filter_category=fc)
        return f"Exported {len(paths)} Markdown file(s) to {out}"
    if fmt == "obsidian":
        paths = export_obsidian(conn, out, filter_tag=ft, filter_category=fc)
        return f"Exported {len(paths)} Obsidian note(s) to {out}"
    return f"ERROR: unknown format '{format}'. Choose json, markdown, or obsidian."


# ---------------------------------------------------------------------------
# kb_stats
# ---------------------------------------------------------------------------

@mcp.tool()
def kb_stats(db: str = "") -> str:
    """Return statistics about the knowledge base.

    Reports total number of stored entries and how many distinct source files
    have been ingested.

    Args:
        db: Override the default KB database path.
    """
    from ocr_kb.kb.store import stats

    conn, s = _get_conn_and_settings(db)
    st = stats(conn)
    return (
        f"Database:       {s.kb_db_path}\n"
        f"Total entries:  {st['total_entries']}\n"
        f"Unique sources: {st['unique_sources']}"
    )


# ---------------------------------------------------------------------------
# kb_compile_wiki
# ---------------------------------------------------------------------------

@mcp.tool()
def kb_compile_wiki(
    limit: int = 0,
    since: str = "",
    db: str = "",
) -> str:
    """Synthesize a structured wiki from the knowledge base using the LLM.

    Runs concept extraction on every stored page, then generates dedicated
    concept pages with definitions, evidence, metrics, limitations, and
    cross-references. Each concept page is written to settings.markdown_dir/concepts/.

    Incremental mode (since) only processes entries newer than the given date,
    making it practical to run after each ingest session.

    Args:
        limit: Maximum number of pages to process (0 = all).
        since: ISO date string — skip entries older than this, e.g. "2024-06-01".
        db: Override the default KB database path.
    """
    from ocr_kb.wiki.compiler import compile_wiki

    _, s = _get_conn_and_settings(db)
    _log("compiling wiki…")
    result = compile_wiki(s, limit=limit or None, since=since or None)
    return (
        f"Wiki compiled.\n"
        f"Processed {result['processed_entries']} pages.\n"
        f"Synthesized {result['concepts_found']} concepts.\n"
        f"Output: {s.markdown_dir}"
    )


# ---------------------------------------------------------------------------
# kb_lint_wiki
# ---------------------------------------------------------------------------

@mcp.tool()
def kb_lint_wiki(db: str = "") -> str:
    """Check the wiki for broken links, orphan concept pages, and missing front matter.

    Reports issues by category. A clean wiki returns "Wiki is healthy."

    Args:
        db: Override the default KB database path.
    """
    from ocr_kb.wiki.linter import lint_wiki

    _, s = _get_conn_and_settings(db)
    report = lint_wiki(s)

    if "error" in report:
        return f"ERROR: {report['error'][0]}"

    lines: list[str] = []
    has_issues = False
    for key, items in report.items():
        if items:
            has_issues = True
            lines.append(f"\n{key.upper().replace('_', ' ')} ({len(items)}):")
            lines.extend(f"  • {item}" for item in items)

    if not has_issues:
        return "Wiki is healthy — no issues found."
    total = sum(len(v) for v in report.values())
    return f"Found {total} issue(s):" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

def run() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="ocr-kb MCP server")
    parser.add_argument(
        "--transport", choices=["stdio", "sse"], default="stdio",
        help="Transport protocol (default: stdio)",
    )
    args = parser.parse_args()
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    run()
