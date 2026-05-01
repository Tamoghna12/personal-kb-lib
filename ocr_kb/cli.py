"""CLI entry-point — ocr-kb <command> [options]"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

from ocr_kb.ingest.loader import SUPPORTED_SUFFIXES
from ocr_kb.kb.exporter import (
    _entry_to_markdown,
    export_json,
    export_markdown,
    export_obsidian,
)
from ocr_kb.kb.indexer import filtered_search, fts_search, hybrid_search, semantic_search
from ocr_kb.kb.ingestion import page_result_to_entry as _result_to_entry, save_file_result as _save_file_result_impl
from ocr_kb.kb.schema import KBEntry
from ocr_kb.kb.store import (
    delete as _db_delete,
    delete_by_source,
    dlq_get_retryable,
    dlq_list,
    dlq_mark_resolved,
    dlq_push,
    get,
    get_recent,
    init_db,
    save,
    update_category,
    update_tags,
)
from ocr_kb.kb.store import stats as _kb_stats
from ocr_kb.monitoring import (
    IngestTimer,
    SearchTimer,
    log_failed_document,
    log_ingest_summary,
    log_search_query,
    setup_logging,
)
from ocr_kb.pipeline import FileResult, PageResult, process_path
from ocr_kb.postprocess import is_blank
from ocr_kb.settings import Settings, get_settings

app = typer.Typer(name="ocr-kb", add_completion=False, no_args_is_help=True)
console = Console()


# ---------------------------------------------------------------------------
# Helpers shared across commands
# ---------------------------------------------------------------------------

def _db_conn(db_path: Optional[Path]):
    s = get_settings()
    path = db_path or s.kb_db_path
    return init_db(path), path


def _save_file_result(
    file_result: FileResult,
    conn,
    tags: str,
    category: str,
) -> tuple[int, list[str]]:
    return _save_file_result_impl(file_result, conn, get_settings(), tags, category)


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------

@app.command()
def ingest(
    path: Path = typer.Argument(..., help="File or directory to ingest."),
    mode: str = typer.Option("text", "--mode", "-m", help="text or html."),
    pages: Optional[str] = typer.Option(None, "--pages", help='Page range, e.g. "1,3-5".'),
    tags: str = typer.Option("", "--tags", "-t", help="Comma-separated tags."),
    category: str = typer.Option("", "--category", "-c", help="Category label."),
    key_points: bool = typer.Option(False, "--key-points", help="Extract key points via text model."),
    summary: bool = typer.Option(False, "--summary", help="Generate summary via text model."),
    vision_model: Optional[str] = typer.Option(None, "--vision-model", help="Override vision/OCR model name for this run."),
    text_model: Optional[str] = typer.Option(None, "--text-model", help="Override text/enrichment model name for this run."),
    db: Optional[Path] = typer.Option(None, "--db", help="Path to SQLite DB."),
    auto_compile: bool = typer.Option(False, "--auto-compile", help="Compile the wiki after ingestion."),
    auto_lint: bool = typer.Option(False, "--auto-lint", help="Lint the wiki after compilation."),
) -> None:
    """Ingest a file or folder into the knowledge base."""
    if not path.exists():
        console.print(f"[red]Path not found:[/red] {path}")
        raise typer.Exit(1)

    s = get_settings()
    conn, db_path = _db_conn(db)

    setup_logging(s.kb_dir / "logs")
    if s.enable_embeddings:
        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            console.print(
                "[yellow]Warning:[/yellow] ENABLE_EMBEDDINGS=true but "
                "[bold]sentence-transformers[/bold] is not installed. "
                "Embeddings will be skipped. Install with: "
                "[dim]pip install sentence-transformers[/dim]"
            )

    console.print(f"[bold]Ingesting[/bold] {path}  →  [cyan]{db_path}[/cyan]")
    if vision_model:
        console.print(f"  [dim]vision model:[/dim] {vision_model}")
    if text_model:
        console.print(f"  [dim]text model:  [/dim] {text_model}")

    total_saved = 0
    all_errors: list[str] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(f"[cyan]{path.name}[/cyan]", total=None)

        def _on_page(current: int, total: int) -> None:
            progress.update(task, total=total, completed=current,
                            description=f"[cyan]{path.name}[/cyan]  page {current}/{total}")

        results = process_path(
            path, s,
            mode=mode,  # type: ignore[arg-type]
            page_range=pages,
            extract_key_points=key_points,
            extract_summary=summary,
            vision_model=vision_model or None,
            text_model=text_model or None,
            on_page=_on_page,
        )

    batch_table = Table(title="Ingest Summary", show_lines=False)
    batch_table.add_column("File", style="cyan")
    batch_table.add_column("Pages", style="green", width=6)
    batch_table.add_column("Errors", style="red", width=7)
    batch_table.add_column("Time (s)", style="dim", width=8)

    for file_result in results:
        label = file_result.source_path.name
        with IngestTimer() as t:
            saved, errors = _save_file_result(file_result, conn, tags, category)
        elapsed = t["elapsed_secs"]
        total_saved += saved
        all_errors.extend(errors)
        icon = "✓" if saved else "⚠"
        console.print(f"  {icon}  [green]{label}[/green]  {saved} page(s) saved")
        log_ingest_summary(str(file_result.source_path), saved, errors, elapsed)
        batch_table.add_row(label, str(saved), str(len(errors)), f"{elapsed:.2f}")
        if not file_result.succeeded and errors:
            error_summary = "; ".join(errors[:3])
            dlq_push(conn, str(file_result.source_path), error_summary, max_retries=s.max_retries)
            log_failed_document(str(file_result.source_path), error_summary, 0)
            console.print(f"  [yellow]→ Added to dead-letter queue[/yellow]")

    if len(results) > 1:
        console.print(batch_table)

    if all_errors:
        console.print("\n[yellow]Errors:[/yellow]")
        for e in all_errors:
            console.print(f"  [red]•[/red] {e}")

    console.print(f"\n[bold green]Done.[/bold green] {total_saved} entr{'y' if total_saved == 1 else 'ies'} saved.")

    if auto_compile and total_saved > 0:
        console.print("\n[bold]Auto-compiling Wiki...[/bold]")
        from ocr_kb.wiki.compiler import compile_wiki
        stats = compile_wiki(s)
        console.print(f"[bold green]Wiki compiled.[/bold green] {stats['concepts_found']} concepts synthesized.")

        if auto_lint:
            console.print("\n[bold]Auto-linting Wiki...[/bold]")
            from ocr_kb.wiki.linter import lint_wiki
            report = lint_wiki(s)
            has_issues = any(items for k, items in report.items() if k != "error")
            if not has_issues and "error" not in report:
                console.print("[bold green]No issues found.[/bold green]")
            else:
                total_issues = sum(len(v) for v in report.values()) if "error" not in report else 1
                console.print(f"[bold red]{total_issues} issue(s). Run 'ocr-kb lint-wiki' for details.[/bold red]")


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

@app.command()
def search(
    query: str = typer.Argument(..., help="Search query."),
    limit: int = typer.Option(10, "--limit", "-n", help="Max results."),
    source: Optional[str] = typer.Option(None, "--source", "-s", help="Filter by source filename (substring)."),
    category: Optional[str] = typer.Option(None, "--category", "-c", help="Filter by exact category."),
    tag: Optional[str] = typer.Option(None, "--tag", "-t", help="Filter by tag (substring)."),
    after: Optional[str] = typer.Option(None, "--after", help="Only entries after ISO date, e.g. 2024-01-01."),
    db: Optional[Path] = typer.Option(None, "--db", help="Path to SQLite DB."),
) -> None:
    """Full-text search the knowledge base with optional filters."""
    conn, _ = _db_conn(db)
    s = get_settings()
    setup_logging(s.kb_dir / "logs")

    if any([source, category, tag, after]):
        with SearchTimer() as t:
            results = filtered_search(conn, query, source=source, category=category,
                                      tag=tag, after=after, limit=limit)
        log_search_query(query, len(results), t["elapsed_ms"])
    else:
        with SearchTimer() as t:
            results = hybrid_search(conn, query, s, limit=limit)
        log_search_query(query, len(results), t["elapsed_ms"])

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        raise typer.Exit(0)

    table = Table(title=f'Results for "{query}"', show_lines=True)
    table.add_column("ID", style="dim", width=5)
    table.add_column("Source", style="cyan")
    table.add_column("Pg", width=4)
    table.add_column("Category", style="green")
    table.add_column("Tags", style="blue")
    table.add_column("Excerpt", no_wrap=False, max_width=60)

    for entry in results:
        excerpt = (entry.markdown or entry.raw_text)[:200].replace("\n", " ")
        table.add_row(
            str(entry.id),
            Path(entry.source_path).name,
            str(entry.page_number),
            entry.category or "—",
            entry.tags or "—",
            excerpt,
        )

    console.print(table)


# ---------------------------------------------------------------------------
# show
# ---------------------------------------------------------------------------

@app.command()
def show(
    entry_id: int = typer.Argument(..., help="Entry ID from 'search' or 'recent'."),
    open_file: bool = typer.Option(False, "--open", help="Open the source file with xdg-open."),
    db: Optional[Path] = typer.Option(None, "--db", help="Path to SQLite DB."),
) -> None:
    """Show full content of a single knowledge base entry."""
    conn, _ = _db_conn(db)
    entry = get(conn, entry_id)
    if entry is None:
        console.print(f"[red]No entry with id={entry_id}[/red]")
        raise typer.Exit(1)

    console.print(Panel(
        f"[bold]{Path(entry.source_path).name}[/bold]  •  page {entry.page_number}  •  "
        f"id={entry.id}  •  {entry.created_at.strftime('%Y-%m-%d %H:%M')}",
        title="[cyan]KB Entry[/cyan]",
    ))

    meta_table = Table(show_header=False, box=None)
    meta_table.add_column(style="bold dim", width=14)
    meta_table.add_column()
    meta_table.add_row("Source", entry.source_path)
    meta_table.add_row("Category", entry.category or "—")
    meta_table.add_row("Tags", entry.tags or "—")
    if entry.summary:
        meta_table.add_row("Summary", entry.summary[:300])
    if entry.key_points:
        meta_table.add_row("Key Points", entry.key_points[:300])
    console.print(meta_table)

    console.rule("[dim]Content[/dim]")
    console.print(entry.markdown or entry.raw_text)

    if open_file:
        src = Path(entry.source_path)
        if not src.exists():
            console.print(f"[red]Source file not found:[/red] {src}")
        else:
            opener = "xdg-open" if sys.platform.startswith("linux") else "open"
            subprocess.Popen([opener, str(src)])
            console.print(f"[dim]Opening {src.name}...[/dim]")


# ---------------------------------------------------------------------------
# recent
# ---------------------------------------------------------------------------

@app.command()
def recent(
    limit: int = typer.Option(20, "--limit", "-n", help="Number of entries to show."),
    db: Optional[Path] = typer.Option(None, "--db", help="Path to SQLite DB."),
) -> None:
    """List the most recently ingested entries."""
    conn, _ = _db_conn(db)
    entries = get_recent(conn, limit=limit)

    if not entries:
        console.print("[yellow]No entries in the knowledge base yet.[/yellow]")
        raise typer.Exit(0)

    table = Table(title=f"Recent entries (last {limit})", show_lines=True)
    table.add_column("ID", style="dim", width=5)
    table.add_column("Ingested", style="dim", width=16)
    table.add_column("Source", style="cyan")
    table.add_column("Pg", width=4)
    table.add_column("Category", style="green")
    table.add_column("Tags", style="blue")

    for entry in entries:
        table.add_row(
            str(entry.id),
            entry.created_at.strftime("%Y-%m-%d %H:%M"),
            Path(entry.source_path).name,
            str(entry.page_number),
            entry.category or "—",
            entry.tags or "—",
        )

    console.print(table)


# ---------------------------------------------------------------------------
# tag
# ---------------------------------------------------------------------------

@app.command()
def tag(
    entry_id: int = typer.Argument(..., help="Entry ID to modify."),
    add: str = typer.Option("", "--add", help="Comma-separated tags to add."),
    remove: str = typer.Option("", "--remove", help="Comma-separated tags to remove."),
    set_category: Optional[str] = typer.Option(None, "--category", "-c", help="Replace category."),
    db: Optional[Path] = typer.Option(None, "--db", help="Path to SQLite DB."),
) -> None:
    """Add or remove tags (and optionally change category) without re-ingesting."""
    conn, _ = _db_conn(db)
    entry = get(conn, entry_id)
    if entry is None:
        console.print(f"[red]No entry with id={entry_id}[/red]")
        raise typer.Exit(1)

    current = {t.strip() for t in entry.tags.split(",") if t.strip()}
    if add:
        current |= {t.strip() for t in add.split(",") if t.strip()}
    if remove:
        current -= {t.strip() for t in remove.split(",") if t.strip()}

    new_tags = ",".join(sorted(current))
    update_tags(conn, entry_id, new_tags)
    console.print(f"[green]Tags updated:[/green] {new_tags or '(none)'}")

    if set_category is not None:
        update_category(conn, entry_id, set_category)
        console.print(f"[green]Category set:[/green] {set_category}")


# ---------------------------------------------------------------------------
# retag  (bulk tag editing via search query)
# ---------------------------------------------------------------------------

@app.command()
def retag(
    query: str = typer.Option(..., "--query", "-q", help="FTS query to select entries."),
    add: str = typer.Option("", "--add", help="Tags to add to every matched entry."),
    remove: str = typer.Option("", "--remove", help="Tags to remove from every matched entry."),
    limit: int = typer.Option(100, "--limit", "-n", help="Max entries to modify."),
    db: Optional[Path] = typer.Option(None, "--db", help="Path to SQLite DB."),
) -> None:
    """Bulk-edit tags on all entries matching a search query."""
    conn, _ = _db_conn(db)
    entries = fts_search(conn, query, limit=limit)
    if not entries:
        console.print("[yellow]No entries matched.[/yellow]")
        raise typer.Exit(0)

    add_set = {t.strip() for t in add.split(",") if t.strip()}
    remove_set = {t.strip() for t in remove.split(",") if t.strip()}

    updated = 0
    for entry in entries:
        current = {t.strip() for t in entry.tags.split(",") if t.strip()}
        current = (current | add_set) - remove_set
        update_tags(conn, entry.id, ",".join(sorted(current)))  # type: ignore[arg-type]
        updated += 1

    console.print(f"[green]Updated tags on {updated} entr{'y' if updated == 1 else 'ies'}.[/green]")


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------

@app.command()
def delete(
    entry_id: Optional[int] = typer.Option(None, "--id", help="Delete a single entry by ID."),
    source: Optional[str] = typer.Option(None, "--source", "-s", help="Delete all entries from this source path."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
    db: Optional[Path] = typer.Option(None, "--db", help="Path to SQLite DB."),
) -> None:
    """Remove entries from the knowledge base."""
    if entry_id is None and source is None:
        console.print("[red]Specify --id <N> or --source <path>.[/red]")
        raise typer.Exit(1)

    conn, _ = _db_conn(db)

    if entry_id is not None:
        entry = get(conn, entry_id)
        if entry is None:
            console.print(f"[red]No entry with id={entry_id}[/red]")
            raise typer.Exit(1)
        if not yes:
            typer.confirm(
                f"Delete entry {entry_id} ({Path(entry.source_path).name} p.{entry.page_number})?",
                abort=True,
            )
        _db_delete(conn, entry_id)
        console.print(f"[green]Deleted entry {entry_id}.[/green]")

    elif source is not None:
        count = delete_by_source(conn, source)
        if count == 0:
            console.print(f"[yellow]No entries found for source:[/yellow] {source}")
        else:
            if not yes:
                typer.confirm(f"Delete {count} entry/entries from '{source}'?", abort=True)
                count = delete_by_source(conn, source)  # re-run after confirm
            console.print(f"[green]Deleted {count} entry/entries from {source}.[/green]")


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------

@app.command()
def export(
    out: Path = typer.Option(..., "--out", "-o", help="Output path (file or directory)."),
    fmt: str = typer.Option("markdown", "--format", "-f", help="markdown, json, or obsidian."),
    filter_tag: Optional[str] = typer.Option(None, "--filter-tag", help="Only export entries with this tag."),
    filter_category: Optional[str] = typer.Option(None, "--filter-category", help="Only export entries in this category."),
    db: Optional[Path] = typer.Option(None, "--db", help="Path to SQLite DB."),
) -> None:
    """Export knowledge base entries to Markdown, JSON, or Obsidian."""
    conn, _ = _db_conn(db)

    fmt = fmt.lower()
    if fmt == "markdown":
        paths = export_markdown(conn, out, filter_tag=filter_tag, filter_category=filter_category)
        console.print(f"[green]Exported {len(paths)} Markdown file(s) to {out}[/green]")
    elif fmt == "json":
        path = export_json(conn, out, filter_tag=filter_tag, filter_category=filter_category)
        console.print(f"[green]Exported JSON to {path}[/green]")
    elif fmt == "obsidian":
        paths = export_obsidian(conn, out, filter_tag=filter_tag, filter_category=filter_category)
        console.print(f"[green]Exported {len(paths)} Obsidian note(s) to {out}[/green]")
    else:
        console.print(f"[red]Unknown format:[/red] {fmt}. Choose markdown, json, or obsidian.")
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------

@app.command(name="stats")
def stats_cmd(
    db: Optional[Path] = typer.Option(None, "--db", help="Path to SQLite DB."),
) -> None:
    """Show knowledge base statistics."""
    conn, db_path = _db_conn(db)
    s = _kb_stats(conn)

    table = Table(title=f"KB Stats — {db_path}", show_header=False)
    table.add_column("Metric", style="bold")
    table.add_column("Value", style="cyan")
    table.add_row("Total entries", str(s["total_entries"]))
    table.add_row("Unique sources", str(s["unique_sources"]))
    console.print(table)


# ---------------------------------------------------------------------------
# watch
# ---------------------------------------------------------------------------

@app.command()
def watch(
    directory: Optional[Path] = typer.Option(None, "--dir", "-d", help="Directory to watch."),
    tags: str = typer.Option("", "--tags", "-t", help="Tags for auto-ingested files."),
    category: str = typer.Option("", "--category", "-c", help="Category for auto-ingested files."),
    db: Optional[Path] = typer.Option(None, "--db", help="Path to SQLite DB."),
    interval: Optional[int] = typer.Option(None, "--interval", help="Poll interval in seconds."),
    auto_compile: bool = typer.Option(False, "--auto-compile", help="Compile wiki after new files."),
    auto_lint: bool = typer.Option(False, "--auto-lint", help="Lint wiki after compilation."),
) -> None:
    """Watch a folder and automatically ingest new files."""
    from watchdog.events import FileCreatedEvent, FileSystemEventHandler
    from watchdog.observers import Observer

    s = get_settings()
    watch_dir = directory or s.watch_dir
    poll = interval or s.watch_interval
    conn, db_path = _db_conn(db)

    class _Handler(FileSystemEventHandler):
        def on_created(self, event: FileCreatedEvent) -> None:  # type: ignore[override]
            if event.is_directory:
                return
            p = Path(str(event.src_path))
            if p.suffix.lower() not in SUPPORTED_SUFFIXES:
                return
            console.print(f"[cyan]New file:[/cyan] {p.name}")
            results = process_path(p, s)
            total_saved = 0
            for fr in results:
                saved, errors = _save_file_result(fr, conn, tags, category)
                console.print(f"  → {saved} page(s) saved")
                for e in errors:
                    console.print(f"  [red]error:[/red] {e}")
                total_saved += saved

            if auto_compile and total_saved > 0:
                console.print("  → [bold]Auto-compiling Wiki...[/bold]")
                from ocr_kb.wiki.compiler import compile_wiki
                stats = compile_wiki(s)
                console.print(f"  → [bold green]{stats['concepts_found']} concepts.[/bold green]")

                if auto_lint:
                    from ocr_kb.wiki.linter import lint_wiki
                    report = lint_wiki(s)
                    has_issues = any(len(v) > 0 for k, v in report.items() if k != "error")
                    if not has_issues and "error" not in report:
                        console.print("  → [bold green]Wiki healthy.[/bold green]")
                    else:
                        total_issues = sum(len(v) for v in report.values()) if "error" not in report else 1
                        console.print(f"  → [bold red]{total_issues} issue(s).[/bold red]")

    observer = Observer()
    observer.schedule(_Handler(), str(watch_dir), recursive=False)
    observer.start()
    console.print(f"[bold]Watching[/bold] {watch_dir}  (Ctrl-C to stop)")
    try:
        while True:
            time.sleep(poll)
    except KeyboardInterrupt:
        pass
    finally:
        observer.stop()
        observer.join()


# ---------------------------------------------------------------------------
# wiki
# ---------------------------------------------------------------------------

@app.command(name="compile-wiki")
def compile_wiki_cmd(
    limit: Optional[int] = typer.Option(None, "--limit", "-n", help="Max pages to process."),
    since: Optional[str] = typer.Option(None, "--since", help='Only process entries after ISO date, e.g. "2024-06-01".'),
) -> None:
    """Batch-process database entries to synthesize concepts and links."""
    from ocr_kb.wiki.compiler import compile_wiki
    s = get_settings()
    label = f"entries after {since}" if since else "all entries"
    console.print(f"[bold]Compiling Wiki[/bold] ({label})…")
    stats = compile_wiki(s, limit=limit, since=since)
    console.print(
        f"[bold green]Done.[/bold green] Processed {stats['processed_entries']} pages, "
        f"synthesized {stats['concepts_found']} concepts."
    )
    console.print(f"Wiki location: [cyan]{s.markdown_dir}[/cyan]")


@app.command(name="lint-wiki")
def lint_wiki_cmd() -> None:
    """Health-check the wiki for broken links, orphan pages, or missing front matter."""
    from ocr_kb.wiki.linter import lint_wiki
    s = get_settings()
    console.print("[bold]Linting Wiki...[/bold]")
    report = lint_wiki(s)

    if "error" in report:
        console.print(f"[red]Error:[/red] {report['error'][0]}")
        return

    has_issues = False
    for key, items in report.items():
        if items:
            has_issues = True
            console.print(f"\n[bold yellow]{key.upper().replace('_', ' ')}[/bold yellow] ({len(items)})")
            for item in items:
                console.print(f"  • {item}")

    if not has_issues:
        console.print("[bold green]No issues found! Wiki is healthy.[/bold green]")
    else:
        console.print(f"\n[bold red]Found {sum(len(v) for v in report.values())} total issues.[/bold red]")


# ---------------------------------------------------------------------------
# ask  (RAG query over the knowledge base via Gemma)
# ---------------------------------------------------------------------------

@app.command()
def ask(
    question: str = typer.Argument(..., help="Natural-language question about your documents."),
    context_size: int = typer.Option(5, "--context", "-n", help="Number of documents to retrieve."),
    chunk_chars: Optional[int] = typer.Option(None, "--chunk-chars", help="Override chars per document snippet."),
    text_model: Optional[str] = typer.Option(None, "--text-model", help="Override text model for this RAG call."),
    db: Optional[Path] = typer.Option(None, "--db", help="Path to SQLite DB."),
) -> None:
    """Ask a question about the knowledge base (retrieval-augmented generation)."""
    from ocr_kb.model import run_enrichment
    from ocr_kb.prompts import format_rag_prompt

    conn, _ = _db_conn(db)
    s = get_settings()
    setup_logging(s.kb_dir / "logs")
    chars = chunk_chars if chunk_chars is not None else s.rag_chunk_chars

    with SearchTimer() as t:
        hits = hybrid_search(conn, question, s, limit=context_size)
    log_search_query(question, len(hits), t["elapsed_ms"])

    if not hits:
        console.print("[yellow]No relevant documents found in the knowledge base.[/yellow]")
        raise typer.Exit(0)

    context_parts: list[str] = []
    for i, entry in enumerate(hits, start=1):
        source = f"{Path(entry.source_path).name} p.{entry.page_number}"
        text = (entry.markdown or entry.raw_text)[:chars]
        context_parts.append(f"[{i}] {source}:\n{text}")

    context = "\n\n---\n\n".join(context_parts)
    prompt = format_rag_prompt(question, context)

    console.print(f"[dim]Searching {len(hits)} document(s)…[/dim]")
    answer = run_enrichment(prompt, s, text_model=text_model or None)
    console.print(f"\n[bold]Answer:[/bold]\n\n{answer}")

    console.print("\n[dim]Sources:[/dim]")
    for i, entry in enumerate(hits, start=1):
        console.print(
            f"  [dim][{i}] {Path(entry.source_path).name}  p.{entry.page_number}"
            f"  id={entry.id}  [{entry.tags or 'no tags'}][/dim]"
        )


# ---------------------------------------------------------------------------
# models  (list available models from the configured backend)
# ---------------------------------------------------------------------------

@app.command(name="models")
def models_cmd() -> None:
    """List models available from the configured backend (Ollama or LM Studio)."""
    import json
    import urllib.error
    import urllib.request

    s = get_settings()

    if s.backend_provider == "ollama":
        ollama_host = s.ollama_base_url.rstrip("/")
        if ollama_host.endswith("/v1"):
            ollama_host = ollama_host[:-3]

        console.print(f"\n[bold]Ollama models[/bold] at [cyan]{ollama_host}[/cyan]\n")
        try:
            with urllib.request.urlopen(f"{ollama_host}/api/tags", timeout=5) as resp:
                data = json.loads(resp.read())
        except urllib.error.URLError as exc:
            console.print(f"[red]Could not reach Ollama:[/red] {exc}")
            console.print(f"[dim]Is Ollama running? Try: ollama serve[/dim]")
            raise typer.Exit(1)

        model_list = data.get("models", [])
        if not model_list:
            console.print("[yellow]No models found. Pull one with: ollama pull llava:7b[/yellow]")
            raise typer.Exit(0)

        table = Table(show_header=True, show_lines=False)
        table.add_column("Model", style="cyan", min_width=30)
        table.add_column("Size", style="dim", width=8)
        table.add_column("Family", style="dim", width=14)
        table.add_column("Type", width=8)

        for m in sorted(model_list, key=lambda x: x["name"]):
            name = m["name"]
            size_gb = f"{m.get('size', 0) / 1e9:.1f} GB"
            details = m.get("details", {})
            families = details.get("families") or [details.get("family", "")]
            family_str = ", ".join(f for f in families if f)
            # Known vision-capable model families
            _VISION_FAMILIES = {"clip", "llava", "glmocr", "minicpm", "qwen2vl", "llava_next"}
            is_vision = any(f in _VISION_FAMILIES for f in families)
            tag = "[green]VISION[/green]" if is_vision else "[blue]TEXT[/blue]  "
            table.add_row(name, size_gb, family_str, tag)

        console.print(table)
        console.print(
            f"\n[dim]Current:  vision=[bold]{s.vision_model_name}[/bold]"
            f"   text=[bold]{s.text_model_name}[/bold][/dim]"
        )
        console.print(
            "[dim]Override per-run with --vision-model / --text-model, or set "
            "OLLAMA_VISION_MODEL / OLLAMA_TEXT_MODEL in local.env[/dim]"
        )

    else:
        # LM Studio — no programmatic model list without calling the API
        console.print(f"\n[bold]LM Studio backend[/bold]\n")
        table = Table(show_header=False, box=None)
        table.add_column(style="bold dim", width=18)
        table.add_column(style="cyan")
        table.add_row("Vision / OCR", f"{s.glm_ocr_model_name}  →  {s.glm_ocr_base_url}")
        table.add_row("Text / RAG", f"{s.gemma_model_name}  →  {s.gemma_base_url}")
        console.print(table)
        console.print(
            "\n[dim]Set BACKEND_PROVIDER=ollama in local.env to switch to Ollama.[/dim]"
        )


# ---------------------------------------------------------------------------
# Dead-Letter Queue (DLQ) management
# ---------------------------------------------------------------------------

dlq_app = typer.Typer(name="dlq", help="Dead-letter queue management.", no_args_is_help=True)
app.add_typer(dlq_app)


@dlq_app.command(name="list")
def dlq_list_cmd(
    db: Optional[Path] = typer.Option(None, "--db", help="Path to SQLite DB."),
) -> None:
    """Show all unresolved dead-letter queue items."""
    conn, _ = _db_conn(db)
    items = dlq_list(conn)
    if not items:
        console.print("[green]Dead-letter queue is empty.[/green]")
        raise typer.Exit(0)

    table = Table(title=f"Dead-Letter Queue ({len(items)} item(s))", show_lines=True)
    table.add_column("Source", style="cyan")
    table.add_column("Retries", width=8)
    table.add_column("Max", width=5)
    table.add_column("Last Failed", style="dim", width=20)
    table.add_column("Error", no_wrap=False, max_width=50)

    for item in items:
        table.add_row(
            Path(item["source_path"]).name,
            str(item["retry_count"]),
            str(item["max_retries"]),
            item["last_failed_at"][:19],
            item["error_message"][:120],
        )
    console.print(table)


@dlq_app.command(name="retry")
def dlq_retry_cmd(
    db: Optional[Path] = typer.Option(None, "--db", help="Path to SQLite DB."),
) -> None:
    """Re-process all retryable dead-letter queue items."""
    conn, _ = _db_conn(db)
    s = get_settings()
    setup_logging(s.kb_dir / "logs")
    items = dlq_get_retryable(conn)

    if not items:
        console.print("[green]No retryable items in the dead-letter queue.[/green]")
        raise typer.Exit(0)

    console.print(f"[bold]Retrying {len(items)} item(s)...[/bold]")

    result_table = Table(title="Retry Results", show_lines=True)
    result_table.add_column("Source", style="cyan")
    result_table.add_column("Result", width=10)
    result_table.add_column("Pages", width=6)

    for item in items:
        source_path = Path(item["source_path"])
        console.print(f"  [dim]→ {source_path.name}[/dim]")
        try:
            file_results = process_path(source_path, s)
            any_saved = 0
            any_errors: list[str] = []
            for fr in file_results:
                saved, errors = _save_file_result(fr, conn, "", "")
                any_saved += saved
                any_errors.extend(errors)
            if any_saved > 0:
                dlq_mark_resolved(conn, item["source_path"])
                result_table.add_row(source_path.name, "[green]success[/green]", str(any_saved))
            else:
                error_summary = "; ".join(any_errors[:3]) if any_errors else "no pages produced"
                dlq_push(conn, item["source_path"], error_summary, max_retries=s.max_retries)
                log_failed_document(item["source_path"], error_summary, item["retry_count"] + 1)
                result_table.add_row(source_path.name, "[red]failed[/red]", "0")
        except Exception as exc:
            dlq_push(conn, item["source_path"], str(exc), max_retries=s.max_retries)
            log_failed_document(item["source_path"], str(exc), item["retry_count"] + 1)
            result_table.add_row(source_path.name, "[red]failed[/red]", "0")

    console.print(result_table)


@dlq_app.command(name="clear")
def dlq_clear_cmd(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation."),
    db: Optional[Path] = typer.Option(None, "--db", help="Path to SQLite DB."),
) -> None:
    """Mark all unresolved dead-letter queue items as resolved (bulk clear)."""
    conn, _ = _db_conn(db)
    items = dlq_list(conn)
    if not items:
        console.print("[green]Dead-letter queue is already empty.[/green]")
        raise typer.Exit(0)
    if not yes:
        typer.confirm(f"Mark {len(items)} item(s) as resolved?", abort=True)
    for item in items:
        dlq_mark_resolved(conn, item["source_path"])
    console.print(f"[green]Cleared {len(items)} item(s) from the dead-letter queue.[/green]")


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

@app.callback()
def main() -> None:
    """Local OCR knowledge-base pipeline — vision extraction + RAG over local LLMs."""


if __name__ == "__main__":
    app()
