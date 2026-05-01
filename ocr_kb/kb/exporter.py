from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from ocr_kb.kb.schema import KBEntry
from ocr_kb.kb.store import get_all


def _filtered_entries(
    conn: sqlite3.Connection,
    filter_tag: str | None,
    filter_category: str | None,
) -> list[KBEntry]:
    entries = get_all(conn)
    if filter_tag:
        entries = [e for e in entries if filter_tag.lower() in e.tags.lower()]
    if filter_category:
        entries = [e for e in entries if e.category.lower() == filter_category.lower()]
    return entries


def _slug(text: str) -> str:
    """Very simple slug: lowercase, spaces → underscores, strip non-alphanumeric."""
    import re
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    return re.sub(r"[\s-]+", "_", text)[:80]


def export_markdown(
    conn: sqlite3.Connection,
    output_dir: Path,
    *,
    filter_tag: str | None = None,
    filter_category: str | None = None,
) -> list[Path]:
    """Write one .md file per entry into *output_dir*.

    Filename: <source_stem>_p<NNN>.md
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = _filtered_entries(conn, filter_tag, filter_category)
    written: list[Path] = []
    for entry in entries:
        stem = Path(entry.source_path).stem
        name = f"{stem}_p{entry.page_number:03d}.md"
        path = output_dir / name
        path.write_text(_entry_to_markdown(entry), encoding="utf-8")
        written.append(path)
    return written


def export_json(
    conn: sqlite3.Connection,
    output_path: Path,
    *,
    filter_tag: str | None = None,
    filter_category: str | None = None,
) -> Path:
    """Serialise entries to a single JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    entries = _filtered_entries(conn, filter_tag, filter_category)
    data = [_entry_to_dict(e) for e in entries]
    output_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return output_path


def export_obsidian(
    conn: sqlite3.Connection,
    vault_dir: Path,
    *,
    filter_tag: str | None = None,
    filter_category: str | None = None,
) -> list[Path]:
    """Write Obsidian-compatible Markdown files with YAML front matter.

    Files go to *vault_dir*/ocr_kb/<source_stem>/<page>.md
    """
    entries = _filtered_entries(conn, filter_tag, filter_category)
    written: list[Path] = []
    for entry in entries:
        stem = Path(entry.source_path).stem
        folder = vault_dir / "ocr_kb" / stem
        folder.mkdir(parents=True, exist_ok=True)
        name = f"p{entry.page_number:03d}.md"
        path = folder / name
        path.write_text(_entry_to_obsidian(entry), encoding="utf-8")
        written.append(path)
    return written


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

def _entry_to_markdown(entry: KBEntry) -> str:
    front_matter = [
        "---",
        f'title: "{Path(entry.source_path).name} — page {entry.page_number}"',
        "type: source-summary",
        f'source_path: "{entry.source_path}"',
        f"page: {entry.page_number}",
        f'category: "{entry.category or ""}"',
        f'tags: "{entry.tags or ""}"',
        f"created: {entry.created_at}",
        "---",
        "",
    ]
    body = []
    if entry.summary:
        body += ["## Summary", "", entry.summary, ""]
    if entry.key_points:
        body += ["## Key Points", "", entry.key_points, ""]
    if entry.enriched_metadata:
        body += ["## Enriched Metadata", "", entry.enriched_metadata, ""]
    body += ["## Content", "", entry.markdown or entry.raw_text]
    return "\n".join(front_matter) + "\n".join(body)


def _entry_to_obsidian(entry: KBEntry) -> str:
    tags_yaml = (
        "\n".join(f"  - {t.strip()}" for t in entry.tags.split(",") if t.strip())
        if entry.tags
        else ""
    )
    front_matter = "\n".join([
        "---",
        f"source: \"{entry.source_path}\"",
        f"page: {entry.page_number}",
        f"category: \"{entry.category}\"",
        "tags:",
        tags_yaml,
        f"created: \"{entry.created_at.isoformat()}\"",
        "---",
        "",
    ])
    body_lines = []
    if entry.summary:
        body_lines += ["## Summary", "", entry.summary, ""]
    if entry.key_points:
        body_lines += ["## Key Points", "", entry.key_points, ""]
    if entry.enriched_metadata:
        body_lines += ["## Enriched Metadata", "", entry.enriched_metadata, ""]
    body_lines += ["## Content", "", entry.markdown or entry.raw_text]
    return front_matter + "\n".join(body_lines)


def _entry_to_dict(entry: KBEntry) -> dict:
    return {
        "id": entry.id,
        "source_path": entry.source_path,
        "page_number": entry.page_number,
        "raw_text": entry.raw_text,
        "markdown": entry.markdown,
        "tags": entry.tags,
        "category": entry.category,
        "key_points": entry.key_points,
        "summary": entry.summary,
        "enriched_metadata": entry.enriched_metadata,
        "created_at": entry.created_at.isoformat(),
    }
