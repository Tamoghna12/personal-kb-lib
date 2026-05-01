from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ocr_kb.kb.schema import KBEntry

_DDL = """
CREATE TABLE IF NOT EXISTS entries (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    source_path   TEXT    NOT NULL,
    page_number   INTEGER NOT NULL,
    chunk_index   INTEGER NOT NULL DEFAULT 0,
    raw_text      TEXT    NOT NULL DEFAULT '',
    markdown      TEXT    NOT NULL DEFAULT '',
    html          TEXT    NOT NULL DEFAULT '',
    layout_blocks TEXT    NOT NULL DEFAULT '[]',
    tags          TEXT    NOT NULL DEFAULT '',
    category      TEXT    NOT NULL DEFAULT '',
    key_points         TEXT    NOT NULL DEFAULT '',
    summary            TEXT    NOT NULL DEFAULT '',
    enriched_metadata  TEXT    NOT NULL DEFAULT '',
    embedding          TEXT,
    created_at         TEXT    NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS entries_fts USING fts5(
    raw_text,
    markdown,
    tags,
    category,
    key_points,
    summary,
    enriched_metadata,
    content='entries',
    content_rowid='id'
);

CREATE TRIGGER IF NOT EXISTS entries_ai AFTER INSERT ON entries BEGIN
    INSERT INTO entries_fts(rowid, raw_text, markdown, tags, category, key_points, summary, enriched_metadata)
    VALUES (new.id, new.raw_text, new.markdown, new.tags, new.category, new.key_points, new.summary, new.enriched_metadata);
END;

CREATE TRIGGER IF NOT EXISTS entries_ad AFTER DELETE ON entries BEGIN
    INSERT INTO entries_fts(entries_fts, rowid, raw_text, markdown, tags, category, key_points, summary, enriched_metadata)
    VALUES ('delete', old.id, old.raw_text, old.markdown, old.tags, old.category, old.key_points, old.summary, old.enriched_metadata);
END;

CREATE TRIGGER IF NOT EXISTS entries_au AFTER UPDATE ON entries BEGIN
    INSERT INTO entries_fts(entries_fts, rowid, raw_text, markdown, tags, category, key_points, summary, enriched_metadata)
    VALUES ('delete', old.id, old.raw_text, old.markdown, old.tags, old.category, old.key_points, old.summary, old.enriched_metadata);
    INSERT INTO entries_fts(rowid, raw_text, markdown, tags, category, key_points, summary, enriched_metadata)
    VALUES (new.id, new.raw_text, new.markdown, new.tags, new.category, new.key_points, new.summary, new.enriched_metadata);
END;
"""

_DLQ_DDL = """
CREATE TABLE IF NOT EXISTS dead_letter_queue (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    source_path     TEXT    NOT NULL,
    error_message   TEXT    NOT NULL,
    retry_count     INTEGER NOT NULL DEFAULT 0,
    max_retries     INTEGER NOT NULL DEFAULT 3,
    first_failed_at TEXT    NOT NULL,
    last_failed_at  TEXT    NOT NULL,
    resolved_at     TEXT
);
"""


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(db_path: Path) -> sqlite3.Connection:
    conn = _connect(db_path)
    conn.executescript(_DDL)
    # Migrate existing databases — ignore error if column already exists.
    try:
        conn.execute("ALTER TABLE entries ADD COLUMN chunk_index INTEGER NOT NULL DEFAULT 0")
        conn.commit()
    except sqlite3.OperationalError:
        pass
    conn.executescript(_DLQ_DDL)
    return conn


def _entry_to_row(entry: KBEntry) -> dict[str, Any]:
    return {
        "source_path": entry.source_path,
        "page_number": entry.page_number,
        "chunk_index": entry.chunk_index,
        "raw_text": entry.raw_text,
        "markdown": entry.markdown,
        "html": entry.html,
        "layout_blocks": entry.layout_blocks,
        "tags": entry.tags,
        "category": entry.category,
        "key_points": entry.key_points,
        "summary": entry.summary,
        "enriched_metadata": entry.enriched_metadata,
        "embedding": json.dumps(entry.embedding) if entry.embedding is not None else None,
        "created_at": entry.created_at.isoformat(),
    }


def _row_to_entry(row: sqlite3.Row) -> KBEntry:
    d = dict(row)
    d["created_at"] = datetime.fromisoformat(d["created_at"])
    raw_emb = d.pop("embedding", None)
    d["embedding"] = json.loads(raw_emb) if raw_emb else None
    return KBEntry(**d)


def save(conn: sqlite3.Connection, entry: KBEntry) -> KBEntry:
    """Upsert: update existing (source_path, page_number) or insert new row.

    On re-ingest the same page is updated in-place, preserving its id so that
    any external references (e.g. show <id>) remain stable.
    """
    row = _entry_to_row(entry)
    existing = conn.execute(
        "SELECT id FROM entries WHERE source_path=? AND page_number=? AND chunk_index=?",
        (entry.source_path, entry.page_number, entry.chunk_index),
    ).fetchone()

    if existing:
        # Always refresh content; only overwrite tags/category when non-empty
        # so a bare re-ingest doesn't erase previously set metadata.
        existing_row = conn.execute(
            "SELECT tags, category FROM entries WHERE id=?", (existing["id"],)
        ).fetchone()
        effective_tags = row["tags"] if row["tags"] else (existing_row["tags"] or "")
        effective_category = row["category"] if row["category"] else (existing_row["category"] or "")

        conn.execute(
            """
            UPDATE entries SET
                raw_text=:raw_text, markdown=:markdown, html=:html,
                layout_blocks=:layout_blocks, tags=:tags, category=:category,
                key_points=:key_points, summary=:summary,
                enriched_metadata=:enriched_metadata, embedding=:embedding
            WHERE id=:id
            """,
            {**row, "tags": effective_tags, "category": effective_category, "id": existing["id"]},
        )
        conn.commit()
        entry.id = existing["id"]
        entry.tags = effective_tags
        entry.category = effective_category
    else:
        cur = conn.execute(
            """
            INSERT INTO entries
                (source_path, page_number, chunk_index, raw_text, markdown, html,
                 layout_blocks, tags, category, key_points, summary,
                 enriched_metadata, embedding, created_at)
            VALUES
                (:source_path, :page_number, :chunk_index, :raw_text, :markdown, :html,
                 :layout_blocks, :tags, :category, :key_points, :summary,
                 :enriched_metadata, :embedding, :created_at)
            """,
            row,
        )
        conn.commit()
        entry.id = cur.lastrowid

    return entry


def get(conn: sqlite3.Connection, entry_id: int) -> KBEntry | None:
    row = conn.execute("SELECT * FROM entries WHERE id = ?", (entry_id,)).fetchone()
    return _row_to_entry(row) if row else None


def get_all(conn: sqlite3.Connection) -> list[KBEntry]:
    rows = conn.execute("SELECT * FROM entries ORDER BY created_at DESC").fetchall()
    return [_row_to_entry(r) for r in rows]


def delete(conn: sqlite3.Connection, entry_id: int) -> bool:
    cur = conn.execute("DELETE FROM entries WHERE id = ?", (entry_id,))
    conn.commit()
    return cur.rowcount > 0


def delete_by_source(conn: sqlite3.Connection, source_path: str) -> int:
    """Delete all entries for a given source file. Returns number of rows removed."""
    cur = conn.execute("DELETE FROM entries WHERE source_path = ?", (source_path,))
    conn.commit()
    return cur.rowcount


def get_recent(conn: sqlite3.Connection, limit: int = 20) -> list[KBEntry]:
    """Return the most recently ingested entries, newest first."""
    rows = conn.execute(
        "SELECT * FROM entries ORDER BY created_at DESC LIMIT ?", (limit,)
    ).fetchall()
    return [_row_to_entry(r) for r in rows]


def update_tags(conn: sqlite3.Connection, entry_id: int, tags: str) -> bool:
    """Replace the tags on an entry. Returns True if the entry existed."""
    cur = conn.execute("UPDATE entries SET tags=? WHERE id=?", (tags, entry_id))
    conn.commit()
    return cur.rowcount > 0


def update_category(conn: sqlite3.Connection, entry_id: int, category: str) -> bool:
    """Replace the category on an entry. Returns True if the entry existed."""
    cur = conn.execute("UPDATE entries SET category=? WHERE id=?", (category, entry_id))
    conn.commit()
    return cur.rowcount > 0


def stats(conn: sqlite3.Connection) -> dict[str, Any]:
    row = conn.execute(
        "SELECT COUNT(*) as total, COUNT(DISTINCT source_path) as sources FROM entries"
    ).fetchone()
    return {"total_entries": row["total"], "unique_sources": row["sources"]}


# ---------------------------------------------------------------------------
# Dead-Letter Queue
# ---------------------------------------------------------------------------

def dlq_push(
    conn: sqlite3.Connection,
    source_path: str,
    error_msg: str,
    max_retries: int = 3,
) -> None:
    """Insert a new DLQ record, or increment retry_count+last_failed_at if one exists.

    An existing resolved record for the same source_path is re-opened
    (resolved_at set back to NULL) so a re-ingestion attempt is tracked.
    """
    now = datetime.now(timezone.utc).isoformat()
    existing = conn.execute(
        "SELECT id, retry_count FROM dead_letter_queue WHERE source_path = ?",
        (source_path,),
    ).fetchone()
    if existing:
        conn.execute(
            """
            UPDATE dead_letter_queue
               SET retry_count = retry_count + 1,
                   last_failed_at = ?,
                   error_message = ?,
                   resolved_at = NULL
             WHERE source_path = ?
            """,
            (now, error_msg, source_path),
        )
    else:
        conn.execute(
            """
            INSERT INTO dead_letter_queue
                (source_path, error_message, retry_count, max_retries,
                 first_failed_at, last_failed_at)
            VALUES (?, ?, 0, ?, ?, ?)
            """,
            (source_path, error_msg, max_retries, now, now),
        )
    conn.commit()


def dlq_list(conn: sqlite3.Connection) -> list[dict]:
    """Return all unresolved DLQ items as plain dicts."""
    rows = conn.execute(
        """
        SELECT id, source_path, error_message, retry_count, max_retries,
               first_failed_at, last_failed_at
          FROM dead_letter_queue
         WHERE resolved_at IS NULL
         ORDER BY last_failed_at DESC
        """
    ).fetchall()
    return [dict(r) for r in rows]


def dlq_mark_resolved(conn: sqlite3.Connection, source_path: str) -> bool:
    """Set resolved_at to now for the given source_path. Returns True if found."""
    now = datetime.now(timezone.utc).isoformat()
    cur = conn.execute(
        "UPDATE dead_letter_queue SET resolved_at = ? WHERE source_path = ?",
        (now, source_path),
    )
    conn.commit()
    return cur.rowcount > 0


def dlq_get_retryable(conn: sqlite3.Connection) -> list[dict]:
    """Return unresolved items where retry_count < max_retries."""
    rows = conn.execute(
        """
        SELECT id, source_path, error_message, retry_count, max_retries,
               first_failed_at, last_failed_at
          FROM dead_letter_queue
         WHERE resolved_at IS NULL
           AND retry_count < max_retries
         ORDER BY last_failed_at ASC
        """
    ).fetchall()
    return [dict(r) for r in rows]
