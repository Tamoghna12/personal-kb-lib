from __future__ import annotations

import json
import logging as _logging
import re
import sqlite3
import time as _time
from typing import TYPE_CHECKING

from ocr_kb.kb.schema import KBEntry
from ocr_kb.kb.store import _row_to_entry

if TYPE_CHECKING:
    from ocr_kb.settings import Settings

_idx_logger = _logging.getLogger("ocr_kb.indexer")


def _sanitize_fts(query: str) -> str:
    """Strip FTS5 special characters so natural-language questions don't crash.

    FTS5 treats these as syntax: " * ^ ( ) - + : @ . ?
    We keep only word characters and spaces, then quote each token so phrases
    with stopwords still produce useful results.
    """
    tokens = re.sub(r"[^\w\s]", " ", query, flags=re.UNICODE).split()
    if not tokens:
        return '""'
    return " OR ".join(f'"{t}"' for t in tokens)


def fts_search(
    conn: sqlite3.Connection,
    query: str,
    limit: int = 20,
) -> list[KBEntry]:
    """Full-text search across raw_text, markdown, tags, category, key_points, summary."""
    rows = conn.execute(
        """
        SELECT e.*
        FROM entries e
        JOIN entries_fts f ON e.id = f.rowid
        WHERE entries_fts MATCH ?
        ORDER BY rank
        LIMIT ?
        """,
        (_sanitize_fts(query), limit),
    ).fetchall()
    return [_row_to_entry(r) for r in rows]


def filtered_search(
    conn: sqlite3.Connection,
    query: str,
    *,
    source: str | None = None,
    category: str | None = None,
    tag: str | None = None,
    after: str | None = None,
    limit: int = 20,
) -> list[KBEntry]:
    """FTS search with optional column-level filters.

    *source*   — substring match on source_path (e.g. a filename stem)
    *category* — exact case-insensitive match
    *tag*      — substring match inside the comma-separated tags field
    *after*    — ISO-8601 date/datetime; only entries with created_at >= this value
    """
    conditions = ["entries_fts MATCH ?"]
    params: list = [_sanitize_fts(query)]

    if source:
        conditions.append("e.source_path LIKE ?")
        params.append(f"%{source}%")
    if category:
        conditions.append("LOWER(e.category) = ?")
        params.append(category.lower())
    if tag:
        conditions.append("LOWER(e.tags) LIKE ?")
        params.append(f"%{tag.lower()}%")
    if after:
        conditions.append("e.created_at >= ?")
        params.append(after)

    params.append(limit)
    where = " AND ".join(conditions)
    rows = conn.execute(
        f"""
        SELECT e.*
        FROM entries e
        JOIN entries_fts f ON e.id = f.rowid
        WHERE {where}
        ORDER BY rank
        LIMIT ?
        """,
        params,
    ).fetchall()
    return [_row_to_entry(r) for r in rows]


def filter_by_tag(
    conn: sqlite3.Connection,
    tag: str,
    limit: int = 100,
) -> list[KBEntry]:
    """Return entries whose tags field contains *tag* (case-insensitive substring)."""
    rows = conn.execute(
        "SELECT * FROM entries WHERE LOWER(tags) LIKE ? ORDER BY created_at DESC LIMIT ?",
        (f"%{tag.lower()}%", limit),
    ).fetchall()
    return [_row_to_entry(r) for r in rows]


def filter_by_source(
    conn: sqlite3.Connection,
    source_path: str,
) -> list[KBEntry]:
    """Return all entries from a specific source file, ordered by page number."""
    rows = conn.execute(
        "SELECT * FROM entries WHERE source_path = ? ORDER BY page_number",
        (source_path,),
    ).fetchall()
    return [_row_to_entry(r) for r in rows]


def semantic_search(
    conn: sqlite3.Connection,
    query_embedding: list[float],
    limit: int = 10,
) -> list[tuple[KBEntry, float]]:
    """Cosine-similarity search over stored embeddings.

    Returns (entry, score) pairs sorted by descending similarity.
    Entries without embeddings are skipped.
    """
    rows = conn.execute(
        "SELECT * FROM entries WHERE embedding IS NOT NULL"
    ).fetchall()

    results: list[tuple[KBEntry, float]] = []
    qmag = _magnitude(query_embedding)
    if qmag == 0:
        return []

    for row in rows:
        entry = _row_to_entry(row)
        if entry.embedding is None:
            continue
        score = _cosine(query_embedding, entry.embedding, qmag)
        results.append((entry, score))

    results.sort(key=lambda t: t[1], reverse=True)
    return results[:limit]


def hybrid_search(
    conn: sqlite3.Connection,
    query: str,
    settings: "Settings",
    limit: int = 20,
) -> list[KBEntry]:
    """Reciprocal Rank Fusion of FTS5 keyword search and semantic vector search.

    When ``settings.enable_embeddings`` is False (or sentence-transformers is not
    installed) this degrades gracefully to pure FTS5 search.
    """
    _t0 = _time.perf_counter()
    fts_results = fts_search(conn, query, limit=limit * 2)

    scores: dict[int, float] = {}
    entry_map: dict[int, KBEntry] = {}

    for rank, entry in enumerate(fts_results):
        eid = entry.id  # type: ignore[arg-type]
        scores[eid] = scores.get(eid, 0.0) + 1.0 / (60 + rank)
        entry_map[eid] = entry

    if settings.enable_embeddings:
        try:
            from ocr_kb.model.embedder import embed_text
            qemb = embed_text(query, settings)
            if qemb:
                for rank, (entry, _score) in enumerate(semantic_search(conn, qemb, limit=limit * 2)):
                    eid = entry.id  # type: ignore[arg-type]
                    scores[eid] = scores.get(eid, 0.0) + 1.0 / (60 + rank)
                    entry_map[eid] = entry
        except Exception:
            pass  # degrade to FTS-only if embedder fails

    sorted_ids = sorted(scores, key=lambda k: scores[k], reverse=True)
    results = [entry_map[eid] for eid in sorted_ids[:limit]]
    _idx_logger.debug(
        "hybrid_search | query=%r | final=%d | elapsed=%.1fms",
        query,
        len(results),
        (_time.perf_counter() - _t0) * 1000,
    )
    return results


# ---------------------------------------------------------------------------
# Vector helpers
# ---------------------------------------------------------------------------

def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _magnitude(v: list[float]) -> float:
    return sum(x * x for x in v) ** 0.5


def _cosine(a: list[float], b: list[float], mag_a: float | None = None) -> float:
    mag_a = mag_a if mag_a is not None else _magnitude(a)
    mag_b = _magnitude(b)
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return _dot(a, b) / (mag_a * mag_b)
