from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class KBEntry:
    source_path: str          # original file path (str for SQLite compatibility)
    page_number: int
    raw_text: str             # plain text from OCR or text file
    markdown: str             # cleaned Markdown
    html: str                 # cleaned HTML fragment (empty string if plain-text mode)
    layout_blocks: str        # JSON-serialised list of LayoutBlock dicts
    tags: str                 # comma-separated tags
    category: str
    created_at: datetime = field(default_factory=_now)
    key_points: str = ""
    summary: str = ""
    enriched_metadata: str = ""
    embedding: Optional[list[float]] = field(default=None, repr=False)
    chunk_index: int = 0
    id: Optional[int] = field(default=None)
