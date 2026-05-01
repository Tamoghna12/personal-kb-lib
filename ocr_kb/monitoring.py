"""Logging, timing, and observability helpers for ocr-kb.

This module is intentionally free of any ocr_kb package imports so it can
be imported early (before Settings is resolved) without circular dependencies.
"""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Generator


_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"
_initialized: bool = False


def setup_logging(log_dir: Path, *, level: int = logging.DEBUG) -> None:
    """Configure root ocr_kb logger with a rotating file handler + stream handler.

    Safe to call multiple times (no-ops after first call).
    """
    global _initialized
    if _initialized:
        return

    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "ocr_kb.log"

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # File handler — DEBUG and above, rotating 10 MB × 5 backups
    fh = RotatingFileHandler(
        log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # Stream handler — WARNING and above (avoid polluting the rich CLI output)
    sh = logging.StreamHandler()
    sh.setLevel(logging.WARNING)
    sh.setFormatter(formatter)

    root = logging.getLogger("ocr_kb")
    root.setLevel(level)
    root.addHandler(fh)
    root.addHandler(sh)

    _initialized = True


_logger = logging.getLogger("ocr_kb.monitoring")


def log_ingest_summary(
    source_path: str,
    pages_saved: int,
    errors: list[str],
    elapsed_secs: float,
) -> None:
    """Log an INFO-level summary after ingesting one file."""
    status = "ok" if not errors else f"{len(errors)} error(s)"
    _logger.info(
        "ingest | source=%s | pages_saved=%d | status=%s | elapsed=%.2fs",
        source_path,
        pages_saved,
        status,
        elapsed_secs,
    )
    for e in errors:
        _logger.warning("ingest_error | source=%s | %s", source_path, e)


def log_search_query(query: str, result_count: int, elapsed_ms: float) -> None:
    """Log a DEBUG-level search event (goes to file only, not to terminal)."""
    _logger.debug(
        "search | query=%r | results=%d | elapsed=%.1fms", query, result_count, elapsed_ms
    )


def log_failed_document(
    source_path: str, error: str, retry_count: int
) -> None:
    """Log a WARNING when a document is pushed to the dead-letter queue."""
    _logger.warning(
        "dlq_push | source=%s | retry_count=%d | error=%s",
        source_path,
        retry_count,
        error,
    )


@contextmanager
def IngestTimer() -> Generator[dict, None, None]:
    """Context manager that records wall-clock seconds.

    Usage::

        with IngestTimer() as t:
            ...
        elapsed = t["elapsed_secs"]
    """
    info: dict = {}
    start = time.perf_counter()
    try:
        yield info
    finally:
        info["elapsed_secs"] = time.perf_counter() - start


@contextmanager
def SearchTimer() -> Generator[dict, None, None]:
    """Context manager that records wall-clock milliseconds.

    Usage::

        with SearchTimer() as t:
            results = hybrid_search(...)
        elapsed_ms = t["elapsed_ms"]
    """
    info: dict = {}
    start = time.perf_counter()
    try:
        yield info
    finally:
        info["elapsed_ms"] = (time.perf_counter() - start) * 1000
