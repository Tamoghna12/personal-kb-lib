"""Academic paper metadata extraction.

Priority:
  1. DOI regex from first-page text → Crossref API (structured, free)
  2. LLM extraction from first-page text (fallback, requires text model)
"""
from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

_log = logging.getLogger(__name__)

_DOI_RE = re.compile(r'\b(10\.\d{4,9}/[^\s"\'<>{}\|\\^`\[\]]+)', re.IGNORECASE)


@dataclass
class DocMetadata:
    title: str = ""
    authors: str = ""      # comma-separated "First Last, First Last"
    year: int | None = None
    doi: str = ""
    abstract: str = ""
    journal: str = ""


def extract_doi(text: str) -> str | None:
    m = _DOI_RE.search(text)
    return m.group(1).rstrip(".,;)") if m else None


def fetch_crossref(doi: str) -> DocMetadata | None:
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="/")
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "ocr-kb/1.0 (mailto:research@example.com)"}
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read())
    except Exception as exc:
        _log.debug("Crossref lookup failed doi=%s: %s", doi, exc)
        return None

    item = data.get("message", {})
    titles = item.get("title", [])
    title = titles[0] if titles else ""
    authors = ", ".join(
        f"{a.get('given', '')} {a.get('family', '')}".strip()
        for a in item.get("author", [])
    )
    year: int | None = None
    for date_key in ("published", "published-print", "published-online", "created"):
        parts = item.get(date_key, {}).get("date-parts", [[]])
        if parts and parts[0]:
            year = int(parts[0][0])
            break
    containers = item.get("container-title", [])
    journal = containers[0] if containers else item.get("publisher", "")
    abstract = re.sub(r"<[^>]+>", " ", item.get("abstract", "")).strip()
    return DocMetadata(title=title, authors=authors, year=year, doi=doi,
                       abstract=abstract, journal=journal)


def _extract_llm(first_page_text: str, settings) -> DocMetadata:
    from ocr_kb.model import run_enrichment
    from ocr_kb.prompts import format_metadata_prompt
    raw = run_enrichment(format_metadata_prompt(first_page_text), settings)
    # Strip thinking-model output (e.g. <think>...</think> blocks)
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    # Strip markdown code fences
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.DOTALL)
    # Extract first JSON object in case the model added prose around it
    json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", raw, re.DOTALL)
    if json_match:
        raw = json_match.group(0)
    try:
        d = json.loads(raw)
        yr = d.get("year")
        return DocMetadata(
            title=d.get("title", ""),
            authors=d.get("authors", ""),
            year=int(yr) if yr and str(yr).isdigit() else None,
            doi=d.get("doi", ""),
            abstract=d.get("abstract", ""),
            journal=d.get("journal", ""),
        )
    except (json.JSONDecodeError, TypeError, ValueError):
        _log.warning("LLM metadata extraction returned unparseable output")
        return DocMetadata()


def extract_metadata(first_page_text: str, settings) -> DocMetadata:
    """DOI → Crossref first; LLM fallback if text model is available."""
    doi = extract_doi(first_page_text)
    crossref_meta: DocMetadata | None = None
    if doi:
        crossref_meta = fetch_crossref(doi)
        if crossref_meta:
            _log.info("Crossref metadata: %s (%s)", crossref_meta.title, doi)
            return crossref_meta

    if settings.model_backend == "glm_only":
        return crossref_meta or DocMetadata(doi=doi or "")

    try:
        meta = _extract_llm(first_page_text, settings)
        if not meta.doi and doi:
            meta.doi = doi
        return meta
    except Exception as exc:
        _log.warning("LLM metadata extraction failed: %s", exc)
        return crossref_meta or DocMetadata(doi=doi or "")
