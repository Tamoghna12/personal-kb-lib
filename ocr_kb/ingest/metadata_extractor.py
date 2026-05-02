"""Academic paper metadata extraction.

Priority per DOI type:
  1. medRxiv / bioRxiv API  (free, structured, no key — for preprints)
  2. Crossref API            (free, structured, no key — for published papers)
  3. LLM extraction          (fallback; requires text model)
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


def _get_json(url: str, timeout: int = 8) -> dict | None:
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "ocr-kb/1.0 (mailto:research@example.com)"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception as exc:
        _log.debug("HTTP GET failed url=%s: %s", url, exc)
        return None


def fetch_preprint(doi: str, first_page_text: str = "") -> DocMetadata | None:
    """Query medRxiv or bioRxiv API for preprint metadata.

    Only supported for DOIs with the 10.1101 prefix (established preprint DOI
    prefix used by Cold Spring Harbor Labs for both servers). Newer DOI formats
    (e.g. 10.64898) are not yet indexed by these APIs.
    """
    if not doi.startswith("10.1101/"):
        return None

    lower = first_page_text.lower()
    if "medrxiv" in lower:
        servers = ["medrxiv", "biorxiv"]
    elif "biorxiv" in lower:
        servers = ["biorxiv", "medrxiv"]
    else:
        servers = ["medrxiv", "biorxiv"]

    encoded = urllib.parse.quote(doi, safe="/")
    for server in servers:
        data = _get_json(f"https://api.{server}.org/details/{server}/{encoded}")
        if not data:
            continue
        collection = data.get("collection", [])
        if not collection:
            continue
        item = collection[0]
        raw_authors = item.get("authors", "")
        # API returns "Last, First; Last, First" → convert to "First Last, First Last"
        authors = ", ".join(
            " ".join(reversed(a.strip().split(", ", 1))) if ", " in a else a.strip()
            for a in raw_authors.split(";")
            if a.strip()
        )
        date_str = item.get("date", "")
        year = int(date_str[:4]) if date_str and date_str[:4].isdigit() else None
        journal = f"{server.capitalize()} preprint"
        _log.info("%s metadata: %s (%s)", server, item.get("title", ""), doi)
        return DocMetadata(
            title=item.get("title", ""),
            authors=authors,
            year=year,
            doi=doi,
            abstract=item.get("abstract", ""),
            journal=journal,
        )
    return None


def fetch_crossref(doi: str) -> DocMetadata | None:
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="/")
    data = _get_json(url)
    if not data:
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
    _log.info("Crossref metadata: %s (%s)", title, doi)
    return DocMetadata(title=title, authors=authors, year=year, doi=doi,
                       abstract=abstract, journal=journal)


def _find_json(text: str) -> dict | None:
    """Extract the first valid JSON object from arbitrary text.

    More robust than a single regex: scans for '{', then walks forward
    tracking brace depth until the matching '}' is found. Handles
    thinking-model prose wrapped around the JSON.
    """
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        escape = False
        for i, ch in enumerate(text[start:], start):
            if escape:
                escape = False
                continue
            if ch == "\\" and in_str:
                escape = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def _extract_llm(first_page_text: str, settings) -> DocMetadata:
    from ocr_kb.model import run_enrichment
    from ocr_kb.prompts import format_metadata_prompt
    raw = run_enrichment(format_metadata_prompt(first_page_text), settings)
    # Strip thinking-model blocks (<think>...</think>)
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    # Strip markdown code fences
    raw = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "")

    d = _find_json(raw)
    if d is None:
        _log.warning("LLM metadata extraction returned unparseable output")
        return DocMetadata()

    yr = d.get("year")
    return DocMetadata(
        title=d.get("title", ""),
        authors=d.get("authors", ""),
        year=int(yr) if yr and str(yr).isdigit() else None,
        doi=d.get("doi", ""),
        abstract=d.get("abstract", ""),
        journal=d.get("journal", ""),
    )


def extract_metadata(first_page_text: str, settings) -> DocMetadata:
    """Extract bibliographic metadata using the best available source.

    Priority: preprint API (medRxiv/bioRxiv) → Crossref → LLM fallback.
    """
    doi = extract_doi(first_page_text)

    if doi:
        lower = first_page_text.lower()
        # Try preprint APIs first for anything that looks like a preprint
        if "medrxiv" in lower or "biorxiv" in lower or "rxiv" in lower:
            meta = fetch_preprint(doi, first_page_text)
            if meta:
                return meta

        # Crossref for published papers
        meta = fetch_crossref(doi)
        if meta:
            return meta

    if settings.model_backend == "glm_only":
        return DocMetadata(doi=doi or "")

    try:
        meta = _extract_llm(first_page_text, settings)
        if not meta.doi and doi:
            meta.doi = doi
        return meta
    except Exception as exc:
        _log.warning("LLM metadata extraction failed: %s", exc)
        return DocMetadata(doi=doi or "")
