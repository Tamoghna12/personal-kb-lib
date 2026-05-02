"""PDF-to-markdown extraction using granite-docling (local VLM via docling library).

Processes the entire PDF document at once using ibm-granite/granite-docling-258M,
producing clean structured markdown per page. Handles both digital and scanned PDFs.

Models are loaded lazily and cached for the process lifetime — first call downloads
weights from HuggingFace (~258M params) and loads them; subsequent calls are fast.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

_log = logging.getLogger(__name__)

_converter_cache: dict[str, object] = {}


def _get_converter(device: str = "cuda") -> object:
    if device not in _converter_cache:
        from docling.datamodel import vlm_model_specs
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import AcceleratorOptions, VlmPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.pipeline.vlm_pipeline import VlmPipeline

        _log.info("Loading granite-docling converter on device=%s (one-time load)", device)
        pipeline_options = VlmPipelineOptions(
            vlm_options=vlm_model_specs.GRANITEDOCLING_TRANSFORMERS,
            accelerator_options=AcceleratorOptions(
                device=device,
                cuda_use_flash_attention2=False,
            ),
        )
        _converter_cache[device] = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=VlmPipeline,
                    pipeline_options=pipeline_options,
                )
            }
        )
    return _converter_cache[device]


def _is_oom(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "outofmemory" in msg


# ---------------------------------------------------------------------------
# Per-page boilerplate stripping
# ---------------------------------------------------------------------------

# Phrases that uniquely identify preprint server / journal page headers/footers.
# Each pattern matches a whole paragraph containing the phrase — the paragraph is dropped.
_BOILERPLATE_FRAGS = [
    "medrxiv preprint doi:",
    "biorxiv preprint doi:",
    "all rights reserved. no reuse allowed",
    "not certified by peer review",
    "the copyright holder for this preprint",
    "downloaded from http",         # journal "Downloaded from" lines
    "this article is protected by copyright",
    "accepted manuscript",          # some journal footers
]


def _strip_page_headers(text: str) -> str:
    """Drop paragraphs that are purely preprint-server or journal boilerplate.

    Targets per-page copyright/license blocks stamped by medRxiv, bioRxiv, and
    common journal publishers on every page of every article.
    """
    paragraphs = re.split(r"\n\n+", text)
    clean = []
    for para in paragraphs:
        lower = para.lower()
        if not any(frag in lower for frag in _BOILERPLATE_FRAGS):
            clean.append(para)
    return "\n\n".join(clean)


# ---------------------------------------------------------------------------
# Manuscript line-number stripping
# ---------------------------------------------------------------------------

# Leading:  "- 22 text..."  or  "22 text..."
_NUM_LEADING = re.compile(r"^[-*•]?\s*(\d{1,4})\s+(\S.*)", re.MULTILINE)
# Trailing: "- text... 64"
_NUM_TRAILING = re.compile(r"^([-*•]?\s*)(.*\S)\s+(\d{1,4})\s*$", re.MULTILINE)
# Heading with trailing number: "## Conclusions: 27" or "## Introduction 35"
_HEADING_TRAIL_NUM_RE = re.compile(r"^(#{1,6}\s+.*?\S):?\s+(\d{2,})\s*$", re.MULTILINE)


def _consecutiveness(nums: list[int]) -> float:
    """Fraction of consecutive pairs (diff == 1) in a number sequence."""
    if len(nums) < 2:
        return 0.0
    diffs = [nums[i + 1] - nums[i] for i in range(len(nums) - 1)]
    return sum(d == 1 for d in diffs) / len(diffs)


def _strip_manuscript_line_numbers(text: str) -> str:
    """Remove sequential line numbers added by preprint servers (medRxiv, bioRxiv).

    Handles both leading ('- 22 text') and trailing ('- text 64') positions.
    Only activates when ≥5 matched numbers form a mostly-consecutive sequence
    (>50% diffs == 1), so ordinary numbered lists are left untouched.
    Also fixes headings that absorbed a trailing number ('## Conclusions: 27').
    """
    lines = text.split("\n")

    # Collect candidates for both positions
    leading: list[tuple[int, int, str]] = []   # (line_idx, number, text_rest)
    trailing: list[tuple[int, int, str]] = []  # (line_idx, number, text_rest)

    for i, line in enumerate(lines):
        s = line.strip()
        m = _NUM_LEADING.match(s)
        if m:
            leading.append((i, int(m.group(1)), m.group(2)))
            continue
        m = _NUM_TRAILING.match(s)
        if m:
            # group(1) = optional bullet prefix; group(2) = text; group(3) = number
            # Store only the text (group 2), dropping the bullet prefix
            trailing.append((i, int(m.group(3)), m.group(2).strip()))

    # Pick whichever position gives a more consecutive sequence
    best: list[tuple[int, int, str]] = []
    for candidates in (leading, trailing):
        if len(candidates) >= 3:
            nums = [n for _, n, _ in candidates]
            if _consecutiveness(nums) > 0.5 and _consecutiveness(nums) > _consecutiveness(
                [n for _, n, _ in best] if best else []
            ):
                best = candidates

    if not best:
        return text

    min_num = min(n for _, n, _ in best)
    max_num = max(n for _, n, _ in best)
    stripped = {i: rest for i, _, rest in best}

    result = []
    for i, line in enumerate(lines):
        if i in stripped:
            result.append(stripped[i])
        else:
            s = line.strip()
            # Drop blank lines that are only a bare line number
            if re.fullmatch(r"\d{1,4}", s) and min_num <= int(s) <= max_num:
                continue
            # Strip dangling "- " bullet left after number removal
            if s.startswith("- ") and not re.match(r"^- \d", s):
                result.append(s[2:])
            else:
                result.append(_HEADING_TRAIL_NUM_RE.sub(r"\1", line))

    return "\n".join(result)


# ---------------------------------------------------------------------------
# Main clean function
# ---------------------------------------------------------------------------

def _clean(text: str) -> str:
    text = _strip_page_headers(text)
    text = _strip_manuscript_line_numbers(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def pdf_to_markdown_docling(
    path: Path,
    page_range: list[int] | None = None,
    device: str = "cuda",
) -> list[tuple[int, str]] | None:
    """Return [(1-based page number, markdown), ...] for a PDF via granite-docling.

    *page_range* is a list of 0-based page indices (consistent with other readers).
    Returns None when docling is not installed or conversion fails.
    """
    try:
        from docling.document_converter import DocumentConverter  # noqa: F401
    except ImportError:
        _log.debug("docling not installed; skipping granite-docling extraction")
        return None

    try:
        converter = _get_converter(device)
        result = converter.convert(source=str(path))  # type: ignore[union-attr]
        doc = result.document
    except Exception as exc:
        if device == "cuda" and _is_oom(exc):
            _log.warning(
                "granite-docling CUDA OOM on %s (VRAM held by another process); "
                "retrying on CPU — this will be slower",
                path.name,
            )
            _converter_cache.pop("cuda", None)
            try:
                converter = _get_converter("cpu")
                result = converter.convert(source=str(path))  # type: ignore[union-attr]
                doc = result.document
            except Exception as cpu_exc:
                _log.warning("granite-docling CPU fallback also failed on %s: %s", path.name, cpu_exc)
                return None
        else:
            _log.warning("granite-docling failed on %s: %s", path.name, exc)
            return None

    total_pages = doc.num_pages()
    pages_to_process: list[int]
    if page_range is not None:
        pages_to_process = [p + 1 for p in page_range]  # convert 0-based to 1-based
    else:
        pages_to_process = list(range(1, total_pages + 1))

    results: list[tuple[int, str]] = []
    for page_no in pages_to_process:
        try:
            text = doc.export_to_markdown(page_no=page_no)
        except Exception as exc:
            _log.warning("docling page export failed for page %d: %s", page_no, exc)
            continue
        text = _clean(text)
        if text:
            results.append((page_no, text))

    return results or None
