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


_LIST_NUM_RE = re.compile(r"^[-*•]?\s*(\d{1,4})\s+(\S.*)", re.MULTILINE)
# Strips a trailing line-number from headings: "## Conclusions: 27" or "## Introduction 35"
# Only matches 2+ digit numbers to avoid stripping "## Figure 3" or "## Step 1".
_HEADING_TRAIL_NUM_RE = re.compile(r"^(#{1,6}\s+.*?\S):?\s+(\d{2,})\s*$", re.MULTILINE)


def _strip_manuscript_line_numbers(text: str) -> str:
    """Remove sequential line numbers added by preprint servers (medRxiv, bioRxiv).

    These appear as '- N text' list items where N is a consecutive integer.
    Only strips when >50% of matched numbers differ by 1 (true line-number sequence).
    Also fixes headings that absorbed a trailing number ('## Conclusions: 27').
    """
    lines = text.split("\n")
    matches: list[tuple[int, int, str]] = []  # (line_idx, number, rest)
    for i, line in enumerate(lines):
        m = _LIST_NUM_RE.match(line.strip())
        if m:
            matches.append((i, int(m.group(1)), m.group(2)))

    if len(matches) < 5:
        return text

    nums = [n for _, n, _ in matches]
    diffs = [nums[j + 1] - nums[j] for j in range(len(nums) - 1)]
    if not diffs or sum(d == 1 for d in diffs) / len(diffs) < 0.5:
        return text

    min_num = min(nums)
    max_num = max(nums)

    stripped = {i: rest for i, _, rest in matches}
    result = []
    for i, line in enumerate(lines):
        if i in stripped:
            result.append(stripped[i])
        else:
            stripped_line = line.strip()
            # Drop lines that are only a bare line number (blank lines in the manuscript)
            if re.fullmatch(r"\d{1,4}", stripped_line) and min_num <= int(stripped_line) <= max_num:
                continue
            # Strip dangling "- " bullet prefix left after number removal
            if stripped_line.startswith("- ") and not re.match(r"^- \d", stripped_line):
                result.append(stripped_line[2:])
            else:
                # Fix "## Heading: 27" → "## Heading"
                result.append(_HEADING_TRAIL_NUM_RE.sub(r"\1", line))
    return "\n".join(result)


def _clean(text: str) -> str:
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
            # Another process (e.g. Ollama) is holding VRAM — evict the broken
            # CUDA converter from cache and retry on CPU.
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
