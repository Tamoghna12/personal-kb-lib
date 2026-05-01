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


def _clean(text: str) -> str:
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
