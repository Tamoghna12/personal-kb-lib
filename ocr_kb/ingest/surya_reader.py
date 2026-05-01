"""Direct Surya OCR for scanned PDF pages.

Loads only DetectionPredictor + RecognitionPredictor (2 models vs. marker's 5).
Pages are rendered via pypdfium2 then fed straight to Surya — no layout
analysis, table detection, or processors.  Much faster than the full marker
pipeline while still giving ML-quality OCR on scanned content.

Falls back gracefully when surya is not installed.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

_log = logging.getLogger(__name__)

_det_predictor = None
_rec_predictor = None


def _load_predictors(device: str = "cpu"):
    global _det_predictor, _rec_predictor
    if _det_predictor is None or _rec_predictor is None:
        _log.info("Loading Surya detection + recognition models on device=%s", device)
        from surya.detection import DetectionPredictor
        from surya.foundation import FoundationPredictor
        from surya.recognition import RecognitionPredictor
        from surya.settings import settings as surya_settings
        _det_predictor = DetectionPredictor(device=device)
        _rec_predictor = RecognitionPredictor(
            FoundationPredictor(
                checkpoint=surya_settings.RECOGNITION_MODEL_CHECKPOINT,
                device=device,
            )
        )
    return _det_predictor, _rec_predictor


def _clean(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def pdf_to_text_surya(
    path: Path,
    page_range: list[int] | None = None,
    device: str = "cpu",
    dpi: int = 150,
) -> list[tuple[int, str]] | None:
    """Return [(1-based page number, text), ...] using Surya OCR.

    *page_range* is a list of 0-based page indices.
    Returns None when surya is not installed or all pages fail.
    """
    try:
        from surya.recognition import RecognitionPredictor  # noqa: F401
    except ImportError:
        _log.debug("surya not installed; skipping surya OCR")
        return None

    try:
        import pypdfium2 as pdfium
        from PIL import Image as _Image

        doc = pdfium.PdfDocument(str(path))
        total = len(doc)
        indices = page_range if page_range is not None else list(range(total))

        scale = dpi / 72.0
        images: list[_Image.Image] = []
        page_nums: list[int] = []
        for idx in indices:
            bm = doc[idx].render(scale=scale)
            images.append(bm.to_pil().convert("RGB"))
            page_nums.append(idx + 1)  # 1-based

        if not images:
            return None

        det_pred, rec_pred = _load_predictors(device)
        results = rec_pred(images, det_predictor=det_pred)

        output: list[tuple[int, str]] = []
        for page_num, ocr_result in zip(page_nums, results):
            lines = [line.text for line in ocr_result.text_lines if line.text.strip()]
            text = _clean("\n".join(lines))
            if text:
                output.append((page_num, text))

        return output or None

    except Exception as exc:
        _log.warning("surya OCR failed on %s: %s", path.name, exc)
        return None
