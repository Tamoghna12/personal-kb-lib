from __future__ import annotations

import base64
import io
import time

from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from PIL import Image

from ocr_kb.model.schema import OcrRequest, OcrResponse
from ocr_kb.settings import Settings

_RETRYABLE = (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError)
_DEFAULT_MAX_TOKENS = 4096
_DEFAULT_TIMEOUT = 120.0  # seconds; complex scientific figures can take 60-120 s


def _encode_image(image: Image.Image, fmt: str = "JPEG") -> str:
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def _resize_for_vram(image: Image.Image, max_pixels: int) -> Image.Image:
    """Downscale image so width*height ≤ max_pixels, preserving aspect ratio.

    Keeps the image unmodified if it already fits. This prevents OOM on
    the RTX 4070 12 GB when the vision model tokenises very large images.
    """
    w, h = image.size
    if w * h <= max_pixels:
        return image
    ratio = (max_pixels / (w * h)) ** 0.5
    new_w, new_h = max(1, int(w * ratio)), max(1, int(h * ratio))
    return image.resize((new_w, new_h), Image.LANCZOS)


def _build_client(settings: Settings, timeout: float = _DEFAULT_TIMEOUT) -> OpenAI:
    return OpenAI(
        base_url=settings.vision_base_url,
        api_key=settings.vision_api_key,
        timeout=timeout,
    )


def run_ocr(
    request: OcrRequest,
    settings: Settings,
    *,
    vision_model: str | None = None,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
) -> OcrResponse:
    """Send one image + prompt to the configured vision model and return the OCR text.

    *vision_model* overrides the model name from settings for this call only.
    Retries on transient failures; does NOT retry on auth or bad-request errors.
    """
    client = _build_client(settings, timeout=settings.vision_timeout)
    model = vision_model or settings.vision_model_name
    img = _resize_for_vram(request.image, settings.max_image_pixels)
    data_uri = f"data:image/jpeg;base64,{_encode_image(img)}"

    last_exc: Exception = RuntimeError("No attempts made")
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": data_uri}},
                            {"type": "text", "text": request.prompt},
                        ],
                    }
                ],
                max_tokens=max_tokens,
            )
            text = resp.choices[0].message.content or ""
            usage = resp.usage
            return OcrResponse(
                text=text,
                model=resp.model,
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
            )
        except (AuthenticationError, BadRequestError):
            raise
        except _RETRYABLE as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (2**attempt))

    raise last_exc
