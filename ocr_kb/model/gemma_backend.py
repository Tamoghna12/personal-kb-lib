from __future__ import annotations

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

from ocr_kb.settings import Settings

_RETRYABLE = (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError)
_DEFAULT_MAX_TOKENS = 4096
_DEFAULT_TIMEOUT = 60.0  # text-only enrichment may be slower than vision OCR


def _build_client(settings: Settings, timeout: float = _DEFAULT_TIMEOUT) -> OpenAI:
    return OpenAI(
        base_url=settings.text_base_url,
        api_key=settings.text_api_key,
        timeout=timeout,
    )


def run_enrichment(
    prompt: str,
    settings: Settings,
    *,
    text_model: str | None = None,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    max_tokens: int | None = None,
) -> str:
    """Send a text-only prompt to the configured text model and return the response.

    *text_model* overrides the model name from settings for this call only.
    *max_tokens* defaults to ``settings.model_max_new_tokens`` so the caller
    never needs to worry about VRAM-safe caps — they're enforced centrally.
    """
    client = _build_client(settings)
    model = text_model or settings.text_model_name
    tokens = max_tokens if max_tokens is not None else settings.model_max_new_tokens

    last_exc: Exception = RuntimeError("No attempts made")
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=tokens,
            )
            return resp.choices[0].message.content or ""
        except (AuthenticationError, BadRequestError):
            raise
        except _RETRYABLE as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (2**attempt))

    raise last_exc
