"""Sentence-transformer embedding helper with lazy model loading."""
from __future__ import annotations

import logging as _logging
from functools import lru_cache

from ocr_kb.settings import Settings

_emb_logger = _logging.getLogger(__name__)


@lru_cache(maxsize=4)
def _get_model(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        _emb_logger.warning(
            "sentence-transformers is not installed; embeddings disabled. "
            "Install it with: pip install sentence-transformers"
        )
        return None
    return SentenceTransformer(model_name)


def embed_text(text: str, settings: Settings) -> list[float] | None:
    """Return a normalized embedding vector for *text*, or None when disabled.

    The model is loaded once and cached for the process lifetime.
    """
    if not settings.enable_embeddings:
        return None
    model = _get_model(settings.embedding_model)
    if model is None:
        return None
    vec = model.encode(text.strip(), normalize_embeddings=True)
    return vec.tolist()
