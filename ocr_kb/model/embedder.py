"""Embedding helper supporting Ollama (via OpenAI-compatible API) and sentence-transformers."""
from __future__ import annotations

import logging as _logging
from functools import lru_cache

from ocr_kb.settings import Settings

_emb_logger = _logging.getLogger(__name__)


def _is_ollama_model(model_name: str) -> bool:
    """Ollama model names use 'name:tag' format (e.g. qwen3-embedding:4b)."""
    return ":" in model_name


@lru_cache(maxsize=4)
def _get_st_model(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        _emb_logger.warning(
            "sentence-transformers is not installed; embeddings disabled. "
            "Install it with: pip install sentence-transformers"
        )
        return None
    return SentenceTransformer(model_name)


def _embed_ollama(text: str, model_name: str, base_url: str) -> list[float] | None:
    try:
        from openai import OpenAI
        client = OpenAI(base_url=base_url, api_key="ollama")
        response = client.embeddings.create(model=model_name, input=text.strip())
        return response.data[0].embedding
    except Exception as exc:
        _emb_logger.warning("Ollama embedding failed for model %s: %s", model_name, exc)
        return None


def embed_text(text: str, settings: Settings) -> list[float] | None:
    """Return an embedding vector for *text*, or None when disabled."""
    if not settings.enable_embeddings:
        return None
    model_name = settings.embedding_model
    if _is_ollama_model(model_name):
        return _embed_ollama(text, model_name, settings.ollama_base_url)
    model = _get_st_model(model_name)
    if model is None:
        return None
    vec = model.encode(text.strip(), normalize_embeddings=True)
    return vec.tolist()
