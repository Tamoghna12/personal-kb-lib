"""Embedding helper: Ollama API, sentence-transformers, and SPECTER2."""
from __future__ import annotations

import logging as _logging
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ocr_kb.settings import Settings

_emb_logger = _logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Routing helpers
# ---------------------------------------------------------------------------

def _is_ollama_model(name: str) -> bool:
    return ":" in name

def _is_specter2(name: str) -> bool:
    return name.lower() == "specter2"

# ---------------------------------------------------------------------------
# sentence-transformers (non-SPECTER2)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4)
def _get_st_model(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        _emb_logger.warning(
            "sentence-transformers not installed; embeddings disabled. "
            "pip install sentence-transformers"
        )
        return None
    return SentenceTransformer(model_name)

# ---------------------------------------------------------------------------
# SPECTER2 (allenai/specter2_base + adapters)
# ---------------------------------------------------------------------------

_specter2_cache: dict = {}

def _get_specter2():
    if "model" not in _specter2_cache:
        try:
            from adapters import AutoAdapterModel
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError(
                "SPECTER2 requires the 'adapters' library. "
                "pip install 'ocr-kb[specter2]'  or  pip install adapters"
            )
        _emb_logger.info("Loading SPECTER2 base model and adapters (one-time load)...")
        tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
        model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
        model.load_adapter("allenai/specter2", source="hf",
                           load_as="proximity", set_active=True)
        model.load_adapter("allenai/specter2_adhoc_query", source="hf",
                           load_as="adhoc_query")
        model.eval()
        _specter2_cache["tokenizer"] = tokenizer
        _specter2_cache["model"] = model
    return _specter2_cache["tokenizer"], _specter2_cache["model"]


def _encode_specter2(text: str, adapter: str) -> list[float] | None:
    import torch
    try:
        tokenizer, model = _get_specter2()
    except ImportError as exc:
        _emb_logger.warning("%s -- falling back to all-MiniLM-L6-v2", exc)
        return None
    model.set_active_adapters(adapter)
    inputs = tokenizer(
        text, return_tensors="pt", max_length=512,
        truncation=True, padding=True,
    )
    with torch.no_grad():
        outputs = model(**inputs)
    emb = outputs.last_hidden_state[:, 0, :]
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb[0].tolist()


def _format_specter2_doc(text: str, title: str = "", abstract: str = "") -> str:
    """SPECTER2 document input: title + SEP + abstract + SEP + body excerpt."""
    sep = "[SEP]"
    parts = [p for p in [title, abstract, text[:400]] if p]
    return f" {sep} ".join(parts)


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------

def _embed_ollama(text: str, model_name: str, base_url: str) -> list[float] | None:
    try:
        from openai import OpenAI
        client = OpenAI(base_url=base_url, api_key="ollama")
        response = client.embeddings.create(model=model_name, input=text.strip())
        return response.data[0].embedding
    except Exception as exc:
        _emb_logger.warning("Ollama embedding failed model=%s: %s", model_name, exc)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def embed_text(
    text: str,
    settings: "Settings",
    *,
    title: str = "",
    abstract: str = "",
) -> list[float] | None:
    """Embed a document chunk. For SPECTER2, title + abstract improve quality."""
    if not settings.enable_embeddings:
        return None
    model_name = settings.embedding_model
    if _is_specter2(model_name):
        doc_input = _format_specter2_doc(text, title=title, abstract=abstract)
        result = _encode_specter2(doc_input, adapter="proximity")
        if result is not None:
            return result
        # adapters not installed -- fall back to sentence-transformers
        model_name = "all-MiniLM-L6-v2"
    if _is_ollama_model(model_name):
        return _embed_ollama(text, model_name, settings.ollama_base_url)
    model = _get_st_model(model_name)
    if model is None:
        return None
    vec = model.encode(text.strip(), normalize_embeddings=True)
    return vec.tolist()


def embed_query(query: str, settings: "Settings") -> list[float] | None:
    """Embed a search query. SPECTER2 uses the adhoc_query adapter."""
    if not settings.enable_embeddings:
        return None
    model_name = settings.embedding_model
    if _is_specter2(model_name):
        result = _encode_specter2(query, adapter="adhoc_query")
        if result is not None:
            return result
        model_name = "all-MiniLM-L6-v2"
    if _is_ollama_model(model_name):
        return _embed_ollama(query, model_name, settings.ollama_base_url)
    model = _get_st_model(model_name)
    if model is None:
        return None
    vec = model.encode(query.strip(), normalize_embeddings=True)
    return vec.tolist()
