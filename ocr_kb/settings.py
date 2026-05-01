from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file="local.env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Backend provider ────────────────────────────────────────────────────
    # "lmstudio" uses the GLM-OCR + Gemma URL/key/model fields below.
    # "ollama"   uses the ollama_* fields below and ignores the LM Studio ones.
    backend_provider: Literal["lmstudio", "ollama"] = "lmstudio"

    # Pipeline mode (which steps run)
    model_backend: Literal["hybrid", "glm_only", "gemma_only"] = "hybrid"

    # ── LM Studio ───────────────────────────────────────────────────────────
    # Vision / OCR model
    glm_ocr_base_url: str = "http://localhost:8000/v1"
    glm_ocr_api_key: str = "glm-ocr"
    glm_ocr_model_name: str = "glm-ocr"

    # Text enrichment + RAG model
    gemma_base_url: str = "http://localhost:1234/v1"
    gemma_api_key: str = "lm-studio"
    gemma_model_name: str = "gemma-4-e4b"

    # ── Ollama ──────────────────────────────────────────────────────────────
    # Single base URL for both vision and text when backend_provider="ollama".
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_vision_model: str = "llava:7b"   # any vision-capable model in Ollama
    ollama_text_model: str = "gemma3:4b"    # any text model in Ollama

    # ── Routing properties (pick the right endpoint based on provider) ──────
    @property
    def vision_base_url(self) -> str:
        return self.ollama_base_url if self.backend_provider == "ollama" else self.glm_ocr_base_url

    @property
    def vision_api_key(self) -> str:
        return "ollama" if self.backend_provider == "ollama" else self.glm_ocr_api_key

    @property
    def vision_model_name(self) -> str:
        return self.ollama_vision_model if self.backend_provider == "ollama" else self.glm_ocr_model_name

    @property
    def text_base_url(self) -> str:
        return self.ollama_base_url if self.backend_provider == "ollama" else self.gemma_base_url

    @property
    def text_api_key(self) -> str:
        return "ollama" if self.backend_provider == "ollama" else self.gemma_api_key

    @property
    def text_model_name(self) -> str:
        return self.ollama_text_model if self.backend_provider == "ollama" else self.gemma_model_name

    # ── Enrichment toggle ───────────────────────────────────────────────────
    enable_gemma_enrichment: bool = False

    # ── Knowledge base paths ────────────────────────────────────────────────
    kb_dir: Path = Path("kb_data")
    kb_db_path: Path = Path("kb_data/kb.db")
    markdown_dir: Path = Path("kb_data/markdown")

    # ── Image ingestion ─────────────────────────────────────────────────────
    image_dpi: int = 150
    min_image_dim: int = 800
    min_pdf_image_dim: int = 1000
    # Extract embedded images (figures, diagrams) from PDF pages and describe them.
    extract_embedded_images: bool = True
    # Minimum pixel area for an embedded image to be sent to the vision model.
    # Filters out tiny icons and decorative elements (default ≈ 100×100).
    embedded_image_min_pixels: int = 10_000

    # ── PDF ingestion: native text extraction ────────────────────────────────
    # When True, extract the text layer from each PDF page first; only fall
    # back to vision-model OCR when the extracted text looks like a scan
    # (empty, very short, or mostly non-alphabetic characters).
    prefer_pdf_text: bool = True
    # Use pdftext for zero-ML native text extraction (instant on digital PDFs).
    # Reads the PDF text layer directly — no models loaded.  Scanned pages
    # return short/empty strings and are routed to OCR automatically.
    use_pdftext: bool = True
    # Use Surya OCR directly: only loads DetectionPredictor + RecognitionPredictor
    # (2 models) instead of the full marker pipeline (5 models + processors).
    # Best for scanned PDFs when you want ML OCR without the full marker overhead.
    use_surya_ocr: bool = False
    # Use marker-pdf for high-quality layout-aware markdown extraction.
    # Runs Surya neural models (layout + OCR) — first call loads models (~10-30 s).
    # When True, marker-pdf is tried before pymupdf4llm and pypdfium2.
    use_marker_pdf: bool = False
    # Device for marker-pdf / Surya models.  Defaults to "cpu" so they don't
    # compete with Ollama/LM Studio for VRAM.  Set to "cuda" only when those
    # backends are not running or you have spare VRAM.
    marker_device: str = "cpu"
    # Use pymupdf4llm for structured markdown extraction on native-text PDFs.
    # Produces proper heading hierarchy, bold/italic, and table markdown.
    # Falls back to raw pypdfium2 text when pymupdf4llm is not installed.
    use_pymupdf4llm: bool = True

    # ── Sub-page chunking ────────────────────────────────────────────────────
    # Split each page into overlapping chunks for more precise retrieval.
    # 0 = disabled (store one entry per page, as before).
    chunk_size: int = 0        # target chars per chunk
    chunk_overlap: int = 100   # overlap chars between consecutive chunks

    # ── Embeddings ──────────────────────────────────────────────────────────
    enable_embeddings: bool = True
    embedding_model: str = "all-MiniLM-L6-v2"
    max_retries: int = 3

    # ── File watcher ────────────────────────────────────────────────────────
    watch_dir: Path = Path("data/inputs")
    watch_interval: int = 5

    # ── VRAM-safe inference limits (defaults tuned for RTX 4070 12 GB) ─────
    # A 9B vision model at 4-bit ≈ 5-6 GB; a 4B text model at 4-bit ≈ 2-3 GB.
    # Together they leave ~3 GB for KV cache and activations.
    max_image_pixels: int = 1_310_720   # 1280×1024 — hard cap before base64-encoding
    model_max_new_tokens: int = 2048    # caps generation length on both backends
    rag_chunk_chars: int = 600          # chars per document snippet in RAG context

    @field_validator("image_dpi")
    @classmethod
    def dpi_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("image_dpi must be positive")
        return v

    @field_validator("min_image_dim", "min_pdf_image_dim")
    @classmethod
    def dim_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("minimum dimension must be positive")
        return v

    @field_validator("watch_interval")
    @classmethod
    def interval_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("watch_interval must be positive")
        return v

    @field_validator("max_image_pixels", "model_max_new_tokens", "rag_chunk_chars")
    @classmethod
    def vram_limits_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("must be positive")
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
