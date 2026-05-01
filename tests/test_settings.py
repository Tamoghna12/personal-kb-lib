"""Tests for ocr_kb.settings — defaults, env overrides, and validation."""

from pathlib import Path

import pytest
from pydantic import ValidationError


def _fresh(monkeypatch, **env_vars) -> "Settings":
    import ocr_kb.settings as mod
    from importlib import reload
    for k, v in env_vars.items():
        monkeypatch.setenv(k, v)
    reload(mod)
    return mod.Settings(_env_file=None)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

class TestDefaults:
    def test_model_backend(self):
        from ocr_kb.settings import Settings
        assert Settings(_env_file=None).model_backend == "hybrid"

    def test_glm_ocr_base_url(self):
        from ocr_kb.settings import Settings
        assert Settings(_env_file=None).glm_ocr_base_url == "http://localhost:8000/v1"

    def test_glm_ocr_api_key(self):
        from ocr_kb.settings import Settings
        assert Settings(_env_file=None).glm_ocr_api_key == "glm-ocr"

    def test_glm_ocr_model_name(self):
        from ocr_kb.settings import Settings
        assert Settings(_env_file=None).glm_ocr_model_name == "glm-ocr"

    def test_gemma_base_url(self):
        from ocr_kb.settings import Settings
        assert Settings(_env_file=None).gemma_base_url == "http://localhost:1234/v1"

    def test_gemma_api_key(self):
        from ocr_kb.settings import Settings
        assert Settings(_env_file=None).gemma_api_key == "lm-studio"

    def test_gemma_model_name(self):
        from ocr_kb.settings import Settings
        assert Settings(_env_file=None).gemma_model_name == "gemma-4-e4b"

    def test_enable_gemma_enrichment_false(self):
        from ocr_kb.settings import Settings
        assert Settings(_env_file=None).enable_gemma_enrichment is False

    def test_kb_dir(self):
        from ocr_kb.settings import Settings
        assert Settings(_env_file=None).kb_dir == Path("kb_data")

    def test_kb_db_path(self):
        from ocr_kb.settings import Settings
        assert Settings(_env_file=None).kb_db_path == Path("kb_data/kb.db")

    def test_markdown_dir(self):
        from ocr_kb.settings import Settings
        assert Settings(_env_file=None).markdown_dir == Path("kb_data/markdown")

    def test_markdown_dir_is_path(self):
        from ocr_kb.settings import Settings
        assert isinstance(Settings(_env_file=None).markdown_dir, Path)

    def test_image_dpi(self):
        from ocr_kb.settings import Settings
        assert Settings(_env_file=None).image_dpi == 150

    def test_min_image_dim(self):
        from ocr_kb.settings import Settings
        assert Settings(_env_file=None).min_image_dim == 800

    def test_min_pdf_image_dim(self):
        from ocr_kb.settings import Settings
        assert Settings(_env_file=None).min_pdf_image_dim == 1000

    def test_enable_embeddings(self):
        from ocr_kb.settings import Settings
        assert Settings(_env_file=None).enable_embeddings is True

    def test_embedding_model(self):
        from ocr_kb.settings import Settings
        assert Settings(_env_file=None).embedding_model == "all-MiniLM-L6-v2"

    def test_max_retries_default(self):
        from ocr_kb.settings import Settings
        assert Settings(_env_file=None).max_retries == 3

    def test_watch_dir(self):
        from ocr_kb.settings import Settings
        assert Settings(_env_file=None).watch_dir == Path("data/inputs")

    def test_watch_interval(self):
        from ocr_kb.settings import Settings
        assert Settings(_env_file=None).watch_interval == 5


# ---------------------------------------------------------------------------
# Environment overrides
# ---------------------------------------------------------------------------

class TestEnvOverrides:
    def test_model_backend_glm_only(self, monkeypatch):
        assert _fresh(monkeypatch, MODEL_BACKEND="glm_only").model_backend == "glm_only"

    def test_model_backend_gemma_only(self, monkeypatch):
        assert _fresh(monkeypatch, MODEL_BACKEND="gemma_only").model_backend == "gemma_only"

    def test_glm_ocr_base_url(self, monkeypatch):
        s = _fresh(monkeypatch, GLM_OCR_BASE_URL="http://gpu-box:8000/v1")
        assert s.glm_ocr_base_url == "http://gpu-box:8000/v1"

    def test_glm_ocr_model_name(self, monkeypatch):
        assert _fresh(monkeypatch, GLM_OCR_MODEL_NAME="glm-ocr-9b").glm_ocr_model_name == "glm-ocr-9b"

    def test_gemma_base_url(self, monkeypatch):
        s = _fresh(monkeypatch, GEMMA_BASE_URL="http://localhost:5678/v1")
        assert s.gemma_base_url == "http://localhost:5678/v1"

    def test_gemma_model_name(self, monkeypatch):
        assert _fresh(monkeypatch, GEMMA_MODEL_NAME="gemma-2-9b").gemma_model_name == "gemma-2-9b"

    def test_enable_gemma_enrichment_true(self, monkeypatch):
        assert _fresh(monkeypatch, ENABLE_GEMMA_ENRICHMENT="true").enable_gemma_enrichment is True

    def test_image_dpi(self, monkeypatch):
        assert _fresh(monkeypatch, IMAGE_DPI="300").image_dpi == 300

    def test_kb_dir(self, monkeypatch):
        assert _fresh(monkeypatch, KB_DIR="/tmp/kb").kb_dir == Path("/tmp/kb")

    def test_watch_interval(self, monkeypatch):
        assert _fresh(monkeypatch, WATCH_INTERVAL="30").watch_interval == 30


# ---------------------------------------------------------------------------
# Type checks
# ---------------------------------------------------------------------------

class TestTypes:
    def test_kb_dir_is_path(self):
        from ocr_kb.settings import Settings
        assert isinstance(Settings(_env_file=None).kb_dir, Path)

    def test_kb_db_path_is_path(self):
        from ocr_kb.settings import Settings
        assert isinstance(Settings(_env_file=None).kb_db_path, Path)

    def test_watch_dir_is_path(self):
        from ocr_kb.settings import Settings
        assert isinstance(Settings(_env_file=None).watch_dir, Path)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

class TestValidation:
    def test_invalid_model_backend(self):
        from ocr_kb.settings import Settings
        with pytest.raises(ValidationError):
            Settings(_env_file=None, model_backend="ollama")

    def test_zero_image_dpi(self):
        from ocr_kb.settings import Settings
        with pytest.raises(ValidationError):
            Settings(_env_file=None, image_dpi=0)

    def test_negative_image_dpi(self):
        from ocr_kb.settings import Settings
        with pytest.raises(ValidationError):
            Settings(_env_file=None, image_dpi=-1)

    def test_zero_min_image_dim(self):
        from ocr_kb.settings import Settings
        with pytest.raises(ValidationError):
            Settings(_env_file=None, min_image_dim=0)

    def test_zero_min_pdf_image_dim(self):
        from ocr_kb.settings import Settings
        with pytest.raises(ValidationError):
            Settings(_env_file=None, min_pdf_image_dim=0)

    def test_zero_watch_interval(self):
        from ocr_kb.settings import Settings
        with pytest.raises(ValidationError):
            Settings(_env_file=None, watch_interval=0)


# ---------------------------------------------------------------------------
# Cached factory
# ---------------------------------------------------------------------------

def test_get_settings_returns_settings_instance():
    from ocr_kb.settings import Settings, get_settings
    get_settings.cache_clear()
    assert isinstance(get_settings(), Settings)


def test_get_settings_is_cached():
    from ocr_kb.settings import get_settings
    get_settings.cache_clear()
    assert get_settings() is get_settings()


# ---------------------------------------------------------------------------
# VRAM-safe inference settings
# ---------------------------------------------------------------------------

class TestVramSettings:
    def test_max_image_pixels_default(self):
        from ocr_kb.settings import Settings
        assert Settings(_env_file=None).max_image_pixels == 1_310_720

    def test_model_max_new_tokens_default(self):
        from ocr_kb.settings import Settings
        assert Settings(_env_file=None).model_max_new_tokens == 2048

    def test_rag_chunk_chars_default(self):
        from ocr_kb.settings import Settings
        assert Settings(_env_file=None).rag_chunk_chars == 600

    def test_max_image_pixels_env_override(self, monkeypatch):
        assert _fresh(monkeypatch, MAX_IMAGE_PIXELS="2000000").max_image_pixels == 2_000_000

    def test_model_max_new_tokens_env_override(self, monkeypatch):
        assert _fresh(monkeypatch, MODEL_MAX_NEW_TOKENS="4096").model_max_new_tokens == 4096

    def test_rag_chunk_chars_env_override(self, monkeypatch):
        assert _fresh(monkeypatch, RAG_CHUNK_CHARS="400").rag_chunk_chars == 400

    def test_zero_max_image_pixels_invalid(self):
        from ocr_kb.settings import Settings
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            Settings(_env_file=None, max_image_pixels=0)

    def test_zero_model_max_new_tokens_invalid(self):
        from ocr_kb.settings import Settings
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            Settings(_env_file=None, model_max_new_tokens=0)

    def test_zero_rag_chunk_chars_invalid(self):
        from ocr_kb.settings import Settings
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            Settings(_env_file=None, rag_chunk_chars=0)
