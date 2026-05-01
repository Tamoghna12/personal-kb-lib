"""Smoke tests — verify the package skeleton imports and settings load cleanly."""

import importlib

import pytest


def test_package_importable():
    ocr_kb = importlib.import_module("ocr_kb")
    assert ocr_kb.__version__ == "0.1.0"


def test_sub_packages_importable():
    for sub in ("ocr_kb.ingest", "ocr_kb.model", "ocr_kb.postprocess", "ocr_kb.kb"):
        importlib.import_module(sub)


def test_settings_defaults():
    from ocr_kb.settings import Settings

    s = Settings(_env_file=None)
    assert s.model_backend == "hybrid"
    assert s.glm_ocr_base_url == "http://localhost:8000/v1"
    assert s.gemma_base_url == "http://localhost:1234/v1"
    assert s.image_dpi == 150
    assert s.enable_embeddings is True


def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("IMAGE_DPI", "300")
    monkeypatch.setenv("ENABLE_EMBEDDINGS", "true")

    from importlib import reload

    import ocr_kb.settings as mod
    reload(mod)
    s = mod.Settings(_env_file=None)
    assert s.image_dpi == 300
    assert s.enable_embeddings is True


def test_cli_importable():
    from ocr_kb.cli import app
    assert app is not None
