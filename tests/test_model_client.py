"""Tests for ocr_kb.model — schema, image encoding, GLM-OCR client, Gemma client."""

from __future__ import annotations

import base64
import io
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from ocr_kb.settings import Settings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _glm_settings(**kwargs) -> Settings:
    defaults = dict(
        glm_ocr_base_url="http://localhost:8000/v1",
        glm_ocr_api_key="test-key",
        glm_ocr_model_name="test-glm-model",
    )
    defaults.update(kwargs)
    return Settings(_env_file=None, **defaults)


def _gemma_settings(**kwargs) -> Settings:
    defaults = dict(
        gemma_base_url="http://localhost:1234/v1",
        gemma_api_key="test-key",
        gemma_model_name="test-gemma-model",
    )
    defaults.update(kwargs)
    return Settings(_env_file=None, **defaults)


def _small_image(w: int = 64, h: int = 64) -> Image.Image:
    return Image.new("RGB", (w, h), color=(10, 20, 30))


def _mock_response(text: str = "hello", model: str = "test-model") -> MagicMock:
    usage = MagicMock()
    usage.prompt_tokens = 10
    usage.completion_tokens = 5
    choice = MagicMock()
    choice.message.content = text
    resp = MagicMock()
    resp.choices = [choice]
    resp.model = model
    resp.usage = usage
    return resp


# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------

class TestSchema:
    def test_ocr_request_fields(self):
        from ocr_kb.model.schema import OcrRequest
        img = _small_image()
        req = OcrRequest(image=img, prompt="read this")
        assert req.image is img
        assert req.prompt == "read this"

    def test_ocr_response_fields(self):
        from ocr_kb.model.schema import OcrResponse
        r = OcrResponse(text="abc", model="m", prompt_tokens=1, completion_tokens=2)
        assert r.text == "abc"
        assert r.model == "m"
        assert r.prompt_tokens == 1
        assert r.completion_tokens == 2


# ---------------------------------------------------------------------------
# _encode_image
# ---------------------------------------------------------------------------

class TestEncodeImage:
    def test_output_is_valid_base64(self):
        from ocr_kb.model.glm_ocr_backend import _encode_image
        b64 = _encode_image(_small_image())
        decoded = base64.b64decode(b64)
        assert len(decoded) > 0

    def test_output_is_valid_jpeg(self):
        from ocr_kb.model.glm_ocr_backend import _encode_image
        b64 = _encode_image(_small_image())
        buf = io.BytesIO(base64.b64decode(b64))
        img = Image.open(buf)
        assert img.format == "JPEG"

    def test_round_trips_image_size(self):
        from ocr_kb.model.glm_ocr_backend import _encode_image
        original = _small_image(128, 96)
        b64 = _encode_image(original)
        buf = io.BytesIO(base64.b64decode(b64))
        recovered = Image.open(buf)
        assert recovered.size == (128, 96)


# ---------------------------------------------------------------------------
# run_ocr — happy path
# ---------------------------------------------------------------------------

class TestRunOcrSuccess:
    def _call(self, text="OCR output", **settings_kwargs):
        from ocr_kb.model.glm_ocr_backend import run_ocr
        from ocr_kb.model.schema import OcrRequest

        s = _glm_settings(**settings_kwargs)
        req = OcrRequest(image=_small_image(), prompt="transcribe")

        with patch("ocr_kb.model.glm_ocr_backend.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            MockOpenAI.return_value = mock_client
            mock_client.chat.completions.create.return_value = _mock_response(text)
            result = run_ocr(req, s, retry_delay=0)

        return result, mock_client

    def test_returns_ocr_response(self):
        from ocr_kb.model.schema import OcrResponse
        result, _ = self._call()
        assert isinstance(result, OcrResponse)

    def test_text_matches_mock(self):
        result, _ = self._call(text="extracted text")
        assert result.text == "extracted text"

    def test_model_name_passed_to_api(self):
        _, mock_client = self._call(glm_ocr_model_name="glm-ocr-9b")
        create_call = mock_client.chat.completions.create.call_args
        assert create_call.kwargs["model"] == "glm-ocr-9b"

    def test_prompt_in_message_content(self):
        from ocr_kb.model.schema import OcrRequest
        from ocr_kb.model.glm_ocr_backend import run_ocr

        s = _glm_settings()
        req = OcrRequest(image=_small_image(), prompt="my specific prompt")

        with patch("ocr_kb.model.glm_ocr_backend.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            MockOpenAI.return_value = mock_client
            mock_client.chat.completions.create.return_value = _mock_response()
            run_ocr(req, s, retry_delay=0)

        content = mock_client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
        text_parts = [p for p in content if p["type"] == "text"]
        assert text_parts[0]["text"] == "my specific prompt"

    def test_image_sent_as_data_uri(self):
        from ocr_kb.model.schema import OcrRequest
        from ocr_kb.model.glm_ocr_backend import run_ocr

        s = _glm_settings()
        req = OcrRequest(image=_small_image(), prompt="p")

        with patch("ocr_kb.model.glm_ocr_backend.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            MockOpenAI.return_value = mock_client
            mock_client.chat.completions.create.return_value = _mock_response()
            run_ocr(req, s, retry_delay=0)

        content = mock_client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
        image_parts = [p for p in content if p["type"] == "image_url"]
        assert image_parts[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")

    def test_token_usage_captured(self):
        result, _ = self._call()
        assert result.prompt_tokens == 10
        assert result.completion_tokens == 5

    def test_none_usage_defaults_to_zero(self):
        from ocr_kb.model.schema import OcrRequest
        from ocr_kb.model.glm_ocr_backend import run_ocr

        s = _glm_settings()
        req = OcrRequest(image=_small_image(), prompt="p")
        mock_resp = _mock_response()
        mock_resp.usage = None

        with patch("ocr_kb.model.glm_ocr_backend.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            MockOpenAI.return_value = mock_client
            mock_client.chat.completions.create.return_value = mock_resp
            result = run_ocr(req, s, retry_delay=0)

        assert result.prompt_tokens == 0
        assert result.completion_tokens == 0

    def test_client_constructed_with_glm_settings(self):
        from ocr_kb.model.schema import OcrRequest
        from ocr_kb.model.glm_ocr_backend import run_ocr

        s = _glm_settings(glm_ocr_base_url="http://gpu-box:8000/v1", glm_ocr_api_key="my-key")
        req = OcrRequest(image=_small_image(), prompt="p")

        with patch("ocr_kb.model.glm_ocr_backend.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            MockOpenAI.return_value = mock_client
            mock_client.chat.completions.create.return_value = _mock_response()
            run_ocr(req, s, retry_delay=0)

        call_kwargs = MockOpenAI.call_args.kwargs
        assert call_kwargs["base_url"] == "http://gpu-box:8000/v1"
        assert call_kwargs["api_key"] == "my-key"
        assert "timeout" in call_kwargs

    def test_max_tokens_default_passed(self):
        from ocr_kb.model.schema import OcrRequest
        from ocr_kb.model.glm_ocr_backend import run_ocr, _DEFAULT_MAX_TOKENS

        s = _glm_settings()
        req = OcrRequest(image=_small_image(), prompt="p")

        with patch("ocr_kb.model.glm_ocr_backend.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            MockOpenAI.return_value = mock_client
            mock_client.chat.completions.create.return_value = _mock_response()
            run_ocr(req, s, retry_delay=0)

        assert mock_client.chat.completions.create.call_args.kwargs["max_tokens"] == _DEFAULT_MAX_TOKENS

    def test_max_tokens_override(self):
        from ocr_kb.model.schema import OcrRequest
        from ocr_kb.model.glm_ocr_backend import run_ocr

        s = _glm_settings()
        req = OcrRequest(image=_small_image(), prompt="p")

        with patch("ocr_kb.model.glm_ocr_backend.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            MockOpenAI.return_value = mock_client
            mock_client.chat.completions.create.return_value = _mock_response()
            run_ocr(req, s, retry_delay=0, max_tokens=512)

        assert mock_client.chat.completions.create.call_args.kwargs["max_tokens"] == 512


# ---------------------------------------------------------------------------
# run_ocr — retry logic
# ---------------------------------------------------------------------------

class TestRunOcrRetry:
    def _patched_run(self, side_effects, max_retries=3):
        from ocr_kb.model.schema import OcrRequest
        from ocr_kb.model.glm_ocr_backend import run_ocr

        s = _glm_settings()
        req = OcrRequest(image=_small_image(), prompt="p")

        with patch("ocr_kb.model.glm_ocr_backend.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            MockOpenAI.return_value = mock_client
            mock_client.chat.completions.create.side_effect = side_effects
            return run_ocr(req, s, max_retries=max_retries, retry_delay=0)

    def test_retries_on_connection_error_then_succeeds(self):
        from openai import APIConnectionError
        result = self._patched_run([
            APIConnectionError(request=MagicMock()),
            _mock_response("recovered"),
        ])
        assert result.text == "recovered"

    def test_retries_on_rate_limit_then_succeeds(self):
        from openai import RateLimitError
        result = self._patched_run([
            RateLimitError("rate limited", response=MagicMock(), body={}),
            _mock_response("ok"),
        ])
        assert result.text == "ok"

    def test_retries_on_internal_server_error(self):
        from openai import InternalServerError
        result = self._patched_run([
            InternalServerError("500", response=MagicMock(), body={}),
            _mock_response("ok"),
        ])
        assert result.text == "ok"

    def test_raises_after_max_retries_exhausted(self):
        from openai import APIConnectionError
        exc = APIConnectionError(request=MagicMock())
        with pytest.raises(APIConnectionError):
            self._patched_run([exc, exc, exc], max_retries=3)

    def test_attempt_count_equals_max_retries(self):
        from openai import APIConnectionError
        from ocr_kb.model.schema import OcrRequest
        from ocr_kb.model.glm_ocr_backend import run_ocr

        s = _glm_settings()
        req = OcrRequest(image=_small_image(), prompt="p")
        exc = APIConnectionError(request=MagicMock())

        with patch("ocr_kb.model.glm_ocr_backend.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            MockOpenAI.return_value = mock_client
            mock_client.chat.completions.create.side_effect = exc
            with pytest.raises(APIConnectionError):
                run_ocr(req, s, max_retries=2, retry_delay=0)
            assert mock_client.chat.completions.create.call_count == 2

    def test_no_retry_on_authentication_error(self):
        from openai import AuthenticationError
        from ocr_kb.model.schema import OcrRequest
        from ocr_kb.model.glm_ocr_backend import run_ocr

        s = _glm_settings()
        req = OcrRequest(image=_small_image(), prompt="p")
        exc = AuthenticationError("bad key", response=MagicMock(), body={})

        with patch("ocr_kb.model.glm_ocr_backend.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            MockOpenAI.return_value = mock_client
            mock_client.chat.completions.create.side_effect = exc
            with pytest.raises(AuthenticationError):
                run_ocr(req, s, max_retries=3, retry_delay=0)
            assert mock_client.chat.completions.create.call_count == 1

    def test_no_retry_on_bad_request(self):
        from openai import BadRequestError
        from ocr_kb.model.schema import OcrRequest
        from ocr_kb.model.glm_ocr_backend import run_ocr

        s = _glm_settings()
        req = OcrRequest(image=_small_image(), prompt="p")
        exc = BadRequestError("bad", response=MagicMock(), body={})

        with patch("ocr_kb.model.glm_ocr_backend.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            MockOpenAI.return_value = mock_client
            mock_client.chat.completions.create.side_effect = exc
            with pytest.raises(BadRequestError):
                run_ocr(req, s, max_retries=3, retry_delay=0)
            assert mock_client.chat.completions.create.call_count == 1


# ---------------------------------------------------------------------------
# run_enrichment (Gemma backend) — happy path
# ---------------------------------------------------------------------------

class TestRunEnrichmentSuccess:
    def _call(self, text="enriched output", **settings_kwargs):
        from ocr_kb.model.gemma_backend import run_enrichment

        s = _gemma_settings(**settings_kwargs)

        with patch("ocr_kb.model.gemma_backend.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            MockOpenAI.return_value = mock_client
            mock_client.chat.completions.create.return_value = _mock_response(text)
            result = run_enrichment("some prompt", s, retry_delay=0)

        return result, mock_client

    def test_returns_string(self):
        result, _ = self._call()
        assert isinstance(result, str)

    def test_text_matches_mock(self):
        result, _ = self._call(text="cleaned text")
        assert result == "cleaned text"

    def test_model_name_passed_to_api(self):
        _, mock_client = self._call(gemma_model_name="gemma-2-9b")
        assert mock_client.chat.completions.create.call_args.kwargs["model"] == "gemma-2-9b"

    def test_prompt_sent_as_user_message(self):
        from ocr_kb.model.gemma_backend import run_enrichment

        s = _gemma_settings()
        with patch("ocr_kb.model.gemma_backend.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            MockOpenAI.return_value = mock_client
            mock_client.chat.completions.create.return_value = _mock_response()
            run_enrichment("my prompt text", s, retry_delay=0)

        msgs = mock_client.chat.completions.create.call_args.kwargs["messages"]
        assert msgs[0] == {"role": "user", "content": "my prompt text"}

    def test_client_constructed_with_gemma_settings(self):
        from ocr_kb.model.gemma_backend import run_enrichment

        s = _gemma_settings(gemma_base_url="http://localhost:5678/v1", gemma_api_key="studio-key")
        with patch("ocr_kb.model.gemma_backend.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            MockOpenAI.return_value = mock_client
            mock_client.chat.completions.create.return_value = _mock_response()
            run_enrichment("p", s, retry_delay=0)

        call_kwargs = MockOpenAI.call_args.kwargs
        assert call_kwargs["base_url"] == "http://localhost:5678/v1"
        assert call_kwargs["api_key"] == "studio-key"
        assert "timeout" in call_kwargs

    def test_empty_response_returns_empty_string(self):
        from ocr_kb.model.gemma_backend import run_enrichment

        s = _gemma_settings()
        mock_resp = _mock_response()
        mock_resp.choices[0].message.content = None

        with patch("ocr_kb.model.gemma_backend.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            MockOpenAI.return_value = mock_client
            mock_client.chat.completions.create.return_value = mock_resp
            result = run_enrichment("p", s, retry_delay=0)

        assert result == ""

    def test_max_tokens_default_passed(self):
        from ocr_kb.model.gemma_backend import run_enrichment

        s = _gemma_settings()
        with patch("ocr_kb.model.gemma_backend.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            MockOpenAI.return_value = mock_client
            mock_client.chat.completions.create.return_value = _mock_response()
            run_enrichment("p", s, retry_delay=0)

        # Default is settings.model_max_new_tokens, not the module constant
        assert mock_client.chat.completions.create.call_args.kwargs["max_tokens"] == s.model_max_new_tokens

    def test_max_tokens_override(self):
        from ocr_kb.model.gemma_backend import run_enrichment

        s = _gemma_settings()
        with patch("ocr_kb.model.gemma_backend.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            MockOpenAI.return_value = mock_client
            mock_client.chat.completions.create.return_value = _mock_response()
            run_enrichment("p", s, retry_delay=0, max_tokens=256)

        assert mock_client.chat.completions.create.call_args.kwargs["max_tokens"] == 256


# ---------------------------------------------------------------------------
# run_enrichment — retry logic
# ---------------------------------------------------------------------------

class TestRunEnrichmentRetry:
    def _patched_run(self, side_effects, max_retries=3):
        from ocr_kb.model.gemma_backend import run_enrichment

        s = _gemma_settings()
        with patch("ocr_kb.model.gemma_backend.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            MockOpenAI.return_value = mock_client
            mock_client.chat.completions.create.side_effect = side_effects
            return run_enrichment("p", s, max_retries=max_retries, retry_delay=0)

    def test_retries_on_connection_error_then_succeeds(self):
        from openai import APIConnectionError
        result = self._patched_run([
            APIConnectionError(request=MagicMock()),
            _mock_response("ok"),
        ])
        assert result == "ok"

    def test_raises_after_max_retries_exhausted(self):
        from openai import APIConnectionError
        exc = APIConnectionError(request=MagicMock())
        with pytest.raises(APIConnectionError):
            self._patched_run([exc, exc, exc], max_retries=3)

    def test_no_retry_on_authentication_error(self):
        from openai import AuthenticationError
        from ocr_kb.model.gemma_backend import run_enrichment

        s = _gemma_settings()
        exc = AuthenticationError("bad key", response=MagicMock(), body={})

        with patch("ocr_kb.model.gemma_backend.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            MockOpenAI.return_value = mock_client
            mock_client.chat.completions.create.side_effect = exc
            with pytest.raises(AuthenticationError):
                run_enrichment("p", s, max_retries=3, retry_delay=0)
            assert mock_client.chat.completions.create.call_count == 1

    def test_no_retry_on_bad_request(self):
        from openai import BadRequestError
        from ocr_kb.model.gemma_backend import run_enrichment

        s = _gemma_settings()
        exc = BadRequestError("bad", response=MagicMock(), body={})

        with patch("ocr_kb.model.gemma_backend.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            MockOpenAI.return_value = mock_client
            mock_client.chat.completions.create.side_effect = exc
            with pytest.raises(BadRequestError):
                run_enrichment("p", s, max_retries=3, retry_delay=0)
            assert mock_client.chat.completions.create.call_count == 1


# ---------------------------------------------------------------------------
# Public re-exports from model.__init__
# ---------------------------------------------------------------------------

def test_model_init_exports():
    from ocr_kb.model import OcrRequest, OcrResponse, run_ocr, run_enrichment
    assert callable(run_ocr)
    assert callable(run_enrichment)


# ---------------------------------------------------------------------------
# VRAM resize helper
# ---------------------------------------------------------------------------

class TestResizeForVram:
    def test_small_image_unchanged(self):
        from ocr_kb.model.glm_ocr_backend import _resize_for_vram
        img = Image.new("RGB", (100, 100))
        result = _resize_for_vram(img, max_pixels=100 * 100)
        assert result.size == (100, 100)

    def test_large_image_downscaled(self):
        from ocr_kb.model.glm_ocr_backend import _resize_for_vram
        img = Image.new("RGB", (2000, 2000))
        result = _resize_for_vram(img, max_pixels=1_000_000)
        w, h = result.size
        assert w * h <= 1_000_000

    def test_aspect_ratio_preserved(self):
        from ocr_kb.model.glm_ocr_backend import _resize_for_vram
        img = Image.new("RGB", (2000, 1000))
        result = _resize_for_vram(img, max_pixels=500_000)
        w, h = result.size
        assert abs(w / h - 2.0) < 0.1

    def test_run_ocr_resizes_large_image(self):
        from ocr_kb.model.glm_ocr_backend import run_ocr
        from ocr_kb.model.schema import OcrRequest
        s = _glm_settings(max_image_pixels=10_000)  # tiny cap

        with patch("ocr_kb.model.glm_ocr_backend.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            MockOpenAI.return_value = mock_client
            mock_client.chat.completions.create.return_value = _mock_response()

            big_image = Image.new("RGB", (1000, 1000))  # 1MP > cap
            run_ocr(OcrRequest(image=big_image, prompt="read"), s, retry_delay=0)

        # Verify a data URI was sent — image was encoded (not skipped)
        call_args = mock_client.chat.completions.create.call_args
        content = call_args.kwargs["messages"][0]["content"]
        data_uri = next(c["image_url"]["url"] for c in content if c["type"] == "image_url")
        # Decode and check the actual sent image is smaller than the original
        import base64, io
        b64 = data_uri.split(",", 1)[1]
        sent_img = Image.open(io.BytesIO(base64.b64decode(b64)))
        assert sent_img.width * sent_img.height <= 10_000
