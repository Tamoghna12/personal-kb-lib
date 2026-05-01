"""Tests for ocr_kb.monitoring — logging setup, timers, log helpers."""
from __future__ import annotations

import logging
import time
from pathlib import Path


class TestSetupLogging:
    def test_creates_log_file(self, tmp_path):
        from ocr_kb import monitoring as m

        m._initialized = False
        m.setup_logging(tmp_path / "logs")
        assert (tmp_path / "logs" / "ocr_kb.log").exists()

    def test_idempotent_double_call(self, tmp_path):
        from ocr_kb import monitoring as m

        m._initialized = False
        m.setup_logging(tmp_path / "logs")
        root = logging.getLogger("ocr_kb")
        handler_count_first = len(root.handlers)
        m.setup_logging(tmp_path / "logs")
        assert len(root.handlers) == handler_count_first

    def test_log_file_receives_info(self, tmp_path):
        from ocr_kb import monitoring as m

        m._initialized = False
        m.setup_logging(tmp_path / "logs")
        logging.getLogger("ocr_kb.test").info("hello test message")
        content = (tmp_path / "logs" / "ocr_kb.log").read_text()
        assert "hello test message" in content


class TestIngestTimer:
    def test_records_elapsed_secs(self):
        from ocr_kb.monitoring import IngestTimer

        with IngestTimer() as t:
            time.sleep(0.01)
        assert "elapsed_secs" in t
        assert t["elapsed_secs"] >= 0.01

    def test_elapsed_is_float(self):
        from ocr_kb.monitoring import IngestTimer

        with IngestTimer() as t:
            pass
        assert isinstance(t["elapsed_secs"], float)


class TestSearchTimer:
    def test_records_elapsed_ms(self):
        from ocr_kb.monitoring import SearchTimer

        with SearchTimer() as t:
            time.sleep(0.01)
        assert "elapsed_ms" in t
        assert t["elapsed_ms"] >= 10.0

    def test_elapsed_ms_is_float(self):
        from ocr_kb.monitoring import SearchTimer

        with SearchTimer() as t:
            pass
        assert isinstance(t["elapsed_ms"], float)


class TestLogHelpers:
    def test_log_ingest_summary_no_crash(self, tmp_path):
        from ocr_kb import monitoring as m

        m._initialized = False
        m.setup_logging(tmp_path / "logs")
        m.log_ingest_summary("/data/doc.pdf", 3, [], 1.23)

    def test_log_ingest_summary_with_errors(self, tmp_path):
        from ocr_kb import monitoring as m

        m._initialized = False
        m.setup_logging(tmp_path / "logs")
        m.log_ingest_summary("/data/doc.pdf", 0, ["Page 1: timeout"], 0.5)
        content = (tmp_path / "logs" / "ocr_kb.log").read_text()
        assert "timeout" in content

    def test_log_search_query_no_crash(self, tmp_path):
        from ocr_kb import monitoring as m

        m._initialized = False
        m.setup_logging(tmp_path / "logs")
        m.log_search_query("neural nets", 5, 12.3)

    def test_log_failed_document_no_crash(self, tmp_path):
        from ocr_kb import monitoring as m

        m._initialized = False
        m.setup_logging(tmp_path / "logs")
        m.log_failed_document("/data/bad.pdf", "connection refused", 2)
        content = (tmp_path / "logs" / "ocr_kb.log").read_text()
        assert "bad.pdf" in content
