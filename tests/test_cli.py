"""Tests for ocr_kb.cli — all five commands via Typer CliRunner."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from ocr_kb.cli import app
from ocr_kb.kb.store import init_db, save
from ocr_kb.kb.schema import KBEntry
from ocr_kb.pipeline import FileResult, PageResult

runner = CliRunner()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _db(tmp_path: Path) -> Path:
    p = tmp_path / "test.db"
    init_db(p)
    return p


def _txt(tmp_path: Path, name: str = "note.txt", content: str = "hello world notes") -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


def _entry(db_path: Path, **kwargs) -> KBEntry:
    conn = init_db(db_path)
    defaults = dict(
        source_path="doc.pdf", page_number=1,
        raw_text="the quick brown fox", markdown="the quick brown fox",
        html="", layout_blocks="[]", tags="test", category="general",
    )
    defaults.update(kwargs)
    return save(conn, KBEntry(**defaults))


def _page_result(text: str = "ocr text", mode: str = "text") -> PageResult:
    return PageResult(
        source_path=Path("file.pdf"),
        page_number=1,
        text=text,
        mode=mode,
    )


def _file_result(text: str = "hello world content here") -> FileResult:
    fr = FileResult(source_path=Path("file.txt"))
    fr.pages.append(_page_result(text))
    return fr


# ---------------------------------------------------------------------------
# App-level
# ---------------------------------------------------------------------------

def test_app_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "ingest" in result.output
    assert "search" in result.output
    assert "export" in result.output
    assert "stats" in result.output
    assert "watch" in result.output


def test_no_args_shows_help():
    result = runner.invoke(app, [])
    # Typer returns exit code 2 when no_args_is_help=True
    assert result.exit_code in (0, 2)
    assert "ingest" in result.output


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------

class TestIngest:
    def test_ingest_text_file_exits_ok(self, tmp_path):
        p = _txt(tmp_path)
        db = _db(tmp_path)
        result = runner.invoke(app, ["ingest", str(p), "--db", str(db)])
        assert result.exit_code == 0

    def test_ingest_text_file_saves_entry(self, tmp_path):
        from ocr_kb.kb.store import get_all
        p = _txt(tmp_path, content="searchable unique phrase")
        db = _db(tmp_path)
        runner.invoke(app, ["ingest", str(p), "--db", str(db)])
        entries = get_all(init_db(db))
        assert len(entries) == 1
        assert "searchable unique phrase" in entries[0].raw_text

    def test_ingest_applies_tags(self, tmp_path):
        from ocr_kb.kb.store import get_all
        p = _txt(tmp_path)
        db = _db(tmp_path)
        runner.invoke(app, ["ingest", str(p), "--db", str(db), "--tags", "ml,ai"])
        entries = get_all(init_db(db))
        assert entries[0].tags == "ml,ai"

    def test_ingest_applies_category(self, tmp_path):
        from ocr_kb.kb.store import get_all
        p = _txt(tmp_path)
        db = _db(tmp_path)
        runner.invoke(app, ["ingest", str(p), "--db", str(db), "--category", "science"])
        entries = get_all(init_db(db))
        assert entries[0].category == "science"

    def test_ingest_directory_processes_all_txt(self, tmp_path):
        from ocr_kb.kb.store import get_all
        _txt(tmp_path, "a.txt", "first file content here")
        _txt(tmp_path, "b.txt", "second file content here")
        db = _db(tmp_path)
        result = runner.invoke(app, ["ingest", str(tmp_path), "--db", str(db)])
        assert result.exit_code == 0
        entries = get_all(init_db(db))
        assert len(entries) == 2

    def test_ingest_missing_path_exits_nonzero(self, tmp_path):
        db = _db(tmp_path)
        result = runner.invoke(app, ["ingest", str(tmp_path / "nope.txt"), "--db", str(db)])
        assert result.exit_code != 0

    def test_ingest_blank_page_skipped(self, tmp_path):
        from ocr_kb.kb.store import get_all
        p = _txt(tmp_path, content="   ")  # blank
        db = _db(tmp_path)
        runner.invoke(app, ["ingest", str(p), "--db", str(db)])
        assert get_all(init_db(db)) == []

    def test_ingest_image_mocks_ocr(self, tmp_path):
        from ocr_kb.kb.store import get_all
        from PIL import Image
        img = tmp_path / "scan.png"
        Image.new("RGB", (200, 200)).save(img)
        db = _db(tmp_path)

        mock_fr = _file_result("mocked ocr result from image")
        with patch("ocr_kb.cli.process_path", return_value=[mock_fr]):
            result = runner.invoke(app, ["ingest", str(img), "--db", str(db)])
        assert result.exit_code == 0
        entries = get_all(init_db(db))
        assert entries[0].raw_text == "mocked ocr result from image"

    def test_ingest_html_mode_stores_html(self, tmp_path):
        from ocr_kb.kb.store import get_all
        db = _db(tmp_path)
        fake_pdf = tmp_path / "doc.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4")  # file must exist for path check
        fr = FileResult(source_path=fake_pdf)
        fr.pages.append(PageResult(
            source_path=fake_pdf, page_number=1,
            text="<h1>Title</h1><p>Body text content here.</p>",
            mode="html",
        ))
        with patch("ocr_kb.cli.process_path", return_value=[fr]):
            runner.invoke(app, ["ingest", str(fake_pdf),
                                "--db", str(db), "--mode", "html"])
        entries = get_all(init_db(db))
        assert entries and ("<h1>" in entries[0].html or "Title" in entries[0].markdown)

    def test_ingest_output_mentions_saved(self, tmp_path):
        p = _txt(tmp_path, content="content long enough to not be blank")
        db = _db(tmp_path)
        result = runner.invoke(app, ["ingest", str(p), "--db", str(db)])
        assert "saved" in result.output.lower() or "done" in result.output.lower()

    def test_ingest_errors_shown_in_output(self, tmp_path):
        db = _db(tmp_path)
        fake = tmp_path / "bad.pdf"
        fake.write_bytes(b"%PDF-1.4")
        fr = FileResult(source_path=fake)
        fr.errors.append("Page 1: model offline")
        with patch("ocr_kb.cli.process_path", return_value=[fr]):
            result = runner.invoke(app, ["ingest", str(fake), "--db", str(db)])
        assert "model offline" in result.output


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

class TestSearch:
    def test_search_no_results_message(self, tmp_path):
        db = _db(tmp_path)
        result = runner.invoke(app, ["search", "xyzzy_not_found", "--db", str(db)])
        assert result.exit_code == 0
        assert "no results" in result.output.lower()

    def test_search_finds_entry(self, tmp_path):
        db = _db(tmp_path)
        _entry(db, raw_text="quantum mechanics fundamentals", markdown="quantum mechanics fundamentals")
        result = runner.invoke(app, ["search", "quantum", "--db", str(db)])
        assert result.exit_code == 0
        assert "quantum" in result.output.lower() or "doc.pdf" in result.output

    def test_search_limit_option(self, tmp_path):
        db = _db(tmp_path)
        for i in range(5):
            _entry(db, page_number=i, raw_text=f"common word entry {i}",
                   markdown=f"common word entry {i}")
        result = runner.invoke(app, ["search", "common", "--db", str(db), "--limit", "2"])
        assert result.exit_code == 0

    def test_search_help(self):
        result = runner.invoke(app, ["search", "--help"])
        assert result.exit_code == 0
        assert "query" in result.output.lower() or "search" in result.output.lower()


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------

class TestExport:
    def test_export_markdown(self, tmp_path):
        db = _db(tmp_path)
        _entry(db)
        out = tmp_path / "out"
        result = runner.invoke(app, ["export", "--format", "markdown", "--out", str(out),
                                     "--db", str(db)])
        assert result.exit_code == 0
        assert any(out.rglob("*.md"))

    def test_export_json(self, tmp_path):
        db = _db(tmp_path)
        _entry(db)
        out = tmp_path / "out.json"
        result = runner.invoke(app, ["export", "--format", "json", "--out", str(out),
                                     "--db", str(db)])
        assert result.exit_code == 0
        assert out.exists()
        data = json.loads(out.read_text())
        assert len(data) == 1

    def test_export_obsidian(self, tmp_path):
        db = _db(tmp_path)
        _entry(db)
        out = tmp_path / "vault"
        result = runner.invoke(app, ["export", "--format", "obsidian", "--out", str(out),
                                     "--db", str(db)])
        assert result.exit_code == 0
        assert any(out.rglob("*.md"))

    def test_export_unknown_format_exits_nonzero(self, tmp_path):
        db = _db(tmp_path)
        out = tmp_path / "out"
        result = runner.invoke(app, ["export", "--format", "docx", "--out", str(out),
                                     "--db", str(db)])
        assert result.exit_code != 0

    def test_export_help(self):
        result = runner.invoke(app, ["export", "--help"])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------

class TestStats:
    def test_stats_empty_db(self, tmp_path):
        db = _db(tmp_path)
        result = runner.invoke(app, ["stats", "--db", str(db)])
        assert result.exit_code == 0
        assert "0" in result.output

    def test_stats_counts_entries(self, tmp_path):
        db = _db(tmp_path)
        _entry(db, page_number=1)
        _entry(db, page_number=2)
        result = runner.invoke(app, ["stats", "--db", str(db)])
        assert result.exit_code == 0
        assert "2" in result.output

    def test_stats_help(self):
        result = runner.invoke(app, ["stats", "--help"])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# watch — only verify registration and help (no live observer in tests)
# ---------------------------------------------------------------------------

class TestWatch:
    def test_watch_help(self):
        result = runner.invoke(app, ["watch", "--help"])
        assert result.exit_code == 0
        assert "--dir" in result.output or "directory" in result.output.lower()

    def test_watch_registered_in_app(self):
        result = runner.invoke(app, ["--help"])
        assert "watch" in result.output


# ---------------------------------------------------------------------------
# ask  (RAG query)
# ---------------------------------------------------------------------------

class TestAsk:
    def test_ask_help(self):
        result = runner.invoke(app, ["ask", "--help"])
        assert result.exit_code == 0
        assert "question" in result.output.lower() or "ask" in result.output.lower()

    def test_ask_registered_in_app(self):
        result = runner.invoke(app, ["--help"])
        assert "ask" in result.output

    def test_ask_no_results_exits_ok(self, tmp_path):
        db = _db(tmp_path)
        result = runner.invoke(app, ["ask", "quantum teleportation", "--db", str(db)])
        assert result.exit_code == 0
        assert "no relevant" in result.output.lower()

    def test_ask_calls_enrichment_with_context(self, tmp_path):
        db = _db(tmp_path)
        _entry(db, raw_text="backpropagation computes gradients", markdown="backpropagation computes gradients")
        with patch("ocr_kb.model.run_enrichment", return_value="Backprop is used to train neural nets.") as mock_enrich:
            result = runner.invoke(app, ["ask", "backpropagation", "--db", str(db)])
        assert result.exit_code == 0
        mock_enrich.assert_called_once()
        prompt_sent = mock_enrich.call_args[0][0]
        assert "backpropagation" in prompt_sent.lower()

    def test_ask_shows_answer_in_output(self, tmp_path):
        db = _db(tmp_path)
        _entry(db, raw_text="deep learning uses many layers", markdown="deep learning uses many layers")
        with patch("ocr_kb.model.run_enrichment", return_value="Deep learning stacks multiple layers."):
            result = runner.invoke(app, ["ask", "deep learning", "--db", str(db)])
        assert "Deep learning stacks multiple layers." in result.output

    def test_ask_context_size_limits_retrieval(self, tmp_path):
        db = _db(tmp_path)
        for i in range(5):
            _entry(db, page_number=i, raw_text=f"entry {i} about neural nets",
                   markdown=f"entry {i} about neural nets")
        prompts_seen: list[str] = []
        def capture(prompt, settings, **kwargs):
            prompts_seen.append(prompt)
            return "answer"
        with patch("ocr_kb.model.run_enrichment", side_effect=capture):
            runner.invoke(app, ["ask", "neural nets", "--db", str(db), "--context", "2"])
        assert len(prompts_seen) == 1
        # Context contains entries [1] and [2] but not [3] (limit=2 from 5 entries)
        assert "[2]" in prompts_seen[0]
        assert "[3]" not in prompts_seen[0]


# ---------------------------------------------------------------------------
# compile-wiki and lint-wiki
# ---------------------------------------------------------------------------

class TestCompileWiki:
    def test_compile_wiki_help(self):
        result = runner.invoke(app, ["compile-wiki", "--help"])
        assert result.exit_code == 0

    def test_compile_wiki_registered_in_app(self):
        result = runner.invoke(app, ["--help"])
        assert "compile-wiki" in result.output or "compile" in result.output

    def test_compile_wiki_calls_compiler(self, tmp_path):
        with patch("ocr_kb.wiki.compiler.compile_wiki", return_value={"processed_entries": 3, "concepts_found": 7}) as mock_cw:
            result = runner.invoke(app, ["compile-wiki"])
        assert result.exit_code == 0
        mock_cw.assert_called_once()
        assert "7" in result.output

    def test_lint_wiki_help(self):
        result = runner.invoke(app, ["lint-wiki", "--help"])
        assert result.exit_code == 0

    def test_lint_wiki_no_issues(self, tmp_path):
        with patch("ocr_kb.wiki.linter.lint_wiki", return_value={"broken_links": [], "orphan_concepts": [], "missing_frontmatter": []}):
            result = runner.invoke(app, ["lint-wiki"])
        assert result.exit_code == 0
        assert "no issues" in result.output.lower()

    def test_lint_wiki_reports_issues(self, tmp_path):
        with patch("ocr_kb.wiki.linter.lint_wiki", return_value={"broken_links": ["concepts/a.md -> missing.md"], "orphan_concepts": [], "missing_frontmatter": []}):
            result = runner.invoke(app, ["lint-wiki"])
        assert result.exit_code == 0
        assert "missing.md" in result.output or "broken" in result.output.lower()

    def test_compile_wiki_since_flag(self, tmp_path):
        with patch("ocr_kb.wiki.compiler.compile_wiki",
                   return_value={"processed_entries": 1, "concepts_found": 2}) as mock_cw:
            result = runner.invoke(app, ["compile-wiki", "--since", "2024-01-01"])
        assert result.exit_code == 0
        call_kwargs = mock_cw.call_args[1]
        assert call_kwargs.get("since") == "2024-01-01"


# ---------------------------------------------------------------------------
# show
# ---------------------------------------------------------------------------

class TestShow:
    def test_show_prints_entry(self, tmp_path):
        db = _db(tmp_path)
        _entry(db, raw_text="unique visible content here", markdown="unique visible content here")
        result = runner.invoke(app, ["show", "1", "--db", str(db)])
        assert result.exit_code == 0
        assert "doc.pdf" in result.output

    def test_show_missing_id_exits_nonzero(self, tmp_path):
        db = _db(tmp_path)
        result = runner.invoke(app, ["show", "999", "--db", str(db)])
        assert result.exit_code != 0

    def test_show_help(self):
        result = runner.invoke(app, ["show", "--help"])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# recent
# ---------------------------------------------------------------------------

class TestRecent:
    def test_recent_empty_db(self, tmp_path):
        db = _db(tmp_path)
        result = runner.invoke(app, ["recent", "--db", str(db)])
        assert result.exit_code == 0

    def test_recent_shows_entries(self, tmp_path):
        db = _db(tmp_path)
        _entry(db, page_number=1)
        _entry(db, page_number=2)
        result = runner.invoke(app, ["recent", "--db", str(db)])
        assert result.exit_code == 0
        assert "doc.pdf" in result.output

    def test_recent_limit(self, tmp_path):
        db = _db(tmp_path)
        for i in range(5):
            _entry(db, page_number=i)
        result = runner.invoke(app, ["recent", "--limit", "2", "--db", str(db)])
        assert result.exit_code == 0

    def test_recent_help(self):
        result = runner.invoke(app, ["recent", "--help"])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# tag / retag
# ---------------------------------------------------------------------------

class TestTag:
    def test_tag_add(self, tmp_path):
        from ocr_kb.kb.store import get
        db = _db(tmp_path)
        e = _entry(db, tags="ml")
        result = runner.invoke(app, ["tag", str(e.id), "--add", "ai", "--db", str(db)])
        assert result.exit_code == 0
        updated = get(init_db(db), e.id)
        assert "ai" in updated.tags
        assert "ml" in updated.tags

    def test_tag_remove(self, tmp_path):
        from ocr_kb.kb.store import get
        db = _db(tmp_path)
        e = _entry(db, tags="ml,draft")
        result = runner.invoke(app, ["tag", str(e.id), "--remove", "draft", "--db", str(db)])
        assert result.exit_code == 0
        updated = get(init_db(db), e.id)
        assert "draft" not in updated.tags

    def test_tag_set_category(self, tmp_path):
        from ocr_kb.kb.store import get
        db = _db(tmp_path)
        e = _entry(db, category="old")
        runner.invoke(app, ["tag", str(e.id), "--category", "science", "--db", str(db)])
        assert get(init_db(db), e.id).category == "science"

    def test_tag_missing_entry_exits_nonzero(self, tmp_path):
        db = _db(tmp_path)
        result = runner.invoke(app, ["tag", "999", "--add", "x", "--db", str(db)])
        assert result.exit_code != 0

    def test_retag_bulk(self, tmp_path):
        from ocr_kb.kb.store import get_all
        db = _db(tmp_path)
        _entry(db, page_number=1, raw_text="neural network training",
               markdown="neural network training", tags="")
        _entry(db, page_number=2, raw_text="neural network layers",
               markdown="neural network layers", tags="")
        result = runner.invoke(app, ["retag", "--query", "neural", "--add", "ml",
                                     "--db", str(db)])
        assert result.exit_code == 0
        entries = get_all(init_db(db))
        assert all("ml" in e.tags for e in entries)

    def test_retag_no_match(self, tmp_path):
        db = _db(tmp_path)
        result = runner.invoke(app, ["retag", "--query", "xyznotfound",
                                     "--add", "x", "--db", str(db)])
        assert result.exit_code == 0
        assert "no entries" in result.output.lower()


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------

class TestDelete:
    def test_delete_by_id(self, tmp_path):
        from ocr_kb.kb.store import get_all
        db = _db(tmp_path)
        e = _entry(db)
        result = runner.invoke(app, ["delete", "--id", str(e.id), "--yes",
                                     "--db", str(db)])
        assert result.exit_code == 0
        assert get_all(init_db(db)) == []

    def test_delete_missing_id_exits_nonzero(self, tmp_path):
        db = _db(tmp_path)
        result = runner.invoke(app, ["delete", "--id", "999", "--yes",
                                     "--db", str(db)])
        assert result.exit_code != 0

    def test_delete_by_source(self, tmp_path):
        from ocr_kb.kb.store import get_all
        db = _db(tmp_path)
        _entry(db, source_path="doc.pdf", page_number=1)
        _entry(db, source_path="doc.pdf", page_number=2)
        result = runner.invoke(app, ["delete", "--source", "doc.pdf", "--yes",
                                     "--db", str(db)])
        assert result.exit_code == 0
        assert get_all(init_db(db)) == []

    def test_delete_no_args_exits_nonzero(self, tmp_path):
        db = _db(tmp_path)
        result = runner.invoke(app, ["delete", "--db", str(db)])
        assert result.exit_code != 0

    def test_delete_help(self):
        result = runner.invoke(app, ["delete", "--help"])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# search with filters
# ---------------------------------------------------------------------------

class TestSearchFilters:
    def test_search_filter_by_source(self, tmp_path):
        db = _db(tmp_path)
        _entry(db, source_path="paper_a.pdf", raw_text="quantum computing",
               markdown="quantum computing")
        _entry(db, source_path="paper_b.pdf", page_number=2,
               raw_text="quantum entanglement",
               markdown="quantum entanglement")
        result = runner.invoke(app, ["search", "quantum", "--source", "paper_a",
                                     "--db", str(db)])
        assert result.exit_code == 0
        assert "paper_a" in result.output
        assert "paper_b" not in result.output

    def test_search_filter_by_tag(self, tmp_path):
        db = _db(tmp_path)
        _entry(db, raw_text="deep learning layers",
               markdown="deep learning layers", tags="ml")
        _entry(db, page_number=2, raw_text="deep learning theory",
               markdown="deep learning theory", tags="bio")
        result = runner.invoke(app, ["search", "deep learning", "--tag", "ml",
                                     "--db", str(db)])
        assert result.exit_code == 0

    def test_search_filter_help(self):
        result = runner.invoke(app, ["search", "--help"])
        assert result.exit_code == 0
        assert "--source" in result.output or "source" in result.output.lower()


# ---------------------------------------------------------------------------
# ask — source map and chunk-chars
# ---------------------------------------------------------------------------

class TestAskEnhancements:
    def test_ask_prints_source_map(self, tmp_path):
        db = _db(tmp_path)
        _entry(db, raw_text="backpropagation computes gradients",
               markdown="backpropagation computes gradients")
        with patch("ocr_kb.model.run_enrichment", return_value="Answer text."):
            result = runner.invoke(app, ["ask", "backpropagation", "--db", str(db)])
        assert result.exit_code == 0
        assert "doc.pdf" in result.output
        assert "[1]" in result.output

    def test_ask_chunk_chars_option(self, tmp_path):
        db = _db(tmp_path)
        # Use a long text with a clearly searchable word at the start
        long_text = "backprop " + "word " * 400  # 2000+ chars
        _entry(db, raw_text=long_text, markdown=long_text)
        prompts_seen: list[str] = []
        def capture(prompt, settings, **kwargs):
            prompts_seen.append(prompt)
            return "answer"
        with patch("ocr_kb.model.run_enrichment", side_effect=capture):
            runner.invoke(app, ["ask", "backprop", "--chunk-chars", "50", "--db", str(db)])
        assert len(prompts_seen) == 1
        # Snippet in context must be <= 50 chars, so the full text was truncated
        context_block = prompts_seen[0].split("CONTEXT:")[1].split("QUESTION:")[0]
        source_text = context_block.split(":\n", 1)[1] if ":\n" in context_block else context_block
        assert len(source_text.strip()) <= 60  # some tolerance for surrounding whitespace


# ---------------------------------------------------------------------------
# export with filters
# ---------------------------------------------------------------------------

class TestExportFilters:
    def test_export_json_filter_tag(self, tmp_path):
        db = _db(tmp_path)
        _entry(db, source_path="a.pdf", tags="ml")
        _entry(db, source_path="b.pdf", page_number=2, tags="bio")
        out = tmp_path / "out.json"
        result = runner.invoke(app, ["export", "--format", "json", "--out", str(out),
                                     "--filter-tag", "ml", "--db", str(db)])
        assert result.exit_code == 0
        import json
        data = json.loads(out.read_text())
        assert len(data) == 1

    def test_export_markdown_filter_category(self, tmp_path):
        db = _db(tmp_path)
        _entry(db, source_path="a.pdf", category="science")
        _entry(db, source_path="b.pdf", page_number=2, category="history")
        out = tmp_path / "out"
        result = runner.invoke(app, ["export", "--format", "markdown", "--out", str(out),
                                     "--filter-category", "science", "--db", str(db)])
        assert result.exit_code == 0
        mds = list(out.rglob("*.md"))
        assert len(mds) == 1


# ---------------------------------------------------------------------------
# Dead-Letter Queue
# ---------------------------------------------------------------------------

class TestDlqCli:
    def test_dlq_list_empty(self, tmp_path):
        db = _db(tmp_path)
        result = runner.invoke(app, ["dlq", "list", "--db", str(db)])
        assert result.exit_code == 0
        assert "empty" in result.output.lower()

    def test_dlq_list_shows_items(self, tmp_path):
        from ocr_kb.kb.store import dlq_push
        db = _db(tmp_path)
        conn = init_db(db)
        dlq_push(conn, "/data/doc.pdf", "some error")
        result = runner.invoke(app, ["dlq", "list", "--db", str(db)])
        assert result.exit_code == 0
        assert "doc.pdf" in result.output

    def test_dlq_clear_empty(self, tmp_path):
        db = _db(tmp_path)
        result = runner.invoke(app, ["dlq", "clear", "--yes", "--db", str(db)])
        assert result.exit_code == 0
        assert "empty" in result.output.lower()

    def test_dlq_clear_marks_resolved(self, tmp_path):
        from ocr_kb.kb.store import dlq_push, dlq_list
        db = _db(tmp_path)
        conn = init_db(db)
        dlq_push(conn, "/data/doc.pdf", "error")
        result = runner.invoke(app, ["dlq", "clear", "--yes", "--db", str(db)])
        assert result.exit_code == 0
        assert dlq_list(conn) == []

    def test_dlq_retry_no_items(self, tmp_path):
        db = _db(tmp_path)
        result = runner.invoke(app, ["dlq", "retry", "--db", str(db)])
        assert result.exit_code == 0
        assert "no retryable" in result.output.lower()
