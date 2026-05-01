"""Tests for ocr_kb.kb — store, FTS search, semantic search, export."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from ocr_kb.kb.schema import KBEntry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def conn(tmp_path):
    from ocr_kb.kb.store import init_db
    return init_db(tmp_path / "test.db")


def _entry(**kwargs) -> KBEntry:
    defaults = dict(
        source_path="doc.pdf",
        page_number=1,
        raw_text="The quick brown fox jumps over the lazy dog.",
        markdown="The quick brown fox jumps over the lazy dog.",
        html="<p>The quick brown fox jumps over the lazy dog.</p>",
        layout_blocks="[]",
        tags="animals,nature",
        category="test",
        key_points="- foxes jump",
        summary="A fox jumps.",
    )
    defaults.update(kwargs)
    return KBEntry(**defaults)


# ---------------------------------------------------------------------------
# init_db
# ---------------------------------------------------------------------------

def test_init_db_creates_file(tmp_path):
    from ocr_kb.kb.store import init_db
    db = tmp_path / "new.db"
    init_db(db)
    assert db.exists()


def test_init_db_idempotent(tmp_path):
    from ocr_kb.kb.store import init_db
    db = tmp_path / "kb.db"
    init_db(db)
    init_db(db)  # second call must not raise


# ---------------------------------------------------------------------------
# save / get / get_all / delete
# ---------------------------------------------------------------------------

class TestStore:
    def test_save_assigns_id(self, conn):
        from ocr_kb.kb.store import save
        e = save(conn, _entry())
        assert e.id is not None
        assert e.id > 0

    def test_get_returns_entry(self, conn):
        from ocr_kb.kb.store import save, get
        saved = save(conn, _entry())
        fetched = get(conn, saved.id)
        assert fetched is not None
        assert fetched.id == saved.id

    def test_get_unknown_id_returns_none(self, conn):
        from ocr_kb.kb.store import get
        assert get(conn, 9999) is None

    def test_round_trip_fields(self, conn):
        from ocr_kb.kb.store import save, get
        e = _entry(source_path="my/file.pdf", page_number=3, category="science")
        saved = save(conn, e)
        fetched = get(conn, saved.id)
        assert fetched.source_path == "my/file.pdf"
        assert fetched.page_number == 3
        assert fetched.category == "science"

    def test_raw_text_preserved(self, conn):
        from ocr_kb.kb.store import save, get
        e = save(conn, _entry(raw_text="unique phrase xyz"))
        assert get(conn, e.id).raw_text == "unique phrase xyz"

    def test_markdown_preserved(self, conn):
        from ocr_kb.kb.store import save, get
        e = save(conn, _entry(markdown="# Title\n\nParagraph"))
        assert get(conn, e.id).markdown == "# Title\n\nParagraph"

    def test_tags_preserved(self, conn):
        from ocr_kb.kb.store import save, get
        e = save(conn, _entry(tags="ml,ai,nlp"))
        assert get(conn, e.id).tags == "ml,ai,nlp"

    def test_created_at_preserved(self, conn):
        from ocr_kb.kb.store import save, get
        dt = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        e = save(conn, _entry(created_at=dt))
        fetched = get(conn, e.id)
        assert fetched.created_at == dt

    def test_get_all_returns_all(self, conn):
        from ocr_kb.kb.store import save, get_all
        save(conn, _entry(page_number=1))
        save(conn, _entry(page_number=2))
        save(conn, _entry(page_number=3))
        assert len(get_all(conn)) == 3

    def test_get_all_empty_db(self, conn):
        from ocr_kb.kb.store import get_all
        assert get_all(conn) == []

    def test_delete_removes_entry(self, conn):
        from ocr_kb.kb.store import save, get, delete
        e = save(conn, _entry())
        assert delete(conn, e.id) is True
        assert get(conn, e.id) is None

    def test_delete_nonexistent_returns_false(self, conn):
        from ocr_kb.kb.store import delete
        assert delete(conn, 9999) is False

    def test_embedding_round_trip(self, conn):
        from ocr_kb.kb.store import save, get
        emb = [0.1, 0.2, 0.3, 0.4]
        e = save(conn, _entry(embedding=emb))
        fetched = get(conn, e.id)
        assert fetched.embedding == pytest.approx(emb)

    def test_null_embedding_round_trip(self, conn):
        from ocr_kb.kb.store import save, get
        e = save(conn, _entry(embedding=None))
        assert get(conn, e.id).embedding is None


class TestStats:
    def test_stats_empty_db(self, conn):
        from ocr_kb.kb.store import stats
        s = stats(conn)
        assert s["total_entries"] == 0
        assert s["unique_sources"] == 0

    def test_stats_counts_entries(self, conn):
        from ocr_kb.kb.store import save, stats
        save(conn, _entry(source_path="a.pdf"))
        save(conn, _entry(source_path="a.pdf", page_number=2))
        save(conn, _entry(source_path="b.pdf"))
        s = stats(conn)
        assert s["total_entries"] == 3
        assert s["unique_sources"] == 2


# ---------------------------------------------------------------------------
# FTS search
# ---------------------------------------------------------------------------

class TestFtsSearch:
    def test_finds_word_in_raw_text(self, conn):
        from ocr_kb.kb.store import save
        from ocr_kb.kb.indexer import fts_search
        save(conn, _entry(raw_text="machine learning fundamentals"))
        results = fts_search(conn, "machine")
        assert len(results) >= 1
        assert any("machine" in r.raw_text for r in results)

    def test_finds_word_in_tags(self, conn):
        from ocr_kb.kb.store import save
        from ocr_kb.kb.indexer import fts_search
        save(conn, _entry(tags="neuroscience,cognition"))
        results = fts_search(conn, "neuroscience")
        assert len(results) >= 1

    def test_finds_word_in_key_points(self, conn):
        from ocr_kb.kb.store import save
        from ocr_kb.kb.indexer import fts_search
        save(conn, _entry(key_points="- entropy increases over time"))
        results = fts_search(conn, "entropy")
        assert len(results) >= 1

    def test_no_match_returns_empty(self, conn):
        from ocr_kb.kb.store import save
        from ocr_kb.kb.indexer import fts_search
        save(conn, _entry(raw_text="completely unrelated content"))
        results = fts_search(conn, "xyzzy123")
        assert results == []

    def test_natural_language_question_does_not_crash(self, conn):
        from ocr_kb.kb.store import save
        from ocr_kb.kb.indexer import fts_search
        save(conn, _entry(raw_text="backpropagation trains neural nets"))
        # FTS5 special chars like ? should not raise OperationalError
        results = fts_search(conn, "What is backpropagation?")
        assert isinstance(results, list)

    def test_limit_respected(self, conn):
        from ocr_kb.kb.store import save
        from ocr_kb.kb.indexer import fts_search
        for i in range(10):
            save(conn, _entry(raw_text=f"word common entry {i}", page_number=i))
        results = fts_search(conn, "common", limit=3)
        assert len(results) <= 3


# ---------------------------------------------------------------------------
# filter helpers
# ---------------------------------------------------------------------------

class TestFilterHelpers:
    def test_filter_by_tag(self, conn):
        from ocr_kb.kb.store import save
        from ocr_kb.kb.indexer import filter_by_tag
        save(conn, _entry(tags="ml,ai"))
        save(conn, _entry(tags="biology", page_number=2))
        results = filter_by_tag(conn, "ml")
        assert len(results) == 1
        assert "ml" in results[0].tags

    def test_filter_by_tag_case_insensitive(self, conn):
        from ocr_kb.kb.store import save
        from ocr_kb.kb.indexer import filter_by_tag
        save(conn, _entry(tags="ML,AI"))
        assert len(filter_by_tag(conn, "ml")) == 1

    def test_filter_by_source(self, conn):
        from ocr_kb.kb.store import save
        from ocr_kb.kb.indexer import filter_by_source
        save(conn, _entry(source_path="paper.pdf", page_number=1))
        save(conn, _entry(source_path="paper.pdf", page_number=2))
        save(conn, _entry(source_path="other.pdf", page_number=1))
        results = filter_by_source(conn, "paper.pdf")
        assert len(results) == 2
        assert all(r.source_path == "paper.pdf" for r in results)

    def test_filter_by_source_ordered_by_page(self, conn):
        from ocr_kb.kb.store import save
        from ocr_kb.kb.indexer import filter_by_source
        save(conn, _entry(source_path="doc.pdf", page_number=3))
        save(conn, _entry(source_path="doc.pdf", page_number=1))
        save(conn, _entry(source_path="doc.pdf", page_number=2))
        pages = [r.page_number for r in filter_by_source(conn, "doc.pdf")]
        assert pages == [1, 2, 3]


# ---------------------------------------------------------------------------
# semantic search
# ---------------------------------------------------------------------------

class TestSemanticSearch:
    def _make_entry_with_emb(self, conn, emb, **kwargs):
        from ocr_kb.kb.store import save
        return save(conn, _entry(embedding=emb, **kwargs))

    def test_returns_closest_entry(self, conn):
        from ocr_kb.kb.indexer import semantic_search
        self._make_entry_with_emb(conn, [1.0, 0.0, 0.0], page_number=1)
        self._make_entry_with_emb(conn, [0.0, 1.0, 0.0], page_number=2)
        results = semantic_search(conn, [1.0, 0.0, 0.0])
        assert results[0][0].page_number == 1

    def test_score_between_0_and_1(self, conn):
        from ocr_kb.kb.indexer import semantic_search
        self._make_entry_with_emb(conn, [1.0, 0.0])
        results = semantic_search(conn, [0.8, 0.6])
        assert 0.0 <= results[0][1] <= 1.0

    def test_skips_entries_without_embedding(self, conn):
        from ocr_kb.kb.store import save
        from ocr_kb.kb.indexer import semantic_search
        save(conn, _entry(embedding=None))
        results = semantic_search(conn, [1.0, 0.0])
        assert results == []

    def test_zero_query_returns_empty(self, conn):
        from ocr_kb.kb.indexer import semantic_search
        self._make_entry_with_emb(conn, [1.0, 0.0])
        assert semantic_search(conn, [0.0, 0.0]) == []

    def test_limit_respected(self, conn):
        from ocr_kb.kb.indexer import semantic_search
        for i in range(5):
            self._make_entry_with_emb(conn, [float(i), 1.0], page_number=i)
        results = semantic_search(conn, [1.0, 1.0], limit=2)
        assert len(results) <= 2

    def test_sorted_descending_by_score(self, conn):
        from ocr_kb.kb.indexer import semantic_search
        self._make_entry_with_emb(conn, [1.0, 0.0], page_number=1)
        self._make_entry_with_emb(conn, [0.6, 0.8], page_number=2)
        results = semantic_search(conn, [1.0, 0.0])
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# exporter
# ---------------------------------------------------------------------------

class TestExportMarkdown:
    def test_creates_one_file_per_entry(self, conn, tmp_path):
        from ocr_kb.kb.store import save
        from ocr_kb.kb.exporter import export_markdown
        save(conn, _entry(source_path="doc.pdf", page_number=1))
        save(conn, _entry(source_path="doc.pdf", page_number=2))
        paths = export_markdown(conn, tmp_path / "out")
        assert len(paths) == 2

    def test_files_exist(self, conn, tmp_path):
        from ocr_kb.kb.store import save
        from ocr_kb.kb.exporter import export_markdown
        save(conn, _entry())
        for p in export_markdown(conn, tmp_path / "out"):
            assert p.exists()

    def test_filename_convention(self, conn, tmp_path):
        from ocr_kb.kb.store import save
        from ocr_kb.kb.exporter import export_markdown
        save(conn, _entry(source_path="notes.pdf", page_number=5))
        paths = export_markdown(conn, tmp_path / "out")
        assert paths[0].name == "notes_p005.md"

    def test_content_contains_markdown(self, conn, tmp_path):
        from ocr_kb.kb.store import save
        from ocr_kb.kb.exporter import export_markdown
        save(conn, _entry(markdown="# My Title"))
        paths = export_markdown(conn, tmp_path / "out")
        assert "# My Title" in paths[0].read_text()


class TestExportJson:
    def test_creates_json_file(self, conn, tmp_path):
        from ocr_kb.kb.store import save
        from ocr_kb.kb.exporter import export_json
        save(conn, _entry())
        path = export_json(conn, tmp_path / "out.json")
        assert path.exists()

    def test_valid_json(self, conn, tmp_path):
        from ocr_kb.kb.store import save
        from ocr_kb.kb.exporter import export_json
        save(conn, _entry())
        path = export_json(conn, tmp_path / "out.json")
        data = json.loads(path.read_text())
        assert isinstance(data, list)
        assert len(data) == 1

    def test_json_fields_present(self, conn, tmp_path):
        from ocr_kb.kb.store import save
        from ocr_kb.kb.exporter import export_json
        save(conn, _entry(category="physics", tags="waves"))
        data = json.loads(export_json(conn, tmp_path / "out.json").read_text())
        assert data[0]["category"] == "physics"
        assert data[0]["tags"] == "waves"

    def test_empty_db_exports_empty_list(self, conn, tmp_path):
        from ocr_kb.kb.exporter import export_json
        data = json.loads(export_json(conn, tmp_path / "out.json").read_text())
        assert data == []


class TestExportObsidian:
    def test_creates_files_in_vault(self, conn, tmp_path):
        from ocr_kb.kb.store import save
        from ocr_kb.kb.exporter import export_obsidian
        save(conn, _entry(source_path="thesis.pdf", page_number=1))
        paths = export_obsidian(conn, tmp_path / "vault")
        assert len(paths) == 1
        assert paths[0].exists()

    def test_file_under_ocr_kb_subfolder(self, conn, tmp_path):
        from ocr_kb.kb.store import save
        from ocr_kb.kb.exporter import export_obsidian
        save(conn, _entry(source_path="thesis.pdf", page_number=2))
        paths = export_obsidian(conn, tmp_path / "vault")
        assert "ocr_kb" in str(paths[0])

    def test_yaml_frontmatter_present(self, conn, tmp_path):
        from ocr_kb.kb.store import save
        from ocr_kb.kb.exporter import export_obsidian
        save(conn, _entry(category="biology"))
        paths = export_obsidian(conn, tmp_path / "vault")
        content = paths[0].read_text()
        assert content.startswith("---")
        assert "category:" in content
        assert "biology" in content


# ---------------------------------------------------------------------------
# Upsert (deduplication on re-ingest)
# ---------------------------------------------------------------------------

class TestUpsert:
    def test_second_save_updates_not_duplicates(self, conn):
        from ocr_kb.kb.store import get_all, save
        save(conn, _entry(raw_text="original text"))
        save(conn, _entry(raw_text="updated text"))
        all_entries = get_all(conn)
        assert len(all_entries) == 1
        assert all_entries[0].raw_text == "updated text"

    def test_upsert_preserves_id(self, conn):
        from ocr_kb.kb.store import get_all, save
        e1 = save(conn, _entry(raw_text="first"))
        e2 = save(conn, _entry(raw_text="second"))
        assert e1.id == e2.id

    def test_different_page_creates_new_entry(self, conn):
        from ocr_kb.kb.store import get_all, save
        save(conn, _entry(page_number=1))
        save(conn, _entry(page_number=2))
        assert len(get_all(conn)) == 2

    def test_different_source_creates_new_entry(self, conn):
        from ocr_kb.kb.store import get_all, save
        save(conn, _entry(source_path="a.pdf", page_number=1))
        save(conn, _entry(source_path="b.pdf", page_number=1))
        assert len(get_all(conn)) == 2


# ---------------------------------------------------------------------------
# delete_by_source / get_recent / update_tags / update_category
# ---------------------------------------------------------------------------

class TestStoreExtras:
    def test_delete_by_source_removes_all_pages(self, conn):
        from ocr_kb.kb.store import delete_by_source, get_all, save
        save(conn, _entry(source_path="paper.pdf", page_number=1))
        save(conn, _entry(source_path="paper.pdf", page_number=2))
        save(conn, _entry(source_path="other.pdf", page_number=1))
        count = delete_by_source(conn, "paper.pdf")
        assert count == 2
        remaining = get_all(conn)
        assert len(remaining) == 1
        assert remaining[0].source_path == "other.pdf"

    def test_delete_by_source_nonexistent_returns_zero(self, conn):
        from ocr_kb.kb.store import delete_by_source
        assert delete_by_source(conn, "ghost.pdf") == 0

    def test_get_recent_newest_first(self, conn):
        from ocr_kb.kb.store import get_recent, save
        from datetime import datetime, timezone
        save(conn, _entry(source_path="a.pdf", page_number=1,
                          raw_text="older", markdown="older"))
        save(conn, _entry(source_path="b.pdf", page_number=1,
                          raw_text="newer", markdown="newer"))
        recent = get_recent(conn, limit=2)
        assert recent[0].source_path == "b.pdf"

    def test_get_recent_limit(self, conn):
        from ocr_kb.kb.store import get_recent, save
        for i in range(5):
            save(conn, _entry(source_path=f"f{i}.pdf", page_number=1,
                              raw_text=f"text {i}", markdown=f"text {i}"))
        assert len(get_recent(conn, limit=3)) == 3

    def test_update_tags_changes_value(self, conn):
        from ocr_kb.kb.store import get, save, update_tags
        e = save(conn, _entry(tags="old"))
        update_tags(conn, e.id, "new,tags")
        assert get(conn, e.id).tags == "new,tags"

    def test_update_tags_nonexistent_returns_false(self, conn):
        from ocr_kb.kb.store import update_tags
        assert update_tags(conn, 9999, "x") is False

    def test_update_category_changes_value(self, conn):
        from ocr_kb.kb.store import get, save, update_category
        e = save(conn, _entry(category="old"))
        update_category(conn, e.id, "science")
        assert get(conn, e.id).category == "science"


# ---------------------------------------------------------------------------
# filtered_search
# ---------------------------------------------------------------------------

class TestFilteredSearch:
    def test_filter_by_source(self, conn):
        from ocr_kb.kb.store import save
        from ocr_kb.kb.indexer import filtered_search
        save(conn, _entry(source_path="paper_a.pdf", raw_text="quantum computing basics",
                          markdown="quantum computing basics"))
        save(conn, _entry(source_path="paper_b.pdf", page_number=2,
                          raw_text="quantum entanglement",
                          markdown="quantum entanglement"))
        results = filtered_search(conn, "quantum", source="paper_a", limit=10)
        assert len(results) == 1
        assert "paper_a" in results[0].source_path

    def test_filter_by_category(self, conn):
        from ocr_kb.kb.store import save
        from ocr_kb.kb.indexer import filtered_search
        save(conn, _entry(raw_text="neural network training", markdown="neural network training",
                          category="ml"))
        save(conn, _entry(page_number=2, raw_text="neural network training",
                          markdown="neural network training", category="bio"))
        results = filtered_search(conn, "neural", category="ml", limit=10)
        assert all(e.category == "ml" for e in results)

    def test_filter_by_tag(self, conn):
        from ocr_kb.kb.store import save
        from ocr_kb.kb.indexer import filtered_search
        save(conn, _entry(raw_text="deep learning methods",
                          markdown="deep learning methods", tags="ml,ai"))
        save(conn, _entry(page_number=2, raw_text="deep learning methods",
                          markdown="deep learning methods", tags="biology"))
        results = filtered_search(conn, "deep learning", tag="ml", limit=10)
        assert all("ml" in e.tags for e in results)

    def test_filter_by_after(self, conn):
        from ocr_kb.kb.store import save
        from ocr_kb.kb.indexer import filtered_search
        from datetime import datetime, timezone
        save(conn, _entry(raw_text="backpropagation algorithm",
                          markdown="backpropagation algorithm"))
        results = filtered_search(conn, "backpropagation", after="2099-01-01", limit=10)
        assert results == []

    def test_no_filters_behaves_like_fts_search(self, conn):
        from ocr_kb.kb.store import save
        from ocr_kb.kb.indexer import filtered_search, fts_search
        save(conn, _entry(raw_text="gradient descent optimization",
                          markdown="gradient descent optimization"))
        r1 = fts_search(conn, "gradient descent")
        r2 = filtered_search(conn, "gradient descent")
        assert len(r1) == len(r2)


# ---------------------------------------------------------------------------
# export with filters
# ---------------------------------------------------------------------------

class TestFilteredExport:
    def test_export_markdown_filter_tag(self, conn, tmp_path):
        from ocr_kb.kb.store import save
        from ocr_kb.kb.exporter import export_markdown
        save(conn, _entry(source_path="a.pdf", tags="ml"))
        save(conn, _entry(source_path="b.pdf", page_number=2, tags="bio"))
        paths = export_markdown(conn, tmp_path / "out", filter_tag="ml")
        assert len(paths) == 1
        assert "a" in paths[0].name

    def test_export_json_filter_category(self, conn, tmp_path):
        from ocr_kb.kb.store import save
        from ocr_kb.kb.exporter import export_json
        import json
        save(conn, _entry(source_path="a.pdf", category="science"))
        save(conn, _entry(source_path="b.pdf", page_number=2, category="history"))
        out = tmp_path / "out.json"
        export_json(conn, out, filter_category="science")
        data = json.loads(out.read_text())
        assert len(data) == 1
        assert data[0]["category"] == "science"

    def test_export_obsidian_no_filter_exports_all(self, conn, tmp_path):
        from ocr_kb.kb.store import save
        from ocr_kb.kb.exporter import export_obsidian
        save(conn, _entry(source_path="a.pdf"))
        save(conn, _entry(source_path="b.pdf", page_number=2,
                          raw_text="other text", markdown="other text"))
        paths = export_obsidian(conn, tmp_path / "vault")
        assert len(paths) == 2


# ---------------------------------------------------------------------------
# chunk_index: store and upsert behaviour
# ---------------------------------------------------------------------------

class TestChunkIndex:
    def test_chunk_index_defaults_to_zero(self, conn):
        from ocr_kb.kb.store import get, save
        e = save(conn, _entry())
        assert get(conn, e.id).chunk_index == 0

    def test_chunk_index_round_trip(self, conn):
        from ocr_kb.kb.store import get, save
        e = save(conn, _entry(chunk_index=3))
        assert get(conn, e.id).chunk_index == 3

    def test_different_chunk_index_creates_separate_entries(self, conn):
        from ocr_kb.kb.store import get_all, save
        save(conn, _entry(chunk_index=1, raw_text="chunk one"))
        save(conn, _entry(chunk_index=2, raw_text="chunk two"))
        entries = get_all(conn)
        assert len(entries) == 2

    def test_upsert_matches_on_chunk_index(self, conn):
        from ocr_kb.kb.store import get_all, save
        save(conn, _entry(chunk_index=1, raw_text="original"))
        save(conn, _entry(chunk_index=1, raw_text="updated"))
        entries = get_all(conn)
        assert len(entries) == 1
        assert entries[0].raw_text == "updated"

    def test_upsert_different_chunk_indices_do_not_collide(self, conn):
        from ocr_kb.kb.store import get_all, save
        save(conn, _entry(chunk_index=0, raw_text="full page"))
        save(conn, _entry(chunk_index=1, raw_text="first chunk"))
        save(conn, _entry(chunk_index=2, raw_text="second chunk"))
        assert len(get_all(conn)) == 3


# ---------------------------------------------------------------------------
# hybrid_search
# ---------------------------------------------------------------------------

class TestHybridSearch:
    def test_returns_fts_results_when_embeddings_disabled(self, conn):
        from ocr_kb.kb.indexer import hybrid_search
        from ocr_kb.kb.store import save
        from ocr_kb.settings import Settings

        s = Settings(_env_file=None, enable_embeddings=False)
        save(conn, _entry(raw_text="gradient descent converges fast"))
        results = hybrid_search(conn, "gradient", s)
        assert len(results) >= 1
        assert any("gradient" in e.raw_text for e in results)

    def test_returns_empty_when_no_match(self, conn):
        from ocr_kb.kb.indexer import hybrid_search
        from ocr_kb.kb.store import save
        from ocr_kb.settings import Settings

        s = Settings(_env_file=None, enable_embeddings=False)
        save(conn, _entry(raw_text="entirely unrelated content"))
        results = hybrid_search(conn, "xyzzy999", s)
        assert results == []

    def test_limit_respected(self, conn):
        from ocr_kb.kb.indexer import hybrid_search
        from ocr_kb.kb.store import save
        from ocr_kb.settings import Settings

        s = Settings(_env_file=None, enable_embeddings=False)
        for i in range(8):
            save(conn, _entry(raw_text=f"common word entry {i}", page_number=i))
        results = hybrid_search(conn, "common word", s, limit=3)
        assert len(results) <= 3

    def test_merges_semantic_results_when_embeddings_enabled(self, conn):
        from unittest.mock import patch
        from ocr_kb.kb.indexer import hybrid_search
        from ocr_kb.kb.store import save
        from ocr_kb.settings import Settings

        s = Settings(_env_file=None, enable_embeddings=True,
                     embedding_model="all-MiniLM-L6-v2")
        # Entry only in FTS
        save(conn, _entry(page_number=1, raw_text="neural nets converge"))
        # Entry only in semantic (no FTS match)
        save(conn, _entry(page_number=2, raw_text="xyzzy placeholder text",
                          embedding=[1.0, 0.0, 0.0]))

        fake_emb = [1.0, 0.0, 0.0]
        with patch("ocr_kb.model.embedder.embed_text", return_value=fake_emb):
            results = hybrid_search(conn, "neural nets", s, limit=10)

        assert len(results) >= 1


# ---------------------------------------------------------------------------
# Dead-Letter Queue
# ---------------------------------------------------------------------------

class TestDlq:
    def test_dlq_push_creates_record(self, conn):
        from ocr_kb.kb.store import dlq_push, dlq_list
        dlq_push(conn, "/data/doc.pdf", "timeout error")
        items = dlq_list(conn)
        assert len(items) == 1
        assert items[0]["source_path"] == "/data/doc.pdf"
        assert items[0]["retry_count"] == 0

    def test_dlq_push_increments_retry_count(self, conn):
        from ocr_kb.kb.store import dlq_push, dlq_list
        dlq_push(conn, "/data/doc.pdf", "error 1")
        dlq_push(conn, "/data/doc.pdf", "error 2")
        items = dlq_list(conn)
        assert len(items) == 1
        assert items[0]["retry_count"] == 1

    def test_dlq_list_returns_only_unresolved(self, conn):
        from ocr_kb.kb.store import dlq_push, dlq_list, dlq_mark_resolved
        dlq_push(conn, "/data/a.pdf", "error")
        dlq_push(conn, "/data/b.pdf", "error")
        dlq_mark_resolved(conn, "/data/a.pdf")
        items = dlq_list(conn)
        assert len(items) == 1
        assert items[0]["source_path"] == "/data/b.pdf"

    def test_dlq_mark_resolved_returns_true_when_found(self, conn):
        from ocr_kb.kb.store import dlq_push, dlq_mark_resolved
        dlq_push(conn, "/data/doc.pdf", "error")
        assert dlq_mark_resolved(conn, "/data/doc.pdf") is True

    def test_dlq_mark_resolved_returns_false_when_not_found(self, conn):
        from ocr_kb.kb.store import dlq_mark_resolved
        assert dlq_mark_resolved(conn, "/data/ghost.pdf") is False

    def test_dlq_get_retryable_respects_max_retries(self, conn):
        from ocr_kb.kb.store import dlq_push, dlq_get_retryable
        dlq_push(conn, "/data/doc.pdf", "error", max_retries=1)
        dlq_push(conn, "/data/doc.pdf", "error2")
        retryable = dlq_get_retryable(conn)
        assert retryable == []

    def test_dlq_get_retryable_includes_eligible_items(self, conn):
        from ocr_kb.kb.store import dlq_push, dlq_get_retryable
        dlq_push(conn, "/data/doc.pdf", "error", max_retries=3)
        retryable = dlq_get_retryable(conn)
        assert len(retryable) == 1

    def test_dlq_empty_initially(self, conn):
        from ocr_kb.kb.store import dlq_list
        assert dlq_list(conn) == []
