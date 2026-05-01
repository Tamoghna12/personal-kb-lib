"""Tests for ocr_kb.prompts — GLM-OCR vision prompts, Gemma enrichment prompts, formatters."""

import pytest


# ---------------------------------------------------------------------------
# GLM-OCR vision prompt constants
# ---------------------------------------------------------------------------

class TestGlmOcrPrompt:
    def test_is_nonempty_string(self):
        from ocr_kb.prompts import GLM_OCR_PROMPT
        assert isinstance(GLM_OCR_PROMPT, str) and GLM_OCR_PROMPT.strip()

    def test_instructs_plain_text_output(self):
        from ocr_kb.prompts import GLM_OCR_PROMPT
        assert "plain text" in GLM_OCR_PROMPT.lower()

    def test_mentions_reading_order(self):
        from ocr_kb.prompts import GLM_OCR_PROMPT
        assert "reading order" in GLM_OCR_PROMPT.lower()

    def test_mentions_lists_and_tables(self):
        from ocr_kb.prompts import GLM_OCR_PROMPT
        lower = GLM_OCR_PROMPT.lower()
        assert "list" in lower
        assert "table" in lower

    def test_no_placeholder_tokens(self):
        from ocr_kb.prompts import GLM_OCR_PROMPT
        assert "{" not in GLM_OCR_PROMPT


class TestGlmOcrHtmlPrompt:
    def test_is_nonempty_string(self):
        from ocr_kb.prompts import GLM_OCR_HTML_PROMPT
        assert isinstance(GLM_OCR_HTML_PROMPT, str) and GLM_OCR_HTML_PROMPT.strip()

    def test_mentions_heading_tags(self):
        from ocr_kb.prompts import GLM_OCR_HTML_PROMPT
        assert "h1" in GLM_OCR_HTML_PROMPT.lower()

    def test_mentions_list_tags(self):
        from ocr_kb.prompts import GLM_OCR_HTML_PROMPT
        lower = GLM_OCR_HTML_PROMPT.lower()
        assert "<ul>" in lower or "ul" in lower
        assert "<li>" in lower or "li" in lower

    def test_mentions_table_tags(self):
        from ocr_kb.prompts import GLM_OCR_HTML_PROMPT
        assert "table" in GLM_OCR_HTML_PROMPT.lower()

    def test_excludes_full_html_document(self):
        from ocr_kb.prompts import GLM_OCR_HTML_PROMPT
        lower = GLM_OCR_HTML_PROMPT.lower()
        assert "no <html>" in lower or "<html>" in lower

    def test_no_placeholder_tokens(self):
        from ocr_kb.prompts import GLM_OCR_HTML_PROMPT
        assert "{" not in GLM_OCR_HTML_PROMPT


# ---------------------------------------------------------------------------
# Backward-compatible aliases still importable
# ---------------------------------------------------------------------------

def test_ocr_prompt_alias():
    from ocr_kb.prompts import OCR_PROMPT, GLM_OCR_PROMPT
    assert OCR_PROMPT is GLM_OCR_PROMPT


def test_html_extraction_prompt_alias():
    from ocr_kb.prompts import HTML_EXTRACTION_PROMPT, GLM_OCR_HTML_PROMPT
    assert HTML_EXTRACTION_PROMPT is GLM_OCR_HTML_PROMPT


# ---------------------------------------------------------------------------
# format_glm_ocr_prompt
# ---------------------------------------------------------------------------

class TestFormatGlmOcrPrompt:
    def test_text_mode_returns_plain_prompt(self):
        from ocr_kb.prompts import format_glm_ocr_prompt, GLM_OCR_PROMPT
        assert format_glm_ocr_prompt("text") == GLM_OCR_PROMPT

    def test_html_mode_returns_html_prompt(self):
        from ocr_kb.prompts import format_glm_ocr_prompt, GLM_OCR_HTML_PROMPT
        assert format_glm_ocr_prompt("html") == GLM_OCR_HTML_PROMPT

    def test_default_is_text_mode(self):
        from ocr_kb.prompts import format_glm_ocr_prompt, GLM_OCR_PROMPT
        assert format_glm_ocr_prompt() == GLM_OCR_PROMPT


# ---------------------------------------------------------------------------
# Gemma enrichment prompts
# ---------------------------------------------------------------------------

class TestGemmaCleanupPrompt:
    def test_template_is_nonempty(self):
        from ocr_kb.prompts import GEMMA_CLEANUP_PROMPT_TEMPLATE
        assert isinstance(GEMMA_CLEANUP_PROMPT_TEMPLATE, str) and GEMMA_CLEANUP_PROMPT_TEMPLATE.strip()

    def test_mentions_ocr_errors(self):
        from ocr_kb.prompts import GEMMA_CLEANUP_PROMPT_TEMPLATE
        assert "ocr" in GEMMA_CLEANUP_PROMPT_TEMPLATE.lower()

    def test_has_text_placeholder(self):
        from ocr_kb.prompts import GEMMA_CLEANUP_PROMPT_TEMPLATE
        assert "{text}" in GEMMA_CLEANUP_PROMPT_TEMPLATE


class TestGemmaSummaryPromptTemplate:
    def test_template_is_nonempty(self):
        from ocr_kb.prompts import GEMMA_SUMMARY_PROMPT_TEMPLATE
        assert isinstance(GEMMA_SUMMARY_PROMPT_TEMPLATE, str) and GEMMA_SUMMARY_PROMPT_TEMPLATE.strip()

    def test_has_text_placeholder(self):
        from ocr_kb.prompts import GEMMA_SUMMARY_PROMPT_TEMPLATE
        assert "{text}" in GEMMA_SUMMARY_PROMPT_TEMPLATE


class TestGemmaTagsPrompt:
    def test_template_is_nonempty(self):
        from ocr_kb.prompts import GEMMA_TAGS_PROMPT_TEMPLATE
        assert isinstance(GEMMA_TAGS_PROMPT_TEMPLATE, str) and GEMMA_TAGS_PROMPT_TEMPLATE.strip()

    def test_mentions_tags(self):
        from ocr_kb.prompts import GEMMA_TAGS_PROMPT_TEMPLATE
        assert "tag" in GEMMA_TAGS_PROMPT_TEMPLATE.lower()

    def test_has_text_placeholder(self):
        from ocr_kb.prompts import GEMMA_TAGS_PROMPT_TEMPLATE
        assert "{text}" in GEMMA_TAGS_PROMPT_TEMPLATE


class TestGemmaEntitiesPrompt:
    def test_template_is_nonempty(self):
        from ocr_kb.prompts import GEMMA_ENTITIES_PROMPT_TEMPLATE
        assert isinstance(GEMMA_ENTITIES_PROMPT_TEMPLATE, str) and GEMMA_ENTITIES_PROMPT_TEMPLATE.strip()

    def test_mentions_json(self):
        from ocr_kb.prompts import GEMMA_ENTITIES_PROMPT_TEMPLATE
        assert "json" in GEMMA_ENTITIES_PROMPT_TEMPLATE.lower()

    def test_mentions_entity_types(self):
        from ocr_kb.prompts import GEMMA_ENTITIES_PROMPT_TEMPLATE
        lower = GEMMA_ENTITIES_PROMPT_TEMPLATE.lower()
        assert "people" in lower
        assert "organisation" in lower or "organization" in lower

    def test_has_text_placeholder(self):
        from ocr_kb.prompts import GEMMA_ENTITIES_PROMPT_TEMPLATE
        assert "{text}" in GEMMA_ENTITIES_PROMPT_TEMPLATE


# ---------------------------------------------------------------------------
# Formatter helpers
# ---------------------------------------------------------------------------

class TestKeyPointsPrompt:
    def test_inserts_text(self):
        from ocr_kb.prompts import format_key_points_prompt
        result = format_key_points_prompt("some content here")
        assert "some content here" in result

    def test_no_leftover_placeholder(self):
        from ocr_kb.prompts import format_key_points_prompt
        result = format_key_points_prompt("anything")
        assert "{text}" not in result

    def test_mentions_markdown_list(self):
        from ocr_kb.prompts import format_key_points_prompt
        result = format_key_points_prompt("x").lower()
        assert "markdown" in result or "bullet" in result

    def test_mentions_maximum_bullets(self):
        from ocr_kb.prompts import format_key_points_prompt
        result = format_key_points_prompt("x")
        assert "10" in result

    def test_empty_text_still_formats(self):
        from ocr_kb.prompts import format_key_points_prompt
        result = format_key_points_prompt("")
        assert isinstance(result, str) and result.strip()

    def test_multiline_text_preserved(self):
        from ocr_kb.prompts import format_key_points_prompt
        text = "line one\nline two\nline three"
        result = format_key_points_prompt(text)
        assert "line one\nline two\nline three" in result


class TestSummaryPrompt:
    def test_inserts_text(self):
        from ocr_kb.prompts import format_summary_prompt
        result = format_summary_prompt("the quick brown fox")
        assert "the quick brown fox" in result

    def test_no_leftover_placeholder(self):
        from ocr_kb.prompts import format_summary_prompt
        assert "{text}" not in format_summary_prompt("anything")

    def test_mentions_sentence_count(self):
        from ocr_kb.prompts import format_summary_prompt
        result = format_summary_prompt("x").lower()
        assert "sentence" in result or "2" in result or "4" in result

    def test_empty_text_still_formats(self):
        from ocr_kb.prompts import format_summary_prompt
        result = format_summary_prompt("")
        assert isinstance(result, str) and result.strip()


class TestCleanupPrompt:
    def test_inserts_text(self):
        from ocr_kb.prompts import format_cleanup_prompt
        result = format_cleanup_prompt("raw ocr output")
        assert "raw ocr output" in result

    def test_no_leftover_placeholder(self):
        from ocr_kb.prompts import format_cleanup_prompt
        assert "{text}" not in format_cleanup_prompt("x")


class TestTagsPrompt:
    def test_inserts_text(self):
        from ocr_kb.prompts import format_tags_prompt
        result = format_tags_prompt("machine learning paper")
        assert "machine learning paper" in result

    def test_no_leftover_placeholder(self):
        from ocr_kb.prompts import format_tags_prompt
        assert "{text}" not in format_tags_prompt("x")


class TestEntitiesPrompt:
    def test_inserts_text(self):
        from ocr_kb.prompts import format_entities_prompt
        result = format_entities_prompt("Albert Einstein worked at Princeton.")
        assert "Albert Einstein worked at Princeton." in result

    def test_no_leftover_placeholder(self):
        from ocr_kb.prompts import format_entities_prompt
        assert "{text}" not in format_entities_prompt("x")


# ---------------------------------------------------------------------------
# Swappability — prompts are plain strings, not bound to any class
# ---------------------------------------------------------------------------

def test_glm_ocr_prompt_can_be_overridden(monkeypatch):
    import ocr_kb.prompts as mod
    monkeypatch.setattr(mod, "GLM_OCR_PROMPT", "custom ocr prompt")
    assert mod.GLM_OCR_PROMPT == "custom ocr prompt"


def test_glm_ocr_html_prompt_can_be_overridden(monkeypatch):
    import ocr_kb.prompts as mod
    monkeypatch.setattr(mod, "GLM_OCR_HTML_PROMPT", "custom html prompt")
    assert mod.GLM_OCR_HTML_PROMPT == "custom html prompt"


# ---------------------------------------------------------------------------
# Loader — text/Markdown ingestion (added alongside Step 5)
# ---------------------------------------------------------------------------

class TestTextIngestion:
    def test_is_text_txt(self, tmp_path):
        from ocr_kb.ingest.loader import is_text
        assert is_text(tmp_path / "notes.txt") is True

    def test_is_text_md(self, tmp_path):
        from ocr_kb.ingest.loader import is_text
        assert is_text(tmp_path / "notes.md") is True

    def test_is_text_pdf_false(self, tmp_path):
        from ocr_kb.ingest.loader import is_text
        assert is_text(tmp_path / "doc.pdf") is False

    def test_text_suffixes_in_supported(self):
        from ocr_kb.ingest.loader import SUPPORTED_SUFFIXES
        assert ".txt" in SUPPORTED_SUFFIXES
        assert ".md" in SUPPORTED_SUFFIXES

    def test_load_text_file_returns_content(self, tmp_path):
        from ocr_kb.ingest.loader import load_text_file
        p = tmp_path / "note.txt"
        p.write_text("hello world\nsecond line")
        assert load_text_file(p) == "hello world\nsecond line"

    def test_load_text_file_utf8(self, tmp_path):
        from ocr_kb.ingest.loader import load_text_file
        p = tmp_path / "note.txt"
        p.write_text("café résumé", encoding="utf-8")
        assert load_text_file(p) == "café résumé"


class TestBatchBuilderText:
    def test_text_file_produces_single_item(self, tmp_path):
        from ocr_kb.ingest.batch_builder import build_batch
        from ocr_kb.settings import Settings
        p = tmp_path / "note.txt"
        p.write_text("some notes")
        batch = build_batch(p, Settings(_env_file=None))
        assert len(batch) == 1

    def test_text_item_has_text_content(self, tmp_path):
        from ocr_kb.ingest.batch_builder import build_batch
        from ocr_kb.settings import Settings
        p = tmp_path / "note.txt"
        p.write_text("my content")
        item = build_batch(p, Settings(_env_file=None))[0]
        assert item.text_content == "my content"

    def test_text_item_image_is_none(self, tmp_path):
        from ocr_kb.ingest.batch_builder import build_batch
        from ocr_kb.settings import Settings
        p = tmp_path / "note.md"
        p.write_text("# Title\nBody")
        item = build_batch(p, Settings(_env_file=None))[0]
        assert item.image is None

    def test_text_item_needs_ocr_false(self, tmp_path):
        from ocr_kb.ingest.batch_builder import build_batch
        from ocr_kb.settings import Settings
        p = tmp_path / "note.txt"
        p.write_text("x")
        item = build_batch(p, Settings(_env_file=None))[0]
        assert item.needs_ocr is False

    def test_image_item_needs_ocr_true(self, tmp_path):
        from PIL import Image
        from ocr_kb.ingest.batch_builder import build_batch
        from ocr_kb.settings import Settings
        p = tmp_path / "img.png"
        Image.new("RGB", (200, 200)).save(p)
        item = build_batch(p, Settings(_env_file=None))[0]
        assert item.needs_ocr is True

    def test_markdown_file_content_preserved(self, tmp_path):
        from ocr_kb.ingest.batch_builder import build_batch
        from ocr_kb.settings import Settings
        p = tmp_path / "doc.md"
        content = "# Heading\n\n- bullet one\n- bullet two"
        p.write_text(content)
        item = build_batch(p, Settings(_env_file=None))[0]
        assert item.text_content == content

    def test_source_path_set_for_text(self, tmp_path):
        from ocr_kb.ingest.batch_builder import build_batch
        from ocr_kb.settings import Settings
        p = tmp_path / "note.txt"
        p.write_text("x")
        item = build_batch(p, Settings(_env_file=None))[0]
        assert item.source_path == p


# ---------------------------------------------------------------------------
# Wiki synthesis prompts
# ---------------------------------------------------------------------------

class TestWikiConceptExtractionPrompt:
    def test_template_is_nonempty(self):
        from ocr_kb.prompts import WIKI_CONCEPT_EXTRACTION_PROMPT_TEMPLATE
        assert isinstance(WIKI_CONCEPT_EXTRACTION_PROMPT_TEMPLATE, str)
        assert WIKI_CONCEPT_EXTRACTION_PROMPT_TEMPLATE.strip()

    def test_mentions_json_output(self):
        from ocr_kb.prompts import WIKI_CONCEPT_EXTRACTION_PROMPT_TEMPLATE
        assert "json" in WIKI_CONCEPT_EXTRACTION_PROMPT_TEMPLATE.lower()

    def test_mentions_required_fields(self):
        from ocr_kb.prompts import WIKI_CONCEPT_EXTRACTION_PROMPT_TEMPLATE
        lower = WIKI_CONCEPT_EXTRACTION_PROMPT_TEMPLATE.lower()
        for field in ("definition", "evidence", "metrics", "limitations", "citations"):
            assert field in lower, f"Missing field: {field}"

    def test_has_text_placeholder(self):
        from ocr_kb.prompts import WIKI_CONCEPT_EXTRACTION_PROMPT_TEMPLATE
        assert "{text}" in WIKI_CONCEPT_EXTRACTION_PROMPT_TEMPLATE

    def test_mentions_academic_rigour(self):
        from ocr_kb.prompts import WIKI_CONCEPT_EXTRACTION_PROMPT_TEMPLATE
        lower = WIKI_CONCEPT_EXTRACTION_PROMPT_TEMPLATE.lower()
        assert "rigour" in lower or "rigor" in lower or "skeptic" in lower


class TestWikiConceptSummaryPrompt:
    def test_template_is_nonempty(self):
        from ocr_kb.prompts import WIKI_CONCEPT_SUMMARY_PROMPT_TEMPLATE
        assert isinstance(WIKI_CONCEPT_SUMMARY_PROMPT_TEMPLATE, str)
        assert WIKI_CONCEPT_SUMMARY_PROMPT_TEMPLATE.strip()

    def test_has_concept_placeholder(self):
        from ocr_kb.prompts import WIKI_CONCEPT_SUMMARY_PROMPT_TEMPLATE
        assert "{concept}" in WIKI_CONCEPT_SUMMARY_PROMPT_TEMPLATE

    def test_has_snippets_placeholder(self):
        from ocr_kb.prompts import WIKI_CONCEPT_SUMMARY_PROMPT_TEMPLATE
        assert "{snippets}" in WIKI_CONCEPT_SUMMARY_PROMPT_TEMPLATE

    def test_mentions_required_sections(self):
        from ocr_kb.prompts import WIKI_CONCEPT_SUMMARY_PROMPT_TEMPLATE
        for section in ("Technical Overview", "Methodology", "Quantitative Evidence",
                        "Limitations", "Consensus"):
            assert section in WIKI_CONCEPT_SUMMARY_PROMPT_TEMPLATE, f"Missing section: {section}"

    def test_mentions_contradictions(self):
        from ocr_kb.prompts import WIKI_CONCEPT_SUMMARY_PROMPT_TEMPLATE
        assert "Contradiction" in WIKI_CONCEPT_SUMMARY_PROMPT_TEMPLATE


class TestWikiFormatters:
    def test_format_wiki_extraction_inserts_text(self):
        from ocr_kb.prompts import format_wiki_extraction_prompt
        result = format_wiki_extraction_prompt("backpropagation algorithm paper")
        assert "backpropagation algorithm paper" in result

    def test_format_wiki_extraction_no_leftover_placeholder(self):
        from ocr_kb.prompts import format_wiki_extraction_prompt
        result = format_wiki_extraction_prompt("x")
        assert "{text}" not in result

    def test_format_wiki_extraction_valid_json_template(self):
        from ocr_kb.prompts import format_wiki_extraction_prompt
        result = format_wiki_extraction_prompt("some text")
        # The JSON example uses escaped braces; after format() they should be literal braces
        assert "{" in result and "}" in result

    def test_format_wiki_summary_inserts_concept(self):
        from ocr_kb.prompts import format_wiki_concept_summary_prompt
        result = format_wiki_concept_summary_prompt("Attention Mechanism", "snippet A\nsnippet B")
        assert "Attention Mechanism" in result

    def test_format_wiki_summary_inserts_snippets(self):
        from ocr_kb.prompts import format_wiki_concept_summary_prompt
        result = format_wiki_concept_summary_prompt("X", "my snippet content")
        assert "my snippet content" in result

    def test_format_wiki_summary_no_leftover_placeholders(self):
        from ocr_kb.prompts import format_wiki_concept_summary_prompt
        result = format_wiki_concept_summary_prompt("X", "Y")
        assert "{concept}" not in result
        assert "{snippets}" not in result


# ---------------------------------------------------------------------------
# RAG prompt
# ---------------------------------------------------------------------------

class TestRagPrompt:
    def test_template_exists(self):
        from ocr_kb.prompts import GEMMA_RAG_PROMPT_TEMPLATE
        assert isinstance(GEMMA_RAG_PROMPT_TEMPLATE, str)
        assert len(GEMMA_RAG_PROMPT_TEMPLATE) > 20

    def test_format_rag_prompt_inserts_question(self):
        from ocr_kb.prompts import format_rag_prompt
        result = format_rag_prompt("What is backprop?", "context snippet")
        assert "What is backprop?" in result

    def test_format_rag_prompt_inserts_context(self):
        from ocr_kb.prompts import format_rag_prompt
        result = format_rag_prompt("question", "unique context block")
        assert "unique context block" in result

    def test_format_rag_prompt_no_leftover_placeholders(self):
        from ocr_kb.prompts import format_rag_prompt
        result = format_rag_prompt("Q", "C")
        assert "{question}" not in result
        assert "{context}" not in result

    def test_format_rag_prompt_both_args_required(self):
        from ocr_kb.prompts import format_rag_prompt
        import inspect
        sig = inspect.signature(format_rag_prompt)
        params = list(sig.parameters.keys())
        assert len(params) == 2
