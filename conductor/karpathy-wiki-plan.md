# Karpathy LLM Wiki Implementation Plan

## Objective
Evolve the existing OCR knowledge base into a structured, interconnected "LLM Wiki" based on Andrej Karpathy's Idea File. This will be achieved using a **Post-Processing Compiler** approach to maintain fast ingestion speeds while keeping the **existing directory structure** (`kb_data/markdown/`).

## Scope & Impact
This transition separates raw data ingestion from knowledge synthesis. The system will extract entities, generate dedicated concept pages, and automatically maintain an index and activity log.

## Key Files & Context
- `CLAUDE.md` / `WIKI_SCHEMA.md`: The central instruction file for the LLM outlining wiki generation conventions.
- `ocr_kb/prompts.py`: New prompts for entity extraction and wiki page formatting.
- `ocr_kb/wiki/compiler.py` (New): Core logic for batch-processing database entries into wiki pages.
- `ocr_kb/wiki/linter.py` (New): Core logic for health-checking the wiki.
- `ocr_kb/cli.py`: New `compile-wiki` and `lint-wiki` commands.
- `kb_data/markdown/`: The target directory for all generated wiki content (sources, concepts, index, log).

## Implementation Steps

### Phase 1: Schema and Prompts
1. **Define Schema:** Create a `WIKI_SCHEMA.md` (or update `CLAUDE.md`) in the project root to instruct the LLM on wiki conventions:
   - Mandatory YAML frontmatter (`title`, `type`, `sources`, `related`, `created`, `updated`).
   - Page types: `source-summary`, `concept`, `entity`.
2. **Design Extraction Prompts:** Add specific prompts to `ocr_kb/prompts.py` to instruct the `gemma-4-e4b` model (via LM Studio) to read raw ingested text and extract a structured JSON list of key concepts and entities.

### Phase 2: The Post-Processing Compiler
1. **Create Compilation Logic (`ocr_kb/wiki/compiler.py`):**
   - Fetch recently ingested pages from the SQLite database.
   - Run the LLM entity extraction prompt over the raw text to identify key concepts (e.g., "Neural Networks", "Backpropagation").
   - For each extracted concept, generate or update a dedicated Markdown file (e.g., `kb_data/markdown/concepts/neural-networks.md`) summarizing the concept and creating backlinks to the specific source pages (e.g., `[deep_learning_p001](../deep_learning_p001.md)`).
2. **Index and Log Maintenance:**
   - Automatically regenerate `kb_data/markdown/index.md` to list all concepts, entities, and sources.
   - Append a timestamped entry to `kb_data/markdown/log.md` detailing what the compiler updated.

### Phase 3: CLI Integration and Linting
1. **Add `compile-wiki` Command:** Expose the compiler logic via a new Typer command in `ocr_kb/cli.py`.
2. **Create Linter Logic (`ocr_kb/wiki/linter.py`):**
   - Implement logic to scan `kb_data/markdown/` for broken links, orphan pages (no inbound links), and missing concept pages.
   - Optionally, pass the wiki structure to the LLM to identify contradictions or suggest new topics to research.
3. **Add `lint-wiki` Command:** Expose the linter via `ocr-kb lint-wiki`.

## Verification & Testing
- Unit tests in `tests/test_wiki.py` to verify the compiler correctly generates markdown files with valid YAML frontmatter and relative links.
- Test `compile-wiki` on the existing `deep_learning.pdf` data to ensure concept pages (like "Deep Learning", "Supervised Learning") are generated and linked correctly to the source pages.
- Test `lint-wiki` to ensure it successfully detects an intentionally orphaned page.

## Rollback Strategy
If the compiler introduces breaking changes to the markdown files, the user can delete the `kb_data/markdown/concepts/` directory, `index.md`, and `log.md`, and re-run `ocr-kb export --format markdown` to restore the raw source pages from the pristine SQLite database.